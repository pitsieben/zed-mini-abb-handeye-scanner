"""
3D Gaussian Splatting / NeRF Dataset Exporter
==============================================
Converts a ZED + ABB Robot scan session into a dataset ready for:
  - 3D Gaussian Splatting (gaussian-splatting repo, COLMAP format)
  - Nerfstudio / gsplat / instant-NGP (transforms.json format)

Usage:
    python tools/export_3dgs_dataset.py --session scans/session_stages [--out scans/3dgs_dataset]

Output structure:
    scans/3dgs_dataset/
        images/                  <- RGB frames for each stage
            stage_001.png
            ...
        sparse/0/                <- COLMAP sparse text format
            cameras.txt
            images.txt
            points3D.txt
        transforms.json          <- NeRF / instant-NGP / nerfstudio format
        HOWTO_3DGS.md            <- Quick-start training instructions

After export, run Gaussian Splatting training with:
    python train.py -s scans/3dgs_dataset --model_path output/my_scan
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.open3d_pipeline import Open3DScannerPipeline


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def export_3dgs_dataset(session_dir, output_dir, calib_path="calibration_result.npz"):
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    stages = manifest.get("stages", [])
    if not stages:
        raise ValueError("No stages found in manifest.")

    intr = manifest.get("intrinsics", {})
    fx = float(intr.get("fx", 700.0))
    fy = float(intr.get("fy", 700.0))
    cx = float(intr.get("cx", 640.0))
    cy = float(intr.get("cy", 360.0))
    w  = int(intr.get("width", 1280))
    h  = int(intr.get("height", 720))

    print(f"\n[3DGS EXPORT] Session: {session_dir}")
    print(f"[3DGS EXPORT] Output:  {output_dir}")
    print(f"[3DGS EXPORT] Stages:  {len(stages)}")
    print(f"[3DGS EXPORT] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} ({w}x{h})")

    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    pipeline = Open3DScannerPipeline(calib_path=calib_path)
    valid_stages = []

    for idx, stage in enumerate(stages):
        rgb_file = stage.get("rgb")
        if not rgb_file:
            print(f"  [SKIP] Stage #{idx+1}: no RGB in manifest.")
            continue
        rgb_src = os.path.join(session_dir, rgb_file) if not os.path.isabs(rgb_file) else rgb_file
        if not os.path.exists(rgb_src):
            print(f"  [SKIP] Stage #{idx+1}: RGB file missing.")
            continue

        pose = stage["pose_flange"]
        T_cam2world = pipeline.compute_cam2base_matrix(*pose)
        T_world2cam = np.linalg.inv(T_cam2world)
        cam_center = T_cam2world[:3, 3]

        img_name = f"stage_{len(valid_stages)+1:03d}.png"
        shutil.copy2(rgb_src, os.path.join(images_dir, img_name))

        valid_stages.append({
            "image_name": img_name,
            "T_cam2world": T_cam2world,
            "T_world2cam": T_world2cam,
            "cam_center": cam_center,
            "ply": stage.get("ply"),
        })
        print(f"  [OK] Stage #{idx+1} -> {img_name}  cam=({cam_center[0]:.3f}, {cam_center[1]:.3f}, {cam_center[2]:.3f}) m")

    if not valid_stages:
        print("[3DGS EXPORT] No valid stages with RGB images. Use a session captured with web_scanner_app.py v2+.")
        return

    print(f"\n[3DGS EXPORT] {len(valid_stages)} valid RGB stages found.")

    # --- transforms.json (NeRF / instant-NGP / nerfstudio) ---
    # ZED: X right, Y down, Z forward (OpenCV) -> NeRF OpenGL: X right, Y up, Z back
    OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])
    transforms = {
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h,
        "camera_model": "PINHOLE",
        "frames": [
            {
                "file_path": f"images/{vs['image_name']}",
                "transform_matrix": (vs["T_cam2world"] @ OPENCV_TO_OPENGL).tolist()
            }
            for vs in valid_stages
        ]
    }
    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"[+] Written: {transforms_path}  ({len(transforms['frames'])} frames)")

    # --- COLMAP cameras.txt ---
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"[+] Written: {cameras_path}")

    # --- COLMAP images.txt ---
    images_path = os.path.join(sparse_dir, "images.txt")
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, vs in enumerate(valid_stages, start=1):
            R_w2c = vs["T_world2cam"][:3, :3]
            t_w2c = vs["T_world2cam"][:3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
            tx, ty, tz = t_w2c
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {vs['image_name']}\n\n")
    print(f"[+] Written: {images_path}")

    # --- COLMAP points3D.txt (seeded from PLY) ---
    points3D_path = os.path.join(sparse_dir, "points3D.txt")
    _write_sparse_points3d(points3D_path, valid_stages, session_dir, pipeline)

    # --- HOWTO README ---
    readme_path = os.path.join(output_dir, "HOWTO_3DGS.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# 3D Gaussian Splatting Quick-Start

This folder was exported from a ZED Mini + ABB Robot scan session.
It contains {len(valid_stages)} posed RGB frames ready for 3DGS / NeRF training.

## Option A: gaussian-splatting (graphdeco-inria, COLMAP format)

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
python train.py -s "{os.path.abspath(output_dir)}" --model_path output/robot_scan
```

## Option B: nerfstudio (transforms.json format)

```bash
pip install nerfstudio
ns-train gaussian-splatting --data "{os.path.abspath(output_dir)}"
```

## Option C: instant-ngp (NVIDIA, transforms.json format)

```bash
./instant-ngp "{os.path.abspath(output_dir)}"
```

## Camera Intrinsics

| fx | fy | cx | cy | Width | Height |
|----|----|----|-----|-------|--------|
| {fx:.2f} | {fy:.2f} | {cx:.2f} | {cy:.2f} | {w} | {h} |
""")
    print(f"[+] Written: {readme_path}")
    print(f"\n[3DGS EXPORT] Complete! Dataset: {os.path.abspath(output_dir)}")
    print(f"  Train with: python train.py -s \"{os.path.abspath(output_dir)}\" --model_path output/robot_scan\n")


def _write_sparse_points3d(points3D_path, valid_stages, session_dir, pipeline):
    try:
        import open3d as o3d
        merged_pts, merged_cols = [], []
        for vs in valid_stages:
            if not vs.get("ply"):
                continue
            ply_path = os.path.join(session_dir, vs["ply"]) if not os.path.isabs(vs["ply"]) else vs["ply"]
            if not os.path.exists(ply_path):
                continue
            pcd = o3d.io.read_point_cloud(ply_path)
            if len(pcd.points) == 0:
                continue
            step = max(1, len(pcd.points) // 500)
            pts = np.asarray(pcd.points)[::step]
            cols = (np.asarray(pcd.colors)[::step] * 255).astype(np.uint8) if pcd.has_colors() \
                   else np.full((len(pts), 3), 128, dtype=np.uint8)
            T = vs["T_cam2world"]
            pts_world = (T[:3, :3] @ pts.T).T + T[:3, 3]
            merged_pts.append(pts_world)
            merged_cols.append(cols)

        with open(points3D_path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            if not merged_pts:
                print("  [points3D] No PLY files found — empty points3D.txt.")
                return
            all_pts  = np.vstack(merged_pts)
            all_cols = np.vstack(merged_cols)
            print(f"  [points3D] Seeding {len(all_pts):,} world-space points.")
            for pid, (pt, col) in enumerate(zip(all_pts, all_cols), start=1):
                f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {col[0]} {col[1]} {col[2]} 1.0\n")
        print(f"[+] Written: {points3D_path}  ({len(all_pts):,} seed points)")
    except Exception as e:
        with open(points3D_path, "w") as f:
            f.write("# 3D point list — empty\n")
        print(f"  [points3D] Could not seed: {e} — empty file written.")


def main():
    parser = argparse.ArgumentParser(description="Export robot scan session to 3DGS / NeRF dataset")
    parser.add_argument("--session", "-s", default="scans/session_stages", help="Session directory")
    parser.add_argument("--out", "-o", default="scans/3dgs_dataset", help="Output dataset directory")
    parser.add_argument("--calib", "-c", default="calibration_result.npz", help="Hand-eye calibration file")
    args = parser.parse_args()
    export_3dgs_dataset(session_dir=args.session, output_dir=args.out, calib_path=args.calib)


if __name__ == "__main__":
    main()
