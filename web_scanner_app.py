"""
ZED Studio // ABB Robot 3D Scanner & Open3D Web Application
============================================================
Full feature parity with the native Robot Scanner Dashboard:
- Hand-Eye calibration transform loading (calibration_result.npz)
- Robot TCP Socket Server (Port 5000) with ACK handshake
- Session stages & manifest.json persistence (scans/session_stages/)
- Resilient ZED camera grab thread with auto-reconnect & offline standby
- 1:1 Millimeter point cloud and watertight mesh export for Rhino
- Open3D Multi-Way ICP and Screened Poisson Surface Reconstruction
- Real-time MJPEG camera & neural depth streams
- High-performance binary point cloud & STL mesh streaming for Three.js
"""

import os
import sys
import time
import json
import socket
import threading
import subprocess
import numpy as np
import cv2
import pyzed.sl as sl
from flask import Flask, render_template, Response, jsonify, request, send_file
from scipy.spatial.transform import Rotation
import scipy.spatial as spatial

# Ensure tools directory is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.open3d_pipeline import Open3DScannerPipeline

# --- Utility Functions for Point Cloud Processing ---

def save_ply_binary(filepath, points, colors):
    """
    Saves points (N, 3) and colors (N, 3) as binary little-endian PLY with 24-bit RGB vertex colors.
    """
    if len(points) == 0:
        return False
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )
    dt = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    arr = np.empty(len(points), dtype=dt)
    arr['x'] = points[:, 0]
    arr['y'] = points[:, 1]
    arr['z'] = points[:, 2]
    arr['red'] = colors[:, 0]
    arr['green'] = colors[:, 1]
    arr['blue'] = colors[:, 2]
    
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)
    print(f"[+] Saved binary PLY ({len(points):,} pts) -> {filepath}")
    return True

def extract_valid_points(point_cloud_mat, min_dist=0.25, max_dist=0.85, subsample_step=1, border_crop=0.06):
    data = point_cloud_mat.get_data()
    H, W, _ = data.shape
    
    if border_crop > 0:
        by = int(H * border_crop)
        bx = int(W * border_crop)
        data = data[by:H-by, bx:W-bx]
        
    if subsample_step > 1:
        data = data[::subsample_step, ::subsample_step]
        
    pts = data[:, :, :3].reshape(-1, 3)
    color_bytes = np.ascontiguousarray(data[:, :, 3]).view(np.uint32).view(np.uint8).reshape(-1, 4)
    colors = color_bytes[:, :3]  # R, G, B
    
    valid_mask = (
        np.isfinite(pts[:, 0]) &
        np.isfinite(pts[:, 1]) &
        np.isfinite(pts[:, 2]) &
        (pts[:, 2] >= min_dist) &
        (pts[:, 2] <= max_dist)
    )
    return pts[valid_mask], colors[valid_mask]

def statistical_outlier_removal(points, colors, nb_neighbors=16, std_ratio=1.2):
    if len(points) < nb_neighbors + 2:
        return points, colors
    tree = spatial.cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1, workers=-1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
    inliers = mean_dist < threshold
    return points[inliers], colors[inliers]

def voxel_grid_average(points, colors, voxel_size=0.002):
    if len(points) == 0:
        return points, colors
    vox_indices = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices, counts = np.unique(
        vox_indices, axis=0, return_inverse=True, return_counts=True
    )
    num_voxels = len(unique_voxels)
    pts_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    cols_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    np.add.at(pts_sum, inverse_indices, points)
    np.add.at(cols_sum, inverse_indices, colors.astype(np.float64))
    avg_pts = (pts_sum / counts[:, None]).astype(np.float32)
    avg_cols = np.clip(cols_sum / counts[:, None], 0, 255).astype(np.uint8)
    return avg_pts, avg_cols

# --- Scan Configuration State ---
DEFAULT_SCAN_CONFIG = {
    "min_depth_m": 0.25,
    "max_depth_m": 0.70,
    "border_crop": 0.06,
    "voxel_size_mm": 2.0,
    "outlier_std_ratio": 1.2,
    "outlier_nb_neighbors": 16,
    "subsample_step": 1,
    "icp_max_distance_mm": 15.0,
    "depth_mode": "NEURAL"
}

def load_scan_config():
    cfg_file = "scan_config.json"
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
                return {**DEFAULT_SCAN_CONFIG, **cfg}
        except Exception as e:
            print(f"[CONFIG LOAD WARNING] {e}")
    return DEFAULT_SCAN_CONFIG.copy()

def save_scan_config(cfg):
    cfg_file = "scan_config.json"
    try:
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"[CONFIG SAVE ERROR] {e}")
        return False

scan_config = load_scan_config()

# --- Flask Server & Global Application State ---

app = Flask(__name__, template_folder="templates", static_folder="static")

calib_file = "calibration_result.npz"
pipeline = Open3DScannerPipeline(calib_path=calib_file)
R_cam2ee = pipeline.R_cam2ee.astype(np.float32)
t_cam2ee = pipeline.t_cam2ee.astype(np.float32)
cam_offset_mm = (t_cam2ee.ravel() * 1000.0).round(1)

output_dir = "scans"
session_dir = os.path.join(output_dir, "session_stages")
os.makedirs(session_dir, exist_ok=True)

# Per-stage RGB + depth save helpers
def save_stage_rgb(filepath, zed_image_mat):
    """Save full-res RGB frame as PNG (used by TSDF and 3DGS pipelines)."""
    try:
        bgra = zed_image_mat.get_data()
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(filepath, bgr)
    except Exception as e:
        print(f"[RGB SAVE WARNING] {e}")

def save_stage_depth(filepath, zed_depth_mat):
    """Save raw metric depth map as float32 NPY (used by TSDF pipeline)."""
    try:
        depth_arr = zed_depth_mat.get_data().copy().astype(np.float32)
        np.save(filepath, depth_arr)
    except Exception as e:
        print(f"[DEPTH SAVE WARNING] {e}")

def get_camera_intrinsics(zed_cam_obj):
    """Read ZED pinhole intrinsics for manifest and TSDF/3DGS pipelines."""
    try:
        info = zed_cam_obj.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        res = info.camera_configuration.resolution
        return {
            "fx": round(float(calib.fx), 4),
            "fy": round(float(calib.fy), 4),
            "cx": round(float(calib.cx), 4),
            "cy": round(float(calib.cy), 4),
            "width": int(res.width),
            "height": int(res.height)
        }
    except Exception:
        return {"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720}

manifest_path = os.path.join(session_dir, "manifest.json")
if os.path.exists(manifest_path):
    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        print(f"[SESSION ONLINE] Loaded manifest with {len(manifest_data.get('stages', []))} stages from {manifest_path}")
    except Exception:
        manifest_data = {"stages": []}
else:
    manifest_data = {"stages": []}
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

accumulated_points = []
accumulated_colors = []
robot_poses_history = []
active_robot_pose = None
stage_count = len(manifest_data.get("stages", []))
robot_connected = False
robot_client_ip = "None"
last_raw_message = "Waiting for robot socket trigger..."
camera_online = False
fps_cam = 0.0
center_dist_str = "--"

# Auto-preload existing workpiece scans if available
def preload_existing_workpiece():
    global accumulated_points, accumulated_colors, robot_poses_history
    latest_ply = os.path.join(output_dir, "latest_workpiece_mm.ply")
    if os.path.exists(latest_ply):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(latest_ply)
            pts = np.asarray(pcd.points, dtype=np.float32) / 1000.0  # Convert mm to meters for Three.js
            if pcd.has_colors():
                cols = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
            else:
                cols = np.full((len(pts), 3), 200, dtype=np.uint8)
            accumulated_points = [pts]
            accumulated_colors = [cols]
            print(f"[PRELOAD] Loaded latest workpiece: {len(pts):,} points into viewer buffer.")
        except Exception as e:
            print(f"[PRELOAD NOTICE] {e}")

preload_existing_workpiece()

# Placeholder frame generator
def make_placeholder_frame(title, subtitle="CAMERA OFFLINE // RESILIENT STANDBY", color=(198, 241, 53)):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (16, 16, 22)
    for x in range(0, 640, 40):
        cv2.line(img, (x, 0), (x, 360), (24, 24, 32), 1)
    for y in range(0, 360, 40):
        cv2.line(img, (0, y), (640, y), (24, 24, 32), 1)
    cx, cy = 320, 180
    cv2.circle(img, (cx, cy), 36, (35, 35, 48), 1)
    cv2.line(img, (cx - 50, cy), (cx + 50, cy), (45, 45, 60), 1)
    cv2.line(img, (cx, cy - 50), (cx, cy + 50), (45, 45, 60), 1)
    cv2.putText(img, title, (cx - 150, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (cx - 155, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 140), 1, cv2.LINE_AA)
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()

latest_rgb_frame = make_placeholder_frame("VIEW: LEFT (CAMERA OFFLINE)", color=(198, 241, 53))
latest_depth_frame = make_placeholder_frame("MEASURE: DEPTH (STANDBY)", color=(56, 189, 248))
frame_lock = threading.Lock()
cam_lock = threading.Lock()

def get_latest_mesh_path():
    stl_mm = os.path.join("scans", "latest_workpiece_mesh_mm.stl")
    if os.path.exists(stl_mm):
        return stl_mm
    obj_mm = os.path.join("scans", "latest_workpiece_mesh_mm.obj")
    if os.path.exists(obj_mm):
        return obj_mm
    return None

pipeline_process = None
pipeline_logs = []
pipeline_lock = threading.Lock()

def _read_process_output(process):
    global pipeline_logs
    # Read line by line with iter
    for line in iter(process.stdout.readline, ''):
        if line:
            with pipeline_lock:
                # Keep logs manageable, max 1000 lines
                if len(pipeline_logs) > 1000:
                    pipeline_logs.pop(0)
                pipeline_logs.append(line)
    process.stdout.close()
    process.wait()
    with pipeline_lock:
        pipeline_logs.append(f"\n[PROCESS EXITED] Code {process.returncode}\n")

def run_open3d_pipeline_async(mode="icp"):
    global pipeline_process, pipeline_logs
    py311_exe = r"C:\Users\pitsi\AppData\Local\Programs\Python\Python311\python.exe"
    script_path = os.path.join("tools", "open3d_pipeline.py")
    
    with pipeline_lock:
        if pipeline_process is not None and pipeline_process.poll() is None:
            return False, "Pipeline already running."
        pipeline_logs = [f"[SYSTEM] Starting Open3D Pipeline in '{mode}' mode...\n"]
    
    if os.path.exists(py311_exe) and os.path.exists(script_path):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        pipeline_process = subprocess.Popen(
            [py311_exe, script_path, "--session", session_dir, "--mode", mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1, # line buffered
            universal_newlines=True
        )
        t = threading.Thread(target=_read_process_output, args=(pipeline_process,), daemon=True)
        t.start()
        return True, "Pipeline started."
    else:
        with pipeline_lock:
            pipeline_logs.append("[ERROR] Python 3.11 or open3d_pipeline.py not found.\n")
        return False, "Python environment not found."

# --- Resilient Background ZED Camera Thread ---

zed_cam = sl.Camera()

def zed_camera_thread():
    global latest_rgb_frame, latest_depth_frame, fps_cam, center_dist_str, camera_online
    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.2
    init_params.depth_maximum_distance = 1.0

    print("[ZED] Initializing camera thread...")
    with cam_lock:
        try:
            if zed_cam.open(init_params) == sl.ERROR_CODE.SUCCESS:
                camera_online = True
                print("[ZED ONLINE] ZED Camera opened successfully.")
            else:
                print("[ZED STANDBY] Camera not detected at start. Running in resilient standby mode...")
        except Exception as e:
            print(f"[ZED EXCEPTION] {e}")

    image_left = sl.Mat()
    image_depth = sl.Mat()
    depth_meas = sl.Mat()

    last_reconnect_time = 0.0
    prev_t = time.time()
    consecutive_grab_fails = 0

    while True:
        if camera_online:
            try:
                with cam_lock:
                    err = zed_cam.grab()
                    if err == sl.ERROR_CODE.SUCCESS:
                        consecutive_grab_fails = 0
                        curr_t = time.time()
                        fps_cam = 0.9 * fps_cam + 0.1 * (1.0 / max(curr_t - prev_t, 1e-4))
                        prev_t = curr_t

                        # Left RGB
                        zed_cam.retrieve_image(image_left, sl.VIEW.LEFT)
                        rgb_bgr = cv2.cvtColor(image_left.get_data(), cv2.COLOR_BGRA2BGR)
                        rgb_small = cv2.resize(rgb_bgr, (640, 360))
                        _, rgb_jpeg = cv2.imencode('.jpg', rgb_small, [cv2.IMWRITE_JPEG_QUALITY, 75])

                        # Colorized Depth
                        zed_cam.retrieve_image(image_depth, sl.VIEW.DEPTH)
                        depth_bgr = cv2.cvtColor(image_depth.get_data(), cv2.COLOR_BGRA2BGR)
                        depth_small = cv2.resize(depth_bgr, (640, 360))
                        _, depth_jpeg = cv2.imencode('.jpg', depth_small, [cv2.IMWRITE_JPEG_QUALITY, 70])

                        # Center distance laser
                        zed_cam.retrieve_measure(depth_meas, sl.MEASURE.DEPTH)
                        cx = depth_meas.get_width() // 2
                        cy = depth_meas.get_height() // 2
                        err_c, d_val = depth_meas.get_value(cx, cy)
                        center_dist_str = f"{d_val:.3f}" if (not np.isnan(d_val) and d_val > 0) else "--"

                        with frame_lock:
                            latest_rgb_frame = rgb_jpeg.tobytes()
                            latest_depth_frame = depth_jpeg.tobytes()
                    else:
                        consecutive_grab_fails += 1
                        # Only enter offline standby if grab fails repeatedly (e.g. 10 consecutive frames)
                        if consecutive_grab_fails > 10:
                            camera_online = False
                            print(f"[ZED DISCONNECTED] Grab failed {consecutive_grab_fails} times ({err}). Entering standby mode...")
                            with frame_lock:
                                latest_rgb_frame = make_placeholder_frame("VIEW: LEFT (DISCONNECTED)", color=(0, 100, 255))
                                latest_depth_frame = make_placeholder_frame("MEASURE: DEPTH (STANDBY)", color=(120, 120, 120))
            except Exception as ex:
                consecutive_grab_fails += 1
                if consecutive_grab_fails > 10:
                    camera_online = False
                    print(f"[ZED EXCEPTION] {ex}")
        else:
            # Standby mode: poll for camera reconnection every 3.0s
            curr_t = time.time()
            if curr_t - last_reconnect_time > 3.0:
                last_reconnect_time = curr_t
                try:
                    with cam_lock:
                        zed_cam.close()
                        if zed_cam.open(init_params) == sl.ERROR_CODE.SUCCESS:
                            camera_online = True
                            consecutive_grab_fails = 0
                            print("\n[ZED RECONNECTED] Camera successfully reconnected!\n")
                except Exception:
                    pass
            time.sleep(0.04)
        time.sleep(0.015)

# --- Background TCP Socket Server for ABB Robot ---

def tcp_server_thread():
    global robot_connected, robot_client_ip, last_raw_message
    global active_robot_pose, robot_poses_history, stage_count
    global accumulated_points, accumulated_colors, manifest_data
    
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 5000
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((SERVER_HOST, SERVER_PORT))
            s.listen(1)
            print(f"[TCP SERVER] Listening on {SERVER_HOST}:{SERVER_PORT} for ABB Robot...")
        except Exception as e:
            print(f"[TCP SERVER WARNING] Port {SERVER_PORT} bind issue: {e}. (Another dashboard may be running).")
            return
            
        while True:
            try:
                conn, addr = s.accept()
                robot_client_ip = addr[0]
                robot_connected = True
                print(f"\n[ROBOT LINK ONLINE] Connected from {addr[0]}!\n")
                
                with conn:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        msg = data.decode('utf-8').strip()
                        last_raw_message = msg
                        
                        if msg == "DONE":
                            print("\n[ROBOT MESSAGE] Robot completed entire scanning toolpath!\n")
                            if len(accumulated_points) > 0:
                                all_pts = np.vstack(accumulated_points)
                                all_col = np.vstack(accumulated_colors)
                                nb_n = int(scan_config.get("outlier_nb_neighbors", 16))
                                std_r = float(scan_config.get("outlier_std_ratio", 1.2))
                                vox_m = float(scan_config.get("voxel_size_mm", 2.0)) / 1000.0
                                final_pts, final_col = statistical_outlier_removal(all_pts, all_col, nb_neighbors=nb_n, std_ratio=std_r)
                                final_pts, final_col = voxel_grid_average(final_pts, final_col, voxel_size=vox_m)
                                ts = int(time.time())
                                
                                save_path_m = os.path.join(output_dir, f"robot_workpiece_{ts}.ply")
                                save_ply_binary(save_path_m, final_pts, final_col)
                                
                                save_path_mm = os.path.join(output_dir, f"robot_workpiece_{ts}_mm.ply")
                                save_ply_binary(save_path_mm, final_pts * 1000.0, final_col)
                                
                                latest_mm_path = os.path.join(output_dir, "latest_workpiece_mm.ply")
                                save_ply_binary(latest_mm_path, final_pts * 1000.0, final_col)
                                
                                run_open3d_pipeline_async()
                            else:
                                run_open3d_pipeline_async()
                        else:
                            try:
                                parts = [float(v.strip()) for v in msg.split(',')]
                                if len(parts) == 7:
                                    # ABB Format: X, Y, Z, Qw, Qx, Qy, Qz
                                    tx_mm, ty_mm, tz_mm, qw, qx, qy, qz = parts
                                    t_base2ee = np.array([tx_mm / 1000.0, ty_mm / 1000.0, tz_mm / 1000.0], dtype=np.float32)
                                    rot_base2ee = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
                                elif len(parts) == 6:
                                    # KUKA Format: X, Y, Z, A, B, C (Euler Z-Y-X in degrees)
                                    tx_mm, ty_mm, tz_mm, a_deg, b_deg, c_deg = parts
                                    t_base2ee = np.array([tx_mm / 1000.0, ty_mm / 1000.0, tz_mm / 1000.0], dtype=np.float32)
                                    
                                    # Convert KUKA A, B, C (Intrinsic Z-Y-X) to Rotation matrix
                                    r = Rotation.from_euler('ZYX', [a_deg, b_deg, c_deg], degrees=True)
                                    rot_base2ee = r.as_matrix().astype(np.float32)
                                    
                                    # Save quaternions to match universal manifest format
                                    qx, qy, qz, qw = r.as_quat()
                                else:
                                    raise ValueError(f"Unknown coordinate format. Expected 6 or 7 values, got {len(parts)}.")
                                
                                    R_base2cam = rot_base2ee @ R_cam2ee
                                    t_base2cam = rot_base2ee @ t_cam2ee.ravel() + t_base2ee
                                    
                                    active_robot_pose = {
                                        'rot': R_base2cam.tolist(),
                                        'trans': t_base2cam.tolist(),
                                        'trans_mm': [tx_mm, ty_mm, tz_mm],
                                        'quat': [qx, qy, qz, qw]
                                    }
                                    robot_poses_history.append(t_base2cam.tolist())
                                    
                                    if camera_online:
                                        point_cloud = sl.Mat()
                                        image_left_stage = sl.Mat()
                                        depth_meas_stage = sl.Mat()
                                        with cam_lock:
                                            # Capture a fresh, synchronised frame at this exact robot pose
                                            zed_cam.grab()
                                            zed_cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                                            zed_cam.retrieve_image(image_left_stage, sl.VIEW.LEFT)
                                            zed_cam.retrieve_measure(depth_meas_stage, sl.MEASURE.DEPTH)

                                        min_d = float(scan_config.get("min_depth_m", 0.25))
                                        max_d = float(scan_config.get("max_depth_m", 0.70))
                                        step = int(scan_config.get("subsample_step", 1))
                                        crop = float(scan_config.get("border_crop", 0.06))
                                        nb_n = int(scan_config.get("outlier_nb_neighbors", 16))
                                        std_r = float(scan_config.get("outlier_std_ratio", 1.2))

                                        pts_cam, col_cam = extract_valid_points(
                                            point_cloud, min_dist=min_d, max_dist=max_d, subsample_step=step, border_crop=crop
                                        )
                                        pts_cam, col_cam = statistical_outlier_removal(pts_cam, col_cam, nb_neighbors=nb_n, std_ratio=std_r)

                                        stage_count += 1
                                        stage_base  = f"stage_{stage_count:03d}"
                                        stage_ply   = stage_base + ".ply"
                                        stage_rgb   = stage_base + ".png"
                                        stage_depth = stage_base + "_depth.npy"

                                        # Save PLY point cloud (ICP + Poisson pipeline)
                                        save_ply_binary(os.path.join(session_dir, stage_ply), pts_cam, col_cam)

                                        # Save full-res RGB image (TSDF + 3DGS pipelines)
                                        save_stage_rgb(os.path.join(session_dir, stage_rgb), image_left_stage)

                                        # Save raw float32 depth map in metres (TSDF pipeline)
                                        save_stage_depth(os.path.join(session_dir, stage_depth), depth_meas_stage)

                                        # Write camera intrinsics into manifest once (first stage only)
                                        if "intrinsics" not in manifest_data:
                                            manifest_data["intrinsics"] = get_camera_intrinsics(zed_cam)
                                            print(f"[SESSION] Camera intrinsics saved: {manifest_data['intrinsics']}")

                                        manifest_data["stages"].append({
                                            "index": stage_count,
                                            "ply": stage_ply,
                                            "rgb": stage_rgb,
                                            "depth_npy": stage_depth,
                                            "pose_flange": [tx_mm, ty_mm, tz_mm, qw, qx, qy, qz],
                                            "time": time.strftime("%H:%M:%S")
                                        })
                                        with open(manifest_path, "w") as f:
                                            json.dump(manifest_data, f, indent=2)

                                        pts_base = pts_cam @ R_base2cam.T + t_base2cam
                                        accumulated_points.append(pts_base)
                                        accumulated_colors.append(col_cam)
                                        print(f"[STAGE #{stage_count}] Captured ({len(pts_cam):,} pts) | RGB+Depth saved | Flange: ({tx_mm:.1f}, {ty_mm:.1f}, {tz_mm:.1f}) mm -> {stage_ply}")
                                    else:
                                        print(f"[STAGE NOTICE] Robot waypoint received ({tx_mm:.1f}, {ty_mm:.1f}, {tz_mm:.1f}), but camera is OFFLINE.")
                            except Exception as pe:
                                print(f"[PARSE ERROR] {pe}")
                                
                        conn.sendall(b"ACK")
            except Exception as err:
                print(f"[TCP ERROR] {err}")
            finally:
                robot_connected = False
                print("[ROBOT LINK OFFLINE] Connection closed.")

# --- Flask Web Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

def gen_mjpeg_stream(get_frame_func):
    while True:
        with frame_lock:
            frame = get_frame_func()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video/rgb')
def video_rgb():
    return Response(gen_mjpeg_stream(lambda: latest_rgb_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/depth')
def video_depth():
    return Response(gen_mjpeg_stream(lambda: latest_depth_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    total_pts = sum(len(p) for p in accumulated_points)
    mesh_path = get_latest_mesh_path()
    mesh_available = mesh_path is not None
    mesh_info = {
        'available': mesh_available,
        'name': os.path.basename(mesh_path) if mesh_path else 'None',
        'size_mb': round(os.path.getsize(mesh_path) / (1024 * 1024), 2) if mesh_available else 0
    }
    
    return jsonify({
        'camera_online': camera_online,
        'robot_connected': robot_connected,
        'robot_ip': robot_client_ip,
        'stage_count': stage_count,
        'total_points': total_pts,
        'center_distance': center_dist_str,
        'fps': fps_cam,
        'active_pose': active_robot_pose,
        'last_payload': last_raw_message,
        'trajectory': robot_poses_history,
        'mesh_stats': mesh_info
    })

@app.route('/api/points')
def api_points():
    """
    Binary point cloud stream for Three.js WebGL:
    [X, Y, Z (float32)] + [R, G, B (uint8)] = 15 bytes per point.
    Transmitted in meters matching robot base world coordinates.
    """
    if len(accumulated_points) == 0:
        return Response(b'', mimetype='application/octet-stream')
        
    all_pts = np.vstack(accumulated_points)
    all_col = np.vstack(accumulated_colors)
    
    # Subsample if large for 60 FPS WebGL responsiveness
    max_web_points = 200000
    if len(all_pts) > max_web_points:
        step = len(all_pts) // max_web_points + 1
        sub_pts = all_pts[::step]
        sub_col = all_col[::step]
    else:
        sub_pts = all_pts
        sub_col = all_col
        
    point_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
    ])
    
    buf = np.empty(len(sub_pts), dtype=point_dtype)
    buf['x'] = sub_pts[:, 0]
    buf['y'] = sub_pts[:, 1]
    buf['z'] = sub_pts[:, 2]
    buf['r'] = sub_col[:, 0]
    buf['g'] = sub_col[:, 1]
    buf['b'] = sub_col[:, 2]
    
    return Response(buf.tobytes(), mimetype='application/octet-stream')

@app.route('/api/mesh')
def api_mesh():
    """
    Serves the latest reconstructed 3D mesh in binary STL format for Three.js.
    """
    stl_path = os.path.join("scans", "latest_workpiece_mesh_mm.stl")
    if os.path.exists(stl_path):
        return send_file(stl_path, mimetype="application/octet-stream")
    return Response(b'', status=404)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    global scan_config
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            if not isinstance(data, dict):
                return jsonify({'success': False, 'message': 'Invalid JSON format'}), 400
                
            # Validate and update configuration keys
            if 'min_depth_m' in data:
                scan_config['min_depth_m'] = max(0.15, min(float(data['min_depth_m']), 2.0))
            if 'max_depth_m' in data:
                scan_config['max_depth_m'] = max(scan_config['min_depth_m'] + 0.05, min(float(data['max_depth_m']), 3.0))
            if 'border_crop' in data:
                scan_config['border_crop'] = max(0.0, min(float(data['border_crop']), 0.25))
            if 'voxel_size_mm' in data:
                scan_config['voxel_size_mm'] = max(0.5, min(float(data['voxel_size_mm']), 20.0))
            if 'outlier_std_ratio' in data:
                scan_config['outlier_std_ratio'] = max(0.5, min(float(data['outlier_std_ratio']), 3.0))
            if 'outlier_nb_neighbors' in data:
                scan_config['outlier_nb_neighbors'] = max(4, min(int(data['outlier_nb_neighbors']), 64))
            if 'subsample_step' in data:
                scan_config['subsample_step'] = max(1, min(int(data['subsample_step']), 8))
            if 'icp_max_distance_mm' in data:
                scan_config['icp_max_distance_mm'] = max(1.0, min(float(data['icp_max_distance_mm']), 50.0))
                
            save_scan_config(scan_config)
            print(f"[CONFIG UPDATED] New scan parameters: {scan_config}")
            return jsonify({'success': True, 'config': scan_config, 'message': 'Scan parameters updated and saved.'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error saving config: {str(e)}'}), 500
            
    return jsonify({'success': True, 'config': scan_config})

@app.route('/api/action/save', methods=['POST'])
def api_save():
    global accumulated_points, accumulated_colors
    if len(accumulated_points) == 0:
        return jsonify({'success': False, 'message': 'No points in buffer to save!'})
        
    all_pts = np.vstack(accumulated_points)
    all_col = np.vstack(accumulated_colors)
    nb_n = int(scan_config.get("outlier_nb_neighbors", 16))
    std_r = float(scan_config.get("outlier_std_ratio", 1.2))
    vox_m = float(scan_config.get("voxel_size_mm", 2.0)) / 1000.0
    final_pts, final_col = statistical_outlier_removal(all_pts, all_col, nb_neighbors=nb_n, std_ratio=std_r)
    final_pts, final_col = voxel_grid_average(final_pts, final_col, voxel_size=vox_m)
    ts = int(time.time())
    
    save_path_m = os.path.join(output_dir, f"robot_workpiece_{ts}.ply")
    save_ply_binary(save_path_m, final_pts, final_col)
    
    save_path_mm = os.path.join(output_dir, f"robot_workpiece_{ts}_mm.ply")
    save_ply_binary(save_path_mm, final_pts * 1000.0, final_col)
    
    latest_mm_path = os.path.join(output_dir, "latest_workpiece_mm.ply")
    save_ply_binary(latest_mm_path, final_pts * 1000.0, final_col)
    
    run_open3d_pipeline_async()
    
    return jsonify({
        'success': True,
        'message': f"Workpiece saved! Exported 1:1 mm PLY for Rhino and started Open3D pipeline."
    })

@app.route('/api/action/clean', methods=['POST'])
def api_clean():
    global accumulated_points, accumulated_colors
    if len(accumulated_points) == 0:
        return jsonify({'success': False, 'message': 'No points in buffer to clean.'})
        
    all_pts = np.vstack(accumulated_points)
    all_col = np.vstack(accumulated_colors)
    nb_n = int(scan_config.get("outlier_nb_neighbors", 16))
    std_r = float(scan_config.get("outlier_std_ratio", 1.2))
    vox_m = float(scan_config.get("voxel_size_mm", 2.0)) / 1000.0
    clean_pts, clean_col = statistical_outlier_removal(all_pts, all_col, nb_neighbors=nb_n, std_ratio=std_r)
    clean_pts, clean_col = voxel_grid_average(clean_pts, clean_col, voxel_size=vox_m)
    
    accumulated_points = [clean_pts]
    accumulated_colors = [clean_col]
    
    return jsonify({
        'success': True,
        'message': f"Filtered buffer to {len(clean_pts):,} surface points.",
        'points': len(clean_pts)
    })

@app.route('/api/action/open3d_pipeline', methods=['POST'])
def api_open3d_pipeline():
    """
    Executes the Open3D pipeline with chosen mode (icp, tsdf, or both).
    """
    req = request.get_json() or {}
    mode = req.get("mode", "icp")
    detail = req.get("detail", "medium")
    
    # Save the requested detail level to scan_config.json
    try:
        if os.path.exists("scan_config.json"):
            with open("scan_config.json", "r") as f:
                cfg = json.load(f)
        else:
            cfg = {}
        cfg["mesh_detail"] = detail
        with open("scan_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save mesh_detail to config: {e}")

    success, msg = run_open3d_pipeline_async(mode)
    return jsonify({
        'success': success,
        'message': msg
    })

@app.route('/api/pipeline/logs')
def api_pipeline_logs():
    with pipeline_lock:
        logs = "".join(pipeline_logs)
        is_running = pipeline_process is not None and pipeline_process.poll() is None
    return jsonify({
        'running': is_running,
        'logs': logs
    })

@app.route('/api/action/clear', methods=['POST'])
def api_clear():
    global accumulated_points, accumulated_colors, robot_poses_history, stage_count, manifest_data
    accumulated_points = []
    accumulated_colors = []
    robot_poses_history = []
    stage_count = 0
    manifest_data["stages"].clear()
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    # Also delete the orphaned files in the session directory to prevent confusion
    try:
        for filename in os.listdir(session_dir):
            if filename.endswith(".ply") or filename.endswith(".png") or filename.endswith(".npy"):
                os.remove(os.path.join(session_dir, filename))
    except Exception as e:
        print(f"[WEB ACTION] Notice during file cleanup: {e}")
        
    print("[WEB ACTION] Reset workpiece scan buffer, session stages, and deleted old files.")
    return jsonify({'success': True, 'message': 'Scan buffer and session stages reset.'})

if __name__ == "__main__":
    # Start background ZED camera loop
    t_cam = threading.Thread(target=zed_camera_thread, daemon=True)
    t_cam.start()
    
    # Start background TCP server loop for ABB robot
    t_tcp = threading.Thread(target=tcp_server_thread, daemon=True)
    t_tcp.start()
    
    print("\n" + "=" * 68)
    print("           ROBOT 3D SCANNING STUDIO // WEB APPLICATION")
    print("=" * 68)
    print(" -> Web Studio Interface:  http://localhost:8080")
    print(" -> Robot TCP Listen Port: 5000")
    print(f" -> Eye-in-Hand TCP Offset: X={cam_offset_mm[0]}mm, Y={cam_offset_mm[1]}mm, Z={cam_offset_mm[2]}mm")
    print("=" * 68 + "\n")
    
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
