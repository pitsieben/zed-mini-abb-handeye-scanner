import pyzed.sl as sl
import cv2
import numpy as np
import socket
import threading
import queue
import os
import sys
import time
import subprocess
import json
from scipy.spatial.transform import Rotation
import scipy.spatial as spatial

def save_ply_binary(filepath, points, colors):
    """
    Saves points (N, 3) and colors (N, 3) as binary little-endian PLY with 24-bit RGB vertex colors.
    """
    if len(points) == 0:
        print("[PLY ERROR] No points to save!")
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
    
    vertex_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    arr = np.empty(len(points), dtype=vertex_dtype)
    arr['x'] = points[:, 0]
    arr['y'] = points[:, 1]
    arr['z'] = points[:, 2]
    arr['red'] = colors[:, 0]
    arr['green'] = colors[:, 1]
    arr['blue'] = colors[:, 2]
    
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)
    return True

def extract_valid_points(point_cloud_mat, min_dist=0.25, max_dist=0.85, subsample_step=1, border_crop=0.06):
    """
    Extracts valid (X, Y, Z) and true (R, G, B) from sl.Mat(XYZRGBA).
    - Crops noisy stereo image borders (outer 6%) to eliminate edge fringes.
    - Discards background points beyond max_dist.
    - Discards NaNs and Infs.
    """
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
    """
    Removes sparse airborne speckles, flying pixels, and stray edge noise using KD-Tree statistics.
    """
    if len(points) < nb_neighbors + 2:
        return points, colors
    tree = spatial.cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1, workers=-1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
    inliers = mean_dist < threshold
    return points[inliers], colors[inliers]

def fine_icp_alignment(source_pts, target_pts, max_iterations=8, max_distance=0.035):
    """
    Refines alignment of source_pts onto target_pts using Point-to-Point ICP with SVD.
    Eliminates double-surface ghosting caused by minor kinematic/hand-eye tolerances.
    """
    if len(source_pts) < 100 or len(target_pts) < 100:
        return source_pts, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        
    step_tgt = max(1, len(target_pts) // 25000)
    step_src = max(1, len(source_pts) // 20000)
    tgt_sub = target_pts[::step_tgt]
    src_sub = source_pts[::step_src]
    
    tree = spatial.cKDTree(tgt_sub)
    curr_sub = src_sub.copy()
    total_R = np.eye(3, dtype=np.float32)
    total_t = np.zeros(3, dtype=np.float32)
    
    for _ in range(max_iterations):
        distances, indices = tree.query(curr_sub, workers=-1)
        valid = distances < max_distance
        if np.sum(valid) < 100:
            break
            
        s_m = curr_sub[valid]
        t_m = tgt_sub[indices[valid]]
        s_c = np.mean(s_m, axis=0)
        t_c = np.mean(t_m, axis=0)
        
        H = (s_m - s_c).T @ (t_m - t_c)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
            
        t = t_c - s_c @ R.T
        curr_sub = curr_sub @ R.T + t
        total_R = R @ total_R
        total_t = total_t @ R.T + t
        
    aligned_source = source_pts @ total_R.T + total_t
    return aligned_source, total_R, total_t

def voxel_grid_average(points, colors, voxel_size=0.002):
    """
    Averages points and colors within each voxel grid cell, blending overlapping passes into a smooth surface.
    """
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

class Robot3DViewport:
    """
    3D viewport visualizing the robot base space, toolpath trajectory,
    active robot end-effector pose, and the accumulating workpiece point cloud.
    """
    def __init__(self, width=930, height=700):
        self.w = width
        self.h = height
        
        # Orbit camera parameters
        self.yaw = -0.55
        self.pitch = 0.45
        self.distance = 3.5
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Reference origin at Robot Base (0,0,0)
        
        # Mouse interaction
        self.is_dragging_orbit = False
        self.is_dragging_zoom = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

    def handle_mouse(self, event, local_x, local_y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging_orbit = True
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging_orbit = False
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_dragging_zoom = True
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_dragging_zoom = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = local_x - self.last_mouse_x
            dy = local_y - self.last_mouse_y
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y
            
            if self.is_dragging_orbit:
                self.yaw += dx * 0.006
                self.pitch = np.clip(self.pitch + dy * 0.006, -1.3, 1.3)
            elif self.is_dragging_zoom:
                self.distance = np.clip(self.distance + dy * 0.015, 0.4, 12.0)

    def reset_view(self, auto_target=None):
        self.yaw = -0.55
        self.pitch = 0.45
        self.distance = 3.5
        if auto_target is not None:
            self.target = auto_target.copy()
        else:
            self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def render(self, accumulated_pts, accumulated_cols, robot_poses_history, active_pose):
        view_img = np.full((self.h, self.w, 3), 16, dtype=np.uint8)
        
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        
        R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        R_view = R_pitch @ R_yaw
        
        focal = self.h * 1.05
        cx, cy = self.w / 2.0, self.h / 2.0
        
        def project(pts):
            P_rel = (pts - self.target) @ R_view.T
            Z_view = P_rel[:, 2] + self.distance
            valid = Z_view > 0.05
            u = np.zeros(len(pts), dtype=np.int32)
            v = np.zeros(len(pts), dtype=np.int32)
            u[valid] = np.clip(((P_rel[valid, 0] / Z_view[valid]) * focal + cx).astype(np.int32), -1000, 3000)
            v[valid] = np.clip(((-P_rel[valid, 1] / Z_view[valid]) * focal + cy).astype(np.int32), -1000, 3000)
            return u, v, Z_view, valid

        # --- 1. Robot Base World Grid (at Z=0 plane, centered at 0,0,0) ---
        grid_lines = []
        for x in np.linspace(-2.5, 2.5, 11):
            grid_lines.append((np.array([x, -2.5, 0.0], dtype=np.float32), np.array([x, 2.5, 0.0], dtype=np.float32)))
        for y in np.linspace(-2.5, 2.5, 11):
            grid_lines.append((np.array([-2.5, y, 0.0], dtype=np.float32), np.array([2.5, y, 0.0], dtype=np.float32)))
            
        for p1, p2 in grid_lines:
            u, v, z, val = project(np.vstack([p1, p2]))
            if val[0] and val[1]:
                cv2.line(view_img, (u[0], v[0]), (u[1], v[1]), (26, 36, 26), 1, cv2.LINE_AA)

        # Base Origin Axes at (0, 0, 0)
        origin_axes = np.array([
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],  # X Red
            [0.0, 0.4, 0.0],  # Y Green
            [0.0, 0.0, 0.4]   # Z Blue
        ], dtype=np.float32)
        ou, ov, oz, oval = project(origin_axes)
        if oval[0]:
            if oval[1]: cv2.line(view_img, (ou[0], ov[0]), (ou[1], ov[1]), (0, 0, 255), 2, cv2.LINE_AA)
            if oval[2]: cv2.line(view_img, (ou[0], ov[0]), (ou[2], ov[2]), (0, 255, 0), 2, cv2.LINE_AA)
            if oval[3]: cv2.line(view_img, (ou[0], ov[0]), (ou[3], ov[3]), (255, 100, 0), 2, cv2.LINE_AA)
            cv2.putText(view_img, "BASE (0,0,0)", (ou[0] + 8, ov[0] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)

        # --- 2. Robot Path Trajectory Lines ---
        if len(robot_poses_history) > 1:
            traj_pts = np.array([p['trans'] for p in robot_poses_history], dtype=np.float32)
            tu, tv, tz, tval = project(traj_pts)
            for i in range(len(traj_pts) - 1):
                if tval[i] and tval[i+1]:
                    cv2.line(view_img, (tu[i], tv[i]), (tu[i+1], tv[i+1]), (0, 200, 255), 1, cv2.LINE_AA)

        # --- 3. Past Robot Poses Markers ---
        for i, p in enumerate(robot_poses_history):
            pt = p['trans'].reshape(1, 3)
            pu, pv, pz, pval = project(pt)
            if pval[0] and 0 <= pu[0] < self.w and 0 <= pv[0] < self.h:
                cv2.circle(view_img, (pu[0], pv[0]), 3, (0, 160, 220), -1)
                cv2.putText(view_img, f"#{i+1}", (pu[0] + 5, pv[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # --- 4. Accumulated Workpiece Point Cloud in Robot Coordinates (Depth Sorted) ---
        if accumulated_pts is not None and len(accumulated_pts) > 0:
            if len(accumulated_pts) > 80000:
                step = len(accumulated_pts) // 80000 + 1
                vis_pts = accumulated_pts[::step]
                vis_cols = accumulated_cols[::step]
            else:
                vis_pts = accumulated_pts
                vis_cols = accumulated_cols
                
            pu, pv, pz, pval = project(vis_pts)
            in_screen = pval & (pu >= 0) & (pu < self.w) & (pv >= 0) & (pv < self.h)
            
            # Back-to-front depth sorting to eliminate inverted depth artifacts
            screen_indices = np.where(in_screen)[0]
            if len(screen_indices) > 0:
                depth_order = np.argsort(-pz[screen_indices])
                sorted_idx = screen_indices[depth_order]
                
                valid_u = pu[sorted_idx]
                valid_v = pv[sorted_idx]
                bgr_colors = vis_cols[sorted_idx][:, ::-1] # RGB -> BGR
                
                view_img[valid_v, valid_u] = bgr_colors
                # 2x2 splatting
                v_plus = np.clip(valid_v + 1, 0, self.h - 1)
                u_plus = np.clip(valid_u + 1, 0, self.w - 1)
                view_img[v_plus, valid_u] = bgr_colors
                view_img[valid_v, u_plus] = bgr_colors

        # --- 5. Active Robot Tool Flange Pose & Coordinate Triad (Rendered on top for clarity) ---
        if active_pose is not None:
            R_ee = active_pose['rot']
            t_ee = active_pose['trans']
            
            axes_pts = np.array([
                t_ee,
                t_ee + R_ee[:, 0] * 0.15,  # X Red
                t_ee + R_ee[:, 1] * 0.15,  # Y Green
                t_ee + R_ee[:, 2] * 0.15   # Z Blue
            ], dtype=np.float32)
            
            au, av, az, aval = project(axes_pts)
            if aval[0]:
                if aval[1]: cv2.line(view_img, (au[0], av[0]), (au[1], av[1]), (0, 0, 255), 3, cv2.LINE_AA) # X
                if aval[2]: cv2.line(view_img, (au[0], av[0]), (au[2], av[2]), (0, 255, 0), 3, cv2.LINE_AA) # Y
                if aval[3]: cv2.line(view_img, (au[0], av[0]), (au[3], av[3]), (255, 120, 0), 3, cv2.LINE_AA) # Z
                cv2.circle(view_img, (au[0], av[0]), 5, (0, 255, 255), -1)
                cv2.putText(view_img, "ACTIVE TCP", (au[0] + 10, av[0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Borders and watermark
        cv2.rectangle(view_img, (0, 0), (self.w - 1, self.h - 1), (60, 60, 60), 1)
        cv2.putText(view_img, "ROBOT BASE SPACE 3D MONITOR (REF: 0,0,0)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
        cv2.putText(view_img, "[Left-Drag: Orbit  |  Right-Drag: Zoom  |  'R': Recenter (0,0,0)]", (15, self.h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        return view_img

def main():
    # --- 1. Load Hand-Eye Calibration Transform if available ---
    R_cam2ee = np.eye(3, dtype=np.float32)
    t_cam2ee = np.zeros((3, 1), dtype=np.float32)
    calib_file = "calibration_result.npz"
    calib_loaded = False
    cam_offset_mm = np.zeros(3)
    
    if os.path.exists(calib_file):
        try:
            calib_data = np.load(calib_file)
            R_cam2ee = calib_data['R'].astype(np.float32)
            t_cam2ee = calib_data['t'].astype(np.float32)
            calib_loaded = True
            cam_offset_mm = (t_cam2ee.ravel() * 1000.0).round(1)
            print(f"\n=======================================================")
            print(f"[CALIBRATION ONLINE] Loaded Eye-in-Hand matrix from {calib_file}")
            print(f" -> Camera TCP Offset: X={cam_offset_mm[0]:.1f}mm, Y={cam_offset_mm[1]:.1f}mm, Z={cam_offset_mm[2]:.1f}mm")
            print(f"=======================================================\n")
        except Exception as e:
            print(f"[CALIBRATION WARNING] Failed to load {calib_file}: {e}")
    else:
        print("[CALIBRATION NOTICE] 'calibration_result.npz' not found yet.")
        print(" -> Using identity offset. Run robot_calibration_dashboard.py to solve exact lens offset.")

    # --- 1b. Load Scan Configuration Parameters ---
    scan_config = {
        "min_depth_m": 0.25,
        "max_depth_m": 0.70,
        "border_crop": 0.06,
        "voxel_size_mm": 2.0,
        "outlier_std_ratio": 1.2,
        "outlier_nb_neighbors": 16,
        "subsample_step": 1,
        "icp_max_distance_mm": 15.0
    }
    if os.path.exists("scan_config.json"):
        try:
            with open("scan_config.json", "r") as f:
                scan_config.update(json.load(f))
            print(f"[CONFIG ONLINE] Loaded scan_config.json: Range={scan_config['min_depth_m']}m-{scan_config['max_depth_m']}m | Voxel={scan_config['voxel_size_mm']}mm")
        except Exception as ce:
            print(f"[CONFIG NOTICE] Could not load scan_config.json: {ce}")

    # --- 2. Initialize ZED Camera ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.2
    init_params.depth_maximum_distance = 1.0

    print("Connecting to ZED Camera...")
    camera_online = False
    try:
        if zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            camera_online = True
            print("[CAMERA ONLINE] ZED Camera connected successfully.")
        else:
            print("[CAMERA STANDBY] ZED camera not detected. Running dashboard in resilient standby mode...")
    except Exception as e:
        print(f"[CAMERA STANDBY] Camera open exception: {e}. Running in resilient standby mode...")

    image_left = sl.Mat()
    image_depth = sl.Mat()
    point_cloud = sl.Mat()

    # --- 3. TCP Server Setup in Background Thread ---
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 5000
    
    robot_queue = queue.Queue()
    robot_connected = False
    robot_client_addr = None
    last_raw_message = "Waiting for robot..."
    
    def tcp_server_thread():
        nonlocal robot_connected, robot_client_addr, last_raw_message
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((SERVER_HOST, SERVER_PORT))
            s.listen(1)
            print(f"[TCP SERVER] Listening on {SERVER_HOST}:{SERVER_PORT} for ABB Robot...")
            
            while True:
                conn, addr = s.accept()
                robot_client_addr = addr
                robot_connected = True
                print(f"\n[ROBOT CONNECTED] Robot connected from {addr}!\n")
                
                with conn:
                    while True:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                            msg = data.decode('utf-8').strip()
                            last_raw_message = msg
                            
                            # Push to queue for the main thread
                            resp_event = threading.Event()
                            robot_queue.put((msg, resp_event))
                            
                            # Wait for main thread to grab the snapshot and acknowledge
                            resp_event.wait(timeout=5.0)
                            
                            # Send ACK back to robot to unlock next move
                            conn.sendall(b"ACK")
                            
                        except Exception as ex:
                            print(f"[TCP ERROR] {ex}")
                            break
                            
                robot_connected = False
                print("[ROBOT DISCONNECTED] Client connection closed.")

    server_t = threading.Thread(target=tcp_server_thread, daemon=True)
    server_t.start()

    # --- 4. Workpiece Accumulator & Viewport ---
    output_dir = "scans"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    session_dir = os.path.join(output_dir, "session_stages")
    os.makedirs(session_dir, exist_ok=True)
    manifest_path = os.path.join(session_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            print(f"[SESSION ONLINE] Loaded {len(manifest_data.get('stages', []))} existing stages from {manifest_path}")
        except Exception:
            manifest_data = {"stages": []}
    else:
        manifest_data = {"stages": []}
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

    def run_open3d_pipeline_async():
        py311_exe = r"C:\Users\pitsi\AppData\Local\Programs\Python\Python311\python.exe"
        script_path = os.path.join("tools", "open3d_pipeline.py")
        if os.path.exists(py311_exe) and os.path.exists(script_path):
            print("\n" + "=" * 65)
            print("[OPEN3D AUTO-RUN] Launching Open3D Multi-Way ICP & Surface Fusion...")
            print("=" * 65)
            subprocess.Popen([py311_exe, script_path, "--session", session_dir])
        else:
            print("[NOTICE] Run 'run_open3d_pipeline.bat' to execute Open3D Multi-Way ICP.")

    accumulated_points = []
    accumulated_colors = []
    robot_poses_history = []
    active_robot_pose = None
    stage_count = 0
    trigger_flash_frames = 0

    viewport_w = 930
    viewport_h = 700
    viewport_x = 530
    viewport_y = 65
    viewer_3d = Robot3DViewport(width=viewport_w, height=viewport_h)

    # UI Buttons bounds
    btn_save_rect = (530, 15, 690, 55)
    btn_clean_rect = (705, 15, 875, 55)
    btn_open3d_rect = (890, 15, 1070, 55)
    btn_clear_rect = (1085, 15, 1215, 55)
    btn_reset_rect = (1230, 15, 1360, 55)

    def on_mouse(event, x, y, flags, param):
        nonlocal accumulated_points, accumulated_colors, robot_poses_history, stage_count
        
        if viewport_x <= x < viewport_x + viewport_w and viewport_y <= y < viewport_y + viewport_h:
            viewer_3d.handle_mouse(event, x - viewport_x, y - viewport_y)
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Save final workpiece button
            sx1, sy1, sx2, sy2 = btn_save_rect
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                if len(accumulated_points) > 0:
                    all_pts = np.vstack(accumulated_points)
                    all_col = np.vstack(accumulated_colors)
                    print(f"\n[SAVING WORKPIECE] Filtering {len(all_pts):,} raw points...")
                    final_pts, final_col = statistical_outlier_removal(all_pts, all_col, nb_neighbors=16, std_ratio=1.2)
                    final_pts, final_col = voxel_grid_average(final_pts, final_col, voxel_size=0.002)
                    ts = int(time.time())
                    save_path_m = os.path.join(output_dir, f"robot_workpiece_{ts}.ply")
                    save_ply_binary(save_path_m, final_pts, final_col)
                    
                    # 1:1 Millimeter PLY for direct Rhino / Grasshopper reference
                    save_path_mm = os.path.join(output_dir, f"robot_workpiece_{ts}_mm.ply")
                    save_ply_binary(save_path_mm, final_pts * 1000.0, final_col)
                    
                    # Overwrite live reference link
                    latest_mm_path = os.path.join(output_dir, "latest_workpiece_mm.ply")
                    save_ply_binary(latest_mm_path, final_pts * 1000.0, final_col)
                    
                    print(f"[SAVED] Standard (meters): {save_path_m}")
                    print(f"[SAVED] Rhino 1:1 (mm):   {save_path_mm}")
                    print(f"[RHINO READY] Updated live link: {latest_mm_path}\n")
                    run_open3d_pipeline_async()
                else:
                    print("No points collected yet in memory. Running Open3D pipeline on session stages...")
                    run_open3d_pipeline_async()

            # Clean & De-noise current scan in memory button
            cx1, cy1, cx2, cy2 = btn_clean_rect
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                if len(accumulated_points) > 0:
                    all_pts = np.vstack(accumulated_points)
                    all_col = np.vstack(accumulated_colors)
                    nb_n = int(scan_config.get("outlier_nb_neighbors", 16))
                    std_r = float(scan_config.get("outlier_std_ratio", 1.2))
                    vox_m = float(scan_config.get("voxel_size_mm", 2.0)) / 1000.0
                    clean_pts, clean_col = statistical_outlier_removal(all_pts, all_col, nb_neighbors=nb_n, std_ratio=std_r)
                    clean_pts, clean_col = voxel_grid_average(clean_pts, clean_col, voxel_size=vox_m)
                    accumulated_points = [clean_pts]
                    accumulated_colors = [clean_col]
                    print(f"[CLEAN COMPLETE] Workpiece filtered to {len(clean_pts):,} surface points.\n")
                else:
                    print("No points in buffer to clean.")

            # Run Open3D Multi-Way ICP button
            ox1, oy1, ox2, oy2 = btn_open3d_rect
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                print("\n[MANUAL TRIGGER] Launching Open3D Multi-Way ICP & Surface Reconstruction...")
                run_open3d_pipeline_async()
                    
            # Clear scan button
            clx1, cly1, clx2, cly2 = btn_clear_rect
            if clx1 <= x <= clx2 and cly1 <= y <= cly2:
                accumulated_points = []
                accumulated_colors = []
                robot_poses_history = []
                stage_count = 0
                manifest_data["stages"].clear()
                with open(os.path.join(session_dir, "manifest.json"), "w") as f:
                    json.dump(manifest_data, f, indent=2)
                print("[CLEARED] Reset workpiece scan buffer and session stages.")
                
            # Reset 3D View button
            rx1, ry1, rx2, ry2 = btn_reset_rect
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                if active_robot_pose is not None:
                    viewer_3d.reset_view(auto_target=active_robot_pose['trans'])
                else:
                    viewer_3d.reset_view()

    window_name = "ABB Robot 3D Scanning Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    canvas_w = 1480
    canvas_h = 780
    prev_time = time.time()
    fps = 0.0

    print("\n=======================================================")
    print("ABB Robot 3D Scanning Studio Ready!")
    print(f"IP: Enter your PC's IP into RAPID (Port: {SERVER_PORT})")
    print("=======================================================\n")

    last_reconnect_time = time.time()
    last_left_bgr = None
    last_depth_bgr = None

    while True:
        has_new_frame = False
        if camera_online:
            try:
                err = zed.grab()
                if err == sl.ERROR_CODE.SUCCESS:
                    has_new_frame = True
                else:
                    camera_online = False
                    print(f"\n[CAMERA DISCONNECTED] ZED Camera grab failed ({err}). Dashboard running in resilient standby...")
            except Exception as e:
                camera_online = False
                print(f"\n[CAMERA EXCEPTION] Grab error: {e}. Dashboard running in resilient standby...")
        else:
            # When camera is offline, poll for reconnect every 3.0s
            curr_t = time.time()
            if curr_t - last_reconnect_time > 3.0:
                last_reconnect_time = curr_t
                try:
                    zed.close()
                    if zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
                        camera_online = True
                        print("\n[CAMERA RECONNECTED] ZED Camera successfully reconnected!\n")
                except Exception:
                    pass
            # Avoid pegging CPU when offline
            time.sleep(0.02)

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-4))
        prev_time = curr_time

        # 1. Check for incoming Robot trigger messages (ALWAYS PROCESSED)
        if not robot_queue.empty():
            msg, resp_event = robot_queue.get()
            
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
                    
                    # 1:1 Millimeter PLY for direct Rhino / Grasshopper reference
                    save_path_mm = os.path.join(output_dir, f"robot_workpiece_{ts}_mm.ply")
                    save_ply_binary(save_path_mm, final_pts * 1000.0, final_col)
                    
                    # Overwrite live reference link
                    latest_mm_path = os.path.join(output_dir, "latest_workpiece_mm.ply")
                    save_ply_binary(latest_mm_path, final_pts * 1000.0, final_col)
                    
                    print(f"[AUTO-SAVED] Standard (meters): {save_path_m}")
                    print(f"[AUTO-SAVED] Rhino 1:1 (mm):   {save_path_mm}")
                    print(f"[RHINO READY] Updated live link: {latest_mm_path}\n")
                    
                    # Trigger Open3D Multi-Way ICP & Surface Fusion Pipeline
                    run_open3d_pipeline_async()
                else:
                    print("[AUTO-SAVING NOTICE] No accumulated points in memory, but launching Open3D pipeline on session stages...")
                    run_open3d_pipeline_async()
                resp_event.set()
            else:
                try:
                    # Format: "X,Y,Z,Q1,Q2,Q3,Q4"
                    # RAPID sends mm: X, Y, Z in mm. Q1=w, Q2=x, Q3=y, Q4=z
                    parts = [float(val.strip()) for val in msg.split(',')]
                    if len(parts) == 7:
                        tx_mm, ty_mm, tz_mm, qw, qx, qy, qz = parts
                        
                        # Convert mm to meters for robot base frame
                        t_base2ee = np.array([tx_mm / 1000.0, ty_mm / 1000.0, tz_mm / 1000.0], dtype=np.float32)
                        
                        # Scipy Rotation uses [x, y, z, w]
                        rot_base2ee = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
                        
                        # Hand-Eye transform: T_base2cam = T_base2ee * T_ee2cam
                        R_base2cam = rot_base2ee @ R_cam2ee
                        t_base2cam = rot_base2ee @ t_cam2ee.ravel() + t_base2ee
                        
                        active_robot_pose = {'rot': R_base2cam, 'trans': t_base2cam}
                        robot_poses_history.append({'rot': R_base2cam, 'trans': t_base2cam})
                        
                        
                        if camera_online:
                            try:
                                min_d = float(scan_config.get("min_depth_m", 0.25))
                                max_d = float(scan_config.get("max_depth_m", 0.70))
                                step = int(scan_config.get("subsample_step", 1))
                                crop = float(scan_config.get("border_crop", 0.06))
                                nb_n = int(scan_config.get("outlier_nb_neighbors", 16))
                                std_r = float(scan_config.get("outlier_std_ratio", 1.2))
                                
                                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                                pts_cam, col_cam = extract_valid_points(point_cloud, min_dist=min_d, max_dist=max_d, subsample_step=step, border_crop=crop)
                                pts_cam, col_cam = statistical_outlier_removal(pts_cam, col_cam, nb_neighbors=nb_n, std_ratio=std_r)
                                
                                # 1. Save raw individual scan stage in camera frame for Open3D Multi-Way ICP
                                stage_count += 1
                                stage_filename = f"stage_{stage_count:03d}.ply"
                                stage_filepath = os.path.join(session_dir, stage_filename)
                                save_ply_binary(stage_filepath, pts_cam, col_cam)
                                
                                # 2. Record stage metadata in session manifest
                                manifest_data["stages"].append({
                                    "index": stage_count,
                                    "ply": stage_filename,
                                    "pose_flange": [tx_mm, ty_mm, tz_mm, qw, qx, qy, qz],
                                    "time": time.strftime("%H:%M:%S")
                                })
                                with open(os.path.join(session_dir, "manifest.json"), "w") as f:
                                    json.dump(manifest_data, f, indent=2)
                                
                                # 3. Transform points into Robot Base Coordinates: P_base = P_cam * R^T + t (for live display)
                                pts_base = pts_cam @ R_base2cam.T + t_base2cam
                                accumulated_points.append(pts_base)
                                accumulated_colors.append(col_cam)
                                trigger_flash_frames = 8  # Flash trigger visual indicator
                                
                                print(f"[STAGE #{stage_count}] Captured ({len(pts_cam):,} pts) | Flange: ({tx_mm:.1f}, {ty_mm:.1f}, {tz_mm:.1f}) mm -> Saved to {stage_filename}")
                            except Exception as e:
                                print(f"[CAPTURE ERROR] Failed to grab frame: {e}")
                        else:
                            print(f"[CAPTURE NOTICE] Waypoint ({tx_mm:.1f}, {ty_mm:.1f}, {tz_mm:.1f}) received, but ZED camera is OFFLINE.")
                            
                except Exception as err:
                    print(f"[PARSE ERROR] Failed to parse pose: {err}")
                    
                # Acknowledge robot so it moves to next waypoint
                resp_event.set()

        # 2. Retrieve 2D RGB & Depth views for monitoring
        left_w = 500
        left_h = 281
        if has_new_frame:
            try:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                left_bgr = cv2.cvtColor(image_left.get_data(), cv2.COLOR_BGRA2BGR)
                zed.retrieve_image(image_depth, sl.VIEW.DEPTH)
                depth_bgr = cv2.cvtColor(image_depth.get_data(), cv2.COLOR_BGRA2BGR)
                rgb_small = cv2.resize(left_bgr, (left_w, left_h))
                depth_small = cv2.resize(depth_bgr, (left_w, left_h))
                last_left_bgr = rgb_small
                last_depth_bgr = depth_small
            except Exception:
                camera_online = False

        if not camera_online or last_left_bgr is None:
            if last_left_bgr is not None:
                rgb_small = cv2.addWeighted(last_left_bgr, 0.35, np.zeros_like(last_left_bgr), 0.65, 0)
            else:
                rgb_small = np.full((left_h, left_w, 3), 24, dtype=np.uint8)
            cv2.putText(rgb_small, "CAMERA OFFLINE / DISCONNECTED", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 100, 255), 2)
            cv2.putText(rgb_small, "Auto-reconnecting every 3s...", (115, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            if last_depth_bgr is not None:
                depth_small = cv2.addWeighted(last_depth_bgr, 0.35, np.zeros_like(last_depth_bgr), 0.65, 0)
            else:
                depth_small = np.full((left_h, left_w, 3), 20, dtype=np.uint8)
            cv2.putText(depth_small, "DEPTH SENSOR STANDBY", (110, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 120, 120), 1)

        # Build Dark Studio Canvas
        canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)

        # Visual trigger flash overlay when a stage is captured
        if trigger_flash_frames > 0:
            cv2.rectangle(rgb_small, (0, 0), (left_w, left_h), (0, 255, 0), 6)
            cv2.putText(rgb_small, f"STAGE #{stage_count} CAPTURED!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            trigger_flash_frames -= 1

        # Crosshairs
        if camera_online:
            ch_x = left_w // 2
            ch_y = left_h // 2
            cv2.drawMarker(rgb_small, (ch_x, ch_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.drawMarker(depth_small, (ch_x, ch_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

        # Paste Left RGB (y: 20..301, x: 15..515)
        canvas[20:20+left_h, 15:15+left_w] = rgb_small
        cv2.rectangle(canvas, (15, 20), (15+left_w, 20+left_h), (50, 50, 50), 1)
        cv2.putText(canvas, "VIEW: LEFT (RGB)", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Paste Left Depth (y: 315..596, x: 15..515)
        canvas[315:315+left_h, 15:15+left_w] = depth_small
        cv2.rectangle(canvas, (15, 315), (15+left_w, 315+left_h), (50, 50, 50), 1)
        cv2.putText(canvas, "MEASURE: DEPTH (COLORIZED)", (25, 337), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Robot Telemetry Box (Bottom Left: y: 610..765)
        card_y = 610
        cv2.rectangle(canvas, (15, card_y), (15+left_w, 765), (26, 26, 26), -1)
        cv2.rectangle(canvas, (15, card_y), (15+left_w, 765), (55, 55, 55), 1)
        
        status_color = (0, 255, 100) if robot_connected else (0, 160, 255)
        conn_text = f"CONNECTED ({robot_client_addr[0]})" if robot_connected else "WAITING FOR ROBOT (PORT 5000)"
        
        cam_status_str = f"ONLINE ({fps:.0f} FPS)" if camera_online else "OFFLINE (STANDBY)"
        cam_status_col = (0, 255, 100) if camera_online else (0, 100, 255)

        total_pts_count = sum(len(p) for p in accumulated_points)
        
        cv2.putText(canvas, f"CAMERA LINK: {cam_status_str}", (25, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cam_status_col, 1)
        cv2.putText(canvas, f"ROBOT LINK:  {conn_text}", (25, card_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        cv2.putText(canvas, f"Last Payload: {last_raw_message[:44]}", (25, card_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        
        if active_robot_pose is not None:
            t_p = active_robot_pose['trans']
            cv2.putText(canvas, f"Active Pose: X:{t_p[0]:.3f}m  Y:{t_p[1]:.3f}m  Z:{t_p[2]:.3f}m", (25, card_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(canvas, "Active Pose: None", (25, card_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
            
        cv2.putText(canvas, f"Stages Captured: {stage_count}  |  Workpiece Points: {total_pts_count:,}", (25, card_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 230, 210), 1)

        # --- Right Column: Header & 3D Viewport ---
        # Buttons
        sx1, sy1, sx2, sy2 = btn_save_rect
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (20, 140, 40), -1)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (180, 180, 180), 1)
        cv2.putText(canvas, "SAVE WORKPIECE", (sx1 + 10, sy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        clx1, cly1, clx2, cly2 = btn_clean_rect
        cv2.rectangle(canvas, (clx1, cly1), (clx2, cly2), (45, 100, 160), -1)
        cv2.rectangle(canvas, (clx1, cly1), (clx2, cly2), (180, 180, 180), 1)
        cv2.putText(canvas, "CLEAN & DE-NOISE", (clx1 + 8, cly1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        ox1, oy1, ox2, oy2 = btn_open3d_rect
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), (130, 60, 20), -1)
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), (180, 180, 180), 1)
        cv2.putText(canvas, "RUN OPEN3D ICP", (ox1 + 14, oy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        cx1, cy1, cx2, cy2 = btn_clear_rect
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (140, 140, 140), 1)
        cv2.putText(canvas, "CLEAR SCAN", (cx1 + 12, cy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

        rx1, ry1, rx2, ry2 = btn_reset_rect
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (140, 140, 140), 1)
        cv2.putText(canvas, "RECENTER", (rx1 + 16, ry1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

        # Flatten accumulated points for rendering in 3D
        if len(accumulated_points) > 0:
            flat_pts = np.vstack(accumulated_points)
            flat_col = np.vstack(accumulated_colors)
        else:
            flat_pts = None
            flat_col = None

        # Render 3D Scene Viewport
        rendered_3d = viewer_3d.render(flat_pts, flat_col, robot_poses_history, active_robot_pose)
        canvas[viewport_y:viewport_y+viewport_h, viewport_x:viewport_x+viewport_w] = rendered_3d

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(10)
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            if active_robot_pose is not None:
                viewer_3d.reset_view(auto_target=active_robot_pose['trans'])
            else:
                viewer_3d.reset_view()
        elif key == ord('p') or key == ord('P'):
            print("\n[HOTKEY 'P'] Launching Open3D Multi-Way ICP & Surface Reconstruction...")
            run_open3d_pipeline_async()

    cv2.destroyAllWindows()
    try:
        zed.close()
    except Exception:
        pass
    print("Camera and studio closed.")

if __name__ == "__main__":
    main()
