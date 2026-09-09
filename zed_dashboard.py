import pyzed.sl as sl
import cv2
import numpy as np
import os
import sys
import time

def save_ply_binary(filepath, points, colors):
    """
    Saves points (N, 3) and colors (N, 3) as binary little-endian PLY with true 24-bit RGB vertex colors.
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
    # Colors in arr are stored as R, G, B
    arr['red'] = colors[:, 0]
    arr['green'] = colors[:, 1]
    arr['blue'] = colors[:, 2]
    
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)
    return True

def extract_valid_points(point_cloud_mat, min_dist=0.2, max_dist=1.0, subsample_step=1):
    """
    Extracts valid (X, Y, Z) and true (R, G, B) from sl.Mat(XYZRGBA).
    Discards NaNs, Infs, and background points beyond max_dist.
    """
    data = point_cloud_mat.get_data()
    if subsample_step > 1:
        data = data[::subsample_step, ::subsample_step]
        
    pts = data[:, :, :3].reshape(-1, 3)
    
    # 4th float is packed 32-bit RGBA integer
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

def voxel_downsample(points, colors, voxel_size=0.002):
    """
    Fast grid deduplication using NumPy to merge overlapping point clouds cleanly.
    """
    if len(points) == 0:
        return points, colors
    vox_indices = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(vox_indices, axis=0, return_index=True)
    return points[unique_idx], colors[unique_idx]

class Interactive3DViewport:
    """
    Real-time interactive 3D viewport rendered directly into the dashboard.
    Supports orbit, zoom, pan, 3D ground grid, live camera frustum (IMU), and colored point cloud.
    """
    def __init__(self, width=900, height=680):
        self.w = width
        self.h = height
        
        # Orbit camera parameters
        self.yaw = -0.45       # Radians around Y axis
        self.pitch = 0.35      # Radians looking down
        self.distance = 1.35   # Distance from target in meters
        self.target = np.array([0.0, 0.05, 0.55], dtype=np.float32) # Look-at point
        
        # Mouse tracking state
        self.is_dragging_orbit = False
        self.is_dragging_zoom = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Precompute a 3D ground grid (plane at Y = 0.35m below target)
        self.grid_lines, self.grid_crosses = self._build_ground_grid()
        
    def _build_ground_grid(self):
        lines = []
        crosses = []
        floor_y = 0.35
        # Grid range: X from -0.8 to +0.8m, Z from 0.0 to 1.4m
        xs = np.linspace(-0.6, 0.6, 7)
        zs = np.linspace(0.1, 1.3, 7)
        
        for x in xs:
            lines.append((np.array([x, floor_y, 0.1], dtype=np.float32), np.array([x, floor_y, 1.3], dtype=np.float32)))
        for z in zs:
            lines.append((np.array([-0.6, floor_y, z], dtype=np.float32), np.array([0.6, floor_y, z], dtype=np.float32)))
            
        for x in xs:
            for z in zs:
                crosses.append(np.array([x, floor_y, z], dtype=np.float32))
                
        return lines, crosses

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
                self.yaw += dx * 0.007
                self.pitch = np.clip(self.pitch + dy * 0.007, -1.3, 1.3)
            elif self.is_dragging_zoom:
                self.distance = np.clip(self.distance + dy * 0.008, 0.3, 4.0)

    def reset_view(self):
        self.yaw = -0.45
        self.pitch = 0.35
        self.distance = 1.35
        self.target = np.array([0.0, 0.05, 0.55], dtype=np.float32)

    def render(self, pts_world, colors_rgb, cam_R, cam_t):
        """
        Renders the 3D scene:
        1. Ground grid with crosses.
        2. Live camera frustum (pyramid) + coordinate triad from real-time IMU/tracking.
        3. Real-time colored 3D point cloud.
        """
        # Canvas background (ZED Studio Dark #131313)
        view_img = np.full((self.h, self.w, 3), 18, dtype=np.uint8)
        
        # Compute orbit camera view matrix
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        
        # Orbit rotation matrix (World -> Orbit View)
        # Yaw around Y, pitch around X
        R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        R_view = R_pitch @ R_yaw
        
        focal = self.h * 1.05
        cx, cy = self.w / 2.0, self.h / 2.0
        
        def project(pts):
            # Transform pts: (P - target) @ R_view.T
            P_rel = (pts - self.target) @ R_view.T
            # Offset along optical axis by orbit distance
            Z_view = P_rel[:, 2] + self.distance
            
            valid = Z_view > 0.05
            u = np.zeros(len(pts), dtype=np.int32)
            v = np.zeros(len(pts), dtype=np.int32)
            
            u[valid] = np.clip(((P_rel[valid, 0] / Z_view[valid]) * focal + cx).astype(np.int32), -1000, 3000)
            v[valid] = np.clip(((P_rel[valid, 1] / Z_view[valid]) * focal + cy).astype(np.int32), -1000, 3000)
            return u, v, Z_view, valid

        # --- 1. Draw Ground Grid ---
        for p1, p2 in self.grid_lines:
            pts_pair = np.vstack([p1, p2])
            u, v, z, val = project(pts_pair)
            if val[0] and val[1]:
                cv2.line(view_img, (u[0], v[0]), (u[1], v[1]), (38, 48, 38), 1, cv2.LINE_AA)
                
        # Draw grid crosshairs (+) like ZED Depth Viewer
        if len(self.grid_crosses) > 0:
            c_pts = np.array(self.grid_crosses, dtype=np.float32)
            cu, cv, cz, cval = project(c_pts)
            for i in range(len(c_pts)):
                if cval[i] and 0 <= cu[i] < self.w and 0 <= cv[i] < self.h:
                    cv2.drawMarker(view_img, (cu[i], cv[i]), (55, 110, 65), cv2.MARKER_CROSS, 6, 1)

        # --- 2. Draw 3D Camera Frustum & Coordinate Triad from IMU/Pose ---
        if cam_R is not None and cam_t is not None:
            # Frustum geometry in camera local space: apex at (0,0,0), base rectangle at Z = 0.12m
            frust_d = 0.12
            frust_w = 0.09
            frust_h = 0.05
            
            local_frust = np.array([
                [0.0, 0.0, 0.0],                          # 0: Apex (Camera center)
                [-frust_w, -frust_h, frust_d],            # 1: Top-Left
                [ frust_w, -frust_h, frust_d],            # 2: Top-Right
                [ frust_w,  frust_h, frust_d],            # 3: Bottom-Right
                [-frust_w,  frust_h, frust_d],            # 4: Bottom-Left
                # Axes endpoints (length 0.14m)
                [0.14, 0.0, 0.0],                         # 5: X Axis (Red)
                [0.0, 0.14, 0.0],                         # 6: Y Axis (Green)
                [0.0, 0.0, 0.20]                          # 7: Z Optical Axis (Blue)
            ], dtype=np.float32)
            
            # Transform frustum points to world space
            world_frust = local_frust @ cam_R.T + cam_t
            fu, fv, fz, fval = project(world_frust)
            
            # Draw Frustum pyramid lines (bright cyan/yellow)
            frust_col = (0, 240, 255)
            if fval[0]:
                for corner_idx in (1, 2, 3, 4):
                    if fval[corner_idx]:
                        cv2.line(view_img, (fu[0], fv[0]), (fu[corner_idx], fv[corner_idx]), frust_col, 2, cv2.LINE_AA)
                        
                # Connect base rectangle: 1-2, 2-3, 3-4, 4-1
                for i1, i2 in ((1, 2), (2, 3), (3, 4), (4, 1)):
                    if fval[i1] and fval[i2]:
                        cv2.line(view_img, (fu[i1], fv[i1]), (fu[i2], fv[i2]), frust_col, 2, cv2.LINE_AA)
                        
                # Draw Coordinate Triad (Red=X, Green=Y, Blue=Z optical axis)
                apex_pt = (fu[0], fv[0])
                if fval[5]: cv2.line(view_img, apex_pt, (fu[5], fv[5]), (0, 0, 255), 2, cv2.LINE_AA)   # X Red
                if fval[6]: cv2.line(view_img, apex_pt, (fu[6], fv[6]), (0, 255, 0), 2, cv2.LINE_AA)   # Y Green
                if fval[7]: cv2.line(view_img, apex_pt, (fu[7], fv[7]), (255, 120, 0), 3, cv2.LINE_AA) # Z Blue optical axis
                
                # Small sphere at camera center
                cv2.circle(view_img, apex_pt, 4, (0, 255, 255), -1)

        # --- 3. Render 3D Point Cloud in True Color ---
        if pts_world is not None and len(pts_world) > 0:
            pu, pv, pz, pval = project(pts_world)
            
            in_screen = pval & (pu >= 0) & (pu < self.w) & (pv >= 0) & (pv < self.h)
            
            valid_u = pu[in_screen]
            valid_v = pv[in_screen]
            
            # Colors: colors_rgb is (N, 3) in [R, G, B]. OpenCV needs [B, G, R]
            bgr_colors = colors_rgb[in_screen][:, ::-1]
            
            # Fast vectorized buffer plotting (splatting 2x2 for solid visibility)
            view_img[valid_v, valid_u] = bgr_colors
            
            # Dilation for crisp point appearance
            v_plus = np.clip(valid_v + 1, 0, self.h - 1)
            u_plus = np.clip(valid_u + 1, 0, self.w - 1)
            view_img[v_plus, valid_u] = bgr_colors
            view_img[valid_v, u_plus] = bgr_colors

        # Viewport border & watermark
        cv2.rectangle(view_img, (0, 0), (self.w - 1, self.h - 1), (60, 60, 60), 1)
        cv2.putText(view_img, "3D POINT CLOUD & IMU POSE VIEW", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        cv2.putText(view_img, "[Left-Drag: Orbit  |  Right-Drag: Zoom  |  'R': Reset View]", (15, self.h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        return view_img


def main():
    zed = sl.Camera()

    # 1. Initialize camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.2
    init_params.depth_maximum_distance = 1.0  # Background cutoff

    print("Opening ZED Camera...")
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED Camera")
        sys.exit(1)

    # 2. Enable Positional Tracking
    tracking_params = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("Warning: Positional tracking could not be initialized.")

    # Image containers
    image_left = sl.Mat()
    image_depth = sl.Mat()
    point_cloud = sl.Mat()
    cam_pose = sl.Pose()

    # Output directory
    output_dir = "scans"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Multi-frame accumulator state
    recording = False
    scan_count = 0
    accumulated_points = []
    accumulated_colors = []
    last_capture_time = 0.0
    capture_interval = 0.35  # Capture 1 full snapshot every 350ms
    total_accumulated_pts = 0
    frames_recorded = 0

    # 3D Viewport Instance
    viewport_w = 930
    viewport_h = 700
    viewport_x = 530
    viewport_y = 65
    viewer_3d = Interactive3DViewport(width=viewport_w, height=viewport_h)

    # Buttons bounds
    btn_record_rect = (530, 15, 780, 55)
    btn_snap_rect = (800, 15, 1020, 55)
    btn_reset_rect = (1040, 15, 1200, 55)

    def on_mouse(event, x, y, flags, param):
        nonlocal recording, scan_count, accumulated_points, accumulated_colors
        nonlocal total_accumulated_pts, frames_recorded
        
        # Check if mouse is inside the 3D viewport
        if viewport_x <= x < viewport_x + viewport_w and viewport_y <= y < viewport_y + viewport_h:
            viewer_3d.handle_mouse(event, x - viewport_x, y - viewport_y)
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Multi-frame Scan Toggle
            rx1, ry1, rx2, ry2 = btn_record_rect
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                if not recording:
                    print("\n[MULTI-SCAN] Started accumulating high-res colored snapshots...")
                    accumulated_points = []
                    accumulated_colors = []
                    frames_recorded = 0
                    total_accumulated_pts = 0
                    recording = True
                else:
                    recording = False
                    print(f"\n[MULTI-SCAN] Finalizing {frames_recorded} snapshots...")
                    if len(accumulated_points) > 0:
                        all_pts = np.vstack(accumulated_points)
                        all_col = np.vstack(accumulated_colors)
                        print(f"Total raw points collected: {len(all_pts):,}")
                        
                        # 2mm voxel deduplication
                        print("Applying 2mm voxel filter to merge overlapping points...")
                        final_pts, final_col = voxel_downsample(all_pts, all_col, voxel_size=0.002)
                        print(f"Final merged count: {len(final_pts):,} points")
                        
                        out_path = os.path.join(output_dir, f"merged_scan_{scan_count:04d}.ply")
                        save_ply_binary(out_path, final_pts, final_col)
                        print(f"Saved merged scan to: {out_path}\n")
                        scan_count += 1
                    else:
                        print("No frames were captured.")

            # Single Frame Snapshot Click
            sx1, sy1, sx2, sy2 = btn_snap_rect
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                pts, col = extract_valid_points(point_cloud, min_dist=0.2, max_dist=1.0)
                snap_path = os.path.join(output_dir, f"snapshot_{scan_count:04d}.ply")
                save_ply_binary(snap_path, pts, col)
                print(f"\n[SNAPSHOT] Saved pristine colored pointcloud ({len(pts):,} points) to {snap_path}\n")
                scan_count += 1

            # Reset 3D View
            zx1, zy1, zx2, zy2 = btn_reset_rect
            if zx1 <= x <= zx2 and zy1 <= y <= zy2:
                viewer_3d.reset_view()

    window_name = "ZED Studio 3D Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    canvas_w = 1480
    canvas_h = 780
    prev_time = time.time()
    fps = 0.0

    print("Dashboard started! Realtime 3D preview active.")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-4))
            prev_time = curr_time

            # 1. Retrieve left RGB view & colorized depth view
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            left_bgr = cv2.cvtColor(image_left.get_data(), cv2.COLOR_BGRA2BGR)

            zed.retrieve_image(image_depth, sl.VIEW.DEPTH)
            depth_bgr = cv2.cvtColor(image_depth.get_data(), cv2.COLOR_BGRA2BGR)

            # 2. Retrieve laser distance at center
            depth_val_mat = sl.Mat()
            zed.retrieve_measure(depth_val_mat, sl.MEASURE.DEPTH)
            cx_pix = depth_val_mat.get_width() // 2
            cy_pix = depth_val_mat.get_height() // 2
            err_code, center_dist = depth_val_mat.get_value(cx_pix, cy_pix)

            # 3. Retrieve Camera Tracking Pose (from IMU & Visual Odometry)
            zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
            cam_rot = cam_pose.get_rotation_matrix().r   # 3x3 orientation
            cam_trans = cam_pose.get_translation().get() # [tx, ty, tz]
            
            # Keep preview camera position fixed at origin (prevents drifting out of frame)
            cam_trans_preview = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # 4. Retrieve Point Cloud for 3D Viewport (subsampled for ultra-fast 30 FPS display)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            preview_pts, preview_colors = extract_valid_points(point_cloud, min_dist=0.2, max_dist=1.0, subsample_step=4)
            
            # Rotate preview points with pitch/yaw/roll from IMU, keeping XYZ anchored at origin
            pts_world_preview = preview_pts @ cam_rot.T + cam_trans_preview

            # 5. Handle Multi-frame Scan Accumulation (full-res)
            if recording and (curr_time - last_capture_time >= capture_interval):
                full_pts, full_col = extract_valid_points(point_cloud, min_dist=0.2, max_dist=1.0, subsample_step=1)
                full_world = full_pts @ cam_rot.T + cam_trans
                accumulated_points.append(full_world)
                accumulated_colors.append(full_col)
                frames_recorded += 1
                total_accumulated_pts += len(full_world)
                last_capture_time = curr_time

            # Build Dark Canvas
            canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)

            # Left Panels (500 x 281 each)
            left_w = 500
            left_h = 281
            rgb_small = cv2.resize(left_bgr, (left_w, left_h))
            depth_small = cv2.resize(depth_bgr, (left_w, left_h))

            # Crosshairs
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

            # Telemetry Box (Bottom Left: y: 610..765)
            card_y = 610
            cv2.rectangle(canvas, (15, card_y), (15+left_w, 765), (26, 26, 26), -1)
            cv2.rectangle(canvas, (15, card_y), (15+left_w, 765), (55, 55, 55), 1)
            
            euler = cam_pose.get_euler_angles(radian=False)
            dist_str = f"{center_dist:.3f} m" if not np.isnan(center_dist) else "Out of Range"
            telemetry_lines = [
                f"Depth: NEURAL  |  FPS: {fps:.1f} Hz  |  Center: {dist_str}",
                f"IMU: Roll: {euler[0]:>5.1f}deg  Pitch: {euler[1]:>5.1f}deg  Yaw: {euler[2]:>5.1f}deg",
                f"Scanning: {'RECORDING (' + str(frames_recorded) + ' frames)' if recording else 'STANDBY'}  |  Buffer: {total_accumulated_pts:,} pts"
            ]
            for i, line in enumerate(telemetry_lines):
                cv2.putText(canvas, line, (25, card_y + 25 + (i * 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 230, 210), 1)

            # --- Right Column: Header & Interactive 3D Viewport ---
            # Buttons
            rx1, ry1, rx2, ry2 = btn_record_rect
            btn_rec_color = (0, 0, 180) if recording else (20, 140, 40)
            btn_rec_text = "STOP & MERGE SCANS" if recording else "START MULTI-SCAN"
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), btn_rec_color, -1)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (180, 180, 180), 1)
            cv2.putText(canvas, btn_rec_text, (rx1 + 18, ry1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            sx1, sy1, sx2, sy2 = btn_snap_rect
            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (55, 55, 55), -1)
            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (150, 150, 150), 1)
            cv2.putText(canvas, "SNAPSHOT PLY", (sx1 + 18, sy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            zx1, zy1, zx2, zy2 = btn_reset_rect
            cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (40, 40, 40), -1)
            cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (120, 120, 120), 1)
            cv2.putText(canvas, "RESET VIEW", (zx1 + 16, zy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

            # Render 3D Scene Viewport (camera frustum anchored at origin)
            rendered_3d = viewer_3d.render(pts_world_preview, preview_colors, cam_rot, cam_trans_preview)
            canvas[viewport_y:viewport_y+viewport_h, viewport_x:viewport_x+viewport_w] = rendered_3d

            cv2.imshow(window_name, canvas)

        key = cv2.waitKey(5)
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            viewer_3d.reset_view()

    cv2.destroyAllWindows()
    zed.disable_positional_tracking()
    zed.close()
    print("Camera closed successfully.")

if __name__ == "__main__":
    main()
