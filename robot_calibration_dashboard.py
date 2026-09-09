"""
ABB ROBOT // EYE-IN-HAND CALIBRATION DASHBOARD (OPENCV / DESKTOP STUDIO)
========================================================================
Runs natively in the dedicated Python virtual environment (./venv).

Features:
  - Background TCP socket server on port 5000 listening for RAPID toolpath coordinates.
  - Robust single-grab ZED capture loop (synchronous with OpenCV display).
  - Real-time OpenCV 5.0 ChArUco detection with visual diagnostics:
      * Draws detected corners, IDs, and 3D RGB coordinate axes on board
      * Visual & console feedback on every robot trigger (corners found, distance, success/reject)
  - Interactive 3D Viewport (Robot3DViewport):
      * 3D ChArUco board in world space
      * Robot flange trajectory with numbered waypoint spheres
      * Camera optical frustums at each pose pointing at the board
      * Mouse orbit (left-drag), pan (middle-drag), and zoom (right-drag)
  - Multi-Algorithm Eye-in-Hand Hand-Eye Calibration Solver:
      * Tsai-Lenz (CALIB_HAND_EYE_TSAI)
      * Park-Martin (CALIB_HAND_EYE_PARK)
      * Horaud (CALIB_HAND_EYE_HORAUD)
      * Andreff (CALIB_HAND_EYE_ANDREFF)
      * Daniilidis (CALIB_HAND_EYE_DANIILIDIS)
  - Evaluates consistency & residual error across all poses (T_base2target = const).
  - Automatically exports the high-precision matrix to calibration_result.npz.
"""

import socket
import threading
import queue
import os
import sys
import time
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


# =============================================================================
# 1. CHARUCO BOARD CONFIGURATION (Exact calib.io settings: 7x5, 25/18 mm, DICT_6X6_250)
# =============================================================================
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.025   # 25mm (calib.io)
MARKER_LENGTH = 0.018   # 18mm (calib.io)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Dual orientation detectors (7x5 and 5x7) for seamless detection
CHARUCO_BOARD_7X5 = aruco.CharucoBoard((7, 5), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
CHARUCO_DETECTOR_7X5 = aruco.CharucoDetector(CHARUCO_BOARD_7X5)

CHARUCO_BOARD_5X7 = aruco.CharucoBoard((5, 7), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
CHARUCO_DETECTOR_5X7 = aruco.CharucoDetector(CHARUCO_BOARD_5X7)

def detect_charuco_auto(gray):
    """
    Robust dual-orientation ChArUco detector.
    Tests both (7,5) and (5,7) to match any print rotation.
    Returns: corners (N, 1, 2), ids, markerCorners, markerIds, active_board
    """
    c1, ids1, mc1, mi1 = CHARUCO_DETECTOR_7X5.detectBoard(gray)
    if c1 is not None and len(c1) >= 4:
        return c1.reshape(-1, 1, 2), ids1, mc1, mi1, CHARUCO_BOARD_7X5

    c2, ids2, mc2, mi2 = CHARUCO_DETECTOR_5X7.detectBoard(gray)
    if c2 is not None and len(c2) >= 4:
        return c2.reshape(-1, 1, 2), ids2, mc2, mi2, CHARUCO_BOARD_5X7

    if c1 is not None:
        return c1.reshape(-1, 1, 2), ids1, mc1, mi1, CHARUCO_BOARD_7X5
    if c2 is not None:
        return c2.reshape(-1, 1, 2), ids2, mc2, mi2, CHARUCO_BOARD_5X7
    return None, None, mc1, mi1, CHARUCO_BOARD_7X5

def safe_draw_charuco_corners(img, corners, ids, color=(0, 255, 0)):
    """Safe drawing wrapper preventing OpenCV 5.0 assertion failures."""
    if corners is None or len(corners) == 0:
        return
    try:
        c_reshaped = corners.reshape(-1, 1, 2)
        aruco.drawDetectedCornersCharuco(img, c_reshaped, ids, color)
    except Exception:
        for pt in corners.reshape(-1, 2):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1, cv2.LINE_AA)

# Default fallback intrinsics for ZED Mini HD720 if camera offline
DEFAULT_CAM_MATRIX = np.array([
    [700.0,   0.0, 640.0],
    [  0.0, 700.0, 360.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)
DEFAULT_DIST_COEFFS = np.zeros(5, dtype=np.float64)


# =============================================================================
# 2. INTERACTIVE 3D VIEWPORT (Robot3DViewport)
# =============================================================================
class Calibration3DViewport:
    """
    3D viewport visualizing the robot base space, ChArUco calibration board,
    flange trajectory waypoints, and camera optical frustums.
    """
    def __init__(self, width=930, height=720):
        self.w = width
        self.h = height

        # Orbit camera parameters (centered on ABB robot workspace)
        self.yaw = -0.55
        self.pitch = 0.40
        self.distance = 2.6
        self.target = np.array([0.0, -2.1, 1.0], dtype=np.float32)  # Focus near HandtoEye.modx positions

        # Mouse interaction
        self.is_dragging_orbit = False
        self.is_dragging_pan = False
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

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.is_dragging_pan = True
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_dragging_pan = False

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
            elif self.is_dragging_pan:
                cy, sy = np.cos(self.yaw), np.sin(self.yaw)
                cp, sp = np.cos(self.pitch), np.sin(self.pitch)
                right = np.array([cy, 0, -sy], dtype=np.float32)
                up = np.array([-sy * sp, cp, -cy * sp], dtype=np.float32)
                self.target -= (right * dx - up * dy) * 0.002 * self.distance
            elif self.is_dragging_zoom:
                self.distance = np.clip(self.distance + dy * 0.010, 0.4, 8.0)

    def reset_view(self, auto_target=None):
        self.yaw = -0.55
        self.pitch = 0.40
        self.distance = 2.6
        if auto_target is not None:
            self.target = auto_target.copy()
        else:
            self.target = np.array([0.0, -2.1, 1.0], dtype=np.float32)

    def render(self, recorded_poses, rejected_poses=None, board_pose_world=None, solved_cam2ee=None):
        view_img = np.full((self.h, self.w, 3), 16, dtype=np.uint8)

        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)

        R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        R_view = R_pitch @ R_yaw

        focal = self.h * 1.05
        cx, cy_center = self.w / 2.0, self.h / 2.0

        def project(pts):
            P_rel = (pts - self.target) @ R_view.T
            Z_view = P_rel[:, 2] + self.distance
            valid = Z_view > 0.05
            u = np.zeros(len(pts), dtype=np.int32)
            v = np.zeros(len(pts), dtype=np.int32)
            u[valid] = np.clip(((P_rel[valid, 0] / Z_view[valid]) * focal + cx).astype(np.int32), -1000, 3000)
            v[valid] = np.clip(((-P_rel[valid, 1] / Z_view[valid]) * focal + cy_center).astype(np.int32), -1000, 3000)
            return u, v, Z_view, valid

        # --- 1. Robot Base World Grid (centered around workspace) ---
        grid_lines = []
        tx0, ty0 = float(self.target[0]), float(self.target[1])
        for x in np.linspace(tx0 - 1.2, tx0 + 1.2, 9):
            grid_lines.append((np.array([x, ty0 - 1.2, 0.0], dtype=np.float32), np.array([x, ty0 + 1.2, 0.0], dtype=np.float32)))
        for y in np.linspace(ty0 - 1.2, ty0 + 1.2, 9):
            grid_lines.append((np.array([tx0 - 1.2, y, 0.0], dtype=np.float32), np.array([tx0 + 1.2, y, 0.0], dtype=np.float32)))

        for p1, p2 in grid_lines:
            u, v, z, val = project(np.vstack([p1, p2]))
            if val[0] and val[1]:
                cv2.line(view_img, (u[0], v[0]), (u[1], v[1]), (28, 36, 28), 1, cv2.LINE_AA)

        # Cross markers at grid intersections
        cross_pts = []
        s = 0.02
        for x in np.linspace(tx0 - 1.2, tx0 + 1.2, 9):
            for y in np.linspace(ty0 - 1.2, ty0 + 1.2, 9):
                cross_pts.extend([
                    [x - s, y, 0.0], [x + s, y, 0.0],
                    [x, y - s, 0.0], [x, y + s, 0.0]
                ])
        if len(cross_pts) > 0:
            cu, cv, cz, cval = project(np.array(cross_pts, dtype=np.float32))
            for i in range(0, len(cross_pts), 2):
                if cval[i] and cval[i+1]:
                    cv2.line(view_img, (cu[i], cv[i]), (cu[i+1], cv[i+1]), (45, 65, 45), 1, cv2.LINE_AA)

        # --- 2. ChArUco Board in 3D Space ---
        if board_pose_world is not None:
            bw = SQUARES_X * SQUARE_LENGTH
            bh = SQUARES_Y * SQUARE_LENGTH
            board_corners_local = np.array([
                [0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]
            ], dtype=np.float32)

            R_bw = board_pose_world[:3, :3]
            t_bw = board_pose_world[:3, 3]
            board_corners_world = board_corners_local @ R_bw.T + t_bw

            bu, bv, bz, bval = project(board_corners_world)
            if np.all(bval):
                pts_poly = np.array([[bu[i], bv[i]] for i in range(4)], dtype=np.int32)
                cv2.fillPoly(view_img, [pts_poly], (30, 30, 40))
                cv2.polylines(view_img, [pts_poly], isClosed=True, color=(240, 240, 240), thickness=2, lineType=cv2.LINE_AA)

                # Board 3D Coordinate Axis Triad
                axis_len = 0.09
                axes_local = np.array([
                    [0, 0, 0],
                    [axis_len, 0, 0],
                    [0, axis_len, 0],
                    [0, 0, axis_len]
                ], dtype=np.float32)
                axes_world = axes_local @ R_bw.T + t_bw
                au, av, az, aval = project(axes_world)
                if aval[0]:
                    if aval[1]: cv2.line(view_img, (au[0], av[0]), (au[1], av[1]), (0, 0, 255), 2, cv2.LINE_AA) # X Red
                    if aval[2]: cv2.line(view_img, (au[0], av[0]), (au[2], av[2]), (0, 255, 0), 2, cv2.LINE_AA) # Y Green
                    if aval[3]: cv2.line(view_img, (au[0], av[0]), (au[3], av[3]), (255, 120, 0), 2, cv2.LINE_AA) # Z Blue

        # --- 3. Robot Flange Trajectory & Waypoint Spheres ---
        all_display_poses = recorded_poses
        if len(all_display_poses) > 0:
            flange_pts = np.array([p['t_g2b'].ravel() for p in all_display_poses], dtype=np.float32)
            fu, fv, fz, fval = project(flange_pts)

            # Trajectory Line
            if len(flange_pts) > 1:
                for i in range(len(flange_pts) - 1):
                    if fval[i] and fval[i+1]:
                        cv2.line(view_img, (fu[i], fv[i]), (fu[i+1], fv[i+1]), (198, 241, 53), 2, cv2.LINE_AA)

            # Waypoint Spheres & Frustums
            for idx, p in enumerate(all_display_poses):
                if not fval[idx]:
                    continue
                is_latest = (idx == len(all_display_poses) - 1)
                color = (56, 189, 248) if not is_latest else (198, 241, 53)
                radius = 7 if not is_latest else 10
                cv2.circle(view_img, (fu[idx], fv[idx]), radius, color, -1, cv2.LINE_AA)
                cv2.circle(view_img, (fu[idx], fv[idx]), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view_img, str(idx + 1), (fu[idx] + 10, fv[idx] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

                # Camera Frustum pyramid pointing at board
                R_g2b = p['R_g2b']
                t_g2b = p['t_g2b'].ravel()

                if solved_cam2ee is not None:
                    R_c2e, t_c2e = solved_cam2ee
                    R_c2b = R_g2b @ R_c2e
                    t_c2b = R_g2b @ t_c2e.ravel() + t_g2b
                else:
                    R_c2b = R_g2b
                    t_c2b = t_g2b

                fd, fw, fh = 0.12, 0.08, 0.05
                f_corners_local = np.array([
                    [0, 0, 0],
                    [-fw, -fh, fd],
                    [ fw, -fh, fd],
                    [ fw,  fh, fd],
                    [-fw,  fh, fd]
                ], dtype=np.float32)
                f_corners_world = f_corners_local @ R_c2b.T + t_c2b
                pru, prv, prz, prval = project(f_corners_world)

                if prval[0]:
                    frust_color = (0, 200, 255) if not is_latest else (198, 241, 53)
                    for c_idx in range(1, 5):
                        if prval[c_idx]:
                            cv2.line(view_img, (pru[0], prv[0]), (pru[c_idx], prv[c_idx]), frust_color, 1, cv2.LINE_AA)
                    for c_idx in range(1, 5):
                        nxt = c_idx + 1 if c_idx < 4 else 1
                        if prval[c_idx] and prval[nxt]:
                            cv2.line(view_img, (pru[c_idx], prv[c_idx]), (pru[nxt], prv[nxt]), frust_color, 1, cv2.LINE_AA)

        # Render any rejected poses in amber
        if rejected_poses and len(rejected_poses) > 0:
            rej_pts = np.array([p['t_g2b'].ravel() for p in rejected_poses], dtype=np.float32)
            ru, rv, rz, rval = project(rej_pts)
            for idx in range(len(rejected_poses)):
                if rval[idx]:
                    cv2.circle(view_img, (ru[idx], rv[idx]), 6, (0, 165, 255), -1, cv2.LINE_AA)
                    cv2.putText(view_img, "X", (ru[idx] - 4, rv[idx] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        # Viewport Header Label & Mouse Hint
        cv2.putText(view_img, "3D ROBOT WORKSPACE // CALIBRATION TOOLPATH", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA)
        cv2.putText(view_img, f"Valid Poses: {len(recorded_poses)}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (198, 241, 53), 1, cv2.LINE_AA)
        cv2.putText(view_img, "Left-Drag: Orbit | Middle-Drag: Pan | Right-Drag: Zoom", (16, self.h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1, cv2.LINE_AA)

        return view_img


# =============================================================================
# 3. NATIVE EYE-IN-HAND CALIBRATION SOLVERS & CONSISTENCY EVALUATOR
# =============================================================================
def _skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def _compute_motion_pairs(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, min_rot_deg=0.8):
    """
    Computes relative transformation pairs (A_ij, B_ij) for all pose combinations.
    A_ij = inv(T_g2b_j) @ T_g2b_i (robot gripper relative motion)
    B_ij = T_t2c_j @ inv(T_t2c_i) (camera relative motion)
    Relation: A_ij @ X = X @ B_ij where X = T_cam2gripper
    """
    n = len(R_g2b_list)
    A_list = []
    B_list = []

    for i in range(n):
        for j in range(i + 1, n):
            R_gj, t_gj = R_g2b_list[j], t_g2b_list[j]
            R_gi, t_gi = R_g2b_list[i], t_g2b_list[i]

            R_A = R_gj.T @ R_gi
            t_A = R_gj.T @ (t_gi - t_gj)

            rotvec_A = Rotation.from_matrix(R_A).as_rotvec()
            angle_A = np.degrees(np.linalg.norm(rotvec_A))
            if angle_A < min_rot_deg:
                continue

            R_cj, t_cj = R_t2c_list[j], t_t2c_list[j]
            R_ci, t_ci = R_t2c_list[i], t_t2c_list[i]

            R_B = R_cj @ R_ci.T
            t_B = t_cj - R_B @ t_ci

            A_list.append((R_A, t_A))
            B_list.append((R_B, t_B))

    # Fallback without threshold if rotation angles were subtle
    if len(A_list) < 3:
        for i in range(n):
            for j in range(i + 1, n):
                R_gj, t_gj = R_g2b_list[j], t_g2b_list[j]
                R_gi, t_gi = R_g2b_list[i], t_g2b_list[i]
                R_A = R_gj.T @ R_gi
                t_A = R_gj.T @ (t_gi - t_gj)
                R_cj, t_cj = R_t2c_list[j], t_t2c_list[j]
                R_ci, t_ci = R_t2c_list[i], t_t2c_list[i]
                R_B = R_cj @ R_ci.T
                t_B = t_cj - R_B @ t_ci
                A_list.append((R_A, t_A))
                B_list.append((R_B, t_B))

    return A_list, B_list


def _solve_park_martin(A_list, B_list):
    """Park & Martin (1994) SO(3) closed-form Lie-algebra SVD solver."""
    M = np.zeros((3, 3), dtype=np.float64)
    for (R_A, _), (R_B, _) in zip(A_list, B_list):
        alpha = Rotation.from_matrix(R_A).as_rotvec()
        beta = Rotation.from_matrix(R_B).as_rotvec()
        M += np.outer(alpha, beta)

    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(U @ Vt)
    R_X = U @ np.diag([1.0, 1.0, d]) @ Vt

    C_list, d_list = [], []
    for (R_A, t_A), (R_B, t_B) in zip(A_list, B_list):
        C_list.append(R_A - np.eye(3))
        d_list.append(R_X @ t_B - t_A)

    C = np.vstack(C_list)
    d_vec = np.vstack(d_list)
    t_X, _, _, _ = np.linalg.lstsq(C, d_vec, rcond=None)
    return R_X, t_X.reshape(3, 1)


def _solve_tsai_lenz(A_list, B_list):
    """Tsai & Lenz (1989) closed-form linear least-squares solver."""
    S_list, y_list = [], []
    for (R_A, _), (R_B, _) in zip(A_list, B_list):
        rv_A = Rotation.from_matrix(R_A).as_rotvec()
        th_A = np.linalg.norm(rv_A)
        p_A = 2.0 * np.tan(th_A / 2.0) * (rv_A / th_A) if th_A > 1e-12 else np.zeros(3)

        rv_B = Rotation.from_matrix(R_B).as_rotvec()
        th_B = np.linalg.norm(rv_B)
        p_B = 2.0 * np.tan(th_B / 2.0) * (rv_B / th_B) if th_B > 1e-12 else np.zeros(3)

        S_list.append(_skew(p_A + p_B))
        y_list.append(p_B - p_A)

    S = np.vstack(S_list)
    y = np.concatenate(y_list)
    p_prime, _, _, _ = np.linalg.lstsq(S, y, rcond=None)

    norm_pp = np.linalg.norm(p_prime)
    theta = 2.0 * np.arctan(norm_pp)
    rotvec = theta * (p_prime / norm_pp) if norm_pp > 1e-12 else np.zeros(3)
    R_X = Rotation.from_rotvec(rotvec).as_matrix()

    C_list, d_list = [], []
    for (R_A, t_A), (R_B, t_B) in zip(A_list, B_list):
        C_list.append(R_A - np.eye(3))
        d_list.append(R_X @ t_B - t_A)

    C = np.vstack(C_list)
    d_vec = np.vstack(d_list)
    t_X, _, _, _ = np.linalg.lstsq(C, d_vec, rcond=None)
    return R_X, t_X.reshape(3, 1)


def _refine_nonlinear(R_init, t_init, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
    """Levenberg-Marquardt joint refinement minimizing board 3D spatial variance in world coordinates."""
    def residual_fn(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        t = p[3:6].reshape(3, 1)
        board_pts = []
        for Rg, tg, Rc, tc in zip(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
            tb = Rg @ (R @ tc + t) + tg
            board_pts.append(tb.ravel())
        board_pts = np.array(board_pts)
        centroid = np.mean(board_pts, axis=0)
        return (board_pts - centroid).ravel()

    x0 = np.concatenate([Rotation.from_matrix(R_init).as_rotvec(), t_init.ravel()])
    res = least_squares(residual_fn, x0, method='lm')
    R_opt = Rotation.from_rotvec(res.x[:3]).as_matrix()
    t_opt = res.x[3:6].reshape(3, 1)
    return R_opt, t_opt


def _evaluate_consistency(R_c2g, t_c2g, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
    """
    Evaluates residual consistency across all recorded poses.
    The target board is stationary in base frame: T_t2b = T_g2b @ T_c2g @ T_t2c
    """
    target_positions = []
    target_rotations = []
    for Rg, tg, Rc, tc in zip(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
        t_t2b = Rg @ (R_c2g @ tc + t_c2g) + tg
        R_t2b = Rg @ R_c2g @ Rc
        target_positions.append(t_t2b.ravel())
        target_rotations.append(R_t2b)

    target_positions = np.array(target_positions)
    pos_mean = np.mean(target_positions, axis=0)
    trans_errors = np.linalg.norm(target_positions - pos_mean, axis=1) * 1000.0  # mm
    mean_trans_error_mm = float(np.mean(trans_errors))
    max_trans_error_mm = float(np.max(trans_errors))

    try:
        rot_mean = Rotation.from_matrix(target_rotations).mean()
        rot_errors = [np.degrees(np.linalg.norm((Rotation.from_matrix(R) * rot_mean.inv()).as_rotvec())) for R in target_rotations]
        mean_rot_error_deg = float(np.mean(rot_errors))
    except Exception:
        mean_rot_error_deg = 0.0

    euler_deg = Rotation.from_matrix(R_c2g).as_euler('xyz', degrees=True)
    return {
        'mean_err_mm': mean_trans_error_mm,
        'max_err_mm': max_trans_error_mm,
        'rot_err_deg': mean_rot_error_deg,
        'board_world_pos': pos_mean,
        'euler_deg': euler_deg
    }


def solve_and_evaluate_calibration(recorded_poses):
    """
    Pure Python/NumPy/SciPy Eye-in-Hand calibration suite (compatible with OpenCV 5.0+):
      1. Park & Martin (1994) SO(3) closed-form Lie solver
      2. Tsai & Lenz (1989) linear least-squares solver
      3. Non-Linear Levenberg-Marquardt consistency optimizer
    Evaluates physical consistency (T_t2b = const) and selects optimal transformation.
    """
    if len(recorded_poses) < 4:
        return None, "Minimum 4 pose pairs required (recommended 15+)."

    R_g2b_list = [p['R_g2b'] for p in recorded_poses]
    t_g2b_list = [p['t_g2b'] for p in recorded_poses]
    R_t2c_list = [p['R_t2c'] for p in recorded_poses]
    t_t2c_list = [p['t_t2c'] for p in recorded_poses]

    A_list, B_list = _compute_motion_pairs(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
    if len(A_list) < 3:
        return None, "Not enough distinct robot motions to solve calibration."

    print("\n" + "=" * 70)
    print(f"  SOLVING EYE-IN-HAND CALIBRATION ACROSS {len(recorded_poses)} POSES ({len(A_list)} PAIRS)")
    print("=" * 70)

    breakdown = []
    best_result = None
    min_trans_error = float('inf')

    # Algorithm 1: Park-Martin
    try:
        R_pm, t_pm = _solve_park_martin(A_list, B_list)
        eval_pm = _evaluate_consistency(R_pm, t_pm, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
        entry_pm = {
            'name': 'Park-Martin',
            'R': R_pm,
            't': t_pm,
            **eval_pm
        }
        breakdown.append(entry_pm)
        print(f"[{'Park-Martin':14s}] Mean Jitter: {eval_pm['mean_err_mm']:6.2f} mm | Max: {eval_pm['max_err_mm']:6.2f} mm | Cam Offset: [{t_pm[0,0]*1000:6.1f}, {t_pm[1,0]*1000:6.1f}, {t_pm[2,0]*1000:6.1f}] mm")
        if eval_pm['mean_err_mm'] < min_trans_error:
            min_trans_error = eval_pm['mean_err_mm']
            best_result = entry_pm
    except Exception as err:
        print(f"[{'Park-Martin':14s}] Failed: {err}")

    # Algorithm 2: Tsai-Lenz
    try:
        R_tl, t_tl = _solve_tsai_lenz(A_list, B_list)
        eval_tl = _evaluate_consistency(R_tl, t_tl, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
        entry_tl = {
            'name': 'Tsai-Lenz',
            'R': R_tl,
            't': t_tl,
            **eval_tl
        }
        breakdown.append(entry_tl)
        print(f"[{'Tsai-Lenz':14s}] Mean Jitter: {eval_tl['mean_err_mm']:6.2f} mm | Max: {eval_tl['max_err_mm']:6.2f} mm | Cam Offset: [{t_tl[0,0]*1000:6.1f}, {t_tl[1,0]*1000:6.1f}, {t_tl[2,0]*1000:6.1f}] mm")
        if eval_tl['mean_err_mm'] < min_trans_error:
            min_trans_error = eval_tl['mean_err_mm']
            best_result = entry_tl
    except Exception as err:
        print(f"[{'Tsai-Lenz':14s}] Failed: {err}")

    # Algorithm 3: Non-linear Levenberg-Marquardt Refinement
    if best_result is not None:
        try:
            R_opt, t_opt = _refine_nonlinear(best_result['R'], best_result['t'], R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
            eval_opt = _evaluate_consistency(R_opt, t_opt, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
            entry_opt = {
                'name': 'Nonlinear LM',
                'R': R_opt,
                't': t_opt,
                **eval_opt
            }
            breakdown.append(entry_opt)
            print(f"[{'Nonlinear LM':14s}] Mean Jitter: {eval_opt['mean_err_mm']:6.2f} mm | Max: {eval_opt['max_err_mm']:6.2f} mm | Cam Offset: [{t_opt[0,0]*1000:6.1f}, {t_opt[1,0]*1000:6.1f}, {t_opt[2,0]*1000:6.1f}] mm")
            if eval_opt['mean_err_mm'] < min_trans_error:
                min_trans_error = eval_opt['mean_err_mm']
                best_result = entry_opt
        except Exception as err:
            print(f"[{'Nonlinear LM':14s}] Failed: {err}")

    print("=" * 70 + "\n")
    return best_result, breakdown


# =============================================================================
# 4. MAIN DASHBOARD APPLICATION
# =============================================================================
def main():
    print("\n" + "=" * 65)
    print("      ABB ROBOT // EYE-IN-HAND CALIBRATION DASHBOARD")
    print("=" * 65)

    # --- Initialize ZED Camera ---
    zed = None
    cam_matrix = DEFAULT_CAM_MATRIX
    dist_coeffs = DEFAULT_DIST_COEFFS
    camera_online = False

    try:
        import pyzed.sl as sl
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # Calibration only requires RGB

        print("[ZED] Initializing camera...")
        if zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            camera_online = True
            camera_info = zed.get_camera_information()
            left_cam = camera_info.camera_configuration.calibration_parameters.left_cam
            cam_matrix = np.array([
                [left_cam.fx, 0, left_cam.cx],
                [0, left_cam.fy, left_cam.cy],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.array(left_cam.disto[:5], dtype=np.float64)
            print(f"[ZED ONLINE] fx={left_cam.fx:.1f}, fy={left_cam.fy:.1f}, cx={left_cam.cx:.1f}, cy={left_cam.cy:.1f}")
        else:
            print("[ZED WARNING] Camera open failed. Operating in Standby mode.")
    except Exception as e:
        print(f"[ZED WARNING] ZED SDK exception: {e}. Operating in Standby mode.")

    # --- Background Socket Server for ABB Robot ---
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 5000

    robot_connected = False
    robot_client_ip = "None"
    total_robot_packets = 0
    last_raw_message = "Waiting for robot socket connection..."
    robot_queue = queue.Queue()

    def tcp_server_thread():
        nonlocal robot_connected, robot_client_ip, last_raw_message, total_robot_packets
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((SERVER_HOST, SERVER_PORT))
            s.listen(1)
            print(f"[TCP SERVER] Listening on 0.0.0.0:{SERVER_PORT} for ABB Robot...")

            while True:
                conn, addr = s.accept()
                robot_client_ip = addr[0]
                robot_connected = True
                print(f"\n=======================================================")
                print(f"[ROBOT LINK ONLINE] Robot connected from {addr[0]}:{addr[1]}!")
                print(f"=======================================================\n")

                with conn:
                    while True:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                            msg = data.decode('utf-8').strip()
                            last_raw_message = msg
                            total_robot_packets += 1
                            print(f"[TCP IN #{total_robot_packets}] {msg}")

                            # Pass to main thread for ChArUco detection on latest camera frame
                            resp_event = threading.Event()
                            robot_queue.put((msg, resp_event))

                            # Wait for snapshot capture before sending ACK back to robot
                            resp_event.wait(timeout=5.0)
                            conn.sendall(b"ACK")
                            print(" -> [ACK SENT] Robot unblocked.")

                        except Exception as ex:
                            print(f"[TCP ERROR] {ex}")
                            break

                robot_connected = False
                print("[ROBOT LINK OFFLINE] Connection closed by robot.")

    server_t = threading.Thread(target=tcp_server_thread, daemon=True)
    server_t.start()

    # --- State Variables ---
    recorded_poses = []
    rejected_poses = []
    board_pose_world = None
    solved_result = None
    solved_cam2ee = None
    status_banner_text = "READY // WAITING FOR ROBOT CONNECTION"
    status_banner_time = time.time()
    active_flange_pose = None

    # Viewport & UI Coordinates
    viewport_w = 930
    viewport_h = 720
    viewport_x = 530
    viewport_y = 70
    viewer_3d = Calibration3DViewport(width=viewport_w, height=viewport_h)

    # UI Buttons Rects (x1, y1, x2, y2)
    btn_solve_rect = (530, 15, 750, 55)
    btn_save_rect  = (765, 15, 935, 55)
    btn_test_rect  = (950, 15, 1090, 55)
    btn_clear_rect = (1105, 15, 1225, 55)
    btn_reset_rect = (1240, 15, 1370, 55)

    def trigger_calibration():
        nonlocal solved_result, solved_cam2ee, board_pose_world, status_banner_text, status_banner_time
        if len(recorded_poses) < 4:
            status_banner_text = f"ERROR: Only {len(recorded_poses)} poses recorded. Minimum 4 needed (15+ recommended)!"
            status_banner_time = time.time()
            print(f"[!] Cannot calibrate: {len(recorded_poses)} poses collected. Minimum 4 required.")
            return

        best, breakdown = solve_and_evaluate_calibration(recorded_poses)
        if best:
            solved_result = best
            solved_cam2ee = (best['R'], best['t'])
            R_bw = np.eye(3)
            t_bw = best['board_world_pos']
            T_bw = np.eye(4)
            T_bw[:3, :3] = R_bw
            T_bw[:3, 3] = t_bw
            board_pose_world = T_bw

            status_banner_text = f"CALIBRATED [{best['name']}] Mean Jitter: {best['mean_err_mm']:.2f}mm"
            status_banner_time = time.time()

            # Auto-save to calibration_result.npz
            np.savez("calibration_result.npz", R=best['R'], t=best['t'])
            print(f"[+] Saved high-precision matrix to calibration_result.npz ({best['name']})")

    def on_mouse(event, x, y, flags, param):
        nonlocal recorded_poses, rejected_poses, status_banner_text, status_banner_time, viewer_3d

        if viewport_x <= x < viewport_x + viewport_w and viewport_y <= y < viewport_y + viewport_h:
            viewer_3d.handle_mouse(event, x - viewport_x, y - viewport_y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_solve_rect[0] <= x <= btn_solve_rect[2] and btn_solve_rect[1] <= y <= btn_solve_rect[3]:
                trigger_calibration()

            elif btn_save_rect[0] <= x <= btn_save_rect[2] and btn_save_rect[1] <= y <= btn_save_rect[3]:
                if solved_result:
                    np.savez("calibration_result.npz", R=solved_result['R'], t=solved_result['t'])
                    status_banner_text = "SAVED: calibration_result.npz"
                    status_banner_time = time.time()
                    print("[+] Manually saved calibration_result.npz")
                else:
                    status_banner_text = "ERROR: Run 'SOLVE CALIBRATION' first!"
                    status_banner_time = time.time()

            elif btn_test_rect[0] <= x <= btn_test_rect[2] and btn_test_rect[1] <= y <= btn_test_rect[3]:
                simulate_test_pose()

            elif btn_clear_rect[0] <= x <= btn_clear_rect[2] and btn_clear_rect[1] <= y <= btn_clear_rect[3]:
                recorded_poses.clear()
                rejected_poses.clear()
                status_banner_text = "CLEARED: Calibration buffer reset."
                status_banner_time = time.time()
                print("[*] Cleared all poses.")

            elif btn_reset_rect[0] <= x <= btn_reset_rect[2] and btn_reset_rect[1] <= y <= btn_reset_rect[3]:
                if len(recorded_poses) > 0:
                    viewer_3d.reset_view(auto_target=recorded_poses[-1]['t_g2b'].ravel())
                else:
                    viewer_3d.reset_view()

    def simulate_test_pose():
        nonlocal recorded_poses, status_banner_text, status_banner_time
        idx = len(recorded_poses)
        angle = idx * 0.35
        # Generate poses in typical ABB work envelope
        tx = -0.05 + 0.25 * np.cos(angle)
        ty = -2.10 + 0.25 * np.sin(angle)
        tz = 1.05 + 0.08 * np.sin(idx)

        rot_scipy = Rotation.from_euler('xyz', [180.0 + 15*np.sin(angle), 15*np.cos(angle), 0], degrees=True)
        R_g2b = rot_scipy.as_matrix().astype(np.float64)
        t_g2b = np.array([[tx], [ty], [tz]], dtype=np.float64)

        R_t2c = np.eye(3, dtype=np.float64)
        t_t2c = np.array([[0.02 * np.sin(angle)], [0.01 * np.cos(angle)], [0.42]], dtype=np.float64)

        recorded_poses.append({
            'stage': idx + 1,
            'R_g2b': R_g2b,
            't_g2b': t_g2b,
            'R_t2c': R_t2c,
            't_t2c': t_t2c,
            'corners_count': 32,
            'time': time.strftime("%H:%M:%S")
        })
        status_banner_text = f"SIMULATED POSE #{idx + 1} ADDED"
        status_banner_time = time.time()
        print(f"[TEST] Added synthetic pose #{idx + 1}")

    # Window Setup
    window_name = "ABB Robot // Eye-in-Hand Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1480, 880)
    cv2.setMouseCallback(window_name, on_mouse)

    image_sl = None
    if camera_online:
        import pyzed.sl as sl
        image_sl = sl.Mat()

    latest_camera_frame_bgr = None
    prev_loop_time = time.time()
    fps = 0.0

    print("\n=======================================================")
    print("Calibration Studio Ready!")
    print(f"Waiting for robot connection on Port {SERVER_PORT}...")
    print("=======================================================\n")

    # --- Main Unified Loop ---
    while True:
        # 1. Grab camera frame ONCE per iteration
        has_new_frame = False
        if camera_online:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_sl, sl.VIEW.LEFT)
                frame_bgra = image_sl.get_data()
                latest_camera_frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                has_new_frame = True
        else:
            time.sleep(0.025)

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_loop_time, 1e-4))
        prev_loop_time = curr_time

        # 2. Check for incoming Robot trigger messages
        if not robot_queue.empty():
            msg, resp_event = robot_queue.get()

            if msg == "DONE":
                print("\n[ROBOT DONE] Robot finished toolpath! Solving calibration...")
                trigger_calibration()
                resp_event.set()
            else:
                try:
                    parts = [float(x.strip()) for x in msg.split(',')]
                    if len(parts) == 7:
                        tx_mm, ty_mm, tz_mm, q1, q2, q3, q4 = parts
                        tx, ty, tz = tx_mm / 1000.0, ty_mm / 1000.0, tz_mm / 1000.0
                        t_g2b = np.array([[tx], [ty], [tz]], dtype=np.float64)

                        # ABB quaternions: q1=w, q2=x, q3=y, q4=z
                        rot_scipy = Rotation.from_quat([q2, q3, q4, q1])
                        R_g2b = rot_scipy.as_matrix().astype(np.float64)

                        active_flange_pose = {
                            't': [tx, ty, tz],
                            'q': [q1, q2, q3, q4],
                            'euler': rot_scipy.as_euler('xyz', degrees=True)
                        }

                        # Auto-center 3D viewport on first real robot pose
                        if len(recorded_poses) == 0 and len(rejected_poses) == 0:
                            viewer_3d.reset_view(auto_target=np.array([tx, ty, tz], dtype=np.float32))

                        # Detect board on current frame
                        detected = False
                        corners_found = 0

                        if latest_camera_frame_bgr is not None:
                            gray = cv2.cvtColor(latest_camera_frame_bgr, cv2.COLOR_BGR2GRAY)
                            charucoCorners, charucoIds, markerCorners, markerIds, active_board = detect_charuco_auto(gray)

                            if charucoCorners is not None and charucoIds is not None:
                                corners_found = len(charucoCorners)
                                if corners_found >= 4:
                                    objPoints, imgPoints = active_board.matchImagePoints(charucoCorners, charucoIds)
                                    valid, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cam_matrix, dist_coeffs)

                                    if valid:
                                        R_t2c, _ = cv2.Rodrigues(rvec)
                                        t_t2c = tvec

                                        pose_record = {
                                            'stage': len(recorded_poses) + 1,
                                            'R_g2b': R_g2b,
                                            't_g2b': t_g2b,
                                            'R_t2c': R_t2c,
                                            't_t2c': t_t2c,
                                            'corners_count': corners_found,
                                            'time': time.strftime("%H:%M:%S")
                                        }
                                        recorded_poses.append(pose_record)
                                        detected = True
                                        dist_m = float(np.linalg.norm(tvec))
                                        status_banner_text = f"SUCCESS: Pose #{len(recorded_poses)} recorded ({corners_found} corners, {dist_m:.2f}m)."
                                        status_banner_time = time.time()
                                        print(f"[CAPTURE SUCCESS] Pose #{len(recorded_poses)} paired! ({corners_found} corners, dist={dist_m:.3f}m)")

                        if not detected:
                            rejected_poses.append({
                                't_g2b': t_g2b,
                                'R_g2b': R_g2b,
                                'corners_count': corners_found
                            })
                            status_banner_text = f"WARNING: Pose rejected! Saw {corners_found}/4 corners. Check board in view."
                            status_banner_time = time.time()
                            print(f"[CAPTURE REJECTED] Robot at pose, but camera saw only {corners_found} corners (need 4+). Board out of view or obstructed.")

                except Exception as ex:
                    print(f"[POSE PARSE ERROR] {ex}")

                resp_event.set()

        # 3. Render Canvas
        canvas = np.full((880, 1480, 3), 18, dtype=np.uint8)

        # Video feed card
        cam_feed_img = np.zeros((330, 490, 3), dtype=np.uint8)
        board_detected_now = False
        corner_count_now = 0
        dist_to_board_str = "--"

        if latest_camera_frame_bgr is not None:
            display_frame = latest_camera_frame_bgr.copy()
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            charucoCorners, charucoIds, markerCorners, markerIds, active_board = detect_charuco_auto(gray)
            if markerCorners is not None and len(markerCorners) > 0:
                aruco.drawDetectedMarkers(display_frame, markerCorners, markerIds)

            if charucoCorners is not None and charucoIds is not None and len(charucoCorners) >= 4:
                safe_draw_charuco_corners(display_frame, charucoCorners, charucoIds, (0, 255, 0))
                objPoints, imgPoints = active_board.matchImagePoints(charucoCorners, charucoIds)
                valid, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cam_matrix, dist_coeffs)
                if valid:
                    cv2.drawFrameAxes(display_frame, cam_matrix, dist_coeffs, rvec, tvec, 0.12, 2)
                    board_detected_now = True
                    corner_count_now = len(charucoCorners)
                    dist_to_board_str = f"{float(np.linalg.norm(tvec)):.3f} m"

            cam_feed_img = cv2.resize(display_frame, (490, 330))
        else:
            cam_feed_img[:] = (16, 16, 22)
            for gx in range(0, 490, 35): cv2.line(cam_feed_img, (gx, 0), (gx, 330), (24, 24, 32), 1)
            for gy in range(0, 330, 35): cv2.line(cam_feed_img, (0, gy), (490, gy), (24, 24, 32), 1)
            cx, cy = 245, 165
            cv2.circle(cam_feed_img, (cx, cy), 35, (40, 40, 52), 1)
            cv2.putText(cam_feed_img, "ZED CAMERA: OFFLINE / STANDBY", (cx - 150, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (198, 241, 53), 1, cv2.LINE_AA)
            cv2.putText(cam_feed_img, "CONNECT CAMERA OR CLICK '+ TEST POSE'", (cx - 165, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 140), 1, cv2.LINE_AA)

        cv2.drawMarker(cam_feed_img, (245, 165), (198, 241, 53), cv2.MARKER_CROSS, 20, 1)

        # Draw Video Feed
        canvas[70:400, 20:510] = cam_feed_img
        cv2.rectangle(canvas, (20, 70), (510, 400), (38, 38, 48), 1)

        # Detection Badge
        badge_bg = (24, 45, 18) if board_detected_now else (22, 22, 30)
        badge_color = (198, 241, 53) if board_detected_now else (120, 120, 140)
        badge_text = f"BOARD: DETECTED ({corner_count_now} CORNERS)" if board_detected_now else "BOARD: SEARCHING..."
        cv2.rectangle(canvas, (25, 75), (290, 100), badge_bg, -1)
        cv2.rectangle(canvas, (25, 75), (290, 100), badge_color, 1)
        cv2.putText(canvas, badge_text, (32, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_color, 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (370, 75), (505, 100), (12, 12, 16), -1)
        cv2.rectangle(canvas, (370, 75), (505, 100), (45, 45, 60), 1)
        cv2.putText(canvas, f"DIST: {dist_to_board_str}", (376, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (56, 189, 248), 1, cv2.LINE_AA)

        # Robot Telemetry Card
        cv2.rectangle(canvas, (20, 415), (510, 595), (24, 24, 32), -1)
        cv2.rectangle(canvas, (20, 415), (510, 595), (38, 38, 48), 1)

        cv2.putText(canvas, "ROBOT TELEMETRY // IRC5 TCP LINK", (32, 436), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA)
        status_dot_col = (198, 241, 53) if robot_connected else (100, 100, 110)
        status_str = f"ONLINE ({robot_client_ip})" if robot_connected else "WAITING (PORT 5000)"
        cv2.circle(canvas, (360, 432), 5, status_dot_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, status_str, (372, 436), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_dot_col, 1, cv2.LINE_AA)

        if active_flange_pose:
            t = active_flange_pose['t']
            q = active_flange_pose['q']
            eul = active_flange_pose['euler']
            cv2.putText(canvas, f"Flange X, Y, Z (mm):   {t[0]*1000:7.1f}, {t[1]*1000:7.1f}, {t[2]*1000:7.1f}", (32, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Flange Quat (w,x,y,z): [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]", (32, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Euler R, P, Y (deg):   [{eul[0]:5.1f}, {eul[1]:5.1f}, {eul[2]:5.1f}]", (32, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"Socket Packets Received: {total_robot_packets}", (32, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Waiting for robot socket trigger...", (32, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 130), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (30, 540), (500, 580), (14, 14, 18), -1)
        cv2.rectangle(canvas, (30, 540), (500, 580), (32, 32, 42), 1)
        cv2.putText(canvas, f"Payload: {last_raw_message[:52]}", (38, 563), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1, cv2.LINE_AA)

        # Calibration Results Card
        cv2.rectangle(canvas, (20, 610), (510, 860), (24, 24, 32), -1)
        cv2.rectangle(canvas, (20, 610), (510, 860), (38, 38, 48), 1)
        cv2.putText(canvas, "HAND-EYE CALIBRATION RESULTS", (32, 632), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (198, 241, 53), 1, cv2.LINE_AA)

        if solved_result:
            r = solved_result
            cv2.putText(canvas, f"Optimal Algorithm:  {r['name']}", (32, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Mean Consistency:   {r['mean_err_mm']:.2f} mm  (Max: {r['max_err_mm']:.2f} mm)", (32, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (198, 241, 53), 1, cv2.LINE_AA)

            t_mm = r['t'].ravel() * 1000.0
            cv2.putText(canvas, f"Translation (cam->ee): X:{t_mm[0]:6.1f}  Y:{t_mm[1]:6.1f}  Z:{t_mm[2]:6.1f} mm", (32, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Euler Angles (deg):    R:{r['euler_deg'][0]:5.1f}  P:{r['euler_deg'][1]:5.1f}  Y:{r['euler_deg'][2]:5.1f}", (32, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

            R = r['R']
            for row in range(3):
                cv2.putText(canvas, f"R[{row}]: [{R[row,0]:7.4f}, {R[row,1]:7.4f}, {R[row,2]:7.4f}]", (32, 775 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 160), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Saved to: calibration_result.npz", (32, 848), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (198, 241, 53), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"Valid Poses Captured: {len(recorded_poses)}", (32, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (198, 241, 53), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Rejected (Board Out of View): {len(rejected_poses)}", (32, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255) if len(rejected_poses) > 0 else (120, 120, 130), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Total Robot Triggers: {total_robot_packets}", (32, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Board Settings: 7x5 / 5x7 squares, 25mm square, 18mm marker", (32, 745), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Dictionary: DICT_6X6_250 | Solvers: Park-Martin, Tsai, LM", (32, 765), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1, cv2.LINE_AA)

        # Render 3D Viewport
        view_3d_img = viewer_3d.render(recorded_poses, rejected_poses=rejected_poses, board_pose_world=board_pose_world, solved_cam2ee=solved_cam2ee)
        canvas[viewport_y:viewport_y+viewport_h, viewport_x:viewport_x+viewport_w] = view_3d_img
        cv2.rectangle(canvas, (viewport_x, viewport_y), (viewport_x+viewport_w, viewport_y+viewport_h), (38, 38, 48), 1)

        # Top Header Buttons
        cv2.rectangle(canvas, (20, 15), (34, 29), (198, 241, 53), -1)
        cv2.putText(canvas, "ABB ROBOT // EYE-IN-HAND CALIBRATION STUDIO", (44, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CHARUCO 7x5 // NATIVE HIGH-PRECISION SOLVER", (44, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 130), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (btn_solve_rect[0], btn_solve_rect[1]), (btn_solve_rect[2], btn_solve_rect[3]), (25, 45, 18), -1)
        cv2.rectangle(canvas, (btn_solve_rect[0], btn_solve_rect[1]), (btn_solve_rect[2], btn_solve_rect[3]), (198, 241, 53), 1)
        cv2.putText(canvas, "SOLVE CALIBRATION", (btn_solve_rect[0] + 16, btn_solve_rect[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (198, 241, 53), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (btn_save_rect[0], btn_save_rect[1]), (btn_save_rect[2], btn_save_rect[3]), (20, 38, 48), -1)
        cv2.rectangle(canvas, (btn_save_rect[0], btn_save_rect[1]), (btn_save_rect[2], btn_save_rect[3]), (56, 189, 248), 1)
        cv2.putText(canvas, "SAVE MATRIX", (btn_save_rect[0] + 18, btn_save_rect[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (btn_test_rect[0], btn_test_rect[1]), (btn_test_rect[2], btn_test_rect[3]), (28, 28, 38), -1)
        cv2.rectangle(canvas, (btn_test_rect[0], btn_test_rect[1]), (btn_test_rect[2], btn_test_rect[3]), (55, 55, 75), 1)
        cv2.putText(canvas, "+ TEST POSE", (btn_test_rect[0] + 14, btn_test_rect[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (btn_clear_rect[0], btn_clear_rect[1]), (btn_clear_rect[2], btn_clear_rect[3]), (28, 28, 38), -1)
        cv2.rectangle(canvas, (btn_clear_rect[0], btn_clear_rect[1]), (btn_clear_rect[2], btn_clear_rect[3]), (55, 55, 75), 1)
        cv2.putText(canvas, "CLEAR", (btn_clear_rect[0] + 22, btn_clear_rect[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (btn_reset_rect[0], btn_reset_rect[1]), (btn_reset_rect[2], btn_reset_rect[3]), (28, 28, 38), -1)
        cv2.rectangle(canvas, (btn_reset_rect[0], btn_reset_rect[1]), (btn_reset_rect[2], btn_reset_rect[3]), (55, 55, 75), 1)
        cv2.putText(canvas, "RECENTER", (btn_reset_rect[0] + 16, btn_reset_rect[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1, cv2.LINE_AA)

        # Bottom Status Banner
        banner_col = (198, 241, 53) if (time.time() - status_banner_time < 5.0) else (120, 120, 130)
        cv2.putText(canvas, f"STATUS: {status_banner_text}", (530, 815), cv2.FONT_HERSHEY_SIMPLEX, 0.42, banner_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (1410, 815), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(15) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('c') or key == ord('s'):
            trigger_calibration()
        elif key == ord('t'):
            simulate_test_pose()
        elif key == ord('r'):
            viewer_3d.reset_view()

    if zed and camera_online:
        zed.close()
    cv2.destroyAllWindows()
    print("\n[CALIBRATION DASHBOARD] Shutdown complete.")


if __name__ == "__main__":
    main()
