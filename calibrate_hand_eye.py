import socket
import numpy as np
import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
from scipy.spatial.transform import Rotation

# ==========================================
# CONFIGURATION
# ==========================================
HOST = '0.0.0.0' # Listen on all network interfaces
PORT = 5000      # Must match pc_port in RAPID code

# ChArUco board settings (calib.io printed board)
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.025 # 25mm
MARKER_LENGTH = 0.018 # 18mm
DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
BOARD = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, DICTIONARY)

# ==========================================
# INITIALIZATION
# ==========================================
# Initialize ZED Camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NONE # Only need RGB for calibration

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Get camera intrinsics from ZED SDK
camera_info = zed.get_camera_information()
cam_matrix = np.array([[camera_info.camera_configuration.calibration_parameters.left_cam.fx, 0, camera_info.camera_configuration.calibration_parameters.left_cam.cx],
                       [0, camera_info.camera_configuration.calibration_parameters.left_cam.fy, camera_info.camera_configuration.calibration_parameters.left_cam.cy],
                       [0, 0, 1]])
dist_coeffs = np.array([camera_info.camera_configuration.calibration_parameters.left_cam.disto[0],
                        camera_info.camera_configuration.calibration_parameters.left_cam.disto[1],
                        camera_info.camera_configuration.calibration_parameters.left_cam.disto[2],
                        camera_info.camera_configuration.calibration_parameters.left_cam.disto[3],
                        camera_info.camera_configuration.calibration_parameters.left_cam.disto[4]])

# Data storage for calibration
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# ==========================================
# TCP SERVER & DATA COLLECTION
# ==========================================
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Waiting for robot connection on port {PORT}...")
    
    # This will block until the robot connects
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        
        image_sl = sl.Mat()
        while True:
            # Wait for message from robot
            data = conn.recv(1024)
            if not data:
                break
            
            msg = data.decode('utf-8').strip()
            
            if msg == "DONE":
                print("Robot finished poses.")
                break
                
            print(f"Received pose from robot: {msg}")
            
            # Parse robot pose (X, Y, Z, q1, q2, q3, q4)
            # Note: ABB q1=w, q2=x, q3=y, q4=z
            parts = [float(x) for x in msg.split(',')]
            tx, ty, tz, q1, q2, q3, q4 = parts
            
            # Convert translation to meters (RAPID sends mm)
            tx, ty, tz = tx / 1000.0, ty / 1000.0, tz / 1000.0
            
            # Gripper to base transform (A)
            t_g2b = np.array([[tx], [ty], [tz]])
            # scipy Rotation uses [x, y, z, w]
            r_g2b_rot = Rotation.from_quat([q2, q3, q4, q1]) 
            R_g2b = r_g2b_rot.as_matrix()
            
            # Capture ZED image
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_sl, sl.VIEW.LEFT)
                img = image_sl.get_data()
                gray = cv2.cvtColor(img, cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
                
                # Detect ChArUco markers
                corners, ids, rejected = aruco.detectMarkers(gray, DICTIONARY)
                
                if ids is not None and len(ids) > 4:
                    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)
                    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                        success, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, BOARD, cam_matrix, dist_coeffs, None, None)
                        if success:
                            # Target to camera transform (B)
                            R_t2c, _ = cv2.Rodrigues(rvec)
                            t_t2c = tvec
                            
                            R_gripper2base.append(R_g2b)
                            t_gripper2base.append(t_g2b)
                            R_target2cam.append(R_t2c)
                            t_target2cam.append(t_t2c)
                            
                            print(" -> Successfully detected board and saved pose pair.")
                        else:
                            print(" -> Pose estimation failed.")
                    else:
                        print(" -> Not enough charuco corners.")
                else:
                    print(" -> Not enough aruco markers detected.")
                    
            # Send ACK to robot so it moves to next point
            conn.sendall(b"ACK")

zed.close()

# ==========================================
# HAND-EYE CALIBRATION CALCULATION
# ==========================================
if len(R_gripper2base) > 4:
    print(f"\nStarting calibration with {len(R_gripper2base)} poses...")
    from robot_calibration_dashboard import solve_and_evaluate_calibration
    recorded_poses = [
        {'R_g2b': Rg, 't_g2b': tg, 'R_t2c': Rc, 't_t2c': tc}
        for Rg, tg, Rc, tc in zip(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
    ]
    best_result, breakdown = solve_and_evaluate_calibration(recorded_poses)
    
    if best_result:
        R_cam2gripper = best_result['R']
        t_cam2gripper = best_result['t']
        print("\n========================================")
        print(f"   CALIBRATION RESULTS ({best_result['name']})")
        print("========================================")
        print("Rotation Matrix (Camera to Tool0):\n", R_cam2gripper)
        print("\nTranslation Vector (Camera to Tool0) [meters]:\n", t_cam2gripper)
        print(f"\nResidual Consistency Jitter: Mean {best_result['mean_err_mm']:.2f} mm | Max {best_result['max_err_mm']:.2f} mm")
        
        # Save to file
        np.savez("calibration_result.npz", R=R_cam2gripper, t=t_cam2gripper)
        print("\nSaved to calibration_result.npz")
        print("You can load this later using np.load('calibration_result.npz')")
    else:
        print("Failed to solve calibration from captured poses.")
else:
    print("Not enough successful poses collected for calibration (minimum 5 required, recommended 15+).")
