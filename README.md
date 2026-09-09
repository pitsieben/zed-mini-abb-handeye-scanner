# 🤖 ZED Mini // ABB Robot Hand-Eye Calibration & 3D Scanning Studio

An end-to-end robotic photogrammetry and 3D reconstruction system integrating a **Stereolabs ZED Mini camera** mounted on an **ABB industrial robot arm**. 

The system features bidirectional robot TCP socket synchronization, dual **Web (WebGL)** and **Desktop (OpenCV)** scanning dashboards, an accelerated **Open3D Multi-Way ICP and Screened Poisson meshing pipeline**, and direct **1:1 millimeter live point cloud integration into Rhino / Grasshopper**.

---

## 📐 System Architecture

```
                      +-----------------------------+
                      |   ABB Robot Controller      |
                      |   (RAPID / robtarget)       |
                      +--------------+--------------+
                                     | TCP Socket (Port 5000)
                                     | [X, Y, Z, Q1, Q2, Q3, Q4]
                                     v
+-----------------------+     +-------------------------------+     +-------------------------+
|   Stereolabs ZED M    | --> |  ZED Scanner Studio Engine    | <-- | Calibration Matrix      |
|   (Stereo RGB + Depth)|     |  (Web / Desktop Dashboard)    |     | (calibration_result.npz)|
+-----------------------+     +---------------+---------------+     +-------------------------+
                                              |
                                              v
                              +-------------------------------+
                              | Scans / Session Stages        |
                              | (scans/session_stages/)       |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | Open3D Multi-Way ICP Pipeline |
                              | - Pass-Aware Segmentation     |
                              | - Multi-Scale Coarse-to-Fine  |
                              | - Screened Poisson Meshing    |
                              +---------------+---------------+
                                              |
                                              v
                         +-----------------------------------------+
                         | 1:1 Millimeter Outputs for Rhino CAD    |
                         | - latest_workpiece_mm.ply               |
                         | - latest_workpiece_mesh_mm.obj / .stl   |
                         +--------------------+--------------------+
                                              |
                                              v
                                 +-------------------------+
                                 | Rhino / Grasshopper     |
                                 | (GhPython Live Stream)  |
                                 +-------------------------+
```

---

## 📁 Repository Structure

```text
├── web_scanner_app.py              # 🌐 Web Studio: Flask + Three.js WebGL 3D scanning interface (Port 8080)
├── robot_scanner_dashboard.py      # 🖥️ Desktop Studio: Native OpenCV GUI with real-time 3D orbit viewport
├── robot_calibration_dashboard.py  # 🎯 ChArUco Calibration Dashboard: Live corner detection & 5-solver suite
├── calibrate_hand_eye.py           # 📐 Headless Eye-in-Hand calibration routine
├── zed_dashboard.py                # 📷 Standalone ZED handheld camera monitor & depth inspector
├── run_open3d_pipeline.bat         # ⚡ One-click launcher for Open3D ICP & Poisson reconstruction
├── scan_config.json                # ⚙️ Central configuration for depth clipping, voxel size & ICP thresholds
├── calibration_result.npz          # 💾 Active Hand-Eye transformation matrix (T_cam2ee)
│
├── 📁 grasshopper/                 # Rhino / Grasshopper integration
│   ├── HandtoEyeCalibration.gh    # Grasshopper parametric toolpath generator
│   └── rhino_live_pointcloud.py   # GhPython live streaming component (1:1 mm in Rhino)
│
├── 📁 rapid/                       # ABB Robot RAPID Programs (.modx)
│   ├── HandtoEye.modx             # Scanning routine generated from Grasshopper with TCP handshake
│   ├── Simple_TCP_Test.modx       # Minimal non-motion network ping test
│   └── prc_project.modx           # Baseline PRC project code
│
├── 📁 scans/                       # Output workpiece scans & meshes (1:1 Millimeters)
│   ├── latest_workpiece_mm.ply    # Live reference point cloud for Rhino
│   ├── latest_workpiece_mesh_mm.obj# Watertight textured mesh for Rhino
│   ├── latest_workpiece_mesh_mm.stl# Binary STL surface mesh
│   └── session_stages/            # Individual waypoint point clouds & manifest.json
│
├── 📁 templates/ & static/         # Frontend assets for Web Scanning Studio
│   ├── templates/index.html       # WebGL 3D scanning studio UI with settings modal
│   ├── static/three.min.js        # Three.js 3D rendering engine
│   └── static/OrbitControls.js    # Camera navigation controls
│
└── 📁 tools/                       # Utilities & Core Algorithms
    ├── open3d_pipeline.py         # Pass-Aware Multi-Way ICP & Poisson reconstruction engine
    ├── cad_to_calibration.py      # Converts Rhino CAD assembly measurements to calibration_result.npz
    ├── transform_modx.py          # Injects TCP socket triggers into standard RAPID routines
    ├── test_tcp_server.py         # Standalone TCP test server (Port 5000)
    ├── tcp_tester_gui.py          # Tkinter live robot coordinate monitor
    ├── capture_pointcloud.py      # Single-shot point cloud capture utility
    ├── clean_and_optimize_scan.py # Outlier rejection & voxel decimation utility
    └── fuse_robot_scans.py        # Rigid scan fusion utility
```

---

## ⚡ Quick Start

### 1. Prerequisites
* **Operating System**: Windows 10 / 11 (64-bit)
* **ZED SDK**: 4.x or 5.x with CUDA support
* **Python Environment**:
  * Python virtual environment with `pyzed`, `opencv-python`, `scipy`, `flask`
  * Python 3.11 with `open3d` for the reconstruction pipeline

### 2. Verify Robot Network Link
1. Start the TCP test server on your PC:
   ```powershell
   python tools/test_tcp_server.py
   ```
2. On the ABB teach pendant, run `rapid/Simple_TCP_Test.modx`.
3. Verify that the PC receives the connection and displays the incoming coordinates.

---

## 🎯 Step 1: Hand-Eye Calibration ($T_{cam \to ee}$)

To transform points from the camera frame into the robot base frame, the camera-to-flange transformation ($T_{cam2ee}$) must be calibrated.

### Option A: Physical Measurement from CAD (Recommended — 100% Noise-Free)
If your camera bracket is rigidly modeled in CAD and bolted to the robot flange (`tool0`), you can extract the exact offsets directly from Rhino without image noise:
```powershell
python tools/cad_to_calibration.py --tx 0 --ty 24 --tz 128 --ry 0 --rz -90
```
*(Or run `python tools/cad_to_calibration.py` without arguments for an interactive guide).*

### Option B: ChArUco Automated Calibration Dashboard
When using an optical calibration target:
```powershell
python robot_calibration_dashboard.py
```
* Observe live green ChArUco corner detections on the video feed.
* Execute a toolpath of **15–20 poses** distributed over a dome/hemisphere around the board ($\pm 25^\circ - 35^\circ$ tilt diversity).
* The dashboard solves the transformation using Tsai-Lenz, Park-Martin, and Non-Linear Levenberg-Marquardt, saving to `calibration_result.npz`.

---

## 🚀 Step 2: Running the 3D Scanning Studio

You can run either the **Web Studio** or the **Desktop Studio**:

### Option A: Web Scanning Studio (WebGL / Browser)
```powershell
python web_scanner_app.py
```
Open your browser to: **`http://localhost:8080`**

**Features:**
* **Real-time Video**: HD720 RGB stream and colorized Neural Depth stream with center laser distance meter.
* **Interactive 3D Viewport**: Built on Three.js (Z-UP aligned with robot base origin `(0,0,0)`).
* **Layer Toggles**: Point cloud, shaded surface mesh, wireframe, robot TCP gizmo, and floor grid.
* **Scan Settings Modal**: Click **`⚙️ SCAN SETTINGS`** in the header to customize depth clipping (`min_depth`, `max_depth`), border crop, voxel resolution, and outlier sensitivity before scanning.
* **Resilient Camera Loop**: Automatic background reconnection if the camera cable is jostled.

### Option B: Desktop Scanner Dashboard (OpenCV)
```powershell
python robot_scanner_dashboard.py
```
* High-performance OpenCV native desktop window with back-to-front depth sorting.
* Keybindings:
  * `Left-Click + Drag`: Orbit around base origin `(0,0,0)`
  * `Right-Click + Drag`: Zoom in/out
  * `R`: Recenter view on origin
  * `P`: Run Open3D Multi-Way ICP & Poisson meshing
  * `S`: Save workpiece
  * `C`: Clean & de-noise buffer

---

## ⚡ Step 3: Open3D Reconstruction Pipeline

When your robot completes its toolpath (or upon clicking **`RUN OPEN3D ICP`**), the accelerated pipeline runs automatically:

```powershell
run_open3d_pipeline.bat
```

### Key Capabilities:
1. **Toolpath Pass-Aware Segmentation**:
   Automatically identifies distinct passes across the workpiece (e.g., Left pass tilted $+42^\circ$, Center pass vertical, Right pass tilted $-42^\circ$).
2. **Multi-Scale Inter-Pass ICP**:
   Uses an adaptive coarse search radius ($120\text{ mm}$) to bridge Hand-Eye calibration residuals between wide-angle passes, followed by fine Point-to-Plane ICP ($25\text{ mm}$ with Tukey robust loss). **This eliminates scan duplication/ghosting.**
3. **Screened Poisson Surface Reconstruction**:
   Extracts clean, watertight 3D meshes in **under 20 seconds**.
4. **Native 1:1 Millimeter Exports**:
   Exports directly to:
   * `scans/latest_workpiece_mm.ply`
   * `scans/latest_workpiece_mesh_mm.obj`
   * `scans/latest_workpiece_mesh_mm.stl`

---

## 🦏 Step 4: Rhino & Grasshopper Live Integration

To stream point clouds directly into Rhino without manual importing or scaling:

1. In Rhino 7 or 8, open Grasshopper.
2. Add a **GhPython** component and paste the code from:
   [`grasshopper/rhino_live_pointcloud.py`](grasshopper/rhino_live_pointcloud.py)
3. Connect a Grasshopper **Timer** (e.g. 500 ms) or boolean toggle to the `Refresh` input.
4. The point cloud will update in real time in your Rhino viewport at **1:1 millimeter scale** in native ABB robot base coordinates.

> [!NOTE]
> When importing `.obj` or `.stl` meshes manually into Rhino, ensure the import dialog is set to **`Map Z to Z`** (Z-Up) to match the robot's base coordinate system.

---

## ⚙️ Configuration (`scan_config.json`)

All scanning thresholds can be adjusted in `scan_config.json` or through the Web UI modal:

```json
{
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
```

* **`min_depth_m` / `max_depth_m`**: Clips foreground lens artifacts and background tables/walls (optimal range: $350\text{ mm} - 650\text{ mm}$).
* **`border_crop`**: Eliminates wide-angle lens distortion around the frame edges ($0.06 = 6\%$).
* **`voxel_size_mm`**: Downsampling grid spacing ($2.0\text{ mm}$ gives fine detail and 60 FPS viewport response).
* **`outlier_std_ratio`**: Statistical noise rejection sensitivity (lower = more aggressive cleaning).

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Two duplicate/mirrored objects appear** | Multiple scanning passes offset due to residual calibration angle | Run `run_open3d_pipeline.bat`. The pass-aware multi-scale ICP will automatically register and merge them into one object. |
| **Camera disconnected / blank stream** | USB cable bandwidth or connection drop | The studio runs in resilient standby and auto-reconnects every 3 seconds once the USB link is restored. |
| **Mesh is rotated 90° in Rhino** | Rhino OBJ import defaults to "Map OBJ Y to Rhino Z" | In Rhino's OBJ/STL import dialog, change the setting to **`Map Z to Z`**. |
| **Robot connection timeout** | PC IP mismatch or firewall block | Verify your PC's IP address with `ipconfig` and ensure port 5000 is open in Windows Defender Firewall. |

---

## 📜 License
MIT License. Developed for automated robotic scanning, photogrammetry, and reverse-engineering.
