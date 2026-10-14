# ==============================================================================
# ABB ROBOT 3D SCANNING TOOLPATH DESIGN GUIDELINES (ZED MINI)
# ==============================================================================
# Paste these rules directly into your Grasshopper Python / C# / Panel component
# to ensure high-fidelity Open3D multi-way registration and Poisson meshing.
# ==============================================================================

RECOMMENDED_PARAMETERS = {
    # 1. OPTICAL STANDOFF DISTANCE (ZED Mini Stereo Baseline: 63mm)
    # -------------------------------------------------------------
    # - Absolute Min Sensor Standoff: 200 mm (hardware minimum for ZED Mini)
    # - Optimal Scanning Range:       400 mm - 650 mm
    # - Standoff Limit:               750 mm (depth uncertainty scales with Z^2)
    "standoff_distance_min_mm": 400.0,
    "standoff_distance_max_mm": 650.0,

    # 2. WAYPOINT STEP SIZE & FRAME OVERLAP
    # -------------------------------------
    # At 500mm standoff, the ZED Mini field of view covers ~650mm x 400mm.
    # To guarantee a minimum of 40% - 60% spatial overlap:
    # - Linear Step Distance (along sweep):  180 mm - 250 mm
    # - Lateral Step Distance (between sweeps): 150 mm - 220 mm
    "step_distance_along_path_mm": 200.0,
    "step_distance_between_paths_mm": 180.0,
    "target_overlap_percentage": 50.0,

    # 3. ORIENTATION & INCIDENCE ANGLES
    # ---------------------------------
    # - Keep the camera optical Z-axis normal to the workpiece surface (+/- 25 deg).
    # - Grazing angles (> 45 deg) degrade stereoscopic correlation.
    # - Maximum rotation swing between consecutive waypoints: <= 30.0 deg.
    "max_surface_incidence_angle_deg": 35.0,
    "max_angular_jump_between_poses_deg": 30.0,

    # 4. ROBOT CAPTURE TIMING & SETTLING
    # ----------------------------------
    # - Arm deceleration causes camera vibration on tool mounts.
    # - Zone data: Use "fine" positioning (WaitRob \InPos).
    # - Settling wait time: 0.3 - 0.5 seconds before triggering camera snapshot.
    # - FlexPendant TCP Timeout: >= 15 seconds (prevents premature disconnection).
    "settling_time_seconds": 0.5,
    "robot_socket_timeout_seconds": 15.0,

    # 5. SUBJECT / WORKPIECE STABILITY
    # --------------------------------
    # - Rigid Workpieces (timber, clay, casts): Zero deformation.
    # - Living / Flexible Subjects (humans, clothing):
    #   * Instruct subject to fix eyes on a stationary point and breathe gently.
    #   * Colored ICP will use fabric patterns & textures to bridge micro-movement.
}
