import pyzed.sl as sl
import cv2
import os
import sys
import time

def main():
    # 1. Create a ZED Camera object
    zed = sl.Camera()

    # 2. Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    # Use NEURAL depth mode
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
    init_params.coordinate_units = sl.UNIT.METER          # Standard metric units
    init_params.depth_minimum_distance = 0.2              # 20cm min depth
    
    # HARD CUTOFF: Ignore absolutely everything further than 1 meter away.
    # This prevents the background from being captured at all.
    init_params.depth_maximum_distance = 1.0

    print("Opening ZED Camera (this may take a few seconds)...")
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED Camera")
        sys.exit(1)
        
    # 3. Enable positional tracking (Mandatory for fusing points across space)
    tracking_parameters = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(tracking_parameters) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking")
        sys.exit(1)

    print("Camera opened and tracking enabled successfully.")

    image = sl.Mat()
    
    # State variables
    recording = False
    scan_count = 0
    
    # 4. Setup Spatial Mapping parameters
    mapping_parameters = sl.SpatialMappingParameters()
    # We want a combined point cloud instead of a polygon mesh
    mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    
    # Max resolution supported by ZED Spatial Mapping engine (1cm)
    mapping_parameters.resolution_meter = 0.01
    # Minimum range accepted by ZED Spatial Mapping engine (2m)
    mapping_parameters.range_meter = 2.0
    
    # FORCE COLOR: Explicitly tell the mapping engine to sample and save RGB texture data
    mapping_parameters.save_texture = True
    
    output_dir = "scans"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # OpenCV Mouse Callback
    def mouse_callback(event, x, y, flags, param):
        nonlocal recording, scan_count
        if event == cv2.EVENT_LBUTTONDOWN:
            # Button bounds
            if 20 <= x <= 280 and 20 <= y <= 70:
                if not recording:
                    # START RECORDING / MAPPING
                    print("\nStarting Spatial Mapping...")
                    zed.enable_spatial_mapping(mapping_parameters)
                    recording = True
                else:
                    # STOP RECORDING & SAVE
                    print("\nStopping recording, extracting fused point cloud... Please wait.")
                    recording = False
                    
                    # Extract the fused 3D data from the ZED's memory
                    fused_pc = sl.FusedPointCloud()
                    zed.extract_whole_spatial_map(fused_pc)
                    
                    filename = os.path.join(output_dir, f"combined_scan_{scan_count:04d}.ply")
                    print(f"Saving single combined point cloud to {filename}...")
                    
                    # Save it to disk as binary PLY
                    fused_pc.save(filename, typeMesh=sl.MESH_FILE_FORMAT.PLY_BIN)
                    print("Saved successfully!\n")
                    
                    scan_count += 1
                    
                    # Disable mapping to clear memory for the next scan
                    zed.disable_spatial_mapping()

    # Create the window and attach the mouse listener
    cv2.namedWindow("ZED Scanner")
    cv2.setMouseCallback("ZED Scanner", mouse_callback)

    print("\n=======================================================")
    print("Stream running! Click the button to start the 3D scan.")
    print("Move the camera smoothly around your workpiece.")
    print("Click STOP to fuse everything and save the combined PLY.")
    print("Press 'Q' to exit.")
    print("=======================================================\n")

    while True:
        # Grab a new frame from the camera. 
        # (While mapping is enabled, ZED fuses the 3D points in the background automatically!)
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv2_img = image.get_data()
            
            # Downscale for display
            display_img = cv2.resize(cv2_img, (1280, 720))
            
            # --- Draw the Button ---
            btn_color = (0, 0, 200) if recording else (0, 200, 0)
            btn_text = "STOP & SAVE SCAN" if recording else "START 3D SCAN"
            
            cv2.rectangle(display_img, (20, 20), (280, 70), btn_color, -1)
            cv2.rectangle(display_img, (20, 20), (280, 70), (255, 255, 255), 2)
            cv2.putText(display_img, btn_text, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw Status Texts
            status_text = "MAPPING IN PROGRESS..." if recording else "IDLE"
            status_color = (0, 165, 255) if recording else (0, 255, 0)
            cv2.putText(display_img, f"Status: {status_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_img, f"Combined Scans saved: {scan_count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if recording:
                # Blinking red dot to indicate recording
                if (int(time.time() * 2) % 2) == 0:
                    cv2.circle(display_img, (310, 45), 10, (0, 0, 255), -1)

            cv2.imshow("ZED Scanner", display_img)

        # Check for 'q' key to quit
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break

    # Safety catch: if the user quits while recording, save anyway just in case
    if recording:
        print("Quitting while mapping! Saving backup...")
        fused_pc = sl.FusedPointCloud()
        zed.extract_whole_spatial_map(fused_pc)
        fused_pc.save(os.path.join(output_dir, f"backup_scan_{scan_count:04d}.ply"), typeMesh=sl.MESH_FILE_FORMAT.PLY_BIN)
        zed.disable_spatial_mapping()

    # Cleanup
    cv2.destroyAllWindows()
    zed.disable_positional_tracking()
    zed.close()
    print("Camera closed successfully.")

if __name__ == "__main__":
    main()
