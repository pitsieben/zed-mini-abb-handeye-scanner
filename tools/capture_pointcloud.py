import pyzed.sl as sl
import sys
import os

def main():
    # 1. Create a Camera object
    zed = sl.Camera()

    # 2. Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # High resolution for scanning
    init_params.camera_fps = 30                           # Framerate
    
    # Depth mode is CRITICAL for scanning. 
    # NEURAL gives the best quality but requires more GPU power. 
    # ULTRA is a great balance of speed and quality.
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA 
    
    init_params.coordinate_units = sl.UNIT.METER          # Standard metric units
    init_params.depth_minimum_distance = 0.3              # 30cm min depth (ZED Mini is good at close range)

    # 3. Open the camera
    print("Opening ZED Camera...")
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED Camera: {repr(err)}")
        sys.exit(1)

    print("Camera successfully opened! Capturing point cloud...")

    # 4. Prepare objects to hold the data
    point_cloud = sl.Mat()
    
    # 5. Capture a frame
    # We grab a few frames first to let the auto-exposure settle
    for i in range(15):
        zed.grab()

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # 6. Retrieve the colored point cloud
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        
        # 7. Save the point cloud to a file
        filepath = "scan_output.ply"
        print(f"Saving point cloud to {filepath}...")
        
        # ZED API has a built-in method to save to .ply format
        err = point_cloud.write(filepath, sl.MEM.CPU) 
        
        if err == sl.ERROR_CODE.SUCCESS:
            print("Scan saved successfully!")
            print(f"File location: {os.path.abspath(filepath)}")
        else:
            print("Failed to save point cloud.")
    else:
        print("Failed to grab a frame from the camera.")

    # 8. Safely close the camera
    zed.close()
    print("Camera closed.")

if __name__ == "__main__":
    main()
