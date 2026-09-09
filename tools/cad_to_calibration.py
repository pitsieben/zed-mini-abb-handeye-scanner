"""
CAD to Eye-in-Hand Calibration Converter
=========================================
Converts physical measurements or Rhino CAD assembly coordinates directly into
the 'calibration_result.npz' format used by the scanning studio and Open3D pipeline.

Why use CAD calibration?
- Eliminates ChArUco board detection jitter, reprojection noise, and optical blur ambiguity.
- Since the camera bracket is rigidly manufactured and bolted to the robot flange (tool0),
  the CAD assembly in Rhino gives exact sub-millimeter ground truth.

Frame Conventions:
  - Tool0 (Flange):
      Origin at center of robot mounting flange face.
      +Z points outward along flange normal.
      +X points along alignment notch / pin (down/forward).
      +Y completes the right-handed system.
  - ZED Mini (Camera Frame):
      Origin at optical center of the LEFT camera lens.
      +X points RIGHT (towards right lens).
      +Y points DOWN.
      +Z points FORWARD (camera viewing direction).
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

def create_calibration(tx_mm, ty_mm, tz_mm, rx_deg=0.0, ry_deg=0.0, rz_deg=-90.0, output_npz="calibration_result.npz"):
    # Convert translation from mm to meters
    t_cam2ee = np.array([[tx_mm / 1000.0], [ty_mm / 1000.0], [tz_mm / 1000.0]], dtype=np.float64)

    # Compute rotation matrix from Euler angles (XYZ convention in degrees)
    r = Rotation.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True)
    R_cam2ee = r.as_matrix().astype(np.float64)

    print("\n" + "=" * 65)
    print("      RHINO CAD TO EYE-IN-HAND CALIBRATION")
    print("=" * 65)
    print(f"Left Lens Center in Tool0 (mm):  X={tx_mm:+.1f}, Y={ty_mm:+.1f}, Z={tz_mm:+.1f}")
    print(f"Camera Orientation Euler (deg):  Rx={rx_deg:+.1f} deg, Ry={ry_deg:+.1f} deg, Rz={rz_deg:+.1f} deg")
    print("\nRotation Matrix R_cam2ee (Camera -> Tool0):")
    print(np.round(R_cam2ee, 4))
    print("\nTranslation Vector t_cam2ee (meters):")
    print(np.round(t_cam2ee.ravel(), 4))

    # Backup existing if present
    if os.path.exists(output_npz):
        backup_name = output_npz + ".bak"
        try:
            import shutil
            shutil.copyfile(output_npz, backup_name)
            print(f"\n[INFO] Backed up previous calibration -> {backup_name}")
        except Exception:
            pass

    np.savez(output_npz, R=R_cam2ee, t=t_cam2ee)
    print(f"\n[SUCCESS] Saved calibration file -> {output_npz}")
    print("=" * 65 + "\n")
    return R_cam2ee, t_cam2ee

def main():
    parser = argparse.ArgumentParser(description="Convert Rhino CAD measurements to calibration_result.npz")
    parser.add_argument("--tx", type=float, default=None, help="X offset from tool0 center to left lens in mm")
    parser.add_argument("--ty", type=float, default=None, help="Y offset from tool0 center to left lens in mm")
    parser.add_argument("--tz", type=float, default=None, help="Z offset from tool0 center to left lens in mm")
    parser.add_argument("--rx", type=float, default=0.0, help="Rotation around Tool0 X axis in degrees")
    parser.add_argument("--ry", type=float, default=0.0, help="Rotation around Tool0 Y axis (tilt/pitch) in degrees")
    parser.add_argument("--rz", type=float, default=-90.0, help="Rotation around Tool0 Z axis (yaw) in degrees (default: -90)")
    parser.add_argument("--pitch", type=float, default=None, help="Shortcut for tilt/pitch angle (degrees)")
    parser.add_argument("--out", type=str, default="calibration_result.npz", help="Output npz filename")
    args = parser.parse_args()

    if args.tx is None or args.ty is None or args.tz is None:
        print("=" * 65)
        print("  RHINO CAD -> HAND-EYE CALIBRATION ASSISTANT")
        print("=" * 65)
        print("Measure the vector from the Center of Robot Flange (tool0) to the")
        print("Center of the ZED Mini LEFT Lens in your Rhino 3D model.")
        print("------------------------------------------------------------")
        try:
            tx = float(input("Offset X in mm [default: 0.0]: ") or "0.0")
            ty = float(input("Offset Y in mm [default: 24.0]: ") or "24.0")
            tz = float(input("Offset Z in mm [default: 130.0]: ") or "130.0")
            rx = float(input("Rotation Rx (deg) [default: 0.0]: ") or "0.0")
            ry = float(input("Rotation Ry / Pitch Tilt (deg) [default: 30.0]: ") or "30.0")
            rz = float(input("Rotation Rz / Yaw (deg) [default: -90.0]: ") or "-90.0")
        except KeyboardInterrupt:
            print("\nAborted.")
            return
    else:
        tx, ty, tz = args.tx, args.ty, args.tz
        rx, ry, rz = args.rx, args.ry, args.rz
        if args.pitch is not None:
            ry = args.pitch

    create_calibration(tx, ty, tz, rx, ry, rz, args.out)

if __name__ == "__main__":
    main()
