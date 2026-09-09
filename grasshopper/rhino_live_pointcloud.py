"""
GRASSHOPPER LIVE POINT CLOUD IMPORTER (RHINO 7 / RHINO 8)
=========================================================
Paste this script into a GhPython (Grasshopper Python) component.

INPUTS:
  - FilePath  (string)  : Path to 'scans/latest_workpiece_mm.ply'
  - Refresh   (boolean) : Connect a Boolean Toggle, Button, or GH Timer (e.g. 500ms)
  - Subsample (integer) : 1 = all points, 2 = every 2nd point, etc. (Default: 1)

OUTPUTS:
  - PointCloud : Native Rhino.Geometry.PointCloud with true RGB vertex colors
  - BoundingBox: Workpiece bounding box in robot base coordinates (mm)
  - Info       : Text summary (Point count, dimensions, timestamp)
"""

import os
import struct
import System.Drawing.Color as SysColor
import Rhino.Geometry as rg

# Default parameters
if 'FilePath' not in globals() or not FilePath:
    default_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.getcwd()
    FilePath = os.path.join(default_dir, "scans", "latest_workpiece_mm.ply")

if 'Subsample' not in globals() or Subsample is None or Subsample < 1:
    Subsample = 1

def load_ply_to_rhino_pointcloud(filepath, step=1):
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
    
    try:
        with open(filepath, 'rb') as f:
            # 1. Parse PLY header
            num_vertices = 0
            is_binary = False
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('format binary_little_endian'):
                    is_binary = True
                elif line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line == 'end_header':
                    break
            
            if not is_binary:
                return None, "PLY is not binary_little_endian format."
            if num_vertices == 0:
                return None, "PLY contains 0 vertices."
            
            # 2. Read vertices: each vertex is 3 floats (12 bytes) + 3 bytes RGB (total 15 bytes)
            # Binary layout: x (<f4), y (<f4), z (<f4), red (u1), green (u1), blue (u1)
            raw_bytes = f.read(num_vertices * 15)
            
        pc = rg.PointCloud()
        stride = 15 * step
        total_read = 0
        
        # Fast unpack loop
        for offset in range(0, len(raw_bytes), stride):
            if offset + 15 > len(raw_bytes):
                break
            x, y, z, r, g, b = struct.unpack_from('<fffBBB', raw_bytes, offset)
            pc.Add(rg.Point3d(x, y, z), SysColor.FromArgb(r, g, b))
            total_read += 1
            
        return pc, total_read
        
    except Exception as ex:
        return None, str(ex)

# Execute
pc_obj, status = load_ply_to_rhino_pointcloud(FilePath, step=Subsample)

if pc_obj is not None:
    PointCloud = pc_obj
    bbox = pc_obj.GetBoundingBox(True)
    BoundingBox = bbox
    dx = bbox.Max.X - bbox.Min.X
    dy = bbox.Max.Y - bbox.Min.Y
    dz = bbox.Max.Z - bbox.Min.Z
    Info = (
        "Points: {:,}\n"
        "Bounding Box (mm):\n"
        "  X: {:.1f} to {:.1f} (dx: {:.1f} mm)\n"
        "  Y: {:.1f} to {:.1f} (dy: {:.1f} mm)\n"
        "  Z: {:.1f} to {:.1f} (dz: {:.1f} mm)\n"
        "Source: {}\n"
        "Scale: 1:1 Millimeters (Robot Base Space)"
    ).format(pc_obj.Count, bbox.Min.X, bbox.Max.X, dx, bbox.Min.Y, bbox.Max.Y, dy, bbox.Min.Z, bbox.Max.Z, dz, os.path.basename(FilePath))
else:
    PointCloud = None
    BoundingBox = None
    Info = "Status: {}".format(status)
