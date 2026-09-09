import numpy as np
import scipy.spatial as spatial
import os
import sys
import time
import argparse

def load_ply_binary(filepath):
    """
    Loads a binary PLY file with x, y, z float32 and red, green, blue uchar.
    """
    print(f"[*] Reading {filepath}...")
    with open(filepath, 'rb') as f:
        N = 0
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                N = int(line.split()[-1])
            elif line == 'end_header':
                break
                
        if N == 0:
            raise ValueError("No vertices found in PLY header.")
            
        dt = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        data = np.fromfile(f, dtype=dt, count=N)
        
    pts = np.vstack([data['x'], data['y'], data['z']]).T
    colors = np.vstack([data['red'], data['green'], data['blue']]).T
    print(f"[+] Loaded {len(pts):,} points.")
    return pts, colors

def save_ply_binary(filepath, points, colors):
    """
    Saves points and RGB colors into standard binary little-endian PLY.
    """
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
    dt = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    arr = np.empty(len(points), dtype=dt)
    arr['x'] = points[:, 0]
    arr['y'] = points[:, 1]
    arr['z'] = points[:, 2]
    arr['red'] = colors[:, 0]
    arr['green'] = colors[:, 1]
    arr['blue'] = colors[:, 2]
    
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)
    print(f"[+] Saved {len(points):,} cleaned points to: {filepath}")

def statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=1.2):
    """
    Removes sparse noise, flying pixels, and floating specks using KD-Tree nearest neighbor statistics.
    """
    t0 = time.time()
    print(f"[*] Running Statistical Outlier Removal (k={nb_neighbors}, std={std_ratio})...")
    tree = spatial.cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1, workers=-1)
    
    # Exclude distance to self (column 0)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    mu = np.mean(mean_dist)
    sigma = np.std(mean_dist)
    threshold = mu + std_ratio * sigma
    
    inliers = mean_dist < threshold
    removed = len(points) - np.sum(inliers)
    print(f"[+] Removed {removed:,} outlier points ({time.time()-t0:.2f}s).")
    return points[inliers], colors[inliers]

def voxel_grid_average(points, colors, voxel_size=0.002):
    """
    Averages points and colors within each voxel grid cell, eliminating duplicate surfaces.
    """
    t0 = time.time()
    print(f"[*] Voxel centroid averaging at {voxel_size*1000:.1f}mm resolution...")
    
    vox_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Sort indices to group voxels together
    # Pack 3D voxel index into unique hash/tuple
    unique_voxels, inverse_indices, counts = np.unique(
        vox_indices, axis=0, return_inverse=True, return_counts=True
    )
    
    num_voxels = len(unique_voxels)
    
    # Sum coordinates and colors in each voxel
    pts_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    cols_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    
    np.add.at(pts_sum, inverse_indices, points)
    np.add.at(cols_sum, inverse_indices, colors.astype(np.float64))
    
    avg_pts = (pts_sum / counts[:, None]).astype(np.float32)
    avg_cols = np.clip(cols_sum / counts[:, None], 0, 255).astype(np.uint8)
    
    print(f"[+] Blended {len(points):,} points down to {len(avg_pts):,} surface points ({time.time()-t0:.2f}s).")
    return avg_pts, avg_cols

def crop_bounding_box(points, colors, min_xyz=None, max_xyz=None):
    """
    Crops points to a Volume of Interest (VOI) bounding box to eliminate table, floor, and distant walls.
    """
    mask = np.ones(len(points), dtype=bool)
    if min_xyz is not None:
        mask &= (points[:, 0] >= min_xyz[0]) & (points[:, 1] >= min_xyz[1]) & (points[:, 2] >= min_xyz[2])
    if max_xyz is not None:
        mask &= (points[:, 0] <= max_xyz[0]) & (points[:, 1] <= max_xyz[1]) & (points[:, 2] <= max_xyz[2])
        
    print(f"[+] Cropped bounding box: retained {np.sum(mask):,} of {len(points):,} points.")
    return points[mask], colors[mask]

def main():
    parser = argparse.ArgumentParser(description="Clean, de-noise, and optimize workpiece point clouds.")
    parser.add_argument("input_ply", help="Path to raw scanned PLY file")
    parser.add_argument("--output_ply", default=None, help="Path for cleaned output PLY")
    parser.add_argument("--voxel_size", type=float, default=0.002, help="Voxel size in meters (default: 0.002 = 2mm)")
    parser.add_argument("--neighbors", type=int, default=20, help="Number of neighbors for outlier filtering (default: 20)")
    parser.add_argument("--std_ratio", type=float, default=1.2, help="Standard deviation threshold for outliers (default: 1.2)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_ply):
        print(f"[!] File not found: {args.input_ply}")
        sys.exit(1)
        
    if args.output_ply is None:
        base, ext = os.path.splitext(args.input_ply)
        args.output_ply = f"{base}_clean{ext}"
        
    print("=" * 65)
    print("      WORKPIECE POINT CLOUD CLEANING & OPTIMIZATION")
    print("=" * 65)
    
    pts, colors = load_ply_binary(args.input_ply)
    
    # 1. Statistical Outlier Removal
    pts, colors = statistical_outlier_removal(pts, colors, nb_neighbors=args.neighbors, std_ratio=args.std_ratio)
    
    # 2. Voxel Grid Centroid Averaging (blends duplicate surfaces & smooths surface)
    pts, colors = voxel_grid_average(pts, colors, voxel_size=args.voxel_size)
    
    # 3. Second light pass of SOR to catch any edge fringes remaining after voxelization
    pts, colors = statistical_outlier_removal(pts, colors, nb_neighbors=15, std_ratio=1.5)
    
    # 4. Save result
    save_ply_binary(args.output_ply, pts, colors)
    
    print("=" * 65)
    print(f"[SUCCESS] Cleaned point cloud ready!")
    print(f" -> Output: {args.output_ply}")
    print("=" * 65)

if __name__ == "__main__":
    main()
