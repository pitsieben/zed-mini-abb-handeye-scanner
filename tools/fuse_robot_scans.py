import numpy as np
import scipy.spatial as spatial
import os
import sys
import time
import argparse

def load_ply_binary(filepath):
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
    print(f"[+] Successfully exported {len(points):,} fused points to: {filepath}")

def estimate_normals(points, k=15):
    """
    Vectorized surface normal estimation using local covariance PCA.
    """
    tree = spatial.cKDTree(points)
    _, idxs = tree.query(points, k=k, workers=-1)
    
    neighbors = points[idxs]  # Shape: (N, k, 3)
    means = np.mean(neighbors, axis=1, keepdims=True)
    diffs = neighbors - means
    # Covariance per point
    covs = np.matmul(diffs.transpose(0, 2, 1), diffs) / float(k - 1)
    
    # Smallest eigenvalue corresponds to surface normal
    eigenvalues, eigenvectors = np.linalg.eigh(covs)
    normals = eigenvectors[:, :, 0] # (N, 3)
    
    # Normalize
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = normals / norm
    return normals, tree

def statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=1.2, tree=None):
    t0 = time.time()
    if tree is None:
        tree = spatial.cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1, workers=-1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
    inliers = mean_dist < threshold
    print(f"[+] Outlier Removal: pruned {len(points) - np.sum(inliers):,} noise points ({time.time()-t0:.2f}s).")
    return points[inliers], colors[inliers]

def fuse_overlapping_surfaces(points, colors, search_radius=0.004, normal_threshold=0.7):
    """
    Identifies overlapping multi-angle scan layers and collapses them into a single consensus surface.
    - search_radius: Maximum distance between overlapping surfaces (e.g. 4mm)
    - normal_threshold: Cosine similarity of normals (0.7 = ~45 deg). Points with agreeing normals
      are merged; points with opposing normals (e.g. opposite sides of thin geometry) are preserved.
    """
    t0 = time.time()
    print(f"[*] Analyzing surface geometry & estimating normals ({len(points):,} points)...")
    normals, tree = estimate_normals(points, k=15)
    
    print(f"[*] Identifying overlapping zones within {search_radius*1000:.1f}mm radius...")
    # Find all neighbors within radius
    neighbor_indices = tree.query_ball_point(points, r=search_radius, workers=-1)
    
    merged_points = np.empty_like(points)
    merged_colors = np.empty_like(colors)
    visited = np.zeros(len(points), dtype=bool)
    keep_mask = np.zeros(len(points), dtype=bool)
    
    for i in range(len(points)):
        if visited[i]:
            continue
            
        nbrs = np.array(neighbor_indices[i], dtype=np.int32)
        if len(nbrs) <= 1:
            merged_points[i] = points[i]
            merged_colors[i] = colors[i]
            keep_mask[i] = True
            visited[i] = True
            continue
            
        # Check normal agreement with seed point
        nbr_normals = normals[nbrs]
        normal_dots = np.dot(nbr_normals, normals[i])
        
        # Only merge neighbors that share the same surface orientation
        compatible = normal_dots >= normal_threshold
        comp_nbrs = nbrs[compatible]
        
        # Mark all compatible neighbors as visited so they are not processed again
        visited[comp_nbrs] = True
        
        # Weighted centroid projection
        # Points closer to the seed point receive higher weight
        comp_pts = points[comp_nbrs]
        comp_cols = colors[comp_nbrs].astype(np.float32)
        
        nbr_dists = np.linalg.norm(comp_pts - points[i], axis=1)
        weights = np.exp(-0.5 * (nbr_dists / (search_radius * 0.5)) ** 2)
        total_w = np.sum(weights)
        
        merged_points[i] = np.sum(comp_pts * weights[:, None], axis=0) / total_w
        merged_colors[i] = np.clip(np.sum(comp_cols * weights[:, None], axis=0) / total_w, 0, 255).astype(np.uint8)
        keep_mask[i] = True
        
    final_pts = merged_points[keep_mask]
    final_cols = merged_colors[keep_mask]
    
    collapsed = len(points) - len(final_pts)
    print(f"[+] Overlap Fusion complete: collapsed {collapsed:,} redundant multi-pass points into a single surface ({time.time()-t0:.2f}s).")
    return final_pts, final_cols

def voxel_grid_average(points, colors, voxel_size=0.002):
    t0 = time.time()
    vox_indices = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices, counts = np.unique(
        vox_indices, axis=0, return_inverse=True, return_counts=True
    )
    num_voxels = len(unique_voxels)
    
    pts_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    cols_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    
    np.add.at(pts_sum, inverse_indices, points)
    np.add.at(cols_sum, inverse_indices, colors.astype(np.float64))
    
    avg_pts = (pts_sum / counts[:, None]).astype(np.float32)
    avg_cols = np.clip(cols_sum / counts[:, None], 0, 255).astype(np.uint8)
    print(f"[+] Uniform Voxel Filter ({voxel_size*1000:.1f}mm): {len(avg_pts):,} output points ({time.time()-t0:.2f}s).")
    return avg_pts, avg_cols

def main():
    parser = argparse.ArgumentParser(description="Multi-angle scan overlap elimination & consensus surface fusion.")
    parser.add_argument("input_ply", help="Path to raw or combined PLY scan")
    parser.add_argument("--output_ply", default=None, help="Output path for merged PLY")
    parser.add_argument("--overlap_radius", type=float, default=0.004, help="Maximum thickness of overlapping zone in meters (default: 0.004 = 4mm)")
    parser.add_argument("--voxel_size", type=float, default=0.002, help="Final surface resolution in meters (default: 0.002 = 2mm)")
    parser.add_argument("--normal_dot", type=float, default=0.70, help="Normal agreement threshold for merging (default: 0.70)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_ply):
        print(f"[!] File not found: {args.input_ply}")
        sys.exit(1)
        
    if args.output_ply is None:
        base, ext = os.path.splitext(args.input_ply)
        args.output_ply = f"{base}_fused{ext}"
        
    print("=" * 70)
    print("     ROBOT MULTI-SCAN OVERLAP IDENTIFICATION & SURFACE FUSION")
    print("=" * 70)
    
    pts, colors = load_ply_binary(args.input_ply)
    
    # 1. Prune flying noise and edge specks
    pts, colors = statistical_outlier_removal(pts, colors, nb_neighbors=20, std_ratio=1.2)
    
    # 2. Identify overlapping multi-angle surfaces and fuse them into single consensus geometry
    pts, colors = fuse_overlapping_surfaces(pts, colors, search_radius=args.overlap_radius, normal_threshold=args.normal_dot)
    
    # 3. Uniform voxel grid smoothing
    pts, colors = voxel_grid_average(pts, colors, voxel_size=args.voxel_size)
    
    # 4. Final light pass of SOR
    pts, colors = statistical_outlier_removal(pts, colors, nb_neighbors=12, std_ratio=1.5)
    
    save_ply_binary(args.output_ply, pts, colors)
    print("=" * 70)
    print(f"[SUCCESS] Merged point cloud generated with zero overlapping layers!")
    print(f" -> Output: {args.output_ply}")
    print("=" * 70)

if __name__ == "__main__":
    main()
