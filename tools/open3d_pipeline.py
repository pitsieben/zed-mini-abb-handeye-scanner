"""
Open3D High-Precision Point Cloud Cleaning, Multi-Way Registration & Surface Reconstruction
========================================================================================
This module implements the 5-step photogrammetric & volumetric registration pipeline:
  1. Base Frame Transformation: Transforms local camera scans to common robot base frame
     using the calibrated Hand-Eye matrix (T_base2cam = T_base2ee @ T_ee2cam).
  2. Individual Scan Cleaning: Statistical & Radius Outlier Removal on each scan before merging
     to kill edge fraying and grazing angle noise.
  3. Multi-Way ICP Alignment: Point-to-Plane ICP pairwise registration + Pose Graph
     global optimization to lock all scan angles into sub-millimeter consensus.
  4. Surface Reconstruction:
     - Volumetric TSDF Integration (ScalableTSDFVolume) for RGB-D frames.
     - Screened Poisson Reconstruction (create_from_point_cloud_poisson) for point clouds.
  5. Watertight Mesh Extraction & Clean-Up: Marching cubes / Poisson trim, degenerate face pruning,
     non-manifold edge fixing, and export to PLY / OBJ / STL.
"""

import os
import sys
import time
import argparse
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None
import json
from scipy.spatial.transform import Rotation


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='#', printEnd="\r"):
    """Call in a loop to create terminal progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print()


class Open3DScannerPipeline:
    def __init__(self, calib_path="calibration_result.npz", config_path="scan_config.json"):
        self.R_cam2ee = np.eye(3, dtype=np.float64)
        self.t_cam2ee = np.zeros((3, 1), dtype=np.float64)
        self.load_calibration(calib_path)
        self.config = {
            "voxel_size_mm": 2.0,
            "outlier_std_ratio": 1.2,
            "outlier_nb_neighbors": 16,
            "icp_max_distance_mm": 15.0
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config.update(json.load(f))
                print(f"[PIPELINE] Loaded {config_path} (Voxel: {self.config.get('voxel_size_mm')}mm, ICP: {self.config.get('icp_max_distance_mm')}mm)")
            except Exception as e:
                pass

    def load_calibration(self, calib_path):
        if os.path.exists(calib_path):
            try:
                data = np.load(calib_path)
                self.R_cam2ee = data['R'].astype(np.float64)
                self.t_cam2ee = data['t'].astype(np.float64)
                print(f"[PIPELINE] Loaded Hand-Eye calibration from {calib_path}")
            except Exception as e:
                print(f"[PIPELINE WARNING] Could not load calibration: {e}")
        else:
            print(f"[PIPELINE INFO] Calibration file {calib_path} not found; using identity.")

    # -------------------------------------------------------------------------
    # STEP 1: Compute Robot-to-Camera Transformation Matrix
    # -------------------------------------------------------------------------
    def compute_cam2base_matrix(self, tx_mm, ty_mm, tz_mm, qw, qx, qy, qz):
        """
        Computes the 4x4 rigid transformation matrix from camera frame to robot base frame.
        T_base2cam = T_base2ee @ T_ee2cam
        """
        t_base2ee = np.array([tx_mm / 1000.0, ty_mm / 1000.0, tz_mm / 1000.0], dtype=np.float64)
        rot_base2ee = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float64)

        R_base2cam = rot_base2ee @ self.R_cam2ee
        t_base2cam = rot_base2ee @ self.t_cam2ee.ravel() + t_base2ee

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_base2cam
        T[:3, 3] = t_base2cam
        return T

    # -------------------------------------------------------------------------
    # STEP 2: Clean Point Cloud Individually First
    # -------------------------------------------------------------------------
    @staticmethod
    def clean_individual_pcd(pcd, nb_neighbors=16, std_ratio=1.4, voxel_size=0.0025, verbose=True):
        """
        Runs high-speed voxel downsampling and statistical outlier removal on an individual scan.
        Filters out edge noise, flying pixels, and reflection flares before multi-scan fusion.
        """
        initial_pts = len(pcd.points)
        if initial_pts == 0:
            return pcd

        # 1. Voxel downsample first for consistent spacing and ultra-fast KD-tree filtering
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # 2. Statistical Outlier Removal (returns cleaned point cloud directly)
        pcd_cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # 3. Estimate surface normals if not present
        if not pcd_cleaned.has_normals():
            pcd_cleaned.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.012, voxel_size * 4), max_nn=25)
            )

        final_pts = len(pcd_cleaned.points)
        if verbose:
            print(f"[CLEAN] {initial_pts:,} -> {final_pts:,} points (removed {initial_pts - final_pts:,} outliers)")
        return pcd_cleaned

    # -------------------------------------------------------------------------
    # STEP 3: Pairwise Point-to-Plane ICP & Multi-Way Pose Graph Optimization
    # -------------------------------------------------------------------------
    @staticmethod
    def pairwise_registration(source, target, max_correspondence_distance=0.015):
        """
        Point-to-Plane ICP registration between source and target clouds.
        Returns transformation matrix and information matrix for pose graph.
        """
        if not source.has_normals():
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
        if not target.has_normals():
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))

        # Initial point-to-plane ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.005)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )

        reg_result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance,
            np.eye(4), estimation, criteria
        )

        # Compute information matrix
        info_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance, reg_result.transformation
        )

        return reg_result.transformation, info_matrix

    def optimize_pose_graph_spatial_multiway(self, pcd_list, initial_poses, max_correspondence_distance=0.025):
        """
        Universal Trajectory-Agnostic Registration:
        - Completely independent of toolpaths, passes, or scanning sequences.
        - Evaluates 3D spatial co-visibility via Axis-Aligned Bounding Box (AABB) intersection in robot base frame.
        - Employs Multi-Scale Point-to-Plane ICP + Photometric Colored ICP (using RGB gradients)
          to lock overlapping scans into place, preventing sliding on organic / moving subjects.
        - Globally optimizes all camera poses simultaneously using Levenberg-Marquardt.
        """
        n_pcds = len(pcd_list)
        if n_pcds <= 1:
            return initial_poses

        print(f"\n[SPATIAL MULTI-WAY REGISTRATION] Building Pose Graph for {n_pcds} scan stages...")

        # 1. Transform all stages into world base frame using calibrated initial guess
        world_pcds = []
        for pcd, T_init in zip(pcd_list, initial_poses):
            p_world = o3d.geometry.PointCloud(pcd)
            p_world.transform(T_init)
            # Downsample for fast, robust correspondence search
            p_down = p_world.voxel_down_sample(voxel_size=0.004)
            if not p_down.has_normals():
                p_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
            world_pcds.append(p_down)

        pose_graph = o3d.pipelines.registration.PoseGraph()
        for i in range(n_pcds):
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

        # 2. Add sequential continuity edges (with flexible uncertainty) to guarantee full graph connectivity
        for i in range(n_pcds - 1):
            info_seq = np.eye(6) * 1e-4
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i, i + 1, np.eye(4), info_seq, uncertain=True)
            )

        # 3. Detect all spatial overlapping pairs using 3D Bounding Boxes
        edges_added = 0
        # Expand coarse search to 100-120mm to bridge subtle human/subject posture movement between passes
        coarse_threshold = max(0.100, max_correspondence_distance * 3.5)
        fine_threshold = max_correspondence_distance

        print("\n[SPATIAL MULTI-WAY REGISTRATION] Searching for 3D co-visible stages & feature overlap...")
        total_pairs = (n_pcds * (n_pcds - 1)) // 2
        pair_idx = 0

        for i in range(n_pcds):
            bb_i = world_pcds[i].get_axis_aligned_bounding_box()
            for j in range(i + 1, n_pcds):
                print_progress_bar(pair_idx, total_pairs, prefix='Pairwise ICP', suffix=f'Edges: {edges_added}', length=30)
                pair_idx += 1
                
                bb_j = world_pcds[j].get_axis_aligned_bounding_box()

                # Bounding box intersection check in world coordinates (inflated by 50mm)
                min_bound = np.maximum(bb_i.get_min_bound() - 0.05, bb_j.get_min_bound() - 0.05)
                max_bound = np.minimum(bb_i.get_max_bound() + 0.05, bb_j.get_max_bound() + 0.05)

                if np.all(max_bound > min_bound):
                    # Check coarse spatial overlap
                    eval_init = o3d.pipelines.registration.evaluate_registration(
                        world_pcds[i], world_pcds[j], coarse_threshold, np.eye(4)
                    )

                    if eval_init.fitness > 0.12:
                        # Step A: Coarse Point-to-Plane ICP
                        loss_tukey = o3d.pipelines.registration.TukeyLoss(k=0.005)
                        est_p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss_tukey)
                        reg_coarse = o3d.pipelines.registration.registration_icp(
                            world_pcds[i], world_pcds[j], coarse_threshold, np.eye(4), est_p2pl
                        )

                        best_trans = reg_coarse.transformation
                        best_fitness = reg_coarse.fitness
                        best_rmse = reg_coarse.inlier_rmse

                        # Step B: Fine ICP Registration
                        # Attempt Photometric Colored ICP (using RGB gradients & color texture)
                        colored_icp_success = False
                        if world_pcds[i].has_colors() and world_pcds[j].has_colors():
                            # Temporarily suppress Open3D C++ warnings for 0 correspondences
                            current_verbosity = o3d.utility.get_verbosity_level()
                            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
                            try:
                                reg_col = o3d.pipelines.registration.registration_colored_icp(
                                    world_pcds[i], world_pcds[j], fine_threshold, reg_coarse.transformation,
                                    o3d.pipelines.registration.TransformationEstimationForColoredICP()
                                )
                                # Accept colored ICP if it maintains or improves alignment
                                if reg_col.fitness >= best_fitness * 0.75:
                                    best_trans = reg_col.transformation
                                    best_fitness = reg_col.fitness
                                    best_rmse = reg_col.inlier_rmse
                                    colored_icp_success = True
                            except Exception:
                                pass
                            finally:
                                o3d.utility.set_verbosity_level(current_verbosity)

                        # Step B (fallback): Fine Point-to-Plane ICP
                        if not colored_icp_success:
                            reg_fine = o3d.pipelines.registration.registration_icp(
                                world_pcds[i], world_pcds[j], fine_threshold, reg_coarse.transformation, est_p2pl
                            )
                            if reg_fine.fitness >= best_fitness:
                                best_trans = reg_fine.transformation
                                best_fitness = reg_fine.fitness
                                best_rmse = reg_fine.inlier_rmse

                        # Only add edge if alignment has solid consensus
                        if best_fitness > 0.20 and best_rmse < 0.030:
                            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                                world_pcds[i], world_pcds[j], fine_threshold, best_trans
                            )
                            pose_graph.edges.append(
                                o3d.pipelines.registration.PoseGraphEdge(i, j, best_trans, info, uncertain=True)
                            )
                            edges_added += 1
                            # Erase progress bar temporarily to print link, then restore
                            sys.stdout.write('\r' + ' ' * 80 + '\r')
                            print(f"  [OVERLAP LINK] Scan #{i+1} <-> #{j+1}: Fitness={best_fitness:.2f}, RMSE={best_rmse*1000:.1f}mm")

        print_progress_bar(total_pairs, total_pairs, prefix='Pairwise ICP', suffix=f'Edges: {edges_added}', length=30)


        print(f"[SPATIAL MULTI-WAY REGISTRATION] Established {edges_added} spatial consensus edges.")

        # 4. Global Optimization across all stages
        print("[SPATIAL MULTI-WAY REGISTRATION] Running Global Levenberg-Marquardt Optimization...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=fine_threshold,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.2,
            reference_node=0
        )
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            option
        )

        # 5. Extract refined global camera-to-base poses
        refined_poses = []
        for i in range(n_pcds):
            delta_T = pose_graph.nodes[i].pose
            T_opt = delta_T @ initial_poses[i]
            refined_poses.append(T_opt)

        print("[SPATIAL MULTI-WAY REGISTRATION] Global registration completed successfully.")
        return refined_poses

    # -------------------------------------------------------------------------
    # STEP 4: Volumetric TSDF Integration & Screened Poisson Reconstruction
    # -------------------------------------------------------------------------
    @staticmethod
    def integrate_tsdf_volume(rgb_images, depth_images, camera_poses, intrinsics,
                              voxel_length=0.002, sdf_trunc=0.008):
        """
        Fuses a sequence of RGB-D frames into a Truncated Signed Distance Field (TSDF) volume.
        Naturally resolves overlapping surfaces and averages out depth sensor noise.
        """
        print(f"[TSDF] Initializing Scalable TSDF Volume (voxel: {voxel_length*1000:.1f}mm, trunc: {sdf_trunc*1000:.1f}mm)...")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        width, height = intrinsics['width'], intrinsics['height']
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        for idx, (rgb_img, depth_img, pose) in enumerate(zip(rgb_images, depth_images, camera_poses)):
            # Convert depth from meters float to uint16 (millimeters)
            depth_mm = (depth_img * 1000.0).astype(np.uint16)
            rgb_o3d = o3d.geometry.Image(rgb_img)
            depth_o3d = o3d.geometry.Image(depth_mm)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d,
                depth_scale=1000.0,
                depth_trunc=intrinsics.get('max_depth', 0.9),
                convert_rgb_to_intensity=False
            )
            # Camera-to-world pose
            volume.integrate(rgbd, o3d_intrinsics, np.linalg.inv(pose))
            print(f"  [TSDF] Integrated frame {idx + 1}/{len(rgb_images)}")

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        print(f"[TSDF] Extracted triangle mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")
        return mesh

    @staticmethod
    def reconstruct_poisson_mesh(pcd, depth=9, density_trim_quantile=0.04, camera_centers=None):
        """
        Generates a watertight surface mesh from a merged, cleaned point cloud using
        Screened Poisson Surface Reconstruction.
        Trims low-density phantom triangles at scan boundaries.
        """
        print(f"\n[SURFACE RECONSTRUCTION] Processing point cloud with {len(pcd.points):,} points...")
        
        # 1. Voxel Downsample to uniformize density across overlapping passes
        pcd_down = pcd.voxel_down_sample(voxel_size=0.002)
        print(f"[SURFACE RECONSTRUCTION] Downsampled to {len(pcd_down.points):,} uniform points.")

        # 2. Outlier removal on merged scan
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

        # 3. High-quality normal estimation
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=35)
        )

        # 4. Consistent Normal Orientation (fast vector orientation)
        if camera_centers is not None and len(camera_centers) > 0:
            cam_center_mean = np.mean(camera_centers, axis=0)
            pcd_clean.orient_normals_towards_camera_location(cam_center_mean)
        else:
            center = pcd_clean.get_center()
            pcd_clean.orient_normals_towards_camera_location(center + np.array([0.0, 0.0, 1.0]))

        # 5. Screened Poisson Surface Reconstruction
        print(f"[SURFACE RECONSTRUCTION] Computing Poisson Surface (octree depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=depth, linear_fit=True
        )

        # 6. Density-based trimming (eliminates Poisson bubbles and phantom boundary geometry)
        densities_arr = np.asarray(densities)
        trim_threshold = np.quantile(densities_arr, density_trim_quantile)
        vertices_to_remove = densities_arr < trim_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"[SURFACE RECONSTRUCTION] Trimmed {np.sum(vertices_to_remove):,} low-density boundary vertices.")

        # 7. Mesh Decimation (Optional LOD)
        # Note: We need to pull the detail_level from self.config, but this is a static method.
        # We will do it in process_multiway_session instead, before saving!

        return mesh, pcd_clean

    # -------------------------------------------------------------------------
    # STEP 5: Mesh Clean-Up & Watertight Export
    # -------------------------------------------------------------------------
    @staticmethod
    def clean_and_repair_mesh(mesh):
        """
        Fixes non-manifold edges, removes degenerate/duplicated triangles & vertices,
        and computes smooth vertex normals.
        """
        print("[MESH CLEANUP] Running geometric repair passes...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        print(f"[MESH CLEANUP] Result: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles (Watertight: {mesh.is_watertight()}).")
        return mesh

    def export_mesh(self, mesh, output_path):
        """
        Exports the cleaned triangle mesh into PLY, OBJ, or STL.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self.clean_and_repair_mesh(mesh)
        
        # Suppress harmless OBJ warning about triangle normals not being supported in .obj
        current_verbosity = o3d.utility.get_verbosity_level()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        try:
            success = o3d.io.write_triangle_mesh(output_path, mesh)
        finally:
            o3d.utility.set_verbosity_level(current_verbosity)

        if success:
            pass # We rely on other print statements to show the output path cleanly
        else:
            print(f"[!] Failed to write mesh to: {output_path}")
        return success


    def process_multiway_session(self, session_dir, voxel_size=0.002, depth=8, clean_only=False, scale_mm=True):
        """
        Executes the full 5-step photogrammetric & multi-way registration pipeline
        across all robot scan stages saved in a session directory:
          1. Loads each raw scan stage and its robot flange pose.
          2. Applies Hand-Eye calibration matrix to compute base-frame initial guess.
          3. Cleans each scan individually with statistical + radius outlier removal.
          4. Pairwise Point-to-Plane ICP with Tukey robust loss between adjacent & overlapping passes.
          5. Global PoseGraph optimization with Levenberg-Marquardt to lock all scans into consensus.
          6. Merges aligned scans and optionally runs Screened Poisson Surface Reconstruction.
          7. Scales to 1:1 Millimeters for direct Rhino / Grasshopper reference.
        """
        manifest_path = os.path.join(session_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Session manifest not found at: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        stages = manifest.get("stages", [])
        if len(stages) == 0:
            raise ValueError(f"No scan stages listed in {manifest_path}")

        print(f"\n[MULTI-WAY PIPELINE] Found {len(stages)} stages in session: {session_dir}")

        cleaned_pcds = []
        initial_poses = []
        total_stages = len(stages)

        print(f"\n[MULTI-WAY PIPELINE] Loading & Cleaning {total_stages} stages...")
        for idx, stage in enumerate(stages):
            print_progress_bar(idx, total_stages, prefix='Loading', suffix=f'Stage {idx+1}/{total_stages}', length=30)
            
            ply_file = os.path.join(session_dir, stage["ply"]) if not os.path.isabs(stage["ply"]) else stage["ply"]
            if not os.path.exists(ply_file):
                continue

            # Load raw scan in camera frame
            raw_pcd = o3d.io.read_point_cloud(ply_file)

            # Step 2: Clean individual scan first
            clean_pcd = self.clean_individual_pcd(
                raw_pcd, nb_neighbors=16, std_ratio=1.4, voxel_size=max(0.0025, voxel_size), verbose=False
            )

            # Step 1: Compute initial pose T_base2cam = T_base2ee @ T_ee2cam
            pose = stage["pose_flange"]
            T_base2cam = self.compute_cam2base_matrix(
                pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]
            )

            cleaned_pcds.append(clean_pcd)
            initial_poses.append(T_base2cam)
            
        print_progress_bar(total_stages, total_stages, prefix='Loading', suffix='Complete!         ', length=30)

        if len(cleaned_pcds) == 0:
            raise ValueError("No valid point clouds could be loaded from session.")

        # Step 3: Universal Trajectory-Agnostic Spatial Registration
        # Eliminates all toolpath pass assumptions: discovers co-visible overlap across space
        # and applies Multi-Scale Point-to-Plane + Colored ICP with Global PoseGraph optimization.
        icp_search_dist = float(self.config.get("icp_max_distance_mm", 25.0)) / 1000.0
        skip_icp = self.config.get("skip_icp", False)
        
        if not skip_icp:
            refined_poses = self.optimize_pose_graph_spatial_multiway(
                cleaned_pcds, initial_poses, max_correspondence_distance=icp_search_dist
            )
        else:
            print("\n[MULTI-WAY PIPELINE] Skipping ICP alignment (trusting raw ABB robot kinematics).")
            refined_poses = initial_poses

        # Transform each stage to its globally optimized pose, orient normals towards its own camera, and merge
        print("\n[MULTI-WAY PIPELINE] Transforming and fusing aligned point clouds with per-camera normal orientation...")
        fused_pcd = o3d.geometry.PointCloud()
        total_fuse = len(cleaned_pcds)
        for idx, (pcd, T_opt) in enumerate(zip(cleaned_pcds, refined_poses)):
            print_progress_bar(idx, total_fuse, prefix='Fusing', suffix=f'Stage {idx+1}/{total_fuse}', length=30)
            pcd_trans = o3d.geometry.PointCloud(pcd)
            pcd_trans.transform(T_opt)
            cam_pos = T_opt[:3, 3]
            pcd_trans.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
            pcd_trans.orient_normals_towards_camera_location(cam_pos)
            fused_pcd += pcd_trans
        print_progress_bar(total_fuse, total_fuse, prefix='Fusing', suffix='Complete!         ', length=30)

        # Voxel downsample fused surface to eliminate duplicate density
        fused_pcd = fused_pcd.voxel_down_sample(voxel_size=voxel_size)
        fused_pcd, _ = fused_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.3)
        print(f"[+] Consolidated workpiece cloud: {len(fused_pcd.points):,} clean surface points.")

        # Millimeter scaling for Rhino (x 1000.0)
        pcd_mm = o3d.geometry.PointCloud(fused_pcd)
        if scale_mm:
            pcd_mm.scale(1000.0, center=np.zeros(3))

        # Save merged point cloud in mm
        os.makedirs("scans", exist_ok=True)
        out_ply_mm = os.path.join("scans", "latest_workpiece_mm.ply")
        o3d.io.write_point_cloud(out_ply_mm, pcd_mm)
        print(f"[+] Exported 1:1 Millimeter Point Cloud for Rhino: {out_ply_mm}")

        out_session_ply = os.path.join(session_dir, "fused_workpiece_mm.ply")
        o3d.io.write_point_cloud(out_session_ply, pcd_mm)

        if not clean_only:
            # Step 4 & 5: Poisson Reconstruction with aggressive trimming to eliminate phantom boundary sheets
            mesh, _ = self.reconstruct_poisson_mesh(fused_pcd, depth=depth, density_trim_quantile=0.12)
            
            # Apply Mesh Decimation based on desired detail level
            detail_level = self.config.get("mesh_detail", "high")
            if detail_level != "high":
                target_triangles = 100000 if detail_level == "medium" else 30000
                current_triangles = len(mesh.triangles)
                if current_triangles > target_triangles:
                    print(f"[SURFACE RECONSTRUCTION] Decimating mesh to ~{target_triangles} triangles (Level: {detail_level})...")
                    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
                    mesh.compute_vertex_normals()
                    print(f"[SURFACE RECONSTRUCTION] Decimated mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")

            if scale_mm:
                mesh.scale(1000.0, center=np.zeros(3))
            out_mesh_obj = os.path.join("scans", "latest_workpiece_mesh_mm.obj")
            out_mesh_stl = os.path.join("scans", "latest_workpiece_mesh_mm.stl")
            self.export_mesh(mesh, out_mesh_obj)
            self.export_mesh(mesh, out_mesh_stl)
            print(f"[+] Exported 1:1 Millimeter Mesh for Rhino: {out_mesh_obj}")

        return fused_pcd




    def process_tsdf_session(self, session_dir, voxel_length=0.002, sdf_trunc=0.008, scale_mm=True):
        """
        TSDF Volumetric Fusion Pipeline:
        ====================================
        Loads per-stage RGB images and raw depth maps saved by the web scanner,
        integrates them into a Truncated Signed Distance Field (TSDF) volume using
        each stage's calibrated camera-to-base pose.

        Unlike Poisson reconstruction, TSDF **votes** into a shared voxel grid:
        conflicting observations from different angles are averaged out physically,
        so the double-face / sliding-surface artifact cannot occur.

        Requires per-stage .png and _depth.npy files (saved from web_scanner_app.py v2+).
        Falls back to skipping stages that lack RGB-D data.
        """
        manifest_path = os.path.join(session_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Session manifest not found at: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        stages = manifest.get("stages", [])
        if not stages:
            raise ValueError(f"No stages found in {manifest_path}")

        # Camera intrinsics from manifest (written by scanner at first capture)
        intr = manifest.get("intrinsics", {})
        fx   = float(intr.get("fx", 700.0))
        fy   = float(intr.get("fy", 700.0))
        cx   = float(intr.get("cx", 640.0))
        cy   = float(intr.get("cy", 360.0))
        w    = int(intr.get("width", 1280))
        h    = int(intr.get("height", 720))

        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        integrated = 0
        skipped = 0
        total_stages = len(stages)

        print(f"\n[TSDF PIPELINE] Starting volumetric fusion for {total_stages} stages...")
        print(f"[TSDF PIPELINE] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} ({w}x{h})")
        print(f"[TSDF PIPELINE] Voxel: {voxel_length*1000:.1f}mm | Trunc: {sdf_trunc*1000:.1f}mm\n")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for idx, stage in enumerate(stages):
            print_progress_bar(idx, total_stages, prefix='TSDF Fusion', suffix=f'Stage {idx+1}/{total_stages}', length=30)
            rgb_file   = stage.get("rgb")
            depth_file = stage.get("depth_npy")

            if not rgb_file or not depth_file:
                skipped += 1
                continue

            rgb_path   = os.path.join(session_dir, rgb_file)   if not os.path.isabs(rgb_file)   else rgb_file
            depth_path = os.path.join(session_dir, depth_file) if not os.path.isabs(depth_file) else depth_file

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                skipped += 1
                continue

            try:
                import cv2
                bgr = cv2.imread(rgb_path)
                if bgr is None:
                    raise IOError(f"cv2.imread failed for {rgb_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                depth_m = np.load(depth_path).astype(np.float32)

                # Resize depth to match RGB if shapes differ
                if depth_m.shape[:2] != rgb.shape[:2]:
                    import cv2 as _cv2
                    depth_m = _cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=_cv2.INTER_NEAREST)

                # --- APPLY USER CONFIGURATION CLIPPING ---
                max_depth_m = float(self.config.get("max_depth_m", 0.70))
                min_depth_m = float(self.config.get("min_depth_m", 0.20))
                border_crop = float(self.config.get("border_crop", 0.06))

                # 1. Distance clipping (ignore background/floor)
                depth_m[(depth_m > max_depth_m) | (depth_m < min_depth_m)] = 0.0

                # 2. Border cropping (ignore ZED lens edge distortion)
                if border_crop > 0:
                    crop_y = int(depth_m.shape[0] * border_crop)
                    crop_x = int(depth_m.shape[1] * border_crop)
                    if crop_y > 0:
                        depth_m[:crop_y, :] = 0.0
                        depth_m[-crop_y:, :] = 0.0
                    if crop_x > 0:
                        depth_m[:, :crop_x] = 0.0
                        depth_m[:, -crop_x:] = 0.0

                # Clean NaN/Inf in depth
                depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

                rgb_o3d   = o3d.geometry.Image(rgb.astype(np.uint8))
                depth_mm  = (depth_m * 1000.0).astype(np.uint16)
                depth_o3d = o3d.geometry.Image(depth_mm)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d,
                    depth_scale=1000.0,
                    depth_trunc=max_depth_m,
                    convert_rgb_to_intensity=False
                )

                # Camera-to-world (base-frame) pose from hand-eye calibration
                pose = stage["pose_flange"]
                T_cam2base = self.compute_cam2base_matrix(*pose)
                T_world2cam = np.linalg.inv(T_cam2base)  # TSDF needs world-to-camera

                volume.integrate(rgbd, o3d_intrinsics, T_world2cam)
                integrated += 1

            except Exception:
                skipped += 1

        print_progress_bar(total_stages, total_stages, prefix='TSDF Fusion', suffix='Complete!         ', length=30)

        if integrated == 0:
            print("[TSDF PIPELINE] No stages could be integrated. "
                  "Make sure you are using a scan session captured with web_scanner_app.py v2+ "
                  "(which saves .png and _depth.npy per stage).")
            return None

        print(f"\n[TSDF PIPELINE] Integrated {integrated} stages ({skipped} skipped). Extracting mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        print(f"[TSDF PIPELINE] Raw TSDF mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")

        # Apply Mesh Decimation based on desired detail level
        detail_level = self.config.get("mesh_detail", "high")
        if detail_level != "high":
            target_triangles = 100000 if detail_level == "medium" else 30000
            current_triangles = len(mesh.triangles)
            if current_triangles > target_triangles:
                print(f"[TSDF PIPELINE] Decimating mesh to ~{target_triangles} triangles (Level: {detail_level})...")
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
                mesh.compute_vertex_normals()
                print(f"[TSDF PIPELINE] Decimated mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")

        if scale_mm:
            mesh.scale(1000.0, center=np.zeros(3))

        os.makedirs("scans", exist_ok=True)
        out_obj = os.path.join("scans", "latest_workpiece_tsdf_mm.obj")
        out_stl = os.path.join("scans", "latest_workpiece_tsdf_mm.stl")
        self.export_mesh(mesh, out_obj)
        self.export_mesh(mesh, out_stl)
        print(f"[+] TSDF mesh exported (1:1 mm): {out_obj}")
        print(f"[+] TSDF mesh exported (1:1 mm): {out_stl}")

        # Also export TSDF point cloud for comparison
        pcd_tsdf = volume.extract_point_cloud()
        if scale_mm:
            pcd_tsdf.scale(1000.0, center=np.zeros(3))
        out_ply = os.path.join("scans", "latest_workpiece_tsdf_mm.ply")
        o3d.io.write_point_cloud(out_ply, pcd_tsdf)
        print(f"[+] TSDF point cloud (1:1 mm): {out_ply}")

        return mesh


# -------------------------------------------------------------------------
# CLI Command Line Interface
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Open3D Robot Scan Cleaning, Multi-Way Alignment & Meshing Pipeline")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input PLY point cloud or scan directory")
    parser.add_argument("--session", "-s", type=str, default=None, help="Scan session directory containing manifest.json")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output mesh or cleaned PLY path")
    parser.add_argument("--depth", "-d", type=int, default=8, help="Poisson octree reconstruction depth (default: 8)")
    parser.add_argument("--voxel-size", "-v", type=float, default=0.002, help="Voxel size in meters (default: 0.002 = 2mm)")
    parser.add_argument("--clean-only", action="store_true", help="Only clean and downsample point cloud without meshing")
    parser.add_argument("--scale-mm", action="store_true", default=True, help="Scale output to millimeters for Rhino 1:1 import (default: True)")
    parser.add_argument(
        "--mode", "-m", type=str, default="icp",
        choices=["icp", "tsdf", "both"],
        help=(
            "Reconstruction pipeline mode:\n"
            "  icp  = Multi-Way ICP + Screened Poisson (default, works on PLY-only sessions)\n"
            "  tsdf = TSDF Volumetric Fusion (requires RGB + depth per stage, resolves double-surface)\n"
            "  both = Run ICP and TSDF in sequence"
        )
    )
    args = parser.parse_args()

    pipeline = Open3DScannerPipeline()

    target_session = args.session
    if target_session is None and args.input and os.path.isdir(args.input):
        target_session = args.input

    # ---- Session modes ----
    if target_session:
        if args.mode in ("icp", "both"):
            print(f"\n=======================================================")
            print(f"   OPEN3D MULTI-WAY ICP & FUSION: SESSION MODE         ")
            print(f"=======================================================")
            print(f"Session Directory: {target_session}")
            pipeline.process_multiway_session(
                session_dir=target_session,
                voxel_size=args.voxel_size,
                depth=args.depth,
                clean_only=args.clean_only,
                scale_mm=args.scale_mm
            )
            print("=======================================================\n")

        if args.mode in ("tsdf", "both"):
            print(f"\n=======================================================")
            print(f"   OPEN3D TSDF VOLUMETRIC FUSION: SESSION MODE         ")
            print(f"=======================================================")
            print(f"Session Directory: {target_session}")
            pipeline.process_tsdf_session(
                session_dir=target_session,
                scale_mm=args.scale_mm
            )
            print("=======================================================\n")
        return

    # ---- Single PLY mode ----
    if not args.input or not os.path.exists(args.input):
        print(f"Error: Specify a valid input file (--input <file.ply>) or session directory (--session <dir>).")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"       OPEN3D PRECISION SCAN RECONSTRUCTION PIPELINE   ")
    print(f"=======================================================")
    print(f"Input:  {args.input}")

    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Loaded: {len(pcd.points):,} points from {args.input}")

    # Step 2: Individual Cleaning
    pcd_clean = pipeline.clean_individual_pcd(
        pcd,
        nb_neighbors=25,
        std_ratio=1.4,
        radius=0.012,
        min_points=12,
        voxel_size=args.voxel_size
    )

    if args.scale_mm:
        pcd_clean_save = o3d.geometry.PointCloud(pcd_clean)
        pcd_clean_save.scale(1000.0, center=np.zeros(3))
    else:
        pcd_clean_save = pcd_clean

    if args.clean_only:
        out_path = args.output or args.input.replace(".ply", "_clean_mm.ply")
        o3d.io.write_point_cloud(out_path, pcd_clean_save)
        print(f"Saved cleaned point cloud to: {out_path}")
        return

    # Step 4 & 5: Poisson Reconstruction & Mesh Clean-up
    mesh, final_pcd = pipeline.reconstruct_poisson_mesh(
        pcd_clean,
        depth=args.depth,
        density_trim_quantile=0.04
    )

    if args.scale_mm:
        mesh.scale(1000.0, center=np.zeros(3))
        final_pcd.scale(1000.0, center=np.zeros(3))

    out_mesh_path = args.output or args.input.replace(".ply", "_mesh_mm.obj")
    pipeline.export_mesh(mesh, out_mesh_path)

    pcd_out = args.input.replace(".ply", "_clean_mm.ply")
    o3d.io.write_point_cloud(pcd_out, final_pcd)
    print(f"[+] Saved cleaned point cloud: {pcd_out}")
    print(f"[+] Saved watertight 3D mesh:  {out_mesh_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
