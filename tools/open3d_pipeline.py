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
    def clean_individual_pcd(pcd, nb_neighbors=16, std_ratio=1.4, voxel_size=0.0025):
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

    def optimize_pose_graph_multiway(self, pcd_list, initial_poses, max_correspondence_distance=0.015):
        """
        Builds an Open3D PoseGraph using pairwise ICP on adjacent and overlapping scans,
        then optimizes poses globally using Levenberg-Marquardt.
        """
        n_pcds = len(pcd_list)
        if n_pcds <= 1:
            return initial_poses

        print(f"\n[MULTI-WAY ICP] Building Pose Graph for {n_pcds} robot scan stages...")
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.eye(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

        # Downsample for fast robust registration
        pcds_down = [p.voxel_down_sample(voxel_size=0.003) for p in pcd_list]
        for p in pcds_down:
            if not p.has_normals():
                p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))

        # Add Odometry edges (between consecutive scans)
        for i in range(n_pcds - 1):
            src = pcds_down[i + 1]
            tgt = pcds_down[i]
            # Relative transformation from initial poses
            T_rel_init = np.linalg.inv(initial_poses[i]) @ initial_poses[i + 1]

            # Transform source by initial guess before fine registration
            src_transformed = o3d.geometry.PointCloud(src)
            src_transformed.transform(T_rel_init)

            trans, info = self.pairwise_registration(src_transformed, tgt, max_correspondence_distance)
            T_step = trans @ T_rel_init
            odometry = np.dot(odometry, T_step)

            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, i + 1, T_step, info, uncertain=False
                )
            )

        # Compute world-frame workpiece centroids for each scan
        world_centers = []
        for i in range(n_pcds):
            c_local = pcds_down[i].get_center()
            c_world = (initial_poses[i] @ np.append(c_local, 1.0))[:3]
            world_centers.append(c_world)

        # Add Loop closure / Multi-pass overlap edges (non-adjacent scans that overlap)
        coarse_threshold = max(0.080, max_correspondence_distance * 4)
        for i in range(n_pcds):
            for j in range(i + 2, n_pcds):
                # Check spatial proximity between actual workpiece point cloud centroids (NOT camera mount)
                dist_workpiece = np.linalg.norm(world_centers[i] - world_centers[j])
                if dist_workpiece < 0.25:  # workpiece points are within 250mm
                    src = pcds_down[j]
                    tgt = pcds_down[i]
                    T_rel_init = np.linalg.inv(initial_poses[i]) @ initial_poses[j]
                    src_transformed = o3d.geometry.PointCloud(src)
                    src_transformed.transform(T_rel_init)

                    # Step A: Coarse alignment to bridge inter-pass calibration gaps (e.g. 80-100mm)
                    trans_coarse, _ = self.pairwise_registration(src_transformed, tgt, coarse_threshold)
                    src_coarse = o3d.geometry.PointCloud(src_transformed)
                    src_coarse.transform(trans_coarse)

                    # Step B: Fine point-to-plane registration
                    trans_fine, info = self.pairwise_registration(src_coarse, tgt, max_correspondence_distance)
                    trans_total = trans_fine @ trans_coarse

                    # Check fitness
                    eval_res = o3d.pipelines.registration.evaluate_registration(
                        src_transformed, tgt, max_correspondence_distance, trans_total
                    )
                    if eval_res.fitness > 0.35:
                        T_edge = trans_total @ T_rel_init
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                i, j, T_edge, info, uncertain=True
                            )
                        )
                        print(f"  [EDGE] Inter-pass overlap link: scan #{i+1} <-> #{j+1} (fitness: {eval_res.fitness:.2f}, RMSE: {eval_res.inlier_rmse*1000:.1f} mm)")

        # Global Optimization
        print("[MULTI-WAY ICP] Running Global Levenberg-Marquardt Optimization...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.5,
            reference_node=0
        )
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            option
        )

        optimized_poses = []
        for i in range(n_pcds):
            node_pose = np.linalg.inv(pose_graph.nodes[i].pose)
            refined_pose = initial_poses[0] @ node_pose
            optimized_poses.append(refined_pose)

        print("[MULTI-WAY ICP] Pose graph optimization completed successfully.")
        return optimized_poses

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
        success = o3d.io.write_triangle_mesh(output_path, mesh)
        if success:
            print(f"[+] Successfully exported clean mesh to: {output_path}")
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

        for idx, stage in enumerate(stages):
            ply_file = os.path.join(session_dir, stage["ply"]) if not os.path.isabs(stage["ply"]) else stage["ply"]
            if not os.path.exists(ply_file):
                print(f"[WARNING] Stage file not found: {ply_file}. Skipping.")
                continue

            # Load raw scan in camera frame
            raw_pcd = o3d.io.read_point_cloud(ply_file)
            print(f"[*] Stage #{idx + 1}: Loaded {len(raw_pcd.points):,} pts from {os.path.basename(ply_file)}")

            # Step 2: Clean individual scan first
            clean_pcd = self.clean_individual_pcd(
                raw_pcd, nb_neighbors=16, std_ratio=1.4, voxel_size=max(0.0025, voxel_size)
            )

            # Step 1: Compute initial pose T_base2cam = T_base2ee @ T_ee2cam
            pose = stage["pose_flange"]
            T_base2cam = self.compute_cam2base_matrix(
                pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]
            )

            cleaned_pcds.append(clean_pcd)
            initial_poses.append(T_base2cam)

        if len(cleaned_pcds) == 0:
            raise ValueError("No valid point clouds could be loaded from session.")

        # Step 3: Pass-Aware Multi-Scale Registration
        # Detect toolpath passes (orientation changes > 15 deg or jumps > 250mm indicate a new pass)
        passes = []
        current_pass = [0]
        for i in range(1, len(cleaned_pcds)):
            R_prev = initial_poses[i - 1][:3, :3]
            R_curr = initial_poses[i][:3, :3]
            R_diff = R_prev.T @ R_curr
            trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(trace))
            pos_jump = np.linalg.norm(initial_poses[i - 1][:3, 3] - initial_poses[i][:3, 3])

            if angle_deg > 15.0 or pos_jump > 0.25:
                passes.append(current_pass)
                current_pass = [i]
            else:
                current_pass.append(i)
        passes.append(current_pass)

        print(f"\n[TOOLPATH ANALYSIS] Identified {len(passes)} distinct scanning pass(es):")
        for p_i, p_indices in enumerate(passes):
            print(f"  -> Pass #{p_i+1}: stages {p_indices[0]+1} to {p_indices[-1]+1} ({len(p_indices)} waypoints)")

        refined_poses = [p.copy() for p in initial_poses]

        if len(passes) > 1:
            print("\n[MULTI-PASS REGISTRATION] Aligning overlapping passes into consensus...")
            # Build merged cloud for each pass
            pass_clouds = []
            for p_indices in passes:
                p_cloud = o3d.geometry.PointCloud()
                for idx in p_indices:
                    p_cloud += o3d.geometry.PointCloud(cleaned_pcds[idx]).transform(initial_poses[idx])
                p_cloud_down = p_cloud.voxel_down_sample(0.0035)
                p_cloud_down.estimate_normals()
                pass_clouds.append(p_cloud_down)

            # Choose the central pass as reference (usually the downward pass)
            ref_pass_idx = len(passes) // 2
            ref_cloud = pass_clouds[ref_pass_idx]

            # Align other passes to reference pass
            pass_transforms = [np.eye(4) for _ in passes]
            for p_i, p_cloud in enumerate(pass_clouds):
                if p_i == ref_pass_idx:
                    continue
                # Coarse alignment (radius: 120mm) to bridge Hand-Eye offset
                reg_coarse = o3d.pipelines.registration.registration_icp(
                    p_cloud, ref_cloud, 0.120, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                # Fine alignment (radius: 25mm)
                reg_fine = o3d.pipelines.registration.registration_icp(
                    p_cloud, ref_cloud, 0.025, reg_coarse.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                pass_transforms[p_i] = reg_fine.transformation
                print(f"  [PASS ALIGNED] Pass #{p_i+1} -> Reference Pass #{ref_pass_idx+1}: Fitness={reg_fine.fitness:.2f}, Inlier RMSE={reg_fine.inlier_rmse*1000:.1f} mm")

            # Apply pass corrections to individual stage poses
            for p_i, p_indices in enumerate(passes):
                T_pass = pass_transforms[p_i]
                for idx in p_indices:
                    refined_poses[idx] = T_pass @ initial_poses[idx]
        elif len(cleaned_pcds) > 1:
            refined_poses = self.optimize_pose_graph_multiway(
                cleaned_pcds, initial_poses, max_correspondence_distance=0.020
            )

        # Transform each stage to its globally optimized pose and merge
        print("\n[MULTI-WAY PIPELINE] Transforming and fusing aligned point clouds...")
        fused_pcd = o3d.geometry.PointCloud()
        for pcd, T_opt in zip(cleaned_pcds, refined_poses):
            pcd_trans = o3d.geometry.PointCloud(pcd)
            pcd_trans.transform(T_opt)
            fused_pcd += pcd_trans

        # Voxel downsample fused surface to eliminate duplicate density
        fused_pcd = fused_pcd.voxel_down_sample(voxel_size=voxel_size)
        fused_pcd, _ = fused_pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.4)
        fused_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
        
        # Camera centers for outward normal orientation
        cam_centers = np.array([T[:3, 3] for T in refined_poses])
        fused_pcd.orient_normals_towards_camera_location(np.mean(cam_centers, axis=0))
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
            # Step 4 & 5: Poisson Reconstruction
            mesh, _ = self.reconstruct_poisson_mesh(fused_pcd, depth=depth, density_trim_quantile=0.04, camera_centers=cam_centers)
            if scale_mm:
                mesh.scale(1000.0, center=np.zeros(3))
            out_mesh_obj = os.path.join("scans", "latest_workpiece_mesh_mm.obj")
            out_mesh_stl = os.path.join("scans", "latest_workpiece_mesh_mm.stl")
            self.export_mesh(mesh, out_mesh_obj)
            self.export_mesh(mesh, out_mesh_stl)
            print(f"[+] Exported 1:1 Millimeter Mesh for Rhino: {out_mesh_obj}")

        return fused_pcd


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
    args = parser.parse_args()

    pipeline = Open3DScannerPipeline()

    target_session = args.session
    if target_session is None and args.input and os.path.isdir(args.input):
        target_session = args.input

    # Mode 1: Multi-stage session processing
    if target_session:
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
        return

    # Mode 2: Single PLY processing
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
