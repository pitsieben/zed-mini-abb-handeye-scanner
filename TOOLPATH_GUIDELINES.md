# ABB Robot 3D Scanning Toolpath & Trajectory Guidelines

This guide defines the mathematical and geometric principles for designing automated robot scanning toolpaths (in Grasshopper, KUKA|prc, or ABB RobotStudio) when using the **ZED Mini** stereo camera and the **Open3D Universal Registration Pipeline**.

---

## 1. Golden Rules of 3D Reconstruction

| Parameter | Recommended Spec | Technical Rationale |
| :--- | :--- | :--- |
| **Spatial Overlap** | **40% – 60%** | Multi-way ICP and PoseGraph consensus require overlapping surfaces to compute rigid transforms. |
| **Standoff Distance** | **400 mm – 650 mm** | Stereoscopic disparity resolution is sharpest at 0.4–0.65 m. ZED Mini hardware minimum is 200 mm. |
| **Step Spacing ($\Delta d$)** | **180 mm – 250 mm** | At 500 mm standoff, a 200 mm step yields ~55% viewport overlap. |
| **Surface Incidence Angle** | **$\le 35^\circ$ from normal** | Grazing angles (> 45°) cause severe photometric darkening and stereoscopic depth occlusion. |
| **Inter-Pose Angular Step** | **$\le 30^\circ$** | Prevents feature correspondence loss and lighting flares between adjacent camera poses. |
| **Settling / Dwell Time** | **0.5 seconds** | Allows robot arm inertia and tool mount oscillations to settle before camera exposure. |
| **RAPID Socket Timeout** | **$\ge 15$ seconds** | Prevents RAPID from aborting TCP communication on long axis swings. |

---

## 2. Geometry Types & Ideal Toolpath Strategies

### A. Flat & Freeform Surfaces (Timber slabs, relief carvings, stone)
- **Pattern:** Raster / Serpentine grid.
- **Orientation:** Orthogonal to surface ($Z_{\text{optical}}$ aligned with surface normal).
- **Line Spacing:** 180 mm laterally, 200 mm along path.

### B. Rotational & Cylindrical Objects (Logs, columns, vases)
- **Pattern:** Vertical columns around circumference (e.g., 4 or 6 passes at $90^\circ$ or $60^\circ$ intervals).
- **Trajectory:** Move down in 200 mm steps, rotate around workpiece, move down next column.

### C. Organic / Human Subjects
- **Movement Mitigation:** Inform the subject to hold a steady gaze at a fixed marker.
- **Registration Advantage:** The updated pipeline automatically leverages **Colored ICP** (`registration_colored_icp`), using the RGB texture of clothing patterns, facial features, and skin gradients to lock scans into place even when micro-shifts occur.

---

## 3. Grasshopper / Python Quick Reference Snippet

Paste this block into a Grasshopper Python component or Panel:

```python
# Optimal ZED Mini scanning specifications
STANDOFF_MIN_MM = 400.0
STANDOFF_MAX_MM = 650.0
STEP_LINEAR_MM = 200.0
STEP_LATERAL_MM = 180.0
MAX_INCIDENCE_DEG = 35.0
MAX_ANGULAR_STEP_DEG = 30.0
SETTLING_TIME_SEC = 0.5
```
