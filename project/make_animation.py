"""
Generate a deformation animation from VTK files.
- Warps the mesh by displacement * 10 (magnification factor)
- Colors cells by phase: 0=POWDER (blue), 1=LIQUID (white), 2=SOLID (red)
- Outputs a GIF animation
"""

import os
import glob
import numpy as np
import pyvista as pv
import imageio.v3 as iio

# ---- Settings ----
VTK_DIR = "/mnt/d/Fortran/mechanical/project/jax-fem/applications/thermal_mechanical/data/vtk"
OUTPUT_GIF = "/mnt/d/Fortran/mechanical/project/phase_deformation.gif"
WARP_FACTOR = 10.0
FPS = 8

# ---- Offscreen rendering ----
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass  # may already be running or not needed

# ---- Collect VTK files in order ----
vtk_files = sorted(glob.glob(os.path.join(VTK_DIR, "u_*.vtu")))
print(f"Found {len(vtk_files)} VTK files")

# ---- Phase colormap: 0=POWDER(blue), 1=LIQUID(white), 2=SOLID(red) ----
phase_colors = np.array([
    [0.2, 0.4, 1.0],   # POWDER - blue
    [1.0, 1.0, 1.0],   # LIQUID - white
    [1.0, 0.2, 0.2],   # SOLID  - red
])
phase_labels = ["POWDER", "LIQUID", "SOLID"]

# ---- Determine a fixed camera from the first mesh ----
first_mesh = pv.read(vtk_files[0])
bounds = first_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
cx = (bounds[0] + bounds[1]) / 2
cy = (bounds[2] + bounds[3]) / 2
cz = (bounds[4] + bounds[5]) / 2
dx = bounds[1] - bounds[0]
dy = bounds[3] - bounds[2]
dz = bounds[5] - bounds[4]
max_dim = max(dx, dy, dz)

# Isometric-ish camera
camera_position = [
    (cx + 1.8 * max_dim, cy + 1.2 * max_dim, cz + 1.5 * max_dim),  # position
    (cx, cy, cz),  # focal point
    (0, 0, 1),  # view up
]

# ---- Render each frame ----
frames = []
for i, vtk_file in enumerate(vtk_files):
    step = os.path.basename(vtk_file).replace("u_", "").replace(".vtu", "")
    mesh = pv.read(vtk_file)

    # Get displacement and warp
    disp = mesh.point_data["u"]
    mesh.point_data["displacement"] = disp
    warped = mesh.warp_by_vector("displacement", factor=WARP_FACTOR)

    # Map cell phase to RGB colors
    phase = warped.cell_data["phase"].astype(int)
    phase_clamped = np.clip(phase, 0, 2)
    rgb = phase_colors[phase_clamped]
    warped.cell_data["phase_rgb"] = (rgb * 255).astype(np.uint8)

    # Keep scalar phase for colorbar
    warped.cell_data["phase_val"] = phase.astype(float)

    # ---- Plot ----
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
    plotter.set_background("white")

    # Add mesh colored by phase
    plotter.add_mesh(
        warped,
        scalars="phase_val",
        clim=[0, 2],
        cmap=["#3366FF", "#FFFFFF", "#FF3333"],
        show_edges=True,
        edge_color="gray",
        line_width=0.3,
        opacity=1.0,
        scalar_bar_args={
            "title": "Phase",
            "n_labels": 3,
            "label_font_size": 14,
            "title_font_size": 16,
            "position_x": 0.82,
            "position_y": 0.25,
            "width": 0.12,
            "height": 0.5,
            "fmt": "%.0f",
            "color": "black",
        },
    )

    # Add text annotation for step number
    plotter.add_text(
        f"Step {step}",
        position="upper_left",
        font_size=14,
        color="black",
    )

    # Add phase legend
    plotter.add_legend(
        labels=[
            ("POWDER (0)", "#3366FF"),
            ("LIQUID (1)", "#CCCCCC"),  # use light gray for visibility on white bg
            ("SOLID (2)", "#FF3333"),
        ],
        bcolor=(1.0, 1.0, 1.0, 0.7),
        face="rectangle",
        size=(0.18, 0.12),
        loc="lower right",
    )

    plotter.camera_position = camera_position
    plotter.enable_anti_aliasing("ssaa")

    # Capture frame
    img = plotter.screenshot(return_img=True)
    frames.append(img)
    plotter.close()

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Rendered frame {i+1}/{len(vtk_files)} (step {step})")

# ---- Write GIF ----
print(f"Writing GIF with {len(frames)} frames at {FPS} fps ...")
duration_ms = int(1000 / FPS)
iio.imwrite(
    OUTPUT_GIF,
    frames,
    duration=duration_ms,
    loop=0,
)
file_size_mb = os.path.getsize(OUTPUT_GIF) / (1024 * 1024)
print(f"Done! Output: {OUTPUT_GIF} ({file_size_mb:.1f} MB)")
