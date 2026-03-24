"""
Generate FDM Phase+Deformation Animation GIF from VTS files.
Blue=POWDER(0), White=LIQUID(1), Red=SOLID(2), x10 deformation magnification.
"""
import pyvista as pv
import numpy as np
from PIL import Image
import io
import os

pv.OFF_SCREEN = True

# Paths
vtk_dir = "/mnt/d/Fortran/mechanical/project/fortran_fdm/src/vtk_output"
output_gif = "/mnt/d/Fortran/mechanical/project/fdm_phase_deformation.gif"

# Gather all VTS files in order
steps = list(range(0, 501, 10))
files = [os.path.join(vtk_dir, f"fdm_{s:06d}.vts") for s in steps]
files = [f for f in files if os.path.exists(f)]
print(f"Found {len(files)} VTS files")

# Deformation magnification
mag = 10.0

# Custom colormap for phase: 0=Blue(powder), 1=White(liquid), 2=Red(solid)
from matplotlib.colors import LinearSegmentedColormap
phase_cmap = LinearSegmentedColormap.from_list(
    "phase", [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=256
)

# Read first file to get camera bounds
mesh0 = pv.read(files[0])
disp0 = mesh0.point_data["Displacement"]
warped0 = mesh0.warp_by_vector("Displacement", factor=mag)

frames = []
for i, fpath in enumerate(files):
    print(f"  Frame {i+1}/{len(files)}: {os.path.basename(fpath)}")
    mesh = pv.read(fpath)

    # Warp mesh by displacement
    warped = mesh.warp_by_vector("Displacement", factor=mag)

    # Phase data
    phase = warped.point_data["Phase"].astype(float)

    # Create plotter
    pl = pv.Plotter(off_screen=True, window_size=[1200, 500])
    pl.add_mesh(
        warped,
        scalars=phase,
        clim=[0, 2],
        cmap=phase_cmap,
        show_edges=False,
        scalar_bar_args={
            "title": "Phase",
            "n_labels": 3,
            "label_font_size": 14,
            "title_font_size": 16,
        },
    )

    # Set camera for consistent view
    pl.camera_position = "xy"
    pl.camera.zoom(1.3)

    step_num = steps[i]
    pl.add_text(
        f"FDM Step {step_num} (x{int(mag)} deformation)",
        position="upper_left",
        font_size=12,
    )

    # Render to image
    pl.show(auto_close=False)
    img = pl.screenshot(return_img=True)
    pl.close()

    frames.append(Image.fromarray(img))

# Save as GIF
print(f"Saving GIF with {len(frames)} frames...")
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=100,  # ms per frame
    loop=0,
)
print(f"Saved: {output_gif}")
print(f"File size: {os.path.getsize(output_gif) / 1024 / 1024:.1f} MB")
