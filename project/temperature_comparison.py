"""
Temperature comparison between FEM and FDM along top surface center line
(y=Ly/2, z=Lz) at selected timesteps.
"""
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
fdm_dir = "/mnt/d/Fortran/mechanical/project/fortran_fdm/src/vtk_output"
fem_dir = "/mnt/d/Fortran/mechanical/project/jax-fem/applications/thermal_mechanical/data/vtk"
output_png = "/mnt/d/Fortran/mechanical/project/temperature_comparison.png"

# Domain dimensions (from bounds inspection)
Lx = 0.0005  # 500 um
Ly = 0.0002  # 200 um
Lz = 5e-05   # 50 um

# Target line: y = Ly/2, z = Lz (top surface center)
y_target = Ly / 2.0
z_target = Lz

# Tolerance for selecting points on the line
tol_y = Ly / 20 * 0.6  # ~half a cell in y
tol_z = Lz / 5 * 0.6   # ~half a cell in z

# Timesteps to compare
compare_steps = [100, 250, 500]

def extract_line_temperature_fdm(step):
    """Extract temperature along x at y=Ly/2, z=Lz from FDM VTS file."""
    fpath = os.path.join(fdm_dir, f"fdm_{step:06d}.vts")
    mesh = pv.read(fpath)
    pts = mesh.points
    temp = mesh.point_data["Temperature"]

    # Select points near the target line
    mask = (np.abs(pts[:, 1] - y_target) < tol_y) & (np.abs(pts[:, 2] - z_target) < tol_z)
    x_vals = pts[mask, 0]
    t_vals = temp[mask]

    # Sort by x
    order = np.argsort(x_vals)
    return x_vals[order], t_vals[order]


def extract_line_temperature_fem(step):
    """Extract temperature along x at y=Ly/2, z=Lz from FEM VTU file."""
    fpath = os.path.join(fem_dir, f"u_{step:05d}.vtu")
    mesh = pv.read(fpath)
    pts = mesh.points
    temp = mesh.point_data["sol"]

    # Select points near the target line
    mask = (np.abs(pts[:, 1] - y_target) < tol_y) & (np.abs(pts[:, 2] - z_target) < tol_z)
    x_vals = pts[mask, 0]
    t_vals = temp[mask]

    # Sort by x
    order = np.argsort(x_vals)
    return x_vals[order], t_vals[order]


# Create subplot figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

for idx, step in enumerate(compare_steps):
    ax = axes[idx]

    # FDM data
    x_fdm, t_fdm = extract_line_temperature_fdm(step)
    # FEM data
    x_fem, t_fem = extract_line_temperature_fem(step)

    # Convert x to micrometers for readability
    ax.plot(x_fem * 1e6, t_fem, "b-", linewidth=2, label="FEM (JAX-FEM)")
    ax.plot(x_fdm * 1e6, t_fdm, "r--", linewidth=2, label="FDM (Fortran)")

    ax.set_xlabel(r"x ($\mu$m)", fontsize=13)
    ax.set_ylabel("Temperature (K)", fontsize=13)
    ax.set_title(f"Step {step}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    print(f"Step {step}: FDM {len(x_fdm)} pts, T=[{t_fdm.min():.1f}, {t_fdm.max():.1f}] K | "
          f"FEM {len(x_fem)} pts, T=[{t_fem.min():.1f}, {t_fem.max():.1f}] K")

fig.suptitle(
    r"Temperature along top surface center line ($y=L_y/2$, $z=L_z$)",
    fontsize=15,
    y=1.02,
)
plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_png}")
print(f"File size: {os.path.getsize(output_png) / 1024:.1f} KB")
