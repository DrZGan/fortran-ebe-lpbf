# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project replicates the [jax-fem thermal_mechanical](https://github.com/DrZGan/jax-fem/tree/main/applications/thermal_mechanical) LPBF (Laser Powder Bed Fusion) additive manufacturing simulation using a **Fortran implicit FDM solver with OpenMP**, and compares memory usage and computational speed against the JAX-FEM GPU implementation.

The detailed task plan lives in `project/task.md` — always consult and update it when changing approach.

## Development Principles

- **Always write `task.md` in English.** All planning, status updates, and change logs must be in English.
- **First principles thinking.** Derive from the governing equations and physics, not from copying patterns. Understand *why* before implementing *how*.
- **Code consistency and simplicity above all.** Never sacrifice consistency or simplicity to add a feature. If a new feature would make the codebase less coherent, redesign the approach or defer the feature.
- **Maintainability over cleverness.** Prefer straightforward, readable code. A simple 10-line loop is better than a clever 3-line abstraction that obscures intent.

## Environment Setup

### JAX-FEM (reference solution)
```bash
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(/tmp/bin/micromamba shell hook -s bash)"
micromamba activate jaxfem
cd /mnt/d/Fortran/mechanical/project/jax-fem
python -m applications.thermal_mechanical.example
```
- Conda env: `$HOME/micromamba/envs/jaxfem` (Python 3.11, JAX CUDA 12, PETSc, gmsh)
- Source: `project/jax-fem/` (cloned from deepmodeling/jax-fem, editable install)
- GPU: NVIDIA via WSL2, CUDA 12.8, `CUDA_VISIBLE_DEVICES=0`
- VTK output: `project/jax-fem/applications/thermal_mechanical/data/vtk/`

### Fortran FDM (to be built)
```bash
cd project/fortran_fdm
make            # build
./main          # run simulation
make clean      # cleanup
```
- Compiler: gfortran with `-O3 -march=native -fopenmp`
- No external dependencies (matrix-free solver, built-in VTK output)

## Physics: Coupled Thermal-Mechanical LPBF

One-way coupling (T → u): transient heat equation drives quasi-static mechanical equilibrium.

**Heat equation** (implicit backward Euler):
`ρCp(T^n - T^{n-1})/dt = ∇·(k∇T^n)` with laser Gaussian source as top-surface Neumann BC, convection+radiation on all exposed surfaces, Dirichlet T=300K on bottom.

**Mechanics** (quasi-static): `-∇·σ = 0` with J2 plasticity return-mapping. Thermal strain `ε_th = α_V·ΔT·I` only in SOLID phase. Fixed bottom (u=0), traction-free elsewhere.

**Phase model**: POWDER→LIQUID (T>1623K), LIQUID→SOLID (T<1623K), irreversible.

Material: Inconel 625. Domain: 0.5×0.2×0.05 mm. Grid: 50×20×5. dt=2e-6s, ~500 steps, mechanics every 10th step.

## Architecture: Fortran EBE FEM (Element-by-Element)

Code in `project/fortran_fdm/src/`:

| Module | Role |
|--------|------|
| `mod_parameters.f90` | Material constants, grid setup, laser parameters |
| `mod_thermal.f90` | EBE FEM thermal solver: CG with 8×8 element matrices, 8-color parallel scatter |
| `mod_mechanical.f90` | EBE FEM mechanical solver: CG with 24×24 element stiffness, phase-dependent |
| `mod_phase.f90` | Phase transition logic (node-centered) |
| `mod_io.f90` | VTK structured grid output |
| `main.f90` | Time-stepping loop, coupling logic |

### Key Design Decisions
- **EBE FEM** (not FDM): Precomputed HEX8 element matrices (mass 8×8, stiffness 8×8 for thermal; 24×24 for mechanical). CG matvec via element loop with scatter-add. Gives **exact FEM results** with no global matrix.
- **8-color element coloring**: Race-free OpenMP parallel scatter-add using `(ie%2, je%2, ke%2)` coloring.
- **Structured grid**: No connectivity arrays. Gather/scatter uses `(ie+di, je+dj, ke+dk)` indexing.
- **~2 MB memory**: Only field arrays + precomputed element matrices. No sparse matrix storage.

### Why EBE, Not FDM
FDM (7-point Laplacian + lumped mass) gives **3-5x different peak temperatures** vs FEM due to:
1. FEM HEX8 stiffness is a 27-point stencil (zero face coupling, nonzero edge/corner), not a 7-point Laplacian
2. FEM consistent mass at boundary nodes has half the inertia of interior nodes
3. These differences compound under concentrated sources (laser beam)

EBE FEM uses the exact same element matrices as JAX-FEM, giving identical results.

### JAX-FEM Bug (Fixed)
The original JAX-FEM example used `atol=1e-5` (= dz) in boundary location functions, selecting ~7000 faces instead of ~1000. Fixed to `atol=1e-6`.

## Key Reference Files

- `project/task.md` — Master task plan (always follow and update this)
- `project/jax-fem/applications/thermal_mechanical/example.py` — Reference FEM implementation
- `project/jax-fem/applications/thermal_mechanical/README.md` — Mathematical formulation
- `project/jax-fem/jax_fem/solver.py` — JAX-FEM Newton + linear solver
- `project/jax-fem/jax_fem/problem.py` — FEM problem base class
