module mod_parameters
  implicit none

  ! Precision
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! ============================================================
  ! Material properties: Inconel 625 (SI units)
  ! ============================================================
  real(dp), parameter :: rho       = 8440.0_dp       ! density [kg/m^3]
  real(dp), parameter :: Cp        = 588.0_dp        ! specific heat [J/(kg*K)]
  real(dp), parameter :: k_cond    = 15.0_dp         ! thermal conductivity [W/(m*K)]
  real(dp), parameter :: Tl        = 1623.0_dp       ! liquidus temperature [K]
  real(dp), parameter :: T0        = 300.0_dp        ! ambient / initial temperature [K]

  ! Boundary condition parameters
  real(dp), parameter :: h_conv    = 100.0_dp        ! convection coefficient [W/(m^2*K)]
  real(dp), parameter :: eta       = 0.25_dp         ! laser absorptivity
  real(dp), parameter :: SB_const  = 5.67e-8_dp      ! Stefan-Boltzmann constant
  real(dp), parameter :: emissivity = 0.3_dp         ! emissivity

  ! Mechanical properties
  real(dp), parameter :: E_solid   = 70.0e9_dp       ! Young's modulus, solid [Pa]
  real(dp), parameter :: E_soft    = 0.7e9_dp        ! Young's modulus, powder/liquid [Pa]
  real(dp), parameter :: nu        = 0.3_dp          ! Poisson's ratio
  real(dp), parameter :: sig_yield = 250.0e6_dp      ! yield stress [Pa]
  real(dp), parameter :: alpha_V   = 1.0e-5_dp       ! volumetric thermal expansion [1/K]

  ! ============================================================
  ! Laser parameters
  ! ============================================================
  real(dp), parameter :: laser_vel = 0.5_dp          ! scan velocity [m/s]
  real(dp), parameter :: rb        = 0.05e-3_dp      ! beam radius [m]
  real(dp), parameter :: P_laser   = 100.0_dp        ! laser power [W] (2x original)

  ! ============================================================
  ! Grid parameters
  ! ============================================================
  integer, parameter :: Nx = 50
  integer, parameter :: Ny = 20
  integer, parameter :: Nz = 5

  real(dp), parameter :: Lx = 0.5e-3_dp             ! domain length x [m]
  real(dp), parameter :: Ly = 0.2e-3_dp             ! domain length y [m]
  real(dp), parameter :: Lz = 0.05e-3_dp            ! domain length z [m]

  real(dp), parameter :: dx = Lx / dble(Nx)
  real(dp), parameter :: dy = Ly / dble(Ny)
  real(dp), parameter :: dz = Lz / dble(Nz)

  ! Number of nodes (node-centered grid)
  integer, parameter :: Nnx = Nx + 1
  integer, parameter :: Nny = Ny + 1
  integer, parameter :: Nnz = Nz + 1

  ! ============================================================
  ! Time stepping
  ! ============================================================
  real(dp), parameter :: dt = 2.0e-6_dp              ! time step [s]
  integer, parameter  :: num_steps = 500             ! total number of steps
  integer, parameter  :: mech_interval = 10          ! mechanical solve every N steps

  ! ============================================================
  ! Phase identifiers
  ! ============================================================
  integer, parameter :: PHASE_POWDER = 0
  integer, parameter :: PHASE_LIQUID = 1
  integer, parameter :: PHASE_SOLID  = 2

  ! ============================================================
  ! Solver tolerances
  ! ============================================================
  real(dp), parameter :: cg_tol_thermal = 1.0e-8_dp
  integer, parameter  :: cg_maxiter_thermal = 5000
  real(dp), parameter :: cg_tol_mech = 1.0e-4_dp   ! relaxed from 1e-6 (Step B)
  integer, parameter  :: cg_maxiter_mech = 20000
  integer, parameter  :: newton_maxiter = 10
  real(dp), parameter :: newton_tol = 1.0e-4_dp

  ! ============================================================
  ! Mathematical constants
  ! ============================================================
  real(dp), parameter :: PI = 3.141592653589793_dp

end module mod_parameters
