module mod_mechanical
  use mod_parameters
  implicit none

  private
  public :: solve_mechanical, init_mechanical, cleanup_mechanical, get_stress_yield

  ! Precomputed 24x24 element stiffness matrices (symmetric)
  real(dp) :: Ke_solid(24,24)
  real(dp) :: Ke_soft(24,24)

  ! Per-GP stiffness contributions: Ke = sum_{gp=1}^{8} Ke_gp
  real(dp) :: Ke_gp_solid(24,24,8)
  real(dp) :: Ke_gp_soft(24,24,8)

  ! Thermal coupling matrices (kept for reference)
  real(dp) :: Mth_solid(24,8)
  real(dp) :: Mth_soft(24,8)
  real(dp) :: Fth_solid(24)
  real(dp) :: Fth_soft(24)

  ! Precomputed B matrices at each GP: B_all(6,24,8)
  real(dp) :: B_all(6,24,8)

  ! Precomputed shape functions at each GP: N_gp(8,8)
  ! N_gp(a,g) = shape function of node a evaluated at GP g
  real(dp) :: N_gp(8,8)

  ! GP-level state arrays: (6 components, 8 GPs, Nx, Ny, Nz)
  real(dp), allocatable :: sig_gp(:,:,:,:,:)
  real(dp), allocatable :: eps_gp(:,:,:,:,:)

  ! Temperature state for incremental thermal strain
  real(dp), allocatable :: T_old_for_u(:,:,:)

  ! Yield function output (node-centered)
  real(dp), allocatable :: f_plus(:,:,:)

  ! Element phase cache (Nx, Ny, Nz)
  integer, allocatable :: elem_phase(:,:,:)

  ! Jacobian determinant (uniform grid)
  real(dp) :: det_J_val

contains

  ! ============================================================
  ! Build isotropic elasticity matrix C (6x6 Voigt notation)
  ! ============================================================
  subroutine build_C_matrix(E_val, nu_val, C)
    real(dp), intent(in)  :: E_val, nu_val
    real(dp), intent(out) :: C(6,6)
    real(dp) :: lam, mu

    lam = E_val * nu_val / ((1.0_dp + nu_val) * (1.0_dp - 2.0_dp * nu_val))
    mu  = E_val / (2.0_dp * (1.0_dp + nu_val))

    C = 0.0_dp
    C(1,1) = lam + 2.0_dp*mu; C(1,2) = lam;             C(1,3) = lam
    C(2,1) = lam;             C(2,2) = lam + 2.0_dp*mu; C(2,3) = lam
    C(3,1) = lam;             C(3,2) = lam;             C(3,3) = lam + 2.0_dp*mu
    C(4,4) = mu
    C(5,5) = mu
    C(6,6) = mu
  end subroutine build_C_matrix

  ! ============================================================
  ! Compute element stiffness K_e(24,24) and thermal load vector
  ! for a uniform HEX8 element with given material
  ! ============================================================
  subroutine compute_element_matrices(E_val, nu_val, Ke, Fth, Mth, Ke_gpp)
    real(dp), intent(in)  :: E_val, nu_val
    real(dp), intent(out) :: Ke(24,24)
    real(dp), intent(out) :: Fth(24)
    real(dp), intent(out) :: Mth(24,8)
    real(dp), intent(out) :: Ke_gpp(24,24,8)

    real(dp) :: C(6,6)
    real(dp) :: gp_c(2), gw(2)
    real(dp) :: xi, eta_q, zeta, w, det_J
    real(dp) :: xi_n(8), eta_n(8), zeta_n(8)
    real(dp) :: dN_dx(8), dN_dy(8), dN_dz(8)
    integer  :: gp_idx
    real(dp) :: dN_dxi(8), dN_deta(8), dN_dzeta(8)
    real(dp) :: N(8)
    real(dp) :: B_a(6,3), B_b(6,3)
    real(dp) :: CB(6,3)
    real(dp) :: eps_th(6), C_eps_th(6)
    integer  :: i, j, k, a, b, p, q

    call build_C_matrix(E_val, nu_val, C)

    xi_n   = (/ -1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp /)
    eta_n  = (/ -1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp,-1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp /)
    zeta_n = (/ -1.0_dp,-1.0_dp,-1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp /)

    gp_c(1) = -1.0_dp / sqrt(3.0_dp)
    gp_c(2) =  1.0_dp / sqrt(3.0_dp)
    gw(1) = 1.0_dp
    gw(2) = 1.0_dp

    det_J = (dx/2.0_dp) * (dy/2.0_dp) * (dz/2.0_dp)

    eps_th = (/ 1.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp /)
    C_eps_th = 0.0_dp
    do p = 1, 6
      do q = 1, 6
        C_eps_th(p) = C_eps_th(p) + C(p,q) * eps_th(q)
      end do
    end do
    C_eps_th = C_eps_th * alpha_V

    Ke     = 0.0_dp
    Fth    = 0.0_dp
    Mth    = 0.0_dp
    Ke_gpp = 0.0_dp

    do k = 1, 2
      do j = 1, 2
        do i = 1, 2
          gp_idx = i + 2*(j-1) + 4*(k-1)
          xi     = gp_c(i)
          eta_q  = gp_c(j)
          zeta   = gp_c(k)
          w = gw(i) * gw(j) * gw(k)

          do a = 1, 8
            N(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta_q) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
          end do
          do a = 1, 8
            dN_dxi(a)   = xi_n(a)   * (1.0_dp + eta_n(a)*eta_q) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_deta(a)  = (1.0_dp + xi_n(a)*xi) * eta_n(a)  * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_dzeta(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta_q) * zeta_n(a) / 8.0_dp
          end do

          dN_dx = dN_dxi   * (2.0_dp / dx)
          dN_dy = dN_deta  * (2.0_dp / dy)
          dN_dz = dN_dzeta * (2.0_dp / dz)

          do b = 1, 8
            B_b = 0.0_dp
            B_b(1,1) = dN_dx(b)
            B_b(2,2) = dN_dy(b)
            B_b(3,3) = dN_dz(b)
            B_b(4,1) = dN_dy(b); B_b(4,2) = dN_dx(b)
            B_b(5,1) = dN_dz(b); B_b(5,3) = dN_dx(b)
            B_b(6,2) = dN_dz(b); B_b(6,3) = dN_dy(b)

            CB = 0.0_dp
            do q = 1, 3
              do p = 1, 6
                CB(p,q) = C(p,1)*B_b(1,q) + C(p,2)*B_b(2,q) + C(p,3)*B_b(3,q) &
                        + C(p,4)*B_b(4,q) + C(p,5)*B_b(5,q) + C(p,6)*B_b(6,q)
              end do
            end do

            do a = 1, 8
              B_a = 0.0_dp
              B_a(1,1) = dN_dx(a)
              B_a(2,2) = dN_dy(a)
              B_a(3,3) = dN_dz(a)
              B_a(4,1) = dN_dy(a); B_a(4,2) = dN_dx(a)
              B_a(5,1) = dN_dz(a); B_a(5,3) = dN_dx(a)
              B_a(6,2) = dN_dz(a); B_a(6,3) = dN_dy(a)

              do q = 1, 3
                do p = 1, 3
                  Ke(3*(a-1)+p, 3*(b-1)+q) = Ke(3*(a-1)+p, 3*(b-1)+q) &
                    + (B_a(1,p)*CB(1,q) + B_a(2,p)*CB(2,q) + B_a(3,p)*CB(3,q) &
                     + B_a(4,p)*CB(4,q) + B_a(5,p)*CB(5,q) + B_a(6,p)*CB(6,q)) &
                    * det_J * w
                  Ke_gpp(3*(a-1)+p, 3*(b-1)+q, gp_idx) = &
                    Ke_gpp(3*(a-1)+p, 3*(b-1)+q, gp_idx) &
                    + (B_a(1,p)*CB(1,q) + B_a(2,p)*CB(2,q) + B_a(3,p)*CB(3,q) &
                     + B_a(4,p)*CB(4,q) + B_a(5,p)*CB(5,q) + B_a(6,p)*CB(6,q)) &
                    * det_J * w
                end do
              end do
            end do
          end do

          ! Thermal load and coupling
          do a = 1, 8
            B_a = 0.0_dp
            B_a(1,1) = dN_dx(a)
            B_a(2,2) = dN_dy(a)
            B_a(3,3) = dN_dz(a)
            B_a(4,1) = dN_dy(a); B_a(4,2) = dN_dx(a)
            B_a(5,1) = dN_dz(a); B_a(5,3) = dN_dx(a)
            B_a(6,2) = dN_dz(a); B_a(6,3) = dN_dy(a)

            do p = 1, 3
              Fth(3*(a-1)+p) = Fth(3*(a-1)+p) &
                + (B_a(1,p)*C_eps_th(1) + B_a(2,p)*C_eps_th(2) + B_a(3,p)*C_eps_th(3) &
                 + B_a(4,p)*C_eps_th(4) + B_a(5,p)*C_eps_th(5) + B_a(6,p)*C_eps_th(6)) &
                * det_J * w
              do b = 1, 8
                Mth(3*(a-1)+p, b) = Mth(3*(a-1)+p, b) &
                  + (B_a(1,p)*C_eps_th(1) + B_a(2,p)*C_eps_th(2) + B_a(3,p)*C_eps_th(3) &
                   + B_a(4,p)*C_eps_th(4) + B_a(5,p)*C_eps_th(5) + B_a(6,p)*C_eps_th(6)) &
                  * N(b) * det_J * w
              end do
            end do
          end do

        end do
      end do
    end do
  end subroutine compute_element_matrices

  ! ============================================================
  ! Precompute B matrices and shape functions at all 8 GPs
  ! ============================================================
  subroutine precompute_B_and_N()
    real(dp) :: gp_c(2)
    real(dp) :: xi, eta_q, zeta
    real(dp) :: xi_n(8), eta_n(8), zeta_n(8)
    real(dp) :: dN_dxi(8), dN_deta(8), dN_dzeta(8)
    real(dp) :: dN_dx(8), dN_dy(8), dN_dz(8)
    integer  :: i, j, k, gp_idx, a

    xi_n   = (/ -1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp /)
    eta_n  = (/ -1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp,-1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp /)
    zeta_n = (/ -1.0_dp,-1.0_dp,-1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp /)

    gp_c(1) = -1.0_dp / sqrt(3.0_dp)
    gp_c(2) =  1.0_dp / sqrt(3.0_dp)

    det_J_val = (dx/2.0_dp) * (dy/2.0_dp) * (dz/2.0_dp)

    B_all = 0.0_dp
    N_gp  = 0.0_dp

    do k = 1, 2
      do j = 1, 2
        do i = 1, 2
          gp_idx = i + 2*(j-1) + 4*(k-1)
          xi    = gp_c(i)
          eta_q = gp_c(j)
          zeta  = gp_c(k)

          ! Shape functions at this GP
          do a = 1, 8
            N_gp(a, gp_idx) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta_q) &
                             * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
          end do

          ! Shape function derivatives in reference coords
          do a = 1, 8
            dN_dxi(a)   = xi_n(a)   * (1.0_dp + eta_n(a)*eta_q) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_deta(a)  = (1.0_dp + xi_n(a)*xi) * eta_n(a)  * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_dzeta(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta_q) * zeta_n(a) / 8.0_dp
          end do

          ! Physical gradients (uniform grid)
          dN_dx = dN_dxi   * (2.0_dp / dx)
          dN_dy = dN_deta  * (2.0_dp / dy)
          dN_dz = dN_dzeta * (2.0_dp / dz)

          ! Build B_all(:, 3*(a-1)+1:3*a, gp_idx) for each node a
          do a = 1, 8
            B_all(1, 3*(a-1)+1, gp_idx) = dN_dx(a)
            B_all(2, 3*(a-1)+2, gp_idx) = dN_dy(a)
            B_all(3, 3*(a-1)+3, gp_idx) = dN_dz(a)
            B_all(4, 3*(a-1)+1, gp_idx) = dN_dy(a)
            B_all(4, 3*(a-1)+2, gp_idx) = dN_dx(a)
            B_all(5, 3*(a-1)+1, gp_idx) = dN_dz(a)
            B_all(5, 3*(a-1)+3, gp_idx) = dN_dx(a)
            B_all(6, 3*(a-1)+2, gp_idx) = dN_dz(a)
            B_all(6, 3*(a-1)+3, gp_idx) = dN_dy(a)
          end do

        end do
      end do
    end do
  end subroutine precompute_B_and_N

  ! ============================================================
  ! Initialize module: precompute element matrices, allocate state
  ! ============================================================
  subroutine init_mechanical()
    allocate(sig_gp(6, 8, Nx, Ny, Nz))
    allocate(eps_gp(6, 8, Nx, Ny, Nz))
    allocate(T_old_for_u(Nnx, Nny, Nnz))
    allocate(f_plus(Nnx, Nny, Nnz))
    allocate(elem_phase(Nx, Ny, Nz))

    sig_gp = 0.0_dp
    eps_gp = 0.0_dp
    T_old_for_u = T0
    f_plus = 0.0_dp
    elem_phase = PHASE_POWDER

    ! Precompute element stiffness matrices and thermal load vectors
    call compute_element_matrices(E_solid, nu, Ke_solid, Fth_solid, Mth_solid, Ke_gp_solid)
    call compute_element_matrices(E_soft,  nu, Ke_soft,  Fth_soft,  Mth_soft, Ke_gp_soft)

    ! Precompute B matrices and shape functions at all GPs
    call precompute_B_and_N()
  end subroutine init_mechanical

  subroutine cleanup_mechanical()
    if (allocated(sig_gp))      deallocate(sig_gp)
    if (allocated(eps_gp))      deallocate(eps_gp)
    if (allocated(T_old_for_u)) deallocate(T_old_for_u)
    if (allocated(f_plus))      deallocate(f_plus)
    if (allocated(elem_phase))  deallocate(elem_phase)
  end subroutine cleanup_mechanical

  ! ============================================================
  ! Determine element phase: SOLID if any node is SOLID, else POWDER
  ! ============================================================
  subroutine compute_elem_phases(phase)
    integer, intent(in) :: phase(Nnx, Nny, Nnz)
    integer :: ie, je, ke, di, dj, dk, ph, n_solid

    !$omp parallel do collapse(3) default(shared) private(ie,je,ke,di,dj,dk,ph,n_solid)
    do ke = 1, Nz
      do je = 1, Ny
        do ie = 1, Nx
          n_solid = 0
          do dk = 0, 1
            do dj = 0, 1
              do di = 0, 1
                ph = phase(ie+di, je+dj, ke+dk)
                if (ph == PHASE_SOLID) n_solid = n_solid + 1
              end do
            end do
          end do
          if (n_solid > 0) then
            elem_phase(ie,je,ke) = PHASE_SOLID
          else
            elem_phase(ie,je,ke) = PHASE_POWDER
          end if
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine compute_elem_phases

  ! ============================================================
  ! J2 return mapping: maps trial stress to yield surface
  ! ============================================================
  subroutine j2_return_map(s_trial, s_mapped, f_yield_out)
    real(dp), intent(in)  :: s_trial(6)
    real(dp), intent(out) :: s_mapped(6)
    real(dp), intent(out) :: f_yield_out
    real(dp) :: s_mean, s_dev(6), s_norm, f_yield

    s_mean = (s_trial(1) + s_trial(2) + s_trial(3)) / 3.0_dp
    s_dev(1) = s_trial(1) - s_mean
    s_dev(2) = s_trial(2) - s_mean
    s_dev(3) = s_trial(3) - s_mean
    s_dev(4) = s_trial(4)
    s_dev(5) = s_trial(5)
    s_dev(6) = s_trial(6)

    s_norm = sqrt(1.5_dp * (s_dev(1)**2 + s_dev(2)**2 + s_dev(3)**2 &
                 + 2.0_dp * (s_dev(4)**2 + s_dev(5)**2 + s_dev(6)**2)))

    f_yield = s_norm - sig_yield
    f_yield_out = max(f_yield, 0.0_dp)

    if (f_yield > 0.0_dp .and. s_norm > 1.0e-30_dp) then
      s_mapped(1) = s_mean + s_dev(1) * (1.0_dp - f_yield / s_norm)
      s_mapped(2) = s_mean + s_dev(2) * (1.0_dp - f_yield / s_norm)
      s_mapped(3) = s_mean + s_dev(3) * (1.0_dp - f_yield / s_norm)
      s_mapped(4) = s_dev(4) * (1.0_dp - f_yield / s_norm)
      s_mapped(5) = s_dev(5) * (1.0_dp - f_yield / s_norm)
      s_mapped(6) = s_dev(6) * (1.0_dp - f_yield / s_norm)
    else
      s_mapped = s_trial
    end if
  end subroutine j2_return_map

  ! ============================================================
  ! Compute dT at each GP by interpolating from nodes
  ! dT_gp_arr(8, Nx, Ny, Nz) = interpolated delta-T at each GP
  ! ============================================================
  subroutine compute_dT_gp(T_new, dT_gp_arr)
    real(dp), intent(in)  :: T_new(Nnx, Nny, Nnz)
    real(dp), intent(out) :: dT_gp_arr(8, Nx, Ny, Nz)

    integer  :: ie, je, ke, di, dj, dk, ln, g, a
    real(dp) :: dT_nodes(8)

    !$omp parallel do collapse(3) default(shared) private(ie,je,ke,di,dj,dk,ln,g,a,dT_nodes)
    do ke = 1, Nz
      do je = 1, Ny
        do ie = 1, Nx
          ! Gather nodal dT
          do dk = 0, 1
            do dj = 0, 1
              do di = 0, 1
                ln = 1 + di + 2*dj + 4*dk
                dT_nodes(ln) = T_new(ie+di, je+dj, ke+dk) - T_old_for_u(ie+di, je+dj, ke+dk)
              end do
            end do
          end do

          ! Interpolate to each GP
          do g = 1, 8
            dT_gp_arr(g, ie, je, ke) = 0.0_dp
            do a = 1, 8
              dT_gp_arr(g, ie, je, ke) = dT_gp_arr(g, ie, je, ke) + N_gp(a, g) * dT_nodes(a)
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine compute_dT_gp

  ! ============================================================
  ! Compute residual R from internal stress at each GP
  ! R_e = sum_g B^T * sigma * detJ * w_gp
  ! Uses 8-color element coloring for thread safety
  ! ============================================================
  subroutine compute_residual(ux, uy, uz, dT_gp_arr, phase, Rx, Ry, Rz)
    real(dp), intent(in)  :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    real(dp), intent(in)  :: dT_gp_arr(8, Nx, Ny, Nz)
    integer,  intent(in)  :: phase(Nnx,Nny,Nnz)
    real(dp), intent(out) :: Rx(Nnx,Nny,Nnz), Ry(Nnx,Nny,Nnz), Rz(Nnx,Nny,Nnz)

    real(dp) :: u_e(24), R_e(24), eps_curr(6), eps_inc(6), eps_th(6)
    real(dp) :: s_trial(6), sigma(6), BtSig(24), C_loc(6,6)
    real(dp) :: av, dT_g, f_dum
    integer  :: ie, je, ke, di, dj, dk, ln, dof, g, p, q
    integer  :: color, ic, jc, kc, ph_g

    Rx = 0.0_dp; Ry = 0.0_dp; Rz = 0.0_dp

    do color = 0, 7
      ic = mod(color, 2)
      jc = mod(color/2, 2)
      kc = mod(color/4, 2)

      !$omp parallel do collapse(3) default(shared) &
      !$omp   private(ie,je,ke,u_e,R_e,eps_curr,eps_inc,eps_th,s_trial,sigma,BtSig) &
      !$omp   private(C_loc,av,dT_g,f_dum,di,dj,dk,ln,dof,g,p,q,ph_g)
      do ke = 1 + kc, Nz, 2
        do je = 1 + jc, Ny, 2
          do ie = 1 + ic, Nx, 2

            ! Gather 24 DOFs
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dof = 3*(ln-1)
                  u_e(dof+1) = ux(ie+di, je+dj, ke+dk)
                  u_e(dof+2) = uy(ie+di, je+dj, ke+dk)
                  u_e(dof+3) = uz(ie+di, je+dj, ke+dk)
                end do
              end do
            end do

            R_e = 0.0_dp

            ! Loop over 8 Gauss points
            do g = 1, 8
              ! Strain at GP: eps = B * u_e
              do p = 1, 6
                eps_curr(p) = 0.0_dp
                do q = 1, 24
                  eps_curr(p) = eps_curr(p) + B_all(p, q, g) * u_e(q)
                end do
              end do

              ! Incremental strain
              eps_inc = eps_curr - eps_gp(:, g, ie, je, ke)

              ! Determine phase at this GP (use nearest node = node g)
              ! Node g has local indices: di=mod(g-1,2), dj=mod((g-1)/2,2), dk=(g-1)/4
              di = mod(g-1, 2)
              dj = mod((g-1)/2, 2)
              dk = (g-1) / 4
              ph_g = phase(ie+di, je+dj, ke+dk)

              if (ph_g == PHASE_SOLID) then
                call build_C_matrix(E_solid, nu, C_loc)
                av = alpha_V
              else
                call build_C_matrix(E_soft, nu, C_loc)
                av = 0.0_dp
              end if

              ! Thermal strain
              dT_g = dT_gp_arr(g, ie, je, ke)
              eps_th = 0.0_dp
              eps_th(1) = av * dT_g
              eps_th(2) = av * dT_g
              eps_th(3) = av * dT_g

              ! Trial stress: sigma_trial = sigma_old + C * (eps_inc - eps_thermal)
              do p = 1, 6
                s_trial(p) = sig_gp(p, g, ie, je, ke)
                do q = 1, 6
                  s_trial(p) = s_trial(p) + C_loc(p,q) * (eps_inc(q) - eps_th(q))
                end do
              end do

              ! J2 return mapping
              call j2_return_map(s_trial, sigma, f_dum)

              ! R_e += B^T * sigma * detJ * w_gp (w_gp = 1.0 for 2-point Gauss)
              do p = 1, 24
                BtSig(p) = 0.0_dp
                do q = 1, 6
                  BtSig(p) = BtSig(p) + B_all(q, p, g) * sigma(q)
                end do
              end do
              R_e = R_e + BtSig * det_J_val

            end do  ! GP loop

            ! Scatter R_e to global residual
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dof = 3*(ln-1)
                  Rx(ie+di, je+dj, ke+dk) = Rx(ie+di, je+dj, ke+dk) + R_e(dof+1)
                  Ry(ie+di, je+dj, ke+dk) = Ry(ie+di, je+dj, ke+dk) + R_e(dof+2)
                  Rz(ie+di, je+dj, ke+dk) = Rz(ie+di, je+dj, ke+dk) + R_e(dof+3)
                end do
              end do
            end do

          end do
        end do
      end do
      !$omp end parallel do
    end do

    ! Enforce Dirichlet BC: R=0 at bottom face (k=1)
    Rx(:,:,1) = 0.0_dp
    Ry(:,:,1) = 0.0_dp
    Rz(:,:,1) = 0.0_dp
  end subroutine compute_residual

  ! ============================================================
  ! Update GP state after Newton convergence
  ! Recomputes final stress/strain at each GP and stores them
  ! Also computes f_plus for output
  ! ============================================================
  subroutine update_gp_state(ux, uy, uz, dT_gp_arr, phase)
    real(dp), intent(in) :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    real(dp), intent(in) :: dT_gp_arr(8, Nx, Ny, Nz)
    integer,  intent(in) :: phase(Nnx,Nny,Nnz)

    real(dp) :: u_e(24), eps_curr(6), eps_inc(6), eps_th(6)
    real(dp) :: s_trial(6), sigma(6), C_loc(6,6)
    real(dp) :: av, dT_g, f_yield_g
    real(dp) :: f_plus_elem(8)
    integer  :: ie, je, ke, di, dj, dk, ln, dof, g, p, q, ph_g

    ! Reset f_plus before accumulation
    f_plus = 0.0_dp

    !$omp parallel do collapse(3) default(shared) &
    !$omp   private(ie,je,ke,u_e,eps_curr,eps_inc,eps_th,s_trial,sigma,C_loc) &
    !$omp   private(av,dT_g,f_yield_g,f_plus_elem,di,dj,dk,ln,dof,g,p,q,ph_g)
    do ke = 1, Nz
      do je = 1, Ny
        do ie = 1, Nx

          ! Gather 24 DOFs
          do dk = 0, 1
            do dj = 0, 1
              do di = 0, 1
                ln = 1 + di + 2*dj + 4*dk
                dof = 3*(ln-1)
                u_e(dof+1) = ux(ie+di, je+dj, ke+dk)
                u_e(dof+2) = uy(ie+di, je+dj, ke+dk)
                u_e(dof+3) = uz(ie+di, je+dj, ke+dk)
              end do
            end do
          end do

          do g = 1, 8
            ! Strain at GP
            do p = 1, 6
              eps_curr(p) = 0.0_dp
              do q = 1, 24
                eps_curr(p) = eps_curr(p) + B_all(p, q, g) * u_e(q)
              end do
            end do

            eps_inc = eps_curr - eps_gp(:, g, ie, je, ke)

            ! Phase at this GP
            di = mod(g-1, 2)
            dj = mod((g-1)/2, 2)
            dk = (g-1) / 4
            ph_g = phase(ie+di, je+dj, ke+dk)

            if (ph_g == PHASE_SOLID) then
              call build_C_matrix(E_solid, nu, C_loc)
              av = alpha_V
            else
              call build_C_matrix(E_soft, nu, C_loc)
              av = 0.0_dp
            end if

            dT_g = dT_gp_arr(g, ie, je, ke)
            eps_th = 0.0_dp
            eps_th(1) = av * dT_g
            eps_th(2) = av * dT_g
            eps_th(3) = av * dT_g

            ! Trial stress
            do p = 1, 6
              s_trial(p) = sig_gp(p, g, ie, je, ke)
              do q = 1, 6
                s_trial(p) = s_trial(p) + C_loc(p,q) * (eps_inc(q) - eps_th(q))
              end do
            end do

            ! J2 return mapping
            call j2_return_map(s_trial, sigma, f_yield_g)

            ! Store updated state
            sig_gp(:, g, ie, je, ke) = sigma
            eps_gp(:, g, ie, je, ke) = eps_curr
            f_plus_elem(g) = f_yield_g
          end do

          ! Store element-level f_plus for later scatter
          ! (cannot scatter here safely with collapse(3) — use separate loop)
          do g = 1, 8
            ! Store max f_plus per element for output
            ! We'll scatter in a separate 8-color loop below
          end do

        end do
      end do
    end do
    !$omp end parallel do

    ! Scatter f_plus to nodes using 8-color coloring (race-free)
    block
      integer :: color2, ic2, jc2, kc2
      f_plus = 0.0_dp
      do color2 = 0, 7
        ic2 = mod(color2, 2); jc2 = mod(color2/2, 2); kc2 = mod(color2/4, 2)
        !$omp parallel do collapse(3) default(shared) private(ie,je,ke,g,di,dj,dk)
        do ke = 1+kc2, Nz, 2
          do je = 1+jc2, Ny, 2
            do ie = 1+ic2, Nx, 2
              do g = 1, 8
                di = mod(g-1,2); dj = mod((g-1)/2,2); dk = (g-1)/4
                ! Compute f_yield from stored GP stress
                block
                  real(dp) :: sm, sd(6), sn, fy
                  sm = (sig_gp(1,g,ie,je,ke)+sig_gp(2,g,ie,je,ke)+sig_gp(3,g,ie,je,ke))/3.0_dp
                  sd(1)=sig_gp(1,g,ie,je,ke)-sm; sd(2)=sig_gp(2,g,ie,je,ke)-sm; sd(3)=sig_gp(3,g,ie,je,ke)-sm
                  sd(4)=sig_gp(4,g,ie,je,ke); sd(5)=sig_gp(5,g,ie,je,ke); sd(6)=sig_gp(6,g,ie,je,ke)
                  sn = sqrt(1.5_dp*(sd(1)**2+sd(2)**2+sd(3)**2+2.0_dp*(sd(4)**2+sd(5)**2+sd(6)**2)))
                  fy = max(sn - sig_yield, 0.0_dp)
                  f_plus(ie+di,je+dj,ke+dk) = max(f_plus(ie+di,je+dj,ke+dk), fy)
                end block
              end do
            end do
          end do
        end do
        !$omp end parallel do
      end do
      ! f_plus now contains the MAX yield excess at each node
    end block

  end subroutine update_gp_state

  ! ============================================================
  ! EBE matvec: A*[ux,uy,uz] using 8-color element coloring
  ! ============================================================
  subroutine ebe_matvec_mech(ux, uy, uz, Aux, Auy, Auz, phase)
    real(dp), intent(in)  :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    real(dp), intent(out) :: Aux(Nnx,Nny,Nnz), Auy(Nnx,Nny,Nnz), Auz(Nnx,Nny,Nnz)
    integer,  intent(in)  :: phase(Nnx,Nny,Nnz)

    real(dp) :: xe(24), Axe(24), Ke_loc(24,24)
    integer  :: ie, je, ke, a, b, color, ic, jc, kc
    integer  :: di, dj, dk, ln, dof

    Aux = 0.0_dp; Auy = 0.0_dp; Auz = 0.0_dp

    do color = 0, 7
      ic = mod(color, 2)
      jc = mod(color/2, 2)
      kc = mod(color/4, 2)

      !$omp parallel do collapse(3) default(shared) &
      !$omp   private(ie,je,ke,xe,Axe,Ke_loc,a,b,di,dj,dk,ln,dof)
      do ke = 1 + kc, Nz, 2
        do je = 1 + jc, Ny, 2
          do ie = 1 + ic, Nx, 2
            if (elem_phase(ie,je,ke) /= PHASE_POWDER) then
              Ke_loc = Ke_solid
            else
              Ke_loc = Ke_soft
            end if

            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dof = 3*(ln-1)
                  xe(dof+1) = ux(ie+di, je+dj, ke+dk)
                  xe(dof+2) = uy(ie+di, je+dj, ke+dk)
                  xe(dof+3) = uz(ie+di, je+dj, ke+dk)
                end do
              end do
            end do

            Axe = 0.0_dp
            do b = 1, 24
              do a = 1, 24
                Axe(a) = Axe(a) + Ke_loc(a,b) * xe(b)
              end do
            end do

            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dof = 3*(ln-1)
                  Aux(ie+di, je+dj, ke+dk) = Aux(ie+di, je+dj, ke+dk) + Axe(dof+1)
                  Auy(ie+di, je+dj, ke+dk) = Auy(ie+di, je+dj, ke+dk) + Axe(dof+2)
                  Auz(ie+di, je+dj, ke+dk) = Auz(ie+di, je+dj, ke+dk) + Axe(dof+3)
                end do
              end do
            end do

          end do
        end do
      end do
      !$omp end parallel do
    end do

    ! Enforce Dirichlet: u=0 at k=1 (bottom face)
    Aux(:,:,1) = ux(:,:,1)
    Auy(:,:,1) = uy(:,:,1)
    Auz(:,:,1) = uz(:,:,1)
  end subroutine ebe_matvec_mech

  ! ============================================================
  ! CG solver for 3-component mechanical system using EBE matvec
  ! ============================================================
  subroutine solve_mech_cg(ux, uy, uz, fx, fy, fz, phase)
    real(dp), intent(inout) :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    real(dp), intent(in)    :: fx(Nnx,Nny,Nnz), fy(Nnx,Nny,Nnz), fz(Nnx,Nny,Nnz)
    integer,  intent(in)    :: phase(Nnx,Nny,Nnz)

    real(dp) :: rx(Nnx,Nny,Nnz), ry(Nnx,Nny,Nnz), rz(Nnx,Nny,Nnz)
    real(dp) :: px(Nnx,Nny,Nnz), py(Nnx,Nny,Nnz), pz(Nnx,Nny,Nnz)
    real(dp) :: Apx(Nnx,Nny,Nnz), Apy(Nnx,Nny,Nnz), Apz(Nnx,Nny,Nnz)
    real(dp) :: rr_old, rr_new, pAp, alpha_cg, beta_cg, rnorm, bnorm
    integer  :: iter

    rnorm = 0.0_dp

    ! Enforce Dirichlet on initial guess
    ux(:,:,1) = 0.0_dp; uy(:,:,1) = 0.0_dp; uz(:,:,1) = 0.0_dp

    ! r = f - A*u
    call ebe_matvec_mech(ux, uy, uz, Apx, Apy, Apz, phase)
    rx = fx - Apx; ry = fy - Apy; rz = fz - Apz
    px = rx; py = ry; pz = rz

    rr_old = sum(rx*rx) + sum(ry*ry) + sum(rz*rz)
    bnorm  = sqrt(sum(fx*fx) + sum(fy*fy) + sum(fz*fz))
    if (bnorm < 1.0e-30_dp) return

    do iter = 1, cg_maxiter_mech
      call ebe_matvec_mech(px, py, pz, Apx, Apy, Apz, phase)
      pAp = sum(px*Apx) + sum(py*Apy) + sum(pz*Apz)
      if (abs(pAp) < 1.0e-30_dp) exit
      alpha_cg = rr_old / pAp

      ux = ux + alpha_cg * px
      uy = uy + alpha_cg * py
      uz = uz + alpha_cg * pz

      rx = rx - alpha_cg * Apx
      ry = ry - alpha_cg * Apy
      rz = rz - alpha_cg * Apz

      rr_new = sum(rx*rx) + sum(ry*ry) + sum(rz*rz)
      rnorm = sqrt(rr_new)

      if (rnorm / bnorm < cg_tol_mech) exit

      beta_cg = rr_new / rr_old
      px = rx + beta_cg * px
      py = ry + beta_cg * py
      pz = rz + beta_cg * pz
      rr_old = rr_new
    end do

    if (iter >= cg_maxiter_mech) then
      write(*,'(A,I6,A,ES10.3)') '  WARNING: Mech CG did NOT converge, iter=', iter, ' res=', rnorm/bnorm
    end if

    ! Re-enforce Dirichlet
    ux(:,:,1) = 0.0_dp; uy(:,:,1) = 0.0_dp; uz(:,:,1) = 0.0_dp
  end subroutine solve_mech_cg

  ! ============================================================
  ! Full mechanical solve with Newton iteration (public interface)
  ! ============================================================
  subroutine solve_mechanical(ux, uy, uz, T_new, T_old, phase)
    real(dp), intent(inout) :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    real(dp), intent(in)    :: T_new(Nnx,Nny,Nnz), T_old(Nnx,Nny,Nnz)
    integer,  intent(in)    :: phase(Nnx,Nny,Nnz)

    real(dp) :: Rx(Nnx,Nny,Nnz), Ry(Nnx,Nny,Nnz), Rz(Nnx,Nny,Nnz)
    real(dp) :: dux(Nnx,Nny,Nnz), duy(Nnx,Nny,Nnz), duz(Nnx,Nny,Nnz)
    real(dp) :: neg_Rx(Nnx,Nny,Nnz), neg_Ry(Nnx,Nny,Nnz), neg_Rz(Nnx,Nny,Nnz)
    real(dp) :: dT_gp_arr(8, Nx, Ny, Nz)
    real(dp) :: R_norm, R_norm0
    integer  :: newton_iter

    ! 1. Compute element phases
    call compute_elem_phases(phase)

    ! 2. Compute dT at GP level (interpolate from nodes)
    call compute_dT_gp(T_new, dT_gp_arr)

    ! 3. Newton iteration
    R_norm0 = 0.0_dp
    do newton_iter = 1, newton_maxiter
      ! Compute residual: R = sum_elem sum_gp B^T * sigma * detJ
      call compute_residual(ux, uy, uz, dT_gp_arr, phase, Rx, Ry, Rz)

      R_norm = sqrt(sum(Rx**2) + sum(Ry**2) + sum(Rz**2))
      if (newton_iter == 1) R_norm0 = R_norm

      if (R_norm / max(R_norm0, 1.0e-30_dp) < newton_tol) exit

      ! Solve K * du = -R using elastic K as Jacobian via EBE CG
      dux = 0.0_dp; duy = 0.0_dp; duz = 0.0_dp
      neg_Rx = -Rx; neg_Ry = -Ry; neg_Rz = -Rz
      call solve_mech_cg(dux, duy, duz, neg_Rx, neg_Ry, neg_Rz, phase)

      ux = ux + dux; uy = uy + duy; uz = uz + duz
    end do

    ! 4. Update GP stress/strain state after Newton convergence
    call update_gp_state(ux, uy, uz, dT_gp_arr, phase)

    ! 5. Update temperature reference
    T_old_for_u = T_new
  end subroutine solve_mechanical

  ! ============================================================
  ! Return stress and yield info (public interface)
  ! sxx_out: average sig_gp(1,...) over GPs mapped to nearest node
  ! fplus_out: accumulated f_plus
  ! ============================================================
  subroutine get_stress_yield(sxx_out, fplus_out)
    real(dp), intent(out) :: sxx_out(Nnx,Nny,Nnz), fplus_out(Nnx,Nny,Nnz)
    integer :: ie, je, ke, g, di, dj, dk
    integer :: cnt(Nnx,Nny,Nnz)

    sxx_out = 0.0_dp
    cnt = 0

    ! Scatter GP sxx to nearest node, then average
    do ke = 1, Nz
      do je = 1, Ny
        do ie = 1, Nx
          do g = 1, 8
            di = mod(g-1, 2)
            dj = mod((g-1)/2, 2)
            dk = (g-1) / 4
            sxx_out(ie+di, je+dj, ke+dk) = sxx_out(ie+di, je+dj, ke+dk) + sig_gp(1, g, ie, je, ke)
            cnt(ie+di, je+dj, ke+dk) = cnt(ie+di, je+dj, ke+dk) + 1
          end do
        end do
      end do
    end do

    where (cnt > 0)
      sxx_out = sxx_out / dble(cnt)
    end where

    fplus_out = f_plus
  end subroutine get_stress_yield

end module mod_mechanical
