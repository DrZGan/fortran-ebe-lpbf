module mod_mechanical
  use mod_parameters
  implicit none

  private
  public :: solve_mechanical, init_mechanical, cleanup_mechanical, get_stress_yield

  ! Precomputed 24x24 element stiffness matrices (symmetric)
  real(dp) :: Ke_solid(24,24)   ! E=70GPa, nu=0.3
  real(dp) :: Ke_soft(24,24)    ! E=0.7GPa, nu=0.3

  ! Per-GP stiffness contributions: Ke = Σ_{gp=1}^{8} Ke_gp
  ! For MIXED elements: Ke_mixed = Σ_gp (SOLID? Ke_gp_solid : Ke_gp_soft)
  ! GP g is closest to node g (2×2×2 Gauss on HEX8), so use node phase.
  real(dp) :: Ke_gp_solid(24,24,8)  ! per-GP contribution, solid material
  real(dp) :: Ke_gp_soft(24,24,8)   ! per-GP contribution, soft material

  ! Thermal coupling matrix: Mth(24,8) maps 8 nodal dT to 24 force DOFs
  real(dp) :: Mth_solid(24,8)
  real(dp) :: Mth_soft(24,8)

  ! Keep Fth for backward compat
  real(dp) :: Fth_solid(24)
  real(dp) :: Fth_soft(24)

  ! Internal state arrays
  real(dp), allocatable :: sig_old(:,:,:,:)    ! (6, Nnx, Nny, Nnz)
  real(dp), allocatable :: eps_old(:,:,:,:)    ! (6, Nnx, Nny, Nnz)
  real(dp), allocatable :: T_old_for_u(:,:,:)
  real(dp), allocatable :: f_plus(:,:,:)

  ! Element phase cache (Nx, Ny, Nz)
  integer, allocatable :: elem_phase(:,:,:)

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
  subroutine compute_element_matrices(E_val, nu_val, Ke, Fth, Mth, Ke_gp)
    real(dp), intent(in)  :: E_val, nu_val
    real(dp), intent(out) :: Ke(24,24)
    real(dp), intent(out) :: Fth(24)
    real(dp), intent(out) :: Mth(24,8)
    real(dp), intent(out) :: Ke_gp(24,24,8)  ! per-GP contributions

    real(dp) :: C(6,6)
    real(dp) :: gp(2), gw(2)
    real(dp) :: xi, eta_q, zeta, w, det_J
    real(dp) :: xi_n(8), eta_n(8), zeta_n(8)
    real(dp) :: dN_dx(8), dN_dy(8), dN_dz(8)
    integer  :: gp_idx  ! 1..8 GP counter
    real(dp) :: dN_dxi(8), dN_deta(8), dN_dzeta(8)
    real(dp) :: N(8)
    real(dp) :: B_a(6,3), B_b(6,3)
    real(dp) :: CB(6,3)
    real(dp) :: eps_th(6), C_eps_th(6)
    integer  :: i, j, k, a, b, p, q

    call build_C_matrix(E_val, nu_val, C)

    ! Lexicographic node ordering: ln = 1 + di + 2*dj + 4*dk
    xi_n   = (/ -1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp,-1.0_dp, 1.0_dp /)
    eta_n  = (/ -1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp,-1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp /)
    zeta_n = (/ -1.0_dp,-1.0_dp,-1.0_dp,-1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp /)

    ! 2-point Gauss quadrature
    gp(1) = -1.0_dp / sqrt(3.0_dp)
    gp(2) =  1.0_dp / sqrt(3.0_dp)
    gw(1) = 1.0_dp
    gw(2) = 1.0_dp

    ! Jacobian for uniform hex
    det_J = (dx/2.0_dp) * (dy/2.0_dp) * (dz/2.0_dp)

    ! Thermal strain: eps_thermal = alpha_V * [1,1,1,0,0,0]
    eps_th = (/ 1.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp /)

    ! C * eps_th (unit alpha_V factored later)
    C_eps_th = 0.0_dp
    do p = 1, 6
      do q = 1, 6
        C_eps_th(p) = C_eps_th(p) + C(p,q) * eps_th(q)
      end do
    end do
    ! Include alpha_V
    C_eps_th = C_eps_th * alpha_V

    Ke    = 0.0_dp
    Fth   = 0.0_dp
    Mth   = 0.0_dp
    Ke_gp = 0.0_dp

    ! 2x2x2 Gauss integration
    ! GP index matches node ordering: gp_idx = i + 2*(j-1) + 4*(k-1)
    ! GP(1)↔node(1)=(-1,-1,-1), GP(2)↔node(2)=(+1,-1,-1), etc.
    do k = 1, 2
      do j = 1, 2
        do i = 1, 2
          gp_idx = i + 2*(j-1) + 4*(k-1)
          xi     = gp(i)
          eta_q  = gp(j)
          zeta   = gp(k)
          w = gw(i) * gw(j) * gw(k)

          ! Shape functions and gradients
          do a = 1, 8
            N(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta_q) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
          end do
          do a = 1, 8
            dN_dxi(a)   = xi_n(a)   * (1.0_dp + eta_n(a)*eta_q) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_deta(a)  = (1.0_dp + xi_n(a)*xi) * eta_n(a)  * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_dzeta(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta_q) * zeta_n(a) / 8.0_dp
          end do

          ! Physical gradients
          dN_dx = dN_dxi   * (2.0_dp / dx)
          dN_dy = dN_deta  * (2.0_dp / dy)
          dN_dz = dN_dzeta * (2.0_dp / dz)

          ! Build Ke: K_e(3*(a-1)+p, 3*(b-1)+q) += B_a^T C B_b * det_J * w
          ! B_a is 6x3 strain-displacement for node a:
          ! B_a = [ dN_a/dx   0        0      ]
          !       [ 0         dN_a/dy  0      ]
          !       [ 0         0        dN_a/dz]
          !       [ dN_a/dy   dN_a/dx  0      ]
          !       [ dN_a/dz   0        dN_a/dx]
          !       [ 0         dN_a/dz  dN_a/dy]

          do b = 1, 8
            ! Build B_b
            B_b = 0.0_dp
            B_b(1,1) = dN_dx(b)
            B_b(2,2) = dN_dy(b)
            B_b(3,3) = dN_dz(b)
            B_b(4,1) = dN_dy(b); B_b(4,2) = dN_dx(b)
            B_b(5,1) = dN_dz(b); B_b(5,3) = dN_dx(b)
            B_b(6,2) = dN_dz(b); B_b(6,3) = dN_dy(b)

            ! CB = C * B_b  (6x3)
            CB = 0.0_dp
            do q = 1, 3
              do p = 1, 6
                CB(p,q) = C(p,1)*B_b(1,q) + C(p,2)*B_b(2,q) + C(p,3)*B_b(3,q) &
                        + C(p,4)*B_b(4,q) + C(p,5)*B_b(5,q) + C(p,6)*B_b(6,q)
              end do
            end do

            do a = 1, 8
              ! Build B_a
              B_a = 0.0_dp
              B_a(1,1) = dN_dx(a)
              B_a(2,2) = dN_dy(a)
              B_a(3,3) = dN_dz(a)
              B_a(4,1) = dN_dy(a); B_a(4,2) = dN_dx(a)
              B_a(5,1) = dN_dz(a); B_a(5,3) = dN_dx(a)
              B_a(6,2) = dN_dz(a); B_a(6,3) = dN_dy(a)

              ! K_e block (a,b) = B_a^T * C * B_b * det_J * w  → 3x3 block
              ! K_e(3*(a-1)+p, 3*(b-1)+q) += B_a(:,p)^T . CB(:,q) * det_J * w
              do q = 1, 3
                do p = 1, 3
                  Ke(3*(a-1)+p, 3*(b-1)+q) = Ke(3*(a-1)+p, 3*(b-1)+q) &
                    + (B_a(1,p)*CB(1,q) + B_a(2,p)*CB(2,q) + B_a(3,p)*CB(3,q) &
                     + B_a(4,p)*CB(4,q) + B_a(5,p)*CB(5,q) + B_a(6,p)*CB(6,q)) &
                    * det_J * w
                  ! Also store per-GP contribution
                  Ke_gp(3*(a-1)+p, 3*(b-1)+q, gp_idx) = &
                    Ke_gp(3*(a-1)+p, 3*(b-1)+q, gp_idx) &
                    + (B_a(1,p)*CB(1,q) + B_a(2,p)*CB(2,q) + B_a(3,p)*CB(3,q) &
                     + B_a(4,p)*CB(4,q) + B_a(5,p)*CB(5,q) + B_a(6,p)*CB(6,q)) &
                    * det_J * w
                end do
              end do
            end do
          end do

          ! Thermal load: F_th(3*(a-1)+p) += B_a(:,p)^T . C_eps_th * det_J * w
          ! Thermal coupling: Mth(3*(a-1)+p, b) += B_a(:,p)^T . C_eps_th * N_b * det_J * w
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
              ! Per-GP thermal coupling: Mth(dof_a, node_b) += BtC * N_b
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
  ! Initialize module: precompute element matrices, allocate state
  ! ============================================================
  subroutine init_mechanical()
    allocate(sig_old(6, Nnx, Nny, Nnz))
    allocate(eps_old(6, Nnx, Nny, Nnz))
    allocate(T_old_for_u(Nnx, Nny, Nnz))
    allocate(f_plus(Nnx, Nny, Nnz))
    allocate(elem_phase(Nx, Ny, Nz))

    sig_old = 0.0_dp
    eps_old = 0.0_dp
    T_old_for_u = T0
    f_plus = 0.0_dp
    elem_phase = PHASE_POWDER

    ! Precompute element stiffness matrices and thermal load vectors
    call compute_element_matrices(E_solid, nu, Ke_solid, Fth_solid, Mth_solid, Ke_gp_solid)
    call compute_element_matrices(E_soft,  nu, Ke_soft,  Fth_soft,  Mth_soft, Ke_gp_soft)
  end subroutine init_mechanical

  subroutine cleanup_mechanical()
    if (allocated(sig_old))     deallocate(sig_old)
    if (allocated(eps_old))     deallocate(eps_old)
    if (allocated(T_old_for_u)) deallocate(T_old_for_u)
    if (allocated(f_plus))      deallocate(f_plus)
    if (allocated(elem_phase))  deallocate(elem_phase)
  end subroutine cleanup_mechanical

  ! ============================================================
  ! Determine element phase: SOLID if any node is SOLID, else
  ! majority of 8 nodes
  ! ============================================================
  subroutine compute_elem_phases(phase)
    integer, intent(in) :: phase(Nnx, Nny, Nnz)
    integer :: ie, je, ke, di, dj, dk, ph, n_solid

    !$omp parallel do collapse(3) default(shared) private(ie,je,ke,di,dj,dk,ph,n_solid)
    do ke = 1, Nz
      do je = 1, Ny
        do ie = 1, Nx
          ! SOLID if any node is SOLID, else POWDER
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
  ! EBE matvec: A*[ux,uy,uz] using 8-color element coloring
  ! Gathers 24 DOFs per element, multiplies by Ke, scatters back
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
            ! Element stiffness: SOLID if any node is SOLID, else soft
            ! Per-GP blending tested but causes stress amplification at boundaries.
            ! Binary approach gives better overall results (sxx 1.75x vs 100x+).
            if (elem_phase(ie,je,ke) /= PHASE_POWDER) then
              Ke_loc = Ke_solid
            else
              Ke_loc = Ke_soft
            end if

            ! Gather 24 DOFs: node ordering ln = 1+di+2*dj+4*dk
            ! DOF ordering: (node1_ux, node1_uy, node1_uz, node2_ux, ...)
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

            ! Local matvec: Axe = Ke_loc * xe
            Axe = 0.0_dp
            do b = 1, 24
              do a = 1, 24
                Axe(a) = Axe(a) + Ke_loc(a,b) * xe(b)
              end do
            end do

            ! Scatter-add back to global arrays
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

    ! Enforce Dirichlet: u=0 at k=1 (bottom face) → identity row
    Aux(:,:,1) = ux(:,:,1)
    Auy(:,:,1) = uy(:,:,1)
    Auz(:,:,1) = uz(:,:,1)
  end subroutine ebe_matvec_mech

  ! ============================================================
  ! Assemble thermal body force RHS using EBE
  ! F_e = F_th_unit * avg(DT) over element nodes (SOLID elements only)
  ! ============================================================
  subroutine assemble_thermal_rhs(T_new, phase, fx, fy, fz)
    real(dp), intent(in)  :: T_new(Nnx,Nny,Nnz)
    integer,  intent(in)  :: phase(Nnx,Nny,Nnz)
    real(dp), intent(out) :: fx(Nnx,Nny,Nnz), fy(Nnx,Nny,Nnz), fz(Nnx,Nny,Nnz)

    real(dp) :: fe(24), dT_nodes(8), Mth_loc(24,8)
    integer  :: ie, je, ke, di, dj, dk, ln, dof, color, ic, jc, kc, a, b

    fx = 0.0_dp; fy = 0.0_dp; fz = 0.0_dp

    do color = 0, 7
      ic = mod(color, 2)
      jc = mod(color/2, 2)
      kc = mod(color/4, 2)

      !$omp parallel do collapse(3) default(shared) &
      !$omp   private(ie,je,ke,fe,dT_nodes,Mth_loc,di,dj,dk,ln,dof,a,b)
      do ke = 1 + kc, Nz, 2
        do je = 1 + jc, Ny, 2
          do ie = 1 + ic, Nx, 2
            ! Only elements with at least one SOLID node produce thermal load
            if (elem_phase(ie,je,ke) == PHASE_POWDER) cycle

            Mth_loc = Mth_solid

            ! Gather 8 nodal dT values
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dT_nodes(ln) = T_new(ie+di, je+dj, ke+dk) - T_old_for_u(ie+di, je+dj, ke+dk)
                end do
              end do
            end do

            ! Element thermal force: fe = Mth * dT_nodes (24x8 * 8 = 24)
            fe = 0.0_dp
            do b = 1, 8
              do a = 1, 24
                fe(a) = fe(a) + Mth_loc(a,b) * dT_nodes(b)
              end do
            end do

            ! Scatter-add
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dof = 3*(ln-1)
                  fx(ie+di, je+dj, ke+dk) = fx(ie+di, je+dj, ke+dk) + fe(dof+1)
                  fy(ie+di, je+dj, ke+dk) = fy(ie+di, je+dj, ke+dk) + fe(dof+2)
                  fz(ie+di, je+dj, ke+dk) = fz(ie+di, je+dj, ke+dk) + fe(dof+3)
                end do
              end do
            end do

          end do
        end do
      end do
      !$omp end parallel do
    end do

    ! Dirichlet: f=0 at bottom (u=0 there)
    fx(:,:,1) = 0.0_dp
    fy(:,:,1) = 0.0_dp
    fz(:,:,1) = 0.0_dp
  end subroutine assemble_thermal_rhs

  ! ============================================================
  ! Assemble thermal RHS using TOTAL dT = T - T0
  ! Uses per-node phase and per-GP interpolation
  ! Only SOLID nodes contribute (alpha_V = 0 for POWDER/LIQUID)
  ! ============================================================
  subroutine assemble_thermal_rhs_total(T_new, phase, fx, fy, fz)
    real(dp), intent(in)  :: T_new(Nnx,Nny,Nnz)
    integer,  intent(in)  :: phase(Nnx,Nny,Nnz)
    real(dp), intent(out) :: fx(Nnx,Nny,Nnz), fy(Nnx,Nny,Nnz), fz(Nnx,Nny,Nnz)

    real(dp) :: fe(24), dT_nodes(8), Mth_loc(24,8)
    integer  :: ie, je, ke, di, dj, dk, ln, dof, color, ic, jc, kc, a, b

    fx = 0.0_dp; fy = 0.0_dp; fz = 0.0_dp

    do color = 0, 7
      ic = mod(color, 2)
      jc = mod(color/2, 2)
      kc = mod(color/4, 2)

      !$omp parallel do collapse(3) default(shared) &
      !$omp   private(ie,je,ke,fe,dT_nodes,Mth_loc,di,dj,dk,ln,dof,a,b)
      do ke = 1 + kc, Nz, 2
        do je = 1 + jc, Ny, 2
          do ie = 1 + ic, Nx, 2
            if (elem_phase(ie,je,ke) == PHASE_POWDER) cycle

            Mth_loc = Mth_solid

            ! Gather 8 nodal TOTAL dT = T - T0
            ! Only SOLID nodes contribute thermal strain
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  if (phase(ie+di, je+dj, ke+dk) == PHASE_SOLID) then
                    dT_nodes(ln) = T_new(ie+di, je+dj, ke+dk) - T0
                  else
                    dT_nodes(ln) = 0.0_dp
                  end if
                end do
              end do
            end do

            ! Element thermal force: fe = Mth * dT_nodes
            fe = 0.0_dp
            do b = 1, 8
              do a = 1, 24
                fe(a) = fe(a) + Mth_loc(a,b) * dT_nodes(b)
              end do
            end do

            ! Scatter-add
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  dof = 3*(ln-1)
                  fx(ie+di, je+dj, ke+dk) = fx(ie+di, je+dj, ke+dk) + fe(dof+1)
                  fy(ie+di, je+dj, ke+dk) = fy(ie+di, je+dj, ke+dk) + fe(dof+2)
                  fz(ie+di, je+dj, ke+dk) = fz(ie+di, je+dj, ke+dk) + fe(dof+3)
                end do
              end do
            end do

          end do
        end do
      end do
      !$omp end parallel do
    end do

    fx(:,:,1) = 0.0_dp; fy(:,:,1) = 0.0_dp; fz(:,:,1) = 0.0_dp
  end subroutine assemble_thermal_rhs_total

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
  ! Full mechanical solve (public interface, same signature)
  ! ============================================================
  subroutine solve_mechanical(ux, uy, uz, T_new, T_old, phase)
    real(dp), intent(inout) :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    real(dp), intent(in)    :: T_new(Nnx,Nny,Nnz), T_old(Nnx,Nny,Nnz)
    integer,  intent(in)    :: phase(Nnx,Nny,Nnz)

    real(dp) :: fx(Nnx,Nny,Nnz), fy(Nnx,Nny,Nnz), fz(Nnx,Nny,Nnz)
    real(dp) :: dux(Nnx,Nny,Nnz), duy(Nnx,Nny,Nnz), duz(Nnx,Nny,Nnz)
    real(dp) :: lam, mu, E_local, av, dT_val
    real(dp) :: exx, eyy, ezz, exy, exz, eyz
    real(dp) :: eps_inc(6), eps_th(6), s_trial(6), s_mean, s_dev(6), s_norm, f_yield
    integer  :: i, j, k

    ! 1. Compute element phases
    call compute_elem_phases(phase)

    ! 2. Assemble incremental thermal RHS (dT since last mech solve)
    call assemble_thermal_rhs(T_new, phase, fx, fy, fz)

    ! 3. Solve for displacement INCREMENT, then accumulate
    dux = 0.0_dp; duy = 0.0_dp; duz = 0.0_dp
    call solve_mech_cg(dux, duy, duz, fx, fy, fz, phase)
    ux = ux + dux; uy = uy + duy; uz = uz + duz

    ! 4. Incremental stress update with return mapping
    !$omp parallel do collapse(3) default(shared) &
    !$omp   private(i,j,k,E_local,lam,mu,exx,eyy,ezz,exy,exz,eyz,dT_val,av) &
    !$omp   private(eps_inc,eps_th,s_trial,s_mean,s_dev,s_norm,f_yield)
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          if (phase(i,j,k) == PHASE_SOLID) then
            E_local = E_solid; av = alpha_V
          else
            E_local = E_soft; av = 0.0_dp
          end if
          lam = E_local * nu / ((1.0_dp + nu) * (1.0_dp - 2.0_dp * nu))
          mu  = E_local / (2.0_dp * (1.0_dp + nu))

          call compute_strain_at(ux, uy, uz, i, j, k, exx, eyy, ezz, exy, exz, eyz)

          ! Incremental strain and thermal strain
          eps_inc(1) = exx - eps_old(1,i,j,k)
          eps_inc(2) = eyy - eps_old(2,i,j,k)
          eps_inc(3) = ezz - eps_old(3,i,j,k)
          eps_inc(4) = exy - eps_old(4,i,j,k)
          eps_inc(5) = exz - eps_old(5,i,j,k)
          eps_inc(6) = eyz - eps_old(6,i,j,k)

          dT_val = T_new(i,j,k) - T_old_for_u(i,j,k)
          eps_th = 0.0_dp; eps_th(1) = av*dT_val; eps_th(2) = av*dT_val; eps_th(3) = av*dT_val

          ! Trial stress: σ_trial = σ_old + C:(ε_inc - ε_thermal)
          s_trial(1) = sig_old(1,i,j,k) + lam*((eps_inc(1)-eps_th(1))+(eps_inc(2)-eps_th(2))+(eps_inc(3)-eps_th(3))) &
                     + 2.0_dp*mu*(eps_inc(1)-eps_th(1))
          s_trial(2) = sig_old(2,i,j,k) + lam*((eps_inc(1)-eps_th(1))+(eps_inc(2)-eps_th(2))+(eps_inc(3)-eps_th(3))) &
                     + 2.0_dp*mu*(eps_inc(2)-eps_th(2))
          s_trial(3) = sig_old(3,i,j,k) + lam*((eps_inc(1)-eps_th(1))+(eps_inc(2)-eps_th(2))+(eps_inc(3)-eps_th(3))) &
                     + 2.0_dp*mu*(eps_inc(3)-eps_th(3))
          s_trial(4) = sig_old(4,i,j,k) + 2.0_dp*mu*eps_inc(4)
          s_trial(5) = sig_old(5,i,j,k) + 2.0_dp*mu*eps_inc(5)
          s_trial(6) = sig_old(6,i,j,k) + 2.0_dp*mu*eps_inc(6)

          ! J2 return mapping
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
          f_plus(i,j,k) = max(f_yield, 0.0_dp)

          if (f_yield > 0.0_dp .and. s_norm > 1.0e-30_dp) then
            sig_old(1,i,j,k) = s_mean + s_dev(1)*(1.0_dp - f_yield/s_norm)
            sig_old(2,i,j,k) = s_mean + s_dev(2)*(1.0_dp - f_yield/s_norm)
            sig_old(3,i,j,k) = s_mean + s_dev(3)*(1.0_dp - f_yield/s_norm)
            sig_old(4,i,j,k) = s_dev(4)*(1.0_dp - f_yield/s_norm)
            sig_old(5,i,j,k) = s_dev(5)*(1.0_dp - f_yield/s_norm)
            sig_old(6,i,j,k) = s_dev(6)*(1.0_dp - f_yield/s_norm)
          else
            sig_old(:,i,j,k) = s_trial(:)
          end if

          eps_old(1,i,j,k) = exx; eps_old(2,i,j,k) = eyy; eps_old(3,i,j,k) = ezz
          eps_old(4,i,j,k) = exy; eps_old(5,i,j,k) = exz; eps_old(6,i,j,k) = eyz
        end do
      end do
    end do
    !$omp end parallel do

    T_old_for_u = T_new
  end subroutine solve_mechanical

  ! ============================================================
  ! Compute strain at a node using central differences
  ! ============================================================
  subroutine compute_strain_at(ux, uy, uz, i, j, k, exx, eyy, ezz, exy, exz, eyz)
    real(dp), intent(in)  :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    integer,  intent(in)  :: i, j, k
    real(dp), intent(out) :: exx, eyy, ezz, exy, exz, eyz
    integer :: ip, im, jp, jm, kp, km
    real(dp) :: hx, hy, hz

    ip = min(i+1,Nnx); im = max(i-1,1); hx = dble(ip-im)*dx
    jp = min(j+1,Nny); jm = max(j-1,1); hy = dble(jp-jm)*dy
    kp = min(k+1,Nnz); km = max(k-1,1); hz = dble(kp-km)*dz

    exx = (ux(ip,j,k) - ux(im,j,k)) / hx
    eyy = (uy(i,jp,k) - uy(i,jm,k)) / hy
    ezz = (uz(i,j,kp) - uz(i,j,km)) / hz
    exy = 0.5_dp * ((ux(i,jp,k)-ux(i,jm,k))/hy + (uy(ip,j,k)-uy(im,j,k))/hx)
    exz = 0.5_dp * ((ux(i,j,kp)-ux(i,j,km))/hz + (uz(ip,j,k)-uz(im,j,k))/hx)
    eyz = 0.5_dp * ((uy(i,j,kp)-uy(i,j,km))/hz + (uz(i,jp,k)-uz(i,jm,k))/hy)
  end subroutine compute_strain_at

  ! ============================================================
  ! Return stress and yield info (public interface, same signature)
  ! ============================================================
  subroutine get_stress_yield(sxx_out, fplus_out)
    real(dp), intent(out) :: sxx_out(Nnx,Nny,Nnz), fplus_out(Nnx,Nny,Nnz)
    sxx_out = sig_old(1,:,:,:)
    fplus_out = f_plus
  end subroutine get_stress_yield

end module mod_mechanical
