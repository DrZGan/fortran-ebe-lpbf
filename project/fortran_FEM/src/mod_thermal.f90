module mod_thermal
  use mod_parameters
  implicit none

  private
  public :: init_thermal, solve_thermal_cg, compute_thermal_rhs, ebe_matvec, Ae

  ! Precomputed element matrix: A_e = M_e/dt + K_e  (8x8, symmetric)
  real(dp) :: Ae(8,8)

  ! Precomputed surface load shape functions for top face (4 nodes, 4 Gauss pts)
  ! top_face_N(node, gp) = N_i(xi_gp, eta_gp) on the z=+1 face
  real(dp) :: top_face_N(4,4)
  real(dp) :: top_face_w(4)       ! Gauss weights * det(J_face)
  real(dp) :: top_face_xgp(4,2)  ! physical (x,y) of Gauss points relative to element corner

  ! HEX8 local node ordering matching gmsh/VTK convention:
  ! Node 1: (-1,-1,-1) = (0,0,0)
  ! Node 2: (+1,-1,-1) = (1,0,0)
  ! Node 3: (+1,+1,-1) = (1,1,0)  ← swapped vs lexicographic!
  ! Node 4: (-1,+1,-1) = (0,1,0)  ← swapped vs lexicographic!
  ! Node 5: (-1,-1,+1) = (0,0,1)
  ! Node 6: (+1,-1,+1) = (1,0,1)
  ! Node 7: (+1,+1,+1) = (1,1,1)  ← swapped vs lexicographic!
  ! Node 8: (-1,+1,+1) = (0,1,1)  ← swapped vs lexicographic!
  !
  ! BUT: For our structured grid, the node ordering doesn't matter
  ! because the element matrix is the same for all orderings on a
  ! uniform hex (all nodes are equivalent by symmetry).
  ! What matters is the gather/scatter mapping from (di,dj,dk) to
  ! local node index. We use: ln = 1 + di + 2*dj + 4*dk (lexicographic)

contains

  ! ============================================================
  ! Initialize: compute element matrices
  ! ============================================================
  subroutine init_thermal()
    real(dp) :: Me(8,8), Ke(8,8)
    real(dp) :: gp(2), gw(2)
    real(dp) :: xi, eta, zeta, w, det_J
    real(dp) :: N(8), dN_dxi(8), dN_deta(8), dN_dzeta(8)
    real(dp) :: dN_dx(8), dN_dy(8), dN_dz(8)
    real(dp) :: xi_n(8), eta_n(8), zeta_n(8)
    real(dp) :: gp2d(2), xi_f, eta_f, det_J_face
    integer  :: i, j, k, a, b, fi, fj, fn

    ! Reference node coordinates in LEXICOGRAPHIC order:
    ! ln = 1+di+2*dj+4*dk: (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)
    ! xi(di=0)=-1, xi(di=1)=+1, similarly for eta, zeta
    xi_n   = (/ -1, 1,-1, 1,-1, 1,-1, 1 /)
    eta_n  = (/ -1,-1, 1, 1,-1,-1, 1, 1 /)
    zeta_n = (/ -1,-1,-1,-1, 1, 1, 1, 1 /)

    ! 2-point Gauss quadrature
    gp(1) = -1.0_dp / sqrt(3.0_dp)
    gp(2) =  1.0_dp / sqrt(3.0_dp)
    gw(1) = 1.0_dp
    gw(2) = 1.0_dp

    ! Jacobian for uniform hex: dx/dxi = dx/2, etc.
    det_J = (dx/2.0_dp) * (dy/2.0_dp) * (dz/2.0_dp)

    Me = 0.0_dp
    Ke = 0.0_dp

    ! 2x2x2 Gauss integration
    do k = 1, 2
      do j = 1, 2
        do i = 1, 2
          xi   = gp(i)
          eta  = gp(j)
          zeta = gp(k)
          w = gw(i) * gw(j) * gw(k)

          ! Shape functions
          do a = 1, 8
            N(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
          end do

          ! Shape function gradients in reference coords
          do a = 1, 8
            dN_dxi(a)   = xi_n(a)   * (1.0_dp + eta_n(a)*eta) * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_deta(a)  = (1.0_dp + xi_n(a)*xi) * eta_n(a)  * (1.0_dp + zeta_n(a)*zeta) / 8.0_dp
            dN_dzeta(a) = (1.0_dp + xi_n(a)*xi) * (1.0_dp + eta_n(a)*eta) * zeta_n(a)   / 8.0_dp
          end do

          ! Physical gradients: dN/dx = dN/dxi * dxi/dx = dN/dxi * 2/dx
          dN_dx = dN_dxi   * (2.0_dp / dx)
          dN_dy = dN_deta  * (2.0_dp / dy)
          dN_dz = dN_dzeta * (2.0_dp / dz)

          ! Mass matrix: M_ab = rho*Cp * N_a * N_b * det_J * w
          do b = 1, 8
            do a = 1, 8
              Me(a,b) = Me(a,b) + rho * Cp * N(a) * N(b) * det_J * w
            end do
          end do

          ! Stiffness matrix: K_ab = k * (dN_a/dx * dN_b/dx + dN_a/dy * dN_b/dy + dN_a/dz * dN_b/dz) * det_J * w
          do b = 1, 8
            do a = 1, 8
              Ke(a,b) = Ke(a,b) + k_cond * (dN_dx(a)*dN_dx(b) + dN_dy(a)*dN_dy(b) + dN_dz(a)*dN_dz(b)) * det_J * w
            end do
          end do
        end do
      end do
    end do

    ! Combined element matrix
    Ae = Me / dt + Ke

    ! ============================================================
    ! Surface load: top face (zeta = +1), nodes 5,6,7,8
    ! Face parameterized by (xi, eta), zeta=1 fixed
    ! 2x2 Gauss on face, det_J_face = (dx/2)*(dy/2)
    ! ============================================================
    det_J_face = (dx / 2.0_dp) * (dy / 2.0_dp)
    gp2d = gp  ! same Gauss points

    do fj = 1, 2
      do fi = 1, 2
        fn = (fj-1)*2 + fi  ! face Gauss point index 1..4
        xi_f  = gp2d(fi)
        eta_f = gp2d(fj)

        ! Shape functions on the face: the 4 top-face nodes (5,6,7,8)
        ! N5 = (1-xi)(1-eta)/4, N6 = (1+xi)(1-eta)/4, N7 = (1+xi)(1+eta)/4, N8 = (1-xi)(1+eta)/4
        top_face_N(1, fn) = (1.0_dp - xi_f) * (1.0_dp - eta_f) / 4.0_dp  ! node 5
        top_face_N(2, fn) = (1.0_dp + xi_f) * (1.0_dp - eta_f) / 4.0_dp  ! node 6
        top_face_N(3, fn) = (1.0_dp + xi_f) * (1.0_dp + eta_f) / 4.0_dp  ! node 7
        top_face_N(4, fn) = (1.0_dp - xi_f) * (1.0_dp + eta_f) / 4.0_dp  ! node 8

        top_face_w(fn) = gw(fi) * gw(fj) * det_J_face

        ! Physical position of Gauss point relative to element corner (ie*dx, je*dy)
        top_face_xgp(fn, 1) = dx / 2.0_dp * (1.0_dp + xi_f)   ! x offset from ie*dx
        top_face_xgp(fn, 2) = dy / 2.0_dp * (1.0_dp + eta_f)   ! y offset from je*dy
      end do
    end do

  end subroutine init_thermal

  ! ============================================================
  ! EBE matvec: Ax = Σ_e scatter(A_e * gather(x, e))
  ! Uses 8-color element coloring for OpenMP safety
  ! ============================================================
  subroutine ebe_matvec(x, Ax)
    real(dp), intent(in)  :: x(Nnx, Nny, Nnz)
    real(dp), intent(out) :: Ax(Nnx, Nny, Nnz)

    real(dp) :: xe(8), Axe(8)
    integer  :: ie, je, ke, a, b, color, ic, jc, kc
    integer  :: di, dj, dk, ln

    ! Zero output
    Ax = 0.0_dp

    ! 8-color loop for race-free parallel scatter-add
    do color = 0, 7
      ic = mod(color, 2)
      jc = mod(color/2, 2)
      kc = mod(color/4, 2)

      !$omp parallel do collapse(3) default(shared) private(ie,je,ke,xe,Axe,a,b,di,dj,dk,ln)
      do ke = 1 + kc, Nz, 2
        do je = 1 + jc, Ny, 2
          do ie = 1 + ic, Nx, 2
            ! Gather: 8 node values (use FULL values including Dirichlet)
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  xe(ln) = x(ie+di, je+dj, ke+dk)
                end do
              end do
            end do

            ! Local matvec: Axe = Ae * xe  (Dirichlet columns zeroed)
            Axe = 0.0_dp
            do b = 1, 8
              do a = 1, 8
                Axe(a) = Axe(a) + Ae(a,b) * xe(b)
              end do
            end do

            ! Scatter-add (all nodes including Dirichlet)
            do dk = 0, 1
              do dj = 0, 1
                do di = 0, 1
                  ln = 1 + di + 2*dj + 4*dk
                  Ax(ie+di, je+dj, ke+dk) = Ax(ie+di, je+dj, ke+dk) + Axe(ln)
                end do
              end do
            end do

          end do
        end do
      end do
      !$omp end parallel do
    end do

    ! Enforce Dirichlet: identity row for bottom face
    Ax(:,:,1) = x(:,:,1)

  end subroutine ebe_matvec

  ! ============================================================
  ! Compute RHS = M/dt * T_old + F_surface
  ! Uses EBE assembly for mass term
  ! ============================================================
  subroutine compute_thermal_rhs(T_old, rhs, laser_x, laser_y, laser_on)
    real(dp), intent(in)  :: T_old(Nnx, Nny, Nnz)
    real(dp), intent(out) :: rhs(Nnx, Nny, Nnz)
    real(dp), intent(in)  :: laser_x, laser_y
    logical,  intent(in)  :: laser_on

    real(dp) :: Me_dt(8,8)
    real(dp) :: xe(8), fe(8)
    real(dp) :: q_peak, q_laser, q_conv, q_rad, T_local
    real(dp) :: xgp, ygp, dist2, q_total
    integer  :: ie, je, ke, a, b, color, ic, jc, kc
    integer  :: di, dj, dk, ln, fn
    integer  :: face_nodes(4)

    q_peak = 2.0_dp * eta * P_laser / (PI * rb * rb)
    Me_dt = 0.0_dp

    ! Precompute Me/dt element matrix (same structure as Ae but mass only)
    ! Actually just extract from the init: Me = (Ae - Ke) * dt, so Me/dt = Ae - Ke
    ! But simpler: recompute Me in init. For now, compute Me_dt here.
    ! Me_dt(a,b) = rho*Cp * ∫ N_a*N_b dV / dt
    ! Since all elements are identical, this is the same for all.
    ! We already have Ae = Me/dt + Ke, and Ke is the stiffness part.
    ! Let me just recompute Me/dt from Ae and Ke.
    ! Actually, let me just do a fresh EBE loop with the mass part only.

    ! Simpler: compute Me/dt directly
    call compute_Me_dt(Me_dt)

    rhs = 0.0_dp

    ! Mass term: EBE assembly of Me/dt * T_old
    do color = 0, 7
      ic = mod(color, 2); jc = mod(color/2, 2); kc = mod(color/4, 2)
      !$omp parallel do collapse(3) default(shared) private(ie,je,ke,xe,fe,a,b,di,dj,dk,ln)
      do ke = 1 + kc, Nz, 2
        do je = 1 + jc, Ny, 2
          do ie = 1 + ic, Nx, 2
            ! Gather FULL T_old (including Dirichlet)
            do dk = 0, 1; do dj = 0, 1; do di = 0, 1
              ln = 1 + di + 2*dj + 4*dk
              xe(ln) = T_old(ie+di, je+dj, ke+dk)
            end do; end do; end do

            ! Local matvec: fe = Me_dt * xe
            fe = 0.0_dp
            do b = 1, 8
              do a = 1, 8
                fe(a) = fe(a) + Me_dt(a,b) * xe(b)
              end do
            end do

            ! Scatter-add (all nodes)
            do dk = 0, 1; do dj = 0, 1; do di = 0, 1
              ln = 1 + di + 2*dj + 4*dk
              rhs(ie+di, je+dj, ke+dk) = rhs(ie+di, je+dj, ke+dk) + fe(ln)
            end do; end do; end do
          end do
        end do
      end do
      !$omp end parallel do
    end do

    ! Dirichlet bottom (overwrite after assembly)
    rhs(:,:,1) = T0

    ! ============================================================
    ! Surface flux on top face: F_i = ∫ q*N_i dS
    ! Loop over top-face elements (ke=Nz)
    ! Top face nodes are local 5,6,7,8 (dk=1 nodes)
    ! ============================================================
    ! Face local ordering → element local node:
    ! face(1)=(-1,-1) → (di=0,dj=0,dk=1) = elem 5
    ! face(2)=(+1,-1) → (di=1,dj=0,dk=1) = elem 6
    ! face(3)=(+1,+1) → (di=1,dj=1,dk=1) = elem 8  (NOT 7!)
    ! face(4)=(-1,+1) → (di=0,dj=1,dk=1) = elem 7  (NOT 8!)
    face_nodes = (/ 5, 6, 8, 7 /)  ! corrected mapping

    !$omp parallel do collapse(2) default(shared) &
    !$omp   private(ie,je,fn,xgp,ygp,dist2,T_local,q_laser,q_conv,q_rad,q_total,fe,di,dj,ln,a)
    do je = 1, Ny
      do ie = 1, Nx
        fe = 0.0_dp  ! only use entries 5-8

        do fn = 1, 4
          ! Physical position of face Gauss point
          xgp = (ie - 1) * dx + top_face_xgp(fn, 1)
          ygp = (je - 1) * dy + top_face_xgp(fn, 2)

          ! Interpolate T_old at Gauss point for convection/radiation
          ! Face local nodes: 1→(0,0), 2→(1,0), 3→(1,1), 4→(0,1)
          T_local = top_face_N(1, fn) * T_old(ie,   je,   Nnz) &
                  + top_face_N(2, fn) * T_old(ie+1, je,   Nnz) &
                  + top_face_N(3, fn) * T_old(ie+1, je+1, Nnz) &
                  + top_face_N(4, fn) * T_old(ie,   je+1, Nnz)

          ! Heat flux
          q_laser = 0.0_dp
          if (laser_on) then
            dist2 = (xgp - laser_x)**2 + (ygp - laser_y)**2
            q_laser = q_peak * exp(-2.0_dp * dist2 / (rb * rb))
          end if
          q_conv = h_conv * (T0 - T_local)
          q_rad  = SB_const * emissivity * (T0**4 - T_local**4)
          q_total = q_laser + q_conv + q_rad

          ! Accumulate nodal load: F_a += q * N_a * w * det_J_face
          do a = 1, 4
            fe(face_nodes(a)) = fe(face_nodes(a)) + q_total * top_face_N(a, fn) * top_face_w(fn)
          end do
        end do

        ! Scatter surface load to global nodes
        ! face_nodes 5,6,7,8 map to (di,dj,dk=1):
        ! 5=(0,0,1), 6=(1,0,1), 7=(1,1,1), 8=(0,1,1)
        ! Face node mapping: face_local → element_local → (di,dj)
        ! face 1 → elem 5 → (0,0), face 2 → elem 6 → (1,0)
        ! face 3 → elem 8 → (1,1), face 4 → elem 7 → (0,1)
        rhs(ie,   je,   Nnz) = rhs(ie,   je,   Nnz) + fe(5)  ! face node 1
        rhs(ie+1, je,   Nnz) = rhs(ie+1, je,   Nnz) + fe(6)  ! face node 2
        rhs(ie+1, je+1, Nnz) = rhs(ie+1, je+1, Nnz) + fe(8)  ! face node 3 → elem 8!
        rhs(ie,   je+1, Nnz) = rhs(ie,   je+1, Nnz) + fe(7)  ! face node 4 → elem 7!
      end do
    end do
    !$omp end parallel do

    ! Side wall fluxes (convection + radiation only, much weaker)
    ! x-faces (ie=0 and ie=Nx faces)
    call add_side_flux_x(T_old, rhs, 1)     ! x=0 face
    call add_side_flux_x(T_old, rhs, Nx)     ! x=Lx face
    ! y-faces
    call add_side_flux_y(T_old, rhs, 1)     ! y=0 face
    call add_side_flux_y(T_old, rhs, Ny)     ! y=Ly face

  end subroutine compute_thermal_rhs

  ! ============================================================
  ! Helper: compute Me/dt element matrix
  ! ============================================================
  subroutine compute_Me_dt(Me_dt)
    real(dp), intent(out) :: Me_dt(8,8)
    real(dp) :: gp(2), gw(2), xi, eta, zeta, w, det_J, N(8)
    real(dp) :: xi_n(8), eta_n(8), zeta_n(8)
    integer :: i, j, k, a, b

    xi_n   = (/ -1, 1,-1, 1,-1, 1,-1, 1 /)
    eta_n  = (/ -1,-1, 1, 1,-1,-1, 1, 1 /)
    zeta_n = (/ -1,-1,-1,-1, 1, 1, 1, 1 /)
    gp(1) = -1.0_dp/sqrt(3.0_dp); gp(2) = 1.0_dp/sqrt(3.0_dp)
    gw = 1.0_dp
    det_J = (dx/2.0_dp)*(dy/2.0_dp)*(dz/2.0_dp)

    Me_dt = 0.0_dp
    do k = 1, 2; do j = 1, 2; do i = 1, 2
      xi = gp(i); eta = gp(j); zeta = gp(k)
      w = gw(i)*gw(j)*gw(k)
      do a = 1, 8
        N(a) = (1+xi_n(a)*xi)*(1+eta_n(a)*eta)*(1+zeta_n(a)*zeta)/8.0_dp
      end do
      do b = 1, 8; do a = 1, 8
        Me_dt(a,b) = Me_dt(a,b) + rho*Cp*N(a)*N(b)*det_J*w / dt
      end do; end do
    end do; end do; end do
  end subroutine compute_Me_dt

  ! ============================================================
  ! Side wall flux helpers (simplified: nodal evaluation)
  ! ============================================================
  subroutine add_side_flux_x(T_old, rhs, ie_face)
    real(dp), intent(in)    :: T_old(Nnx, Nny, Nnz)
    real(dp), intent(inout) :: rhs(Nnx, Nny, Nnz)
    integer,  intent(in)    :: ie_face  ! element index (1 or Nx)

    integer :: je, ke, a, dj, dk, fn, node_i
    real(dp) :: fe(8), T_local, q_conv, q_rad, q_total
    real(dp) :: det_Jf, gp_val(2), gw_val(2), eta_f, zeta_f, Nf(4)

    gp_val(1) = -1.0_dp/sqrt(3.0_dp); gp_val(2) = 1.0_dp/sqrt(3.0_dp)
    gw_val = 1.0_dp
    det_Jf = (dy/2.0_dp)*(dz/2.0_dp)

    ! Determine which local face and which global i-index
    if (ie_face == 1) then
      node_i = 1   ! nodes at ie_face (left face: di=0)
    else
      node_i = ie_face + 1  ! nodes at ie_face+1 (right face: di=1)
    end if

    do ke = 2, Nz  ! skip ke=1 (Dirichlet bottom)
      do je = 1, Ny
        fe = 0.0_dp
        do fn = 1, 4
          eta_f  = gp_val(mod(fn-1,2)+1)
          zeta_f = gp_val((fn-1)/2+1)
          ! Interpolate T at face GP
          Nf(1) = (1-eta_f)*(1-zeta_f)/4; Nf(2) = (1+eta_f)*(1-zeta_f)/4
          Nf(3) = (1+eta_f)*(1+zeta_f)/4; Nf(4) = (1-eta_f)*(1+zeta_f)/4
          T_local = 0.0_dp
          do a = 1, 4
            dj = mod(a-1,2); dk = (a-1)/2
            T_local = T_local + Nf(a) * T_old(node_i, je+dj, ke+dk)
          end do
          q_conv = h_conv*(T0 - T_local)
          q_rad = SB_const*emissivity*(T0**4 - T_local**4)
          q_total = q_conv + q_rad
          do a = 1, 4
            dj = mod(a-1,2); dk = (a-1)/2
            rhs(node_i, je+dj, ke+dk) = rhs(node_i, je+dj, ke+dk) &
              + q_total * Nf(a) * gw_val(mod(fn-1,2)+1) * gw_val((fn-1)/2+1) * det_Jf
          end do
        end do
      end do
    end do
  end subroutine add_side_flux_x

  subroutine add_side_flux_y(T_old, rhs, je_face)
    real(dp), intent(in)    :: T_old(Nnx, Nny, Nnz)
    real(dp), intent(inout) :: rhs(Nnx, Nny, Nnz)
    integer,  intent(in)    :: je_face

    integer :: ie, ke, a, di, dk, fn, node_j
    real(dp) :: T_local, q_conv, q_rad, q_total
    real(dp) :: det_Jf, gp_val(2), gw_val(2), xi_f, zeta_f, Nf(4)

    gp_val(1) = -1.0_dp/sqrt(3.0_dp); gp_val(2) = 1.0_dp/sqrt(3.0_dp)
    gw_val = 1.0_dp
    det_Jf = (dx/2.0_dp)*(dz/2.0_dp)

    if (je_face == 1) then; node_j = 1; else; node_j = je_face + 1; end if

    do ke = 2, Nz
      do ie = 1, Nx
        do fn = 1, 4
          xi_f   = gp_val(mod(fn-1,2)+1)
          zeta_f = gp_val((fn-1)/2+1)
          Nf(1) = (1-xi_f)*(1-zeta_f)/4; Nf(2) = (1+xi_f)*(1-zeta_f)/4
          Nf(3) = (1+xi_f)*(1+zeta_f)/4; Nf(4) = (1-xi_f)*(1+zeta_f)/4
          T_local = 0.0_dp
          do a = 1, 4
            di = mod(a-1,2); dk = (a-1)/2
            T_local = T_local + Nf(a) * T_old(ie+di, node_j, ke+dk)
          end do
          q_conv = h_conv*(T0 - T_local)
          q_rad = SB_const*emissivity*(T0**4 - T_local**4)
          q_total = q_conv + q_rad
          do a = 1, 4
            di = mod(a-1,2); dk = (a-1)/2
            rhs(ie+di, node_j, ke+dk) = rhs(ie+di, node_j, ke+dk) &
              + q_total * Nf(a) * gw_val(mod(fn-1,2)+1) * gw_val((fn-1)/2+1) * det_Jf
          end do
        end do
      end do
    end do
  end subroutine add_side_flux_y

  ! ============================================================
  ! CG solver using EBE matvec
  ! ============================================================
  subroutine solve_thermal_cg(T_new, T_old, rhs)
    real(dp), intent(inout) :: T_new(Nnx, Nny, Nnz)
    real(dp), intent(in)    :: T_old(Nnx, Nny, Nnz)
    real(dp), intent(in)    :: rhs(Nnx, Nny, Nnz)

    real(dp) :: r(Nnx, Nny, Nnz), p(Nnx, Nny, Nnz), Ap(Nnx, Nny, Nnz)
    real(dp) :: rr_old, rr_new, pAp, alpha, beta, rnorm, bnorm
    integer  :: iter

    T_new = T_old
    T_new(:,:,1) = T0

    call ebe_matvec(T_new, Ap)
    r = rhs - Ap
    p = r
    rr_old = sum(r * r)
    bnorm = sqrt(sum(rhs * rhs))
    if (bnorm < 1.0e-30_dp) bnorm = 1.0_dp

    do iter = 1, cg_maxiter_thermal
      call ebe_matvec(p, Ap)
      pAp = sum(p * Ap)
      if (abs(pAp) < 1.0e-30_dp) exit
      alpha = rr_old / pAp

      T_new = T_new + alpha * p
      r = r - alpha * Ap
      rr_new = sum(r * r)
      rnorm = sqrt(rr_new)
      if (rnorm / bnorm < cg_tol_thermal) then
        !write(*,'(A,I5,A,ES10.3)') '  Thermal CG converged iter=', iter, ' res=', rnorm/bnorm
        exit
      end if

      beta = rr_new / rr_old
      p = r + beta * p
      rr_old = rr_new
    end do

    if (iter >= cg_maxiter_thermal) then
      write(*,'(A,I5,A,ES10.3)') '  WARNING: Thermal CG did NOT converge, iter=', iter, ' res=', rnorm/bnorm
    end if

    T_new(:,:,1) = T0
  end subroutine solve_thermal_cg

end module mod_thermal
