program lpbf_simulation
  use mod_parameters
  use mod_thermal
  use mod_phase
  use mod_mechanical
  use mod_io
  implicit none

  ! Field arrays
  real(dp), allocatable :: T_old(:,:,:), T_new(:,:,:), rhs(:,:,:)
  real(dp), allocatable :: ux(:,:,:), uy(:,:,:), uz(:,:,:)
  integer,  allocatable :: phase(:,:,:)

  ! Stress/yield arrays for output
  real(dp), allocatable :: sxx_out(:,:,:), fplus_out(:,:,:)

  ! Laser position
  real(dp) :: laser_x, laser_y
  logical  :: laser_on

  ! Timing
  real(dp) :: time, T_max
  integer  :: step
  character(len=512) :: vtk_filename, vtk_dir

  ! Wall clock timing
  integer(8) :: clock_start, clock_end, clock_rate
  real(dp)   :: wall_total, wall_thermal, wall_mech
  integer(8) :: t1, t2

  ! ============================================================
  ! Allocate arrays
  ! ============================================================
  allocate(T_old(Nnx, Nny, Nnz))
  allocate(T_new(Nnx, Nny, Nnz))
  allocate(rhs(Nnx, Nny, Nnz))
  allocate(ux(Nnx, Nny, Nnz))
  allocate(uy(Nnx, Nny, Nnz))
  allocate(uz(Nnx, Nny, Nnz))
  allocate(phase(Nnx, Nny, Nnz))
  allocate(sxx_out(Nnx, Nny, Nnz))
  allocate(fplus_out(Nnx, Nny, Nnz))

  ! ============================================================
  ! Initialize
  ! ============================================================
  T_old = T0
  T_new = T0
  ux = 0.0_dp; uy = 0.0_dp; uz = 0.0_dp
  phase = PHASE_POWDER
  sxx_out = 0.0_dp
  fplus_out = 0.0_dp

  call init_thermal()
  call init_mechanical()

  laser_y = Ly / 2.0_dp
  wall_thermal = 0.0_dp
  wall_mech = 0.0_dp

  ! Create output directory
  vtk_dir = 'vtk_output'
  call execute_command_line('mkdir -p ' // trim(vtk_dir))

  ! Save initial state
  write(vtk_filename, '(A,A,I6.6,A)') trim(vtk_dir), '/fdm_', 0, '.vts'
  call write_vtk(trim(vtk_filename), T_old, ux, uy, uz, phase, sxx_out, fplus_out)

  call system_clock(clock_start, clock_rate)

  write(*,'(A)') '=========================================='
  write(*,'(A)') ' LPBF Thermal-Mechanical FDM Simulation'
  write(*,'(A)') '=========================================='
  write(*,'(A,I0,A,I0,A,I0)') ' Grid: ', Nnx, ' x ', Nny, ' x ', Nnz
  write(*,'(A,I0,A)') ' DOFs: ', Nnx*Nny*Nnz, ' (thermal), x3 (mechanical)'
  write(*,'(A,ES10.3,A,I0)') ' dt = ', dt, ', steps = ', num_steps
  write(*,'(A,ES10.3,A,ES10.3,A,ES10.3)') ' dx = ', dx, ', dy = ', dy, ', dz = ', dz
  write(*,'(A)') '=========================================='

  ! ============================================================
  ! Time loop
  ! ============================================================
  do step = 1, num_steps
    time = step * dt

    ! Update laser position (starts at 0.25*Lx, scans +x)
    ! Matches JAX-FEM: laser_center = [Lx*0.25 + vel*t, Ly/2, Lz]
    laser_x = 0.25_dp * Lx + laser_vel * time
    laser_on = (time < 0.5_dp * Lx / laser_vel)

    ! --- Thermal solve ---
    call system_clock(t1)
    call compute_thermal_rhs(T_old, rhs, laser_x, laser_y, laser_on)
    call solve_thermal_cg(T_new, T_old, rhs)
    call system_clock(t2)
    wall_thermal = wall_thermal + dble(t2 - t1) / dble(clock_rate)

    ! Mechanical solve every mech_interval steps
    if (mod(step, mech_interval) == 0) then
      ! Update phase only at mechanical steps (matches JAX-FEM)
      call update_phase(phase, T_new)

      call system_clock(t1)
      call solve_mechanical(ux, uy, uz, T_new, T_old, phase)
      call system_clock(t2)
      wall_mech = wall_mech + dble(t2 - t1) / dble(clock_rate)

      ! Output — save T_old (pre-solve T, matching JAX-FEM's sol_T_old convention)
      call get_stress_yield(sxx_out, fplus_out)
      write(vtk_filename, '(A,A,I6.6,A)') trim(vtk_dir), '/fdm_', step, '.vts'
      call write_vtk(trim(vtk_filename), T_old, ux, uy, uz, phase, sxx_out, fplus_out)

      write(*,'(A,I5,A,I0,A,F8.1,A,L1,A,ES10.3,A,ES10.3)') &
        ' Step ', step, '/', num_steps, '  T_max=', maxval(T_new), &
        '  laser=', laser_on, '  f_plus_max=', maxval(fplus_out), &
        '  sxx_max=', maxval(sxx_out)
    end if

    ! Progress report
    if (mod(step, 100) == 0 .and. mod(step, mech_interval) /= 0) then
      write(*,'(A,I5,A,I0,A,F8.1,A,L1)') &
        ' Step ', step, '/', num_steps, '  T_max=', maxval(T_new), &
        '  laser=', laser_on
    end if

    ! Advance
    T_old = T_new
  end do

  call system_clock(clock_end)
  wall_total = dble(clock_end - clock_start) / dble(clock_rate)

  write(*,'(A)') '=========================================='
  write(*,'(A)') ' Simulation complete.'
  write(*,'(A,F10.3,A)') ' Total wall time:    ', wall_total, ' s'
  write(*,'(A,F10.3,A)') ' Thermal solve time: ', wall_thermal, ' s'
  write(*,'(A,F10.3,A)') ' Mech solve time:    ', wall_mech, ' s'
  write(*,'(A,F10.3,A)') ' Avg thermal/step:   ', wall_thermal / num_steps * 1000.0_dp, ' ms'
  write(*,'(A,F10.3,A)') ' Avg mech/step:      ', wall_mech / (num_steps / mech_interval) * 1000.0_dp, ' ms'
  write(*,'(A)') '=========================================='

  ! Cleanup
  call cleanup_mechanical()
  deallocate(T_old, T_new, rhs, ux, uy, uz, phase, sxx_out, fplus_out)

end program lpbf_simulation
