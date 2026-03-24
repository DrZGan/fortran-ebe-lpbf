module mod_phase
  use mod_parameters
  implicit none

  private
  public :: update_phase

contains

  ! ============================================================
  ! Update phase field based on temperature
  ! POWDER -> LIQUID when T > Tl
  ! LIQUID -> SOLID  when T < Tl
  ! SOLID is irreversible (stays SOLID)
  ! ============================================================
  subroutine update_phase(phase, T)
    integer,  intent(inout) :: phase(Nnx, Nny, Nnz)
    real(dp), intent(in)    :: T(Nnx, Nny, Nnz)

    integer :: i, j, k

    !$omp parallel do collapse(3) default(shared) private(i,j,k)
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          select case (phase(i,j,k))
          case (PHASE_POWDER)
            if (T(i,j,k) > Tl) then
              phase(i,j,k) = PHASE_LIQUID
            end if
          case (PHASE_LIQUID)
            if (T(i,j,k) < Tl) then
              phase(i,j,k) = PHASE_SOLID
            end if
          case (PHASE_SOLID)
            ! Irreversible: stays solid
            continue
          end select
        end do
      end do
    end do
    !$omp end parallel do

  end subroutine update_phase

end module mod_phase
