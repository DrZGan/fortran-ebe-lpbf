module mod_io
  use mod_parameters
  implicit none

  private
  public :: write_vtk

contains

  ! ============================================================
  ! Write VTK structured grid (.vts) for ParaView
  ! ============================================================
  subroutine write_vtk(filename, T, ux, uy, uz, phase, sxx, fplus)
    character(len=*), intent(in) :: filename
    real(dp), intent(in) :: T(Nnx,Nny,Nnz)
    real(dp), intent(in) :: ux(Nnx,Nny,Nnz), uy(Nnx,Nny,Nnz), uz(Nnx,Nny,Nnz)
    integer,  intent(in) :: phase(Nnx,Nny,Nnz)
    real(dp), intent(in) :: sxx(Nnx,Nny,Nnz)
    real(dp), intent(in) :: fplus(Nnx,Nny,Nnz)

    integer :: iu, i, j, k
    character(len=20) :: fmt_r, fmt_i

    fmt_r = '(ES15.7)'
    fmt_i = '(I8)'

    open(newunit=iu, file=filename, status='replace', action='write')

    ! XML header
    write(iu, '(A)') '<?xml version="1.0"?>'
    write(iu, '(A)') '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">'
    write(iu, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A)') &
      '  <StructuredGrid WholeExtent="0 ', Nnx-1, ' 0 ', Nny-1, ' 0 ', Nnz-1, &
      ' 0 ', Nnx-1, ' 0 ', Nny-1, ' 0 ', Nnz-1, '">'
    write(iu, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A)') &
      '    <Piece Extent="0 ', Nnx-1, ' 0 ', Nny-1, ' 0 ', Nnz-1, &
      ' 0 ', Nnx-1, ' 0 ', Nny-1, ' 0 ', Nnz-1, '">'

    ! Points (coordinates)
    write(iu, '(A)') '      <Points>'
    write(iu, '(A)') '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">'
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          write(iu, '(3ES15.7)') (i-1)*dx, (j-1)*dy, (k-1)*dz
        end do
      end do
    end do
    write(iu, '(A)') '        </DataArray>'
    write(iu, '(A)') '      </Points>'

    ! Point data
    write(iu, '(A)') '      <PointData Scalars="Temperature" Vectors="Displacement">'

    ! Temperature
    write(iu, '(A)') '        <DataArray type="Float64" Name="Temperature" format="ascii">'
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          write(iu, fmt_r) T(i,j,k)
        end do
      end do
    end do
    write(iu, '(A)') '        </DataArray>'

    ! Displacement (vector)
    write(iu, '(A)') '        <DataArray type="Float64" Name="Displacement" ' // &
                      'NumberOfComponents="3" format="ascii">'
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          write(iu, '(3ES15.7)') ux(i,j,k), uy(i,j,k), uz(i,j,k)
        end do
      end do
    end do
    write(iu, '(A)') '        </DataArray>'

    ! Phase
    write(iu, '(A)') '        <DataArray type="Int32" Name="Phase" format="ascii">'
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          write(iu, fmt_i) phase(i,j,k)
        end do
      end do
    end do
    write(iu, '(A)') '        </DataArray>'

    ! Stress_xx
    write(iu, '(A)') '        <DataArray type="Float64" Name="Stress_xx" format="ascii">'
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          write(iu, fmt_r) sxx(i,j,k)
        end do
      end do
    end do
    write(iu, '(A)') '        </DataArray>'

    ! f_plus (yield function)
    write(iu, '(A)') '        <DataArray type="Float64" Name="f_plus" format="ascii">'
    do k = 1, Nnz
      do j = 1, Nny
        do i = 1, Nnx
          write(iu, fmt_r) fplus(i,j,k)
        end do
      end do
    end do
    write(iu, '(A)') '        </DataArray>'

    write(iu, '(A)') '      </PointData>'
    write(iu, '(A)') '    </Piece>'
    write(iu, '(A)') '  </StructuredGrid>'
    write(iu, '(A)') '</VTKFile>'

    close(iu)

  end subroutine write_vtk

end module mod_io
