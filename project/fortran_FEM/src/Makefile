FC = gfortran
FFLAGS = -O3 -march=native -fopenmp -std=f2008 -Wall -Wno-unused-dummy-argument
LDFLAGS = -fopenmp

# Source files in dependency order
SRCS = mod_parameters.f90 \
       mod_phase.f90 \
       mod_thermal.f90 \
       mod_io.f90 \
       mod_mechanical.f90 \
       main.f90

OBJS = $(SRCS:.f90=.o)
MODS = $(wildcard *.mod)

TARGET = lpbf_sim

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(LDFLAGS) -o $@ $^

# Module dependencies
mod_parameters.o: mod_parameters.f90
	$(FC) $(FFLAGS) -c $<

mod_phase.o: mod_phase.f90 mod_parameters.o
	$(FC) $(FFLAGS) -c $<

mod_thermal.o: mod_thermal.f90 mod_parameters.o
	$(FC) $(FFLAGS) -c $<

mod_io.o: mod_io.f90 mod_parameters.o
	$(FC) $(FFLAGS) -c $<

mod_mechanical.o: mod_mechanical.f90 mod_parameters.o
	$(FC) $(FFLAGS) -c $<

main.o: main.f90 mod_parameters.o mod_thermal.o mod_phase.o mod_mechanical.o mod_io.o
	$(FC) $(FFLAGS) -c $<

clean:
	rm -f $(TARGET) *.o *.mod *.vts
