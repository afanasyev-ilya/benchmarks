COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	ifeq ($(MPI), true)
		ARCH_COMPILER=mpicxx
	else
		ARCH_COMPILER=g++
	endif
	Include_Path = -I ./

	ifeq ($(ARCH),kunpeng)
    	Flags = -D __USE_KUNPENG__ -O2 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -fopt-info-vec-all=report.lst -ffast-math -march=armv8.2-a -mtune=tsv110
    endif

    ifeq ($(ARCH),intel)
        Flags = -D __USE_INTEL__ -O2 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -fopt-info-vec-all=report.lst -ffast-math -march=skylake-avx512
    endif

  	Libraries = -fopenmp
  	Library_Path =
    ArchSuffix=_mc
endif

.DEFAULT_GOAL := all

all: create_folders

%_ker: %_k.o create_folders
	$(ARCH_COMPILER) object_files/$< $(Library_Path) $(Libraries) -o ./bin/$@$(ArchSuffix)

%_alg: %_a.o create_folders
	$(ARCH_COMPILER) object_files/$< $(Library_Path) $(Libraries) -o ./bin/$@$(ArchSuffix)

%_k.o: kernels/%/main.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c $< -o object_files/$@

%_a.o: algorithms/%/main.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c $< -o object_files/$@

create_folders:
	-mkdir -p ./bin
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rm -f bin/*$(ArchSuffix)*
