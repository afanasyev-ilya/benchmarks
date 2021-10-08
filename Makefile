COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	ifeq ($(MPI), true)
		ARCH_COMPILER=mpicxx
	else
		ARCH_COMPILER=g++
	endif
	Include_Path = -I ./

	ifeq ($(ARCH),kunpeng)
    	Flags = -D __USE_KUNPENG__ -O2 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -fopt-info-vec-all=report.lst -ffast-math -march=armv8.2-a -mtune=tsv110
    endif

    ifeq ($(ARCH),intel)
        Flags = -D __USE_INTEL__ -O2 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -fopt-info-vec-all=report.lst -ffast-math -march=skylake-avx512
    endif

  	Libraries = -fopenmp
  	Library_Path =
    ArchSuffix=_mc
endif

.DEFAULT_GOAL := all

all: create_folders

euclidean_norm: create_folders euclidean_norm.o
	$(ARCH_COMPILER) object_files/euclidean_norm.o $(Library_Path) $(Libraries) -o ./bin/euclidean_norm$(ArchSuffix)

spmv: create_folders spmv.o
	$(ARCH_COMPILER) object_files/spmv.o $(Library_Path) $(Libraries) -o ./bin/spmv$(ArchSuffix)

l1_bandwidth: create_folders l1_bandwidth.o
	$(ARCH_COMPILER) object_files/l1_bandwidth.o $(Library_Path) $(Libraries) -o ./bin/l1_bandwidth$(ArchSuffix)




spmv.o: algorithms/spmv/spmv_main.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c algorithms/spmv/spmv_main.cpp -o object_files/spmv.o

euclidean_norm.o: algorithms/euclidean_norm/euclidean_norm_main.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c algorithms/euclidean_norm/euclidean_norm_main.cpp -o object_files/euclidean_norm.o

l1_bandwidth.o: kernels/l1_bandwidth/l1_bandwidth_main.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c kernels/l1_bandwidth/l1_bandwidth_main.cpp -o object_files/l1_bandwidth.o


create_folders:
	-mkdir -p ./bin
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rm -f bin/*$(ArchSuffix)*
