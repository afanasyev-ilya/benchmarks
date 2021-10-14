COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	ARCH_COMPILER=g++
	Include_Path = -I ./

	ifeq ($(ARCH),kunpeng)
    	Flags = -D __USE_KUNPENG_920__ -O2 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -fopt-info-vec-all=report.lst -ffast-math -march=armv8.2-a -mtune=tsv110 #-Wno-aggressive-loop-optimizations
    endif

    ifeq ($(ARCH),intel)
        Flags = -D __USE_INTEL__ -O2 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -fopt-info-vec-all=report.lst -ffast-math -march=skylake-avx512
    endif

    ifeq ($(ARCH),a64fx)
        Flags = -D __USE_A64FX__ -O2 -std=c++17 -fopenmp -std=c++11 -Ofast -mcpu=native -fopenmp-simd -funroll-loops -fno-builtin -msve-vector-bits=512 -march=armv8.2-a+sve ${OPENMP} -Wno-write-strings
    endif

  	Libraries = -fopenmp
  	Library_Path =
endif

ifeq ($(COMPILER),icpc)
	ARCH_COMPILER=icpc
	Include_Path = -I ./

    ifeq ($(ARCH),intel)
        Flags = -D __USE_INTEL__ -O2 -std=c++17 -D NOFUNCCALL -qopt-report=1 -qopt-report-phase=vec -qopenmp -ffreestanding -qopt-streaming-stores=always -xCORE-AVX512 -mprefer-vector-width=512
    endif

  	Libraries = -fopenmp -mkl
  	Library_Path =
endif

.DEFAULT_GOAL := all

all: create_folders kernels algorithms

kernels: scatter_ker gather_ker fma_ker compute_latency_ker lehmer_ker L1_bandwidth_ker fib_ker dense_vec_ker interconnect_band_ker interconnect_latency_ker

algorithms: gemm_alg norm_alg stencil_1D_alg primes_alg

%_ker: %_k.o create_folders
	$(ARCH_COMPILER) object_files/$< $(Library_Path) $(Libraries) -o ./bin/$@

%_alg: %_a.o create_folders
	$(ARCH_COMPILER) object_files/$< $(Library_Path) $(Libraries) -o ./bin/$@

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
