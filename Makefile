COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	ifeq ($(MPI), true)
		ARCH_COMPILER=mpicxx
	else
		ARCH_COMPILER=g++
	endif
	Include_Path = -I ./
    Flags = -D __USE_MULTICORE__ $(MPI_Flags) -O3 -fopenmp -ftree-vectorize -std=c++17
  	Libraries = -O3 -fopenmp
  	Library_Path =
    ArchSuffix=_mc
endif

.DEFAULT_GOAL := all

all: create_folders

euclidean_norm: create_folders euclidean_norm.o
	$(ARCH_COMPILER) object_files/euclidean_norm.o $(Library_Path) $(Libraries) -o ./bin/euclidean_norm$(ArchSuffix)

euclidean_norm.o: algorithms/euclidean_norm/euclidean_norm_main.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c algorithms/euclidean_norm/euclidean_norm_main.cpp -o object_files/euclidean_norm.o

create_folders:
	-mkdir -p ./bin
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rm -f bin/*$(ArchSuffix)*
