#ifdef __USE_INTEL__
#define SIMD_SIZE 512
#endif

#ifdef __USE_A64FX__
#define SIMD_SIZE 512
#endif

#ifdef __USE_KUNPENG_920__
#define SIMD_SIZE 128
#endif

#define GEN_SIMD_SIZE 16

#include "common/lib.h"

typedef float base_type;

#define INNER_LOADS 16

#define LINEAR_SIZE 524288000*2
#define RADIUS 10000000

#include "L1_bandwidth.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t size = LINEAR_SIZE;
    int mode = parser.get_mode();
    base_type *a = new base_type[size];
    base_type *b = new base_type[size];

    size_t rad_size = ((size_t) RADIUS) * omp_get_max_threads();
    int *random_accesses = new int[rad_size];
    float **chunk_read = (float **)malloc(sizeof(float *) * omp_get_max_threads());
    float **chunk_write = (float **)malloc(sizeof(float *) * omp_get_max_threads());

    print_size("RADIUS", RADIUS*sizeof(int));
    print_size("radius * threads", rad_size*sizeof(int));
    print_size("size", size);

    size_t bytes_requested = (size_t) RADIUS * (GEN_SIMD_SIZE/sizeof(float)) * (size_t)INNER_LOADS * omp_get_max_threads();
    if (mode == 1 || mode == 5) {
        bytes_requested *= 2;
    }
    size_t flops_requested = (size_t) RADIUS * (size_t)INNER_LOADS * omp_get_max_threads();
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    Init(mode, a, b, chunk_read, chunk_write, size, random_accesses);

    for(int i = 0; i < iterations; i++)
	{
		counter.start_timing();

		float val = kernel(mode, chunk_read, chunk_write, size, random_accesses);

		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
	}

	counter.print_average_counters(true);
    counter.print_bw();
    counter.print_flops();

	delete []a;
	delete []b;
	free(chunk_read);
	free(chunk_write);
	delete[]random_accesses;
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
