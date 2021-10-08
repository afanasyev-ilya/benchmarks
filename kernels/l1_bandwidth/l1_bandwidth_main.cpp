#include "common/lib.h"

typedef float base_type;

#define INNER_LOADS 16

#include "l1_bandwidth.h"

#ifdef __USE_INTEL__
#define SIMD_SIZE 512
#endif

#ifdef __USE_KUNPENG__
#define SIMD_SIZE 128
#endif

void call_kernel(Parser &_parser)
{
    size_t size = _parser.get_length();
    int mode = _parser.get_mode();
    base_type *a = new base_type[size];
    base_type *b = new base_type[size];
    int *random_accesses = new int[RADIUS * omp_get_max_threads()];
    float **chunk_read = (float **)malloc(sizeof(float *) * omp_get_max_threads());
    float **chunk_write = (float **)malloc(sizeof(float *) * omp_get_max_threads());

    std::cout << 4 * sizeof(float) << std::endl;
    #ifndef METRIC_RUN
    size_t bytes_requested = (size_t) RADIUS * (SIMD_SIZE/sizeof(float)) * (size_t)INNER_LOADS * omp_get_max_threads();
    if (mode == 1 || mode == 5) {
        bytes_requested *= 2;
    }
    size_t flops_requested = (size_t) RADIUS * (size_t)INNER_LOADS * omp_get_max_threads();
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    #endif

    #ifdef METRIC_RUN
    int iterations = LOC_REPEAT * USUAL_METRICS_REPEAT;
    #else
    int iterations = LOC_REPEAT;
    #endif

    Init(mode, a, b, chunk_read, chunk_write, size, random_accesses);

    for(int i = 0; i < iterations; i++)
	{
        #ifndef METRIC_RUN
        std::cout << "cache anych" << std::endl;
		//locality::utils::CacheAnnil(3);
		counter.start_timing();
        #endif

		float val = Kernel(mode, chunk_read, chunk_write, size, random_accesses);

        #ifndef METRIC_RUN
		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
        #endif
	}
    
    #ifndef METRIC_RUN
	counter.print_average_counters(true);
    counter.print_bw();
    counter.print_flops();
    #endif

	delete []a;
	delete []b;
	free(chunk_read);
	free(chunk_write);
	delete[]random_accesses;
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
