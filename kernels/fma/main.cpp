#include "common/lib.h"

typedef float base_type;

#define INNER_FMA_ITERATIONS 10000
#define NUM_VECTORS 8

#ifdef __USE_INTEL__
#define SIMD_SIZE 16
#endif

#ifdef __USE_KUNPENG__
#define SIMD_SIZE 4
#endif

#include "fma.h"

void call_kernel(Parser &_parser)
{
    size_t size = 1024*1024;

    print_size("size", size*sizeof(base_type));

    base_type *in_data, *out_data;

    MemoryAPI::allocate_array(&in_data, size);
    MemoryAPI::allocate_array(&out_data, size);

    #ifndef METRIC_RUN
    size_t bytes_requested = ((size_t)size) * (2/*since 2 arrays*/ * sizeof(base_type));
    size_t flops_requested = size*NUM_VECTORS*INNER_FMA_ITERATIONS * 2 /*since FMA*/;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    #endif

    #ifdef METRIC_RUN
    int iterations = LOC_REPEAT * USUAL_METRICS_REPEAT;
    #else
    int iterations = LOC_REPEAT;
    #endif

    init(in_data, out_data, size);

    for(int i = 0; i < iterations; i++)
	{
        #ifndef METRIC_RUN
		counter.start_timing();
        re_init(in_data, out_data, size);
        #endif

		kernel(_parser.get_mode(), in_data, out_data, size);

        #ifndef METRIC_RUN
		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
        #endif
	}

	#ifndef METRIC_RUN
	counter.print_average_counters(true);
    #endif

    MemoryAPI::free_array(in_data);
    MemoryAPI::free_array(out_data);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    print_omp_info();

    call_kernel(parser);
    return 0;
}
