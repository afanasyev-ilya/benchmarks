#include "common/lib.h"

typedef float base_type;

#include "fma.h"

void call_kernel(Parser &_parser)
{
    size_t size = 64*1024;

    print_size("size", size*sizeof(base_type));

    base_type *in_data, *out_data;

    MemoryAPI::allocate_array(&in_data, size);
    MemoryAPI::allocate_array(&out_data, size);

    #ifndef METRIC_RUN
    size_t bytes_requested = ((size_t)size) * (2 * sizeof(base_type))*1000;
    size_t flops_requested = size*16*2*1000;
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
