#include "common/lib.h"

typedef double base_type;
typedef int index_type;

#define RADIUS_IN_ELEMENTS (size_t)((size_t)RADIUS * 1024 / sizeof(base_type))

#include "gather.h"

void call_kernel(Parser &_parser)
{
    size_t size = _parser.get_size();
    size_t small_size = _parser.get_radius();
    base_type *a, data;
    index_type *index;
    MemoryAPI::allocate_array(&a, size);
    MemoryAPI::allocate_array(&index, size);
    MemoryAPI::allocate_array(&data, small_size);

    #ifndef METRIC_RUN
    size_t bytes_requested = ((size_t)size) * (2 * sizeof(base_type) + sizeof(index_type));
    size_t flops_requested = (size_t)size;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    #endif

    #ifdef METRIC_RUN
    int iterations = LOC_REPEAT * USUAL_METRICS_REPEAT;
    Init(a, index, data, (size_t)LENGTH);
    #else
    int iterations = LOC_REPEAT;
    #endif

    for(int i = 0; i < iterations; i++)
	{
        #ifndef METRIC_RUN
		counter.start_timing();
        #endif

		kernel(_parser.get_mode(), a, index, data, size);

        #ifndef METRIC_RUN
		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
        #endif
	}

	#ifndef METRIC_RUN
	counter.print_average_counters(true);
    #endif

    MemoryAPI::free_array(a);
    MemoryAPI::free_array(index);
    MemoryAPI::free_array(data);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
