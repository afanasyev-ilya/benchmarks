#include "common/lib.h"

typedef double base_type;
typedef int index_type;

#include "gather.h"

void print_size(string name, size_t len)
{
    string sizes[] = { "B", "KB", "MB", "GB", "TB" };
    int order = 0;
    while (len >= 1024 && order < 4) {
        order++;
        len = len/1024;
    }
    cout << name << " : " << len << " " << sizes[order] << endl;
}

void call_kernel(Parser &_parser)
{
    size_t large_size = _parser.get_large_size() / sizeof(base_type);
    size_t small_size = _parser.get_small_size() / sizeof(base_type);

    print_size("large_size", large_size*sizeof(base_type));
    print_size("small_size", small_size*sizeof(base_type));

    base_type *large_data, *small_data;
    index_type *indexes;

    MemoryAPI::allocate_array(&large_data, large_size);
    MemoryAPI::allocate_array(&indexes, large_size);
    MemoryAPI::allocate_array(&small_data, small_size);

    #ifndef METRIC_RUN
    size_t bytes_requested = ((size_t)large_size) * (2 * sizeof(base_type) + sizeof(index_type));
    size_t flops_requested = (size_t)large_size;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    #endif

    #ifdef METRIC_RUN
    int iterations = LOC_REPEAT * USUAL_METRICS_REPEAT;
    #else
    int iterations = LOC_REPEAT;
    #endif

    init(large_data, indexes, small_data, large_size, small_size);

    for(int i = 0; i < iterations; i++)
	{
        #ifndef METRIC_RUN
		counter.start_timing();
        re_init(small_data, small_size);
        #endif

		kernel(_parser.get_mode(), large_data, indexes, small_data, large_size);

        #ifndef METRIC_RUN
		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
        #endif
	}

	#ifndef METRIC_RUN
	counter.print_average_counters(true);
    #endif

    MemoryAPI::free_array(large_data);
    MemoryAPI::free_array(indexes);
    MemoryAPI::free_array(small_data);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
