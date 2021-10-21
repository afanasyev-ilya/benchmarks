#include "common/lib.h"
#include "norm.h"

typedef double base_type;

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);

    base_type *x, *y;
    MemoryAPI::allocate_array(&x, size);
    MemoryAPI::allocate_array(&y, size);

    int iterations = LOC_REPEAT;

    init(x, y, size);

    size_t bytes_requested = size*sizeof(base_type)*2;
    size_t flops_requested = size*3;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);

	for(int i = 0; i < iterations; i++)
	{
        counter.start_timing();

        kernel(x, y, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
	}

    counter.print_average_counters(true);

    MemoryAPI::free_array(x);
    MemoryAPI::free_array(y);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

