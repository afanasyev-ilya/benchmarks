#include "common/lib.h"

typedef double base_type;

#define OUTER_ITERS 10000

#include "LLC_bandwidth.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);

    base_type *a, *b, *c;

    MemoryAPI::allocate_array(&a, size);
    MemoryAPI::allocate_array(&b, size);
    MemoryAPI::allocate_array(&c, size);
    print_size("total space for 3 arrays", 3*size*sizeof(base_type));

    size_t bytes_requested = ((size_t)size) * (3 * sizeof(base_type)) * OUTER_ITERS;
    size_t flops_requested = (size_t)size * OUTER_ITERS;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(a, b, c, size);

    for(int i = 0; i < iterations; i++)
	{
        //re_init(a, size);
		counter.start_timing();

		kernel(a, b, c, size);

		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
	}

	counter.print_average_counters(true);

    MemoryAPI::free_array(a);
    MemoryAPI::free_array(b);
    MemoryAPI::free_array(c);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
