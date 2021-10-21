#include "common/lib.h"
#include "prefix_sum.h"

typedef double base_type;

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);
    print_size("large_size", size*sizeof(base_type));

    base_type *x;
    MemoryAPI::allocate_array(&x, size);

    int NTHREADS = omp_get_max_threads();
    size_t bytes_requested = (1 + (3 * size) + ((2 + NTHREADS) * (NTHREADS + 1) / 2)) * sizeof(double);
    size_t flops_requested = ((2 * size) + ((2 + NTHREADS) * (NTHREADS + 1) / 2)) * sizeof(double);
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(x, size);

	for(int i = 0; i < iterations; i++)
	{
        counter.start_timing();

        kernel(x, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
	}

    counter.print_average_counters(true);

    MemoryAPI::free_array(x);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

