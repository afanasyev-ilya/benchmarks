#include "common/lib.h"

#define INNER_ITERATIONS 100
#define NUM_OPERS 16

#include "fib.h"

void call_kernel(Parser &parser)
{
    size_t size = parser.get_size();
    size_t *res;

    MemoryAPI::allocate_array(&res, omp_get_max_threads());

    size_t bytes_requested = omp_get_max_threads() * sizeof(size_t);
    size_t flops_requested = size*omp_get_max_threads()*1;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(res, size);

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();
        re_init(res, size);

        kernel(res, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);

    MemoryAPI::free_array(res);
}


int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
