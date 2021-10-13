#include "common/lib.h"
#include "primes.h"

void call_kernel(Parser &parser)
{
    size_t size = parser.get_size();
    int *array;
    MemoryAPI::allocate_array(&array, size);

    init(array, size);

    int div_flops = 1;
    size_t flops_requested = kernel_cnt(array, size) * div_flops;
    size_t bytes_requested = size * 1.9 * sizeof(int);
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    for(int i = 0; i < LOC_REPEAT; i++)
    {
        counter.start_timing();

        kernel(array, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);
    counter.print_bw();
    counter.print_flops();

    MemoryAPI::free_array(array);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

