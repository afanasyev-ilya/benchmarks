#include "common/lib.h"

#define INNER_ITERATIONS 100
#define NUM_OPERS 16

#include "lehmer.h"

void call_kernel(Parser &parser)
{
    size_t size = 1024*1024;
    print_size("size", size*sizeof(float));

    int *in_data, *out_data;

    MemoryAPI::allocate_array(&in_data, size);
    MemoryAPI::allocate_array(&out_data, size);

    size_t bytes_requested = ((size_t)size) * (2/*since 2 arrays*/ * sizeof(float));
    size_t flops_requested = size*INNER_ITERATIONS*10 * 2;

    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(in_data, out_data, size);

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();
        re_init(in_data, out_data, size);

        kernel(in_data, out_data, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);

    MemoryAPI::free_array(in_data);
    MemoryAPI::free_array(out_data);
}


int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
