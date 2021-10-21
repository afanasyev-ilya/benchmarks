#include "common/lib.h"

typedef double base_type;

#include "interconnect_band.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);
    print_size("size", size*sizeof(base_type));

    base_type *a, *b, *c;

    MemoryAPI::allocate_array(&a, size);
    MemoryAPI::allocate_array(&b, size);
    MemoryAPI::allocate_array(&c, size);

    int mode = parser.get_mode();

    size_t bytes_requested = 0;
    size_t flops_requested = 0;
    if(mode == 0 || mode == 1) {
        bytes_requested = (size_t)(size / 2) * sizeof(base_type) * 3;
        flops_requested = (size_t)(size / 2) * 2;
    }
    if(mode == 2 || mode == 3) {
        bytes_requested = (size_t)(size) * sizeof(base_type) * 3;
        flops_requested = (size_t)(size) * 2;
    }
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(mode, a, b, c, size);

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();

        kernel(mode, a, b, c, size);

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
