#include "common/lib.h"

typedef double base_type;
typedef int index_type;

#include "interconnect_latency.h"

void call_kernel(Parser &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);
    print_size("size", size*sizeof(base_type));

    base_type *result, *accessed;
    index_type *rand_indexes;

    MemoryAPI::allocate_array(&result, size);
    MemoryAPI::allocate_array(&accessed, size);
    MemoryAPI::allocate_array(&rand_indexes, size);

    int mode = parser.get_mode();

    size_t bytes_requested = (size_t)(size) * (sizeof(base_type) * 2 + sizeof(index_type));
    size_t flops_requested = (size_t)(size) * 2;

    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(mode, result, accessed, rand_indexes, size, size);

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();

        kernel(mode, result, accessed, rand_indexes, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);

    MemoryAPI::free_array(result);
    MemoryAPI::free_array(accessed);
    MemoryAPI::free_array(rand_indexes);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
