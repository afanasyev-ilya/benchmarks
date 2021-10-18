#include "common/lib.h"

typedef float base_type;

#include "naive_transpose.h"

void call_kernel(Parser &parser)
{
    size_t size = parser.get_size();
    print_size("matrix size", size*size*sizeof(base_type));
    print_size("matrix row size", size*sizeof(base_type));

    base_type* a;
    base_type* b;
    MemoryAPI::allocate_array(&a, size*size);
    MemoryAPI::allocate_array(&b, size*size);

    size_t bytes_requested = 2.0 * (sizeof(base_type) * size*size);
    size_t flops_requested = size*size;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(a, b, size);

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();

        kernel(parser.get_mode(), a, b, 16, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);

    MemoryAPI::free_array(a);
    MemoryAPI::free_array(b);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

