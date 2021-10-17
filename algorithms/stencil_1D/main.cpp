#include "common/lib.h"

#ifdef __USE_INTEL__
#define __USE_AVX_512__
#define SIMD_SIZE 16
#endif

#ifdef __USE_KUNPENG_920__
#define __USE_ARM_NEON__
#define SIMD_SIZE 4
#endif

#include "stencil_1D.h"

typedef float base_type;

void call_kernel(Parser &parser)
{
    base_type *a;
    base_type *b;
    size_t size = parser.get_size();
    int radius = parser.get_radius();

    MemoryAPI::allocate_array(&a, size);
    MemoryAPI::allocate_array(&b, size);

    size_t bytes_requested = size * sizeof(base_type) * (2*radius + 1); // no *2 since only 1 array in inner loop
    size_t flops_requested = size * (2*radius + 1);
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(a, b, size);

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();

        kernel(parser.get_mode(), a, b, size, radius);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
        std::swap(a,b);
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

