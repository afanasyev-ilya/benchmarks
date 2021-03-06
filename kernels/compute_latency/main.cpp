#include "common/lib.h"

#define INNER_FMA_ITERATIONS 1000
#define NUM_VECTORS 8

#define SIMD_SIZE_S 16

#include "compute_latency.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t size = 1024*1024;
    cout << "DATA TYPE: " << "float" << endl;
    cout << "SIMD_SIZE: " << SIMD_SIZE_S << endl;
    print_size("size", size*sizeof(float));

    float *in_data, *out_data;

    MemoryAPI::allocate_array(&in_data, size);
    MemoryAPI::allocate_array(&out_data, size);

    size_t bytes_requested = ((size_t)size) * (2/*since 2 arrays*/ * sizeof(float));
    size_t flops_requested = size*NUM_VECTORS*INNER_FMA_ITERATIONS * 1 /* FMA + sqrt*/;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(in_data, out_data, size);

    for(int i = 0; i < 10; i++) // heat runs
    {
        re_init(in_data, out_data, size);
        kernel<float>(in_data, out_data, size);
    }

    for(int i = 0; i < iterations; i++)
    {
        re_init(in_data, out_data, size);
        counter.start_timing();

        kernel<float>(in_data, out_data, size);

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
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
