#include "common/lib.h"

#define INNER_FMA_ITERATIONS 10000
#define NUM_VECTORS 8

#ifdef __USE_INTEL__
#define __USE_AVX_512__
#define SIMD_SIZE_S 16
#define SIMD_SIZE_D 8
#endif

#ifdef __USE_KUNPENG_920__
#define __USE_ARM_NEON__
#define SIMD_SIZE_S 4
#define SIMD_SIZE_D 2
#endif

#include "fma.h"

void call_kernel_s(Parser &parser)
{
    size_t size = 1024*1024;
    cout << "DATA TYPE: " << "float" << endl;
    cout << "SIMD_SIZE: " << SIMD_SIZE_S << endl;
    print_size("size", size*sizeof(float));

    float *in_data, *out_data;

    MemoryAPI::allocate_array(&in_data, size);
    MemoryAPI::allocate_array(&out_data, size);

    size_t bytes_requested = ((size_t)size) * (2/*since 2 arrays*/ * sizeof(float));
    size_t flops_requested = size*NUM_VECTORS*INNER_FMA_ITERATIONS * 2 /*since FMA*/;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(in_data, out_data, size);

    for(int i = 0; i < 10; i++) // heat runs
    {
        re_init(in_data, out_data, size);
        kernel<float, SIMD_SIZE_S>(parser.get_opt_mode(), in_data, out_data, size);
    }

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();
        re_init(in_data, out_data, size);

        kernel<float, SIMD_SIZE_S>(parser.get_opt_mode(), in_data, out_data, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);

    MemoryAPI::free_array(in_data);
    MemoryAPI::free_array(out_data);
}

void call_kernel_d(Parser &parser)
{
    size_t size = 1024*1024;
    cout << "DATA TYPE: " << "double" << endl;
    cout << "SIMD_SIZE: " << SIMD_SIZE_D << endl;
    print_size("size", size*sizeof(double));

    double *in_data, *out_data;
    MemoryAPI::allocate_array(&in_data, size);
    MemoryAPI::allocate_array(&out_data, size);

    size_t bytes_requested = ((size_t)size) * (2/*since 2 arrays*/ * sizeof(double));
    size_t flops_requested = size*NUM_VECTORS*INNER_FMA_ITERATIONS * 2 /*since FMA*/;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(in_data, out_data, size);

    for(int i = 0; i < 10; i++) // heat runs
    {
        re_init(in_data, out_data, size);
        kernel<double, SIMD_SIZE_D>(parser.get_opt_mode(), in_data, out_data, size);
    }

    for(int i = 0; i < iterations; i++)
    {
        counter.start_timing();
        re_init(in_data, out_data, size);

        kernel<double, SIMD_SIZE_D>(parser.get_opt_mode(), in_data, out_data, size);

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

    if(parser.get_datatype() == __FLOAT__)
        call_kernel_s(parser);
    else if(parser.get_datatype() == __DOUBLE__)
        call_kernel_d(parser);
    return 0;
}
