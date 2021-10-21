#include "common/lib.h"
#include "gemm.h"

typedef float base_type;

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_size();
    float *A, *B, *C;
    MemoryAPI::allocate_array(&A, size*size);
    MemoryAPI::allocate_array(&B, size*size);
    MemoryAPI::allocate_array(&C, size*size);

    print_size("matrix size", size*size*sizeof(base_type));

    int iterations = LOC_REPEAT;

    init(A, B, C, size);

    size_t bytes_requested = size*size*sizeof(base_type)*3;
    size_t flops_requested = size*size*size*2;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);

	for(int i = 0; i < iterations; i++)
	{
        re_init(A, B, C, size);
        counter.start_timing();

        kernel(A, B, C, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
	}

    counter.print_average_counters(true);

    MemoryAPI::free_array(A);
    MemoryAPI::free_array(B);
    MemoryAPI::free_array(C);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

