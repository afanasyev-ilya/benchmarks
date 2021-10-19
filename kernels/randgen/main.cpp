#include "common/lib.h"
#include "randgen.h"

typedef int base_type;

void call_kernel(Parser &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);

    base_type *x;
    MemoryAPI::allocate_array(&x, size);

    int iterations = LOC_REPEAT;
    
    size_t bytes_requested = size*sizeof(base_type);
    size_t flops_requested = size*12;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);

	for(int i = 0; i < iterations; i++)
	{
        counter.start_timing();

        kernel(x, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
	}

    counter.print_average_counters(true);

    MemoryAPI::free_array(x);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

