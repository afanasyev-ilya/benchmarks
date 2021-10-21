#include "common/lib.h"
#include "sha1.h"

typedef unsigned char base_type;

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);

    base_type *x;
    MemoryAPI::allocate_array(&x, size);

    int iterations = LOC_REPEAT;

    init(x, size);

    size_t bytes_requested = size*sizeof(base_type)*2;
    size_t flops_requested = (size/64)*(4*20*(7/*each r op*/ + 8/*2 rol*/ + (11 + 4/*blk*/)));
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
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

