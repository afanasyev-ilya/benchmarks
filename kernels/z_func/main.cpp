#include "common/lib.h"
#include <cstring>
typedef char base_type;
typedef int index_type;

#include "z_func.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t string_size = 30000;
    vector<double> z(string_size);

    base_type *str;
    MemoryAPI::allocate_array(&str, string_size);

    int iterations = LOC_REPEAT;

    init(str, string_size);
    auto count = z_function_count(z, str, string_size);
    double flops_requested = count.first;
    double bytes_requested = count.second;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);

    for(int i = 0; i < iterations; i++)
	{
		counter.start_timing();

		kernel(z, str, string_size);

		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();

		//init(str, string_size);
        fill(z.begin(), z.end(), 0);
	}

	counter.print_average_counters(true);

    MemoryAPI::free_array(str);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
