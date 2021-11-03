#include "common/lib.h"

#ifdef __USE_INTEL__
typedef double base_type;
typedef int index_type;
#endif

#ifdef __USE_KUNPENG_920__
typedef double base_type;
typedef int index_type;
#endif

#ifdef __USE_A64FX__
typedef double base_type;
typedef long long index_type;
#endif

#include "gather_private.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t large_size = parser.get_large_size() / sizeof(base_type);
    size_t small_size = parser.get_small_size() / sizeof(base_type);

    int threads = omp_get_max_threads();
    print_size("large_size", large_size*sizeof(base_type));
    print_size("small_size", small_size*sizeof(base_type));
    print_size("small_size on each thread", (small_size/threads)*sizeof(base_type));

    base_type *large_data;
    base_type **small_data;
    index_type *indexes;

    MemoryAPI::allocate_array(&large_data, large_size);
    MemoryAPI::allocate_array(&indexes, large_size);
    MemoryAPI::allocate_array(&small_data, threads);
    for(int i = 0; i < threads; i++)
    {
        MemoryAPI::allocate_array(&(small_data[i]), small_size/threads);
    }

    size_t bytes_requested = ((size_t)large_size) * (2 * sizeof(base_type) + sizeof(index_type));
    size_t flops_requested = (size_t)large_size;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(large_data, indexes, small_data, large_size, small_size);
    cout << "init done" << endl;

    for(int i = 0; i < iterations; i++)
	{
		counter.start_timing();
        re_init(small_data, small_size);
        cout << "reinit done" << endl;

		kernel(large_data, indexes, small_data, large_size);

		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();

		/*int error_count = 0;
        for(size_t i = 0; i < large_size; i++)
        {
            if(small_data[indexes[i]] != large_data[i])
            {
                if(error_count < 20)
                    cout << large_data[i] << " vs " << small_data[indexes[i]] << endl;
                error_count++;
            }
        }
        cout << "Error count: " << error_count << endl;*/
	}

	counter.print_average_counters(true);

    MemoryAPI::free_array(large_data);
    MemoryAPI::free_array(indexes);
    for(int i = 0; i < threads; i++)
    {
        MemoryAPI::free_array(small_data[i]);
    }
    MemoryAPI::free_array(small_data);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
