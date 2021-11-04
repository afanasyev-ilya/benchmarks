#include "common/lib.h"

typedef double base_type;
typedef int index_type;

#define SHARED_THREADS_NUM 4

#include "gather_shared.h"

void call_kernel(ParserBenchmark &parser)
{
    int threads_groups = omp_get_max_threads()/SHARED_THREADS_NUM;

    size_t large_size = parser.get_large_size() / sizeof(base_type);
    size_t wall_small_size = parser.get_small_size() / sizeof(base_type);
    size_t small_size = wall_small_size/threads_groups;

    print_size("large_size", large_size*sizeof(base_type));
    print_size("wall_small_size",  wall_small_size*sizeof(base_type));
    print_size("local small_size (shared between each K cores)", small_size*sizeof(base_type));

    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int cpu_num = sched_getcpu();
        printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
    }

    base_type *large_data;
    base_type **small_data;
    index_type *indexes;

    MemoryAPI::allocate_array(&large_data, large_size);
    MemoryAPI::allocate_array(&indexes, large_size);
    MemoryAPI::allocate_array(&small_data, threads_groups);
    for(int i = 0; i < threads_groups; i++)
    {
        MemoryAPI::allocate_array(&(small_data[i]), small_size);
    }

    size_t bytes_requested = ((size_t)large_size) * (2 * sizeof(base_type) + sizeof(index_type));
    size_t flops_requested = (size_t)large_size;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(large_data, indexes, small_data, large_size, small_size);

    for(int i = 0; i < iterations; i++)
	{
        re_init(small_data, small_size);

		counter.start_timing();

		kernel(parser.get_opt_mode(), large_data, indexes, small_data, large_size);

		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
	}

	counter.print_average_counters(true);

    MemoryAPI::free_array(large_data);
    MemoryAPI::free_array(indexes);
    for(int i = 0; i < threads_groups; i++)
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
