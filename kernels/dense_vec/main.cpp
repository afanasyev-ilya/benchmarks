#include "common/lib.h"

typedef double base_type;

#include "dense_vec.h"

void call_kernel(Parser &parser)
{
    size_t size = parser.get_large_size() / sizeof(base_type);
    print_size("large_size", size*sizeof(base_type));

    base_type *a, *b, *c, *d, *e;

    MemoryAPI::allocate_array(&a, size);
    MemoryAPI::allocate_array(&b, size);
    MemoryAPI::allocate_array(&c, size);
    MemoryAPI::allocate_array(&d, size);
    MemoryAPI::allocate_array(&e, size);

    int elems_requested[6] = {2, 2, 3, 3, 4, 5};
    int ops_exec[6] = {1, 1, 1, 2, 2, 3};

    size_t bytes_requested = ((size_t)size) * (elems_requested[parser.get_mode()] * sizeof(base_type));
    size_t flops_requested = (size_t)size;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    init(a, b, c, d, e, size);

    for(int i = 0; i < iterations; i++)
	{
		counter.start_timing();
        //re_init(a, size);

		kernel(parser.get_mode(), a, b, c, d, e, size);

		counter.end_timing();
		counter.update_counters();
		counter.print_local_counters();
	}

	counter.print_average_counters(true);

    MemoryAPI::free_array(a);
    MemoryAPI::free_array(b);
    MemoryAPI::free_array(c);
    MemoryAPI::free_array(d);
    MemoryAPI::free_array(e);
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
