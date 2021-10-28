#include "common/lib.h"
#include "rgb_cmyk.h"

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_size();
    char *array;
    /* size - area size of one frame, RGB uses 3 frames */
    MemoryAPI::allocate_array(&array, 3 * size);

    init(array, size);

    int div_flops = 1;
    size_t bytes_requested = size * 3 * sizeof(char);
    size_t flops_requested = size * 1.9 * sizeof(int); // TODO flops
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    int iterations = LOC_REPEAT;

    for(int i = 0; i < LOC_REPEAT; i++)
    {
        counter.start_timing();

        kernel(array, size);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
    }

    counter.print_average_counters(true);

    MemoryAPI::free_array(array);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}
