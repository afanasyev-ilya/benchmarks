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

    /* 2 passes on initial array - when we count K and other - when we substract K */

    size_t bytes_requested = size * 3 * 2 * sizeof(char);

    /* Each size:
     * 3 op - index count (i*3), i*3 + 1, i*3 + 2
     * 6 op - min - substract and LE (assembler)
     * 6 op - index and K substract on second cycle
     * Total - 15 ops per one cell (3 channels in one cell)
     */

    size_t flops_requested = size * 15;

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
