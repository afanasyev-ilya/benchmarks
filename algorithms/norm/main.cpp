#include "common/lib.h"
#include "norm.h"

typedef double base_type;

void call_kernel(Parser &parser)
{
    size_t size = parser.get_size();
    base_type * y = new base_type[size];
    base_type * x = new base_type[size];

    #ifdef METRIC_RUN
    int iterations = LOC_REPEAT * USUAL_METRICS_REPEAT;
    #else
    int iterations = LOC_REPEAT;
    #endif

    Init(x, y, size);

    #ifndef METRIC_RUN
    size_t bytes_requested = size*sizeof(base_type)*2;
    size_t flops_requested = size*3;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    #endif

	for(int i = 0; i < iterations; i++)
	{
        #ifndef METRIC_RUN
        Init(x, y, size);
		//locality::utils::CacheAnnil(0);

        counter.start_timing();
        #endif

        Kernel(x, y, size);

        #ifndef METRIC_RUN
        counter.end_timing();

        counter.update_counters();

        counter.print_local_counters();
        #endif
	}

    #ifndef METRIC_RUN
    counter.print_average_counters(true);
    #endif

    delete[]y;
    delete[]x;
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

