#include <stdio.h>
#include <sys/time.h>

#include "locality.h"
#include "size.h"
#include <chrono>
#include "../../locutils_new/timers.h"

#include "math_norm.h"

typedef double base_type;

void CallKernel(int core_type)
{
    size_t size = (size_t)LENGTH;
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

        Kernel(core_type, x, y, size);

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

extern "C" int main()
{
    CallKernel((int)MODE);
    return 0;
}

