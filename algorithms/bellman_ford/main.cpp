#define __USE_MULTICORE__

#define __PRINT_SAMPLES_PERFORMANCE_STATS__

#define INT_ELEMENTS_PER_EDGE 8.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 16*VECTOR_LENGTH

#include "common/lib.h"

void call_kernel(ParserBenchmark &parser)
{
    // prepare graph
    /*VGL_Graph graph(VECTOR_CSR_GRAPH);

    // prepare weights and distances
    int source_vertex = 0;
    VerticesArray<float> distances(graph, GATHER);
    EdgesArray<float> weights(graph);

    weights.set_all_random(MAX_WEIGHT);
    size_t bytes_requested = 0;
    size_t flops_requested = 0;

    int iterations = LOC_REPEAT;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);

	for(int i = 0; i < iterations; i++)
	{
        counter.start_timing();

        source_vertex = graph.select_random_nz_vertex(GATHER);
        ShortestPaths::vgl_dijkstra(graph, weights, distances, source_vertex, ALL_ACTIVE,
                                    PULL_TRAVERSAL);

        counter.end_timing();
        counter.update_counters();
        counter.print_local_counters();
	}

    counter.print_average_counters(true);*/
}

int main(int argc, char **argv)
{
    VGL_lib::VGL_RUNTIME::init_library(argc, argv);
    VGL_lib::VGL_RUNTIME::info_message("SSSP");

    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);

    VGL_lib::VGL_RUNTIME::finalize_library();
    return 0;
}

