#define __USE_MULTICORE__

//#define __PRINT_SAMPLES_PERFORMANCE_STATS__

#define INT_ELEMENTS_PER_EDGE 8.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 16*VECTOR_LENGTH

#include "graph_library.h"

#define __VGL_USED__

#include "common/lib.h"

void gen_graph(VGL_Graph &graph, ParserBenchmark &parser)
{
    EdgesContainer edges_container;
    int v = pow(2.0, parser.get_graph_scale());
    //GraphGenerationAPI::R_MAT(edges_container, v, v * _parser.get_avg_degree(), 57, 19, 19, 5, _direction);
    GraphGenerationAPI::random_uniform(edges_container, v, v * 32, DIRECTED_GRAPH);
    edges_container.random_shuffle_edges();
    graph.import(edges_container);
}

void call_kernel(ParserBenchmark &parser)
{
    // prepare graph
    VGL_Graph graph(VECTOR_CSR_GRAPH);
    gen_graph(graph, parser);

    // prepare weights and distances
    int source_vertex = 0;
    VerticesArray<float> distances(graph, GATHER);
    EdgesArray<float> weights(graph);

    weights.set_all_random(MAX_WEIGHT);

    int iterations = LOC_REPEAT;

    performance_stats.reset_timers();
    double avg_perf = 0;
    double t1 = omp_get_wtime();
	for(int i = 0; i < iterations; i++)
	{
        source_vertex = graph.select_random_nz_vertex(GATHER);
        avg_perf += ShortestPaths::vgl_dijkstra(graph, weights, distances, source_vertex, ALL_ACTIVE,
                                                     PULL_TRAVERSAL)/iterations;
	}
    double t2 = omp_get_wtime();

    std::cout << std::fixed;
    std::cout << "avg_time: " << (t2 - t1)/LOC_REPEAT << " s" << std::endl;
    std::cout << "avg_bw: " << performance_stats.get_sustained_bandwidth() << " Gb/s" << std::endl;
    std::cout << "avg_flops: " << avg_perf << " MTEPS" << std::endl;
}

int main(int argc, char **argv)
{
    VGL_RUNTIME::init_library(argc, argv);
    VGL_RUNTIME::info_message("SSSP");

    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);

    VGL_RUNTIME::finalize_library();
    return 0;
}

