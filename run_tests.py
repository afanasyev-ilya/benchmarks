import os
import optparse
import shutil
import subprocess
from scripts.roofline import platform_specs
from scripts.helpers import *
#from scripts.plot import plot_gather_or_scatter
import json
import sys


linear_length = 800000000
max_threads = 64


# all upper borders are inclusive
exec_params = {"gather_ker": {"L1_latency": {"length": "3GB",
                                              "step": "1KB"},
                              "LLC_latency": {"length": "3GB",
                                              "step": "1MB"},
                              "DRAM_latency": {"length": "3GB",
                                               "step": "50MB"}},
               "scatter_ker": {"length": "3GB",
                               "min_small_size": "1KB",
                               "max_small_size": "512MB"},
               "fma_ker": [" -opt-mode gen -datatype flt",
                           " -opt-mode gen -datatype dbl",
                           " -opt-mode opt -datatype flt",
                           " -opt-mode opt -datatype dbl"],
               "compute_latency_ker": [""],
               "lehmer_ker": [""],
               "norm_alg": [" -large-size 2GB" ],
               "L1_bandwidth_ker": [" -mode 0",
                                    " -mode 1",
                                    " -mode 2",
                                    " -mode 3"],
               "gemm_alg": [" -size 10000 "],
               "primes_alg": [" -size 100000 "],
               "fib_ker": [" -size 100000000000 "],
               "dense_vec_ker": [" -large-size 4GB -mode 0",
                                 " -large-size 4GB -mode 1",
                                 " -large-size 4GB -mode 2",
                                 " -large-size 4GB -mode 3",
                                 " -large-size 4GB -mode 4",
                                 " -large-size 4GB -mode 5"],
               "interconnect_band_ker": [" -large-size 4GB -mode 0",
                                         " -large-size 4GB -mode 1",
                                         " -large-size 4GB -mode 2",
                                         " -large-size 4GB -mode 3"],
               "interconnect_latency_ker": [" -large-size 2GB -mode 0",
                                            " -large-size 2GB -mode 1",
                                            " -large-size 2GB -mode 2",
                                            " -large-size 2GB -mode 3"],
               "LLC_bandwidth_ker": [" -large-size 1MB ", " -large-size 3MB ", " -large-size 6MB "],
               "prefix_sum_alg": [" -large-size 23MB "],
               "stencil_1D_alg": [" -size 100000000 -r 7 -mode 0 "],
               "naive_transpose_alg": [" -size 25000 -mode 0 ",
                                       " -size 25000 -mode 1"],
               "sha1_alg": [ " -large-size 1GB "],
               "randgen_ker": [ " -large-size 1GB " ]}


generic_compute_bound = {"compute_latency_ker": "float", "scalar_ker": "scalar", "gemm_alg": "float",
                         "primes_alg": "scalar", "lehmer_ker": "scalar", "fib_ker": "scalar",
                         "sha1_alg": "scalar"}
generic_memory_bound = {"stencil_1D_alg": "L1", "dense_vec_ker": "DRAM", "L1_bandwidth_ker": "L1", "norm_alg": "DRAM",
                        "LLC_bandwidth_ker": "LLC", "prefix_sum_alg": "LLC",
                        "naive_transpose_alg": "DRAM"}


def run_benchmarks(benchmarks_list, options):
    testing_results = {"arch_name": options.arch}
    for benchmark_name in benchmarks_list:
        print("\n DOING " + benchmark_name + " BENCHMARK\n")
        if benchmark_name == "gather_ker":
            benchmark_gather(benchmark_name, exec_params[benchmark_name], options, testing_results)
        elif benchmark_name == "scatter_ker":
            benchmark_scatter(benchmark_name, exec_params[benchmark_name], options, testing_results)
        elif benchmark_name == "fma_ker":
            fma_benchmark(benchmark_name, exec_params[benchmark_name], options, testing_results)
        elif "interconnect" in benchmark_name:
            benchmark_interconnect(benchmark_name, exec_params[benchmark_name], options, testing_results)
        elif benchmark_name == "LLC_bandwidth_ker":
            benchmark_LLC_bandwidth(benchmark_name, exec_params[benchmark_name], options, testing_results)
        elif benchmark_name in generic_compute_bound.keys():
            generic_compute_bound_benchmark(benchmark_name, exec_params[benchmark_name], options,
                                            generic_compute_bound[benchmark_name], testing_results)
        elif benchmark_name in generic_memory_bound.keys():
            generic_memory_bound_benchmark(benchmark_name, exec_params[benchmark_name], options,
                                            generic_memory_bound[benchmark_name], testing_results)
        else:
            generic_benchmark(benchmark_name, [], options, testing_results)

    with open("./output/" + options.arch + ".json", 'w') as outfile:
        print(json.dumps(testing_results, indent=4))
        json.dump(testing_results, outfile)


def run_and_wait(cmd, options):
    print("Running " + cmd)
    os.environ['OMP_NUM_THREADS'] = str(options.threads)
    os.environ['OMP_PROC_BIND'] = "close"

    p = subprocess.Popen(cmd, shell=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode
    p_status = p.wait()
    string_output = output.decode("utf-8")
    return string_output


def generic_benchmark(benchmark_name, benchmark_parameters, options, testing_results):
    cmd = "./bin/" + benchmark_name
    string_output = run_and_wait(cmd, options)
    print(parse_timings(string_output))


def fma_benchmark(benchmark_name, benchmark_parameters, options, testing_results):
    flops = []
    for params in benchmark_parameters:
        cmd = "./bin/" + benchmark_name + " " + params
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)
        flops.append(timings["avg_flops"])

    peak_values = platform_specs[options.arch]
    max_flt_flops = max(flops[0], flops[2])
    max_dbl_flops = max(flops[1], flops[3])
    if flops[0] < flops[2]:
        print("Performance on optimized version is " + str("{:.1f}".format(flops[2]/flops[0])) +
              " times higher for floats.");
    if flops[1] < flops[3]:
        print("Performance on optimized version is " + str("{:.1f}".format(flops[1]/flops[3])) +
              " times higher for double.");
    print("SUSTAINED PERFORMANCE on FLOATS: " + str("{:.1f}".format(max_flt_flops)) + " GFLOP/s, " +
          str("{:.1f}".format(100.0*max_flt_flops/peak_values["peak_performances"]["float"])) + "% of peak[" +
          str(peak_values["peak_performances"]["float"]) + " GFLOP/s]")
    print("SUSTAINED PERFORMANCE on DOUBLES: " + str("{:.1f}".format(max_dbl_flops)) + " GFLOP/s, " +
          str("{:.1f}".format(100.0*max_dbl_flops/peak_values["peak_performances"]["double"])) + "% of peak[" +
          str(peak_values["peak_performances"]["double"]) + " GFLOP/s]")

    testing_results[benchmark_name] = {}
    testing_results[benchmark_name]["flt_perf"] = max_flt_flops
    testing_results[benchmark_name]["flt_efficiency"] = 100.0*max_flt_flops/peak_values["peak_performances"]["float"]
    testing_results[benchmark_name]["dbl_perf"] = max_dbl_flops
    testing_results[benchmark_name]["dbl_efficiency"] = 100.0*max_dbl_flops/peak_values["peak_performances"]["double"]


def benchmark_interconnect(benchmark_name, benchmark_parameters, options, testing_results):
    bandwidths = []
    for params in benchmark_parameters:
        cmd = "./bin/" + benchmark_name + " " + params
        old_threads = options.threads
        options.threads = get_cores_count()*2 # TODO sockets count
        string_output = run_and_wait(cmd, options)
        options.threads = old_threads
        timings = parse_timings(string_output)
        bandwidths.append(timings["avg_bw"])

    testing_results[benchmark_name] = {}
    testing_results[benchmark_name]["local_access_by_one_socket"] = bandwidths[0]
    testing_results[benchmark_name]["remote_access_by_one_socket"] = bandwidths[1]
    testing_results[benchmark_name]["local_access_by_both_sockets"] = bandwidths[2]
    testing_results[benchmark_name]["remote_access_by_both_sockets"] = bandwidths[3]


def generic_compute_bound_benchmark(benchmark_name, benchmark_parameters, options, roof_name, testing_results):
    testing_results[benchmark_name] = {}

    for params in benchmark_parameters:
        cmd = "./bin/" + benchmark_name + " " + params
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)
        perf = timings["avg_flops"]
        peak_values = platform_specs[options.arch]
        print("SUSTAINED PERFORMANCE : " + str("{:.1f}".format(perf)) + " GFLOP/s, " +
              str("{:.1f}".format(100.0*perf/peak_values["peak_performances"][roof_name])) + "% of peak[" +
              str(peak_values["peak_performances"][roof_name]) + " GFLOP/s]")
        testing_results[benchmark_name][params] = {}
        testing_results[benchmark_name][params]["performance"] = perf
        testing_results[benchmark_name][params]["efficiency"] = 100.0 *perf / peak_values["peak_performances"][roof_name]


def generic_memory_bound_benchmark(benchmark_name, benchmark_parameters, options, roof_name, testing_results):
    testing_results[benchmark_name] = {}

    for params in benchmark_parameters:
        cmd = "./bin/" + benchmark_name + " " + params
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)
        bw = timings["avg_bw"]
        peak_values = platform_specs[options.arch]
        print("SUSTAINED PERFORMANCE : " + str("{:.1f}".format(bw)) + " GB/s, " +
              str("{:.1f}".format(100.0*bw/peak_values["bandwidths"][roof_name])) + "% of peak[" +
              str(peak_values["bandwidths"][roof_name]) + " GB/s]")
        testing_results[benchmark_name][params] = {}
        testing_results[benchmark_name][params]["bandwidth"] = bw
        testing_results[benchmark_name][params]["efficiency"] = 100.0*bw/peak_values["bandwidths"][roof_name]


def benchmark_gather_region(benchmark_name, mode, testing_results, min_size, max_size, step_size):
    cur_small_size = min_size
    sizes = []
    bandwidths = []

    while cur_small_size <= max_size:
        formatted_small_size = b2t(cur_small_size)
        formatted_large_size = "2GB"

        cmd = "./bin/" + benchmark_name + " -small-size " + formatted_small_size + " -large-size " + \
              formatted_large_size
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)

        sizes.append(formatted_small_size)
        bandwidths.append(timings["avg_bw"])
        cur_small_size += step_size

    testing_results[benchmark_name + "_" + mode] = {}
    for cur_size, cur_band in zip(sizes, bandwidths):
        testing_results[benchmark_name + "_" + mode][cur_size] = cur_band


def benchmark_gather(benchmark_name, benchmark_parameters, options, testing_results):
    l1_size = get_cache_size("L1")
    l2_size = get_cache_size("L2")
    l3_size = get_cache_size("L3")
    benchmark_gather_region(benchmark_name, "L1_latency", testing_results, t2b("1KB"),  l1_size, t2b("1KB"))
    benchmark_gather_region(benchmark_name, "LLC_latency", testing_results, l2_size*2,  l3_size, t2b("1MB"))
    benchmark_gather_region(benchmark_name, "DRAM_latency", testing_results, l3_size*2,  t2b("1GB"), t2b("50MB"))


def benchmark_scatter(benchmark_name, benchmark_parameters, options, testing_results):
    cur_small_size = 1024
    sizes = []
    bandwidths = []
    while cur_small_size <= 1024 * 1024 * 512:
        formatted_small_size = b2t(cur_small_size)
        formatted_large_size = "1GB"

        cmd = "./bin/" + benchmark_name + " -small-size " + formatted_small_size + " -large-size " +\
              formatted_large_size
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)

        sizes.append(formatted_small_size)
        bandwidths.append(timings["avg_bw"])
        cur_small_size *= 2

    testing_results[benchmark_name] = {}
    for cur_size, cur_band in zip(sizes, bandwidths):
        testing_results[benchmark_name][cur_size] = cur_band


def benchmark_LLC_bandwidth(benchmark_name, benchmark_parameters, options, testing_results):
    prev_LLC_size = get_cache_size(get_prev_LLC_name(options.arch))
    LLC_size = get_cache_size(get_LLC_name(options.arch))
    num_arrays = 3
    one_array_size = (prev_LLC_size*2) / num_arrays

    max_bw = 0
    max_pos = 0

    while one_array_size * num_arrays < LLC_size:
        formatted_small_size = b2t(one_array_size)
        cmd = "./bin/" + benchmark_name + " -large-size " + formatted_small_size
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)

        if timings["avg_bw"] > max_bw:
            max_bw = timings["avg_bw"]
            max_pos = b2t(one_array_size * num_arrays)
        one_array_size *= 1.5

    testing_results[benchmark_name] = {"max_bandwidth": max_bw,
                                       "max_size": max_pos,
                                       "efficiency": 100.0*max_bw/platform_specs[options.arch]["bandwidths"]["LLC"]}


if __name__ == "__main__":
    # create .csv files
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-b', '--bench',
                      action="store", dest="bench",
                      help="specify benchmark to test (or all for testing all available benchmarks)", default="all")
    parser.add_option('-p', '--profile',
                      action="store_true", dest="profile",
                      help="use to collect hardware events available (including top-down analysis)", default=False)
    parser.add_option('-s', '--sockets',
                      action="store", dest="sockets",
                      help="set number of sockets used", default=1)
    parser.add_option('-c', '--compiler',
                      action="store", dest="compiler",
                      help="specify compiler used", default="g++")
    parser.add_option('-t', '--threads',
                      action="store", dest="threads",
                      help="specify thread number", default=get_cores_count())
    parser.add_option('-a', '--arch',
                      action="store", dest="arch",
                      help="specify target architecture", default=get_arch())

    options, args = parser.parse_args()

    if options.arch not in platform_specs:
        print("Unsupported target platform. Please add its specifications into roofline.py")
        exit()

    benchmarks_list = []
    if options.bench == "all":
        benchmarks_list = exec_params.keys()
        make_binaries()
    else:
        benchmarks_list = options.bench.split(",")
    run_benchmarks(benchmarks_list, options)
