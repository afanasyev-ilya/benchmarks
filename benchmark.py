import os
import optparse
import shutil
import subprocess
from scripts.roofline import platform_specs
from scripts.helpers import sizeof_fmt,parse_timings,get_cores_count,get_arch
from scripts.plot import plot_gather_or_scatter


linear_length = 800000000
max_threads = 64


# all upper borders are inclusive
exec_params = {"gather_ker": {"length": "3GB",
                              "min_small_size": "1KB",
                              "max_small_size": "512MB"},
               "scatter_ker": {"length": "3GB",
                               "min_small_size": "1KB",
                               "max_small_size": "512MB"},
               "fma_ker": [" -opt-mode gen -datatype flt",
                           " -opt-mode gen -datatype dbl",
                           " -opt-mode opt -datatype flt",
                           " -opt-mode opt -datatype dbl"],
               "compute_latency_ker": "",
               "scalar_ker": "",
               "L1_bandwidth_ker": "",
               "gemm_alg": " -size 10000 "}


generic_compute_bound = {"compute_latency_ker": "float", "scalar_ker": "scalar", "gemm_alg": "float"}


def run_benchmarks(benchmarks_list, options):
    for benchmark_name in benchmarks_list:
        print("\n DOING " + benchmark_name + " BENCHMARK\n")
        if benchmark_name == "gather_ker" or benchmark_name == "scatter_ker":
            benchmark_gather_scatter(benchmark_name, exec_params[benchmark_name], options)
        elif benchmark_name == "fma_ker":
            fma_benchmark(benchmark_name, exec_params[benchmark_name], options)
        elif benchmark_name == "L1_bandwidth_ker":
            l1_bandwidth_benchmark(benchmark_name, exec_params[benchmark_name], options)
        elif benchmark_name in generic_compute_bound.keys():
            generic_compute_bound_benchmark(benchmark_name, exec_params[benchmark_name], options,
                                            generic_compute_bound[benchmark_name])
        else:
            generic_benchmark(benchmark_name, [], options)


def run_and_wait(cmd, options):
    print("Running " + cmd)
    os.environ['OMP_NUM_THREADS'] = str(options.threads)
    p = subprocess.Popen(cmd, shell=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode
    p_status = p.wait()
    string_output = output.decode("utf-8")
    return string_output


def generic_benchmark(benchmark_name, benchmark_parameters, options):
    cmd = "./bin/" + benchmark_name
    string_output = run_and_wait(cmd, options)
    print(parse_timings(string_output))


def fma_benchmark(benchmark_name, benchmark_parameters, options):
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


def generic_compute_bound_benchmark(benchmark_name, benchmark_parameters, options, roof_name):
    cmd = "./bin/" + benchmark_name + " " + benchmark_parameters
    string_output = run_and_wait(cmd, options)
    timings = parse_timings(string_output)
    perf = timings["avg_flops"]
    peak_values = platform_specs[options.arch]
    print("SUSTAINED PERFORMANCE : " + str("{:.1f}".format(perf)) + " GFLOP/s, " +
          str("{:.1f}".format(100.0*perf/peak_values["peak_performances"][roof_name])) + "% of peak[" +
          str(peak_values["peak_performances"][roof_name]) + " GFLOP/s]")


def l1_bandwidth_benchmark(benchmark_name, benchmark_parameters, options):
    for mode in range(0, 4):
        cmd = "./bin/" + benchmark_name + " -mode " + str(mode)
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)
        bw = timings["avg_bw"]
        peak_l1_bandwidth = platform_specs[options.arch]["bandwidths"]["L1"]
        print("SUSTAINED BANDWIDTH : " + str("{:.1f}".format(bw)) + " GB/s, " +
              str("{:.1f}".format(100.0*bw/peak_l1_bandwidth)) + "% of peak[" + str(peak_l1_bandwidth) + " GB/s]")


def benchmark_gather_scatter(benchmark_name, benchmark_parameters, options):
    cur_small_size = 1024
    sizes = []
    bandwidths = []
    while cur_small_size <= 1024 * 1024 * 512:
        formatted_small_size = sizeof_fmt(cur_small_size)
        formatted_large_size = "1GB"

        cmd = "./bin/" + benchmark_name + " -small-size " + formatted_small_size + " -large-size " +\
              formatted_large_size
        string_output = run_and_wait(cmd, options)
        timings = parse_timings(string_output)

        sizes.append(formatted_small_size)
        bandwidths.append(timings["avg_bw"])
        cur_small_size *= 2
    plot_gather_or_scatter(benchmark_name, sizes, bandwidths)


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
                      help="specify thread number", default=get_arch())

    options, args = parser.parse_args()

    if options.arch not in platform_specs:
        print("Unsupported target platform. Please add its specifications into roofline.py")
        exit()

    benchmarks_list = []
    if options.bench == "all":
        benchmarks_list = exec_params.keys()
    else:
        benchmarks_list = options.bench.split(",")
    run_benchmarks(benchmarks_list, options)
