import os
import optparse
import shutil
import subprocess
from scripts.roofline import kunpeng_characteristics,intel_xeon_characteristics,amd_epyc_characteristics
from scripts.helpers import sizeof_fmt,parse_timings
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
               "fma_ker": {}}


def run_benchmarks(benchmarks_list, options):
    for benchmark_name in benchmarks_list:
        if benchmark_name == "gather_ker" or benchmark_name == "scatter_ker":
            benchmark_gather_scatter(benchmark_name, exec_params[benchmark_name], options)
        else:
            generic_benchmark(benchmark_name, exec_params[benchmark_name], options)


def run_and_wait(cmd, options):
    print("Running " + cmd)
    my_env = os.environ.copy()
    my_env["OMP_NUM_THREADS"] = options.threads
    my_env["OMP_PROC_BIND"] = "close"
    os.environ['OMP_NUM_THREADS'] = str(options.threads)
    p = subprocess.Popen(cmd, shell=True, env=my_env,
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
    print(string_output)


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
                      help="specify thread number", default=None)

    options, args = parser.parse_args()

    benchmarks_list = []
    if options.bench == "all":
        benchmarks_list = exec_params.keys()
    else:
        benchmarks_list = options.bench.split(",")
    run_benchmarks(benchmarks_list, options)
