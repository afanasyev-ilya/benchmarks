import os
import optparse
import shutil
import subprocess
from scripts.roofline import platform_specs
from scripts.helpers import b2t,parse_timings,get_cores_count,get_arch,make_binaries
#from scripts.plot import plot_gather_or_scatter
import json
import pprint
import xlsxwriter
import collections


SHEET_NAME = "main_stats"

first_gather_L1_row = 0
last_gather_L1_row = 0
first_gather_LLC_row = 0
last_gather_LLC_row = 0
first_gather_DRAM_row = 0
last_gather_DRAM_row = 0

first_scatter_row = 0
last_scatter_row = 0


ordered_benchmarks = ["fma_ker", "gemm_alg",  # compute -> vector -> unit
                      "compute_latency_ker",  # compute -> vector -> latency

                      "fib_ker",  # compute -> scalar -> unit
                      "lehmer_ker", "primes_alg",  # compute -> scalar -> latency

                      "L1_bandwidth_ker",  # memory -> bandwidth -> L1
                      "LLC_bandwidth_ker", "prefix_sum_alg",  # memory -> bandwidth -> LLC
                      "dense_vec_ker", "norm_alg",  # memory -> bandwidth -> DRAM
                      "interconnect_band_ker",  # memory -> bandwidth -> interconnect

                      "gather_ker_L1_latency",  # memory -> bandwidth -> L1
                      "gather_ker_LLC_latency",  # memory -> bandwidth -> LLC
                      "gather_ker_DRAM_latency",  # memory -> bandwidth -> DRAM
                      "interconnect_latency_ker",  # memory -> latency -> interconnect
]


def fma_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
    worksheet.write(row, 0, "FMA kernel", cell_format)
    worksheet.write(row, 1, "cpu-bound -> vector-bound -> unit-bound", cell_format)

    flt_stats = str("SUSTAINED PERFORMANCE on FLOATS:\n " + str("{:.1f}".format(data["flt_perf"])) + " GFLOP/s, " +
          str("{:.1f}".format(data["flt_efficiency"])) + "% of peak")
    worksheet.write(row, col, str(flt_stats), cell_format)
    dbl_stats = str("SUSTAINED PERFORMANCE on DOUBLEs:\n " + str("{:.1f}".format(data["dbl_perf"])) + " GFLOP/s, " +
          str("{:.1f}".format(data["dbl_efficiency"])) + "% of peak")
    worksheet.write(row + 1, col, str(dbl_stats), cell_format)
    return row + 2


def gemm_alg(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
    data = list(data.values())[0]

    worksheet.write(row, 0, "GEMM algorithm", cell_format)
    worksheet.write(row, 1, "cpu-bound -> vector-bound -> unit-bound", cell_format)

    flt_stats = str("SUSTAINED PERFORMANCE on FLOATS:\n " + str("{:.1f}".format(data["performance"])) + " GFLOP/s, " +
                    str("{:.1f}".format(data["efficiency"])) + "%\n (of float theoretical peak " +
                    str(platform_specs[arch_name]["peak_performances"]["float"]) + ")")
    worksheet.write(row, col, str(flt_stats), cell_format)
    return row + 1


def compute_latency_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
    data = list(data.values())[0]

    worksheet.write(row, 0, "compute latency kernel\n"
                            "performs sqrt and fma operations", cell_format)
    worksheet.write(row, 1, "cpu-bound -> vector-bound -> latency-bound", cell_format)

    flt_stats = str("Performance:\n " + str("{:.1f}".format(data["performance"])) + " GFLOP/s, " +
                    str("{:.1f}".format(data["efficiency"])) + "% of peak")
    worksheet.write(row, col, str(flt_stats), cell_format)
    return row + 1


def fib_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
    data = list(data.values())[0]

    worksheet.write(row, 0, "fibonachi kernel", cell_format)
    worksheet.write(row, 1, "cpu-bound -> scalar-bound -> unit-bound", cell_format)

    flt_stats = str("Performance:\n " + str("{:.1f}".format(data["performance"])) + " GFLOP/s, " +
                    str("{:.1f}".format(data["efficiency"])) + "% of peak")
    worksheet.write(row, col, str(flt_stats), cell_format)
    return row + 1


def primes_alg(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
    data = list(data.values())[0]

    worksheet.write(row, 0, "primes algorithm, which detects first N prime numbers. \n"
                            "unlike lemher kernel, also has workload balance issues.", cell_format)
    worksheet.write(row, 1, "cpu-bound -> scalar-bound -> latency-bound", cell_format)

    flt_stats = str("Performance:\n " + str("{:.1f}".format(data["performance"])) + " GFLOP/s, " +
                    str("{:.1f}".format(data["efficiency"])) + "%\n (of theoretical scalar peak " +
                    str(platform_specs[arch_name]["peak_performances"]["scalar"]) + ")")
    worksheet.write(row, col, str(flt_stats), cell_format)
    return row + 1


def lemher_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
    data = list(data.values())[0]

    worksheet.write(row, 0, "lemher kernel\n"
                            "used for random numbers generation, for example in rand() gcc implementation.", cell_format)
    worksheet.write(row, 1, "cpu-bound -> scalar-bound -> latency-bound", cell_format)

    flt_stats = str("Performance:\n " + str("{:.1f}".format(data["performance"])) + " GFLOP/s, " +
                    str("{:.1f}".format(data["efficiency"])) + "%\n (of theoretical scalar peak " +
                    str(platform_specs[arch_name]["peak_performances"]["scalar"]) + ")")
    worksheet.write(row, col, str(flt_stats), cell_format)
    return row + 1


def L1_bandwidth_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + 4, 0, "L1 bandwidth kernel\n"
                                              "uses vector instructions to load/store information inside arrays of 16 KB size.\n"
                                              "has multiple modes with instruction level parallelism enabled/disabled.", cell_format)
    worksheet.merge_range(row, 1, row + 4, 1, "memory-bound -> bandwidth-bound -> L1-bound", cell_format)

    num_rows = 0
    mode_comments = ["reads-only, instruction level parallelism available",
                     "reads and writes, instruction level parallelism available",
                     "reads-only, no instruction level parallelism available",
                     "reads-only, array is accessed with random offsets"]
    for exec_param in list(data.values()):
        worksheet.write(row + num_rows, col, "mode: " + mode_comments[num_rows], cell_format)
        bw_stats = str("Bandwidth:\n " + str("{:.1f}".format(exec_param["bandwidth"])) + " GB/s, " +
                        str("{:.1f}".format(exec_param["efficiency"])) + "%\n (of theoretical L1 bandwidth " +
                        str(platform_specs[arch_name]["bandwidths"]["L1"]) + ")")
        worksheet.write(row + num_rows, col + 1, str(bw_stats), cell_format)
        num_rows += 1
    return row + num_rows


def dense_vec_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + 5, 0, "dense vectors kernel (DAXPY)", cell_format)
    worksheet.merge_range(row, 1, row + 5, 1, "memory-bound -> bandwidth-bound -> DRAM-bound", cell_format)
    num_runs = 0

    num_vectors = ["2", "2", "3", "3", "4", "5"]
    for exec_param in list(data.values()):
        bw_stats = str("num vectors: " + num_vectors[num_runs] + "\n" + str("{:.1f}".format(exec_param["bandwidth"])) + " GB/s, " +
                    str("{:.1f}".format(exec_param["efficiency"])) + "% of peak")
        worksheet.write(row + num_runs, col, str(bw_stats), cell_format)
        num_runs += 1
    return row + num_runs


def interconnect_band_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + 4, 0, "interconnect kernel \n, "
                                              "performs either local or remote sequential memory accesses", cell_format)
    worksheet.merge_range(row, 1, row + 4, 1, "memory-bound -> bandwidth-bound -> interconnect-bound", cell_format)
    num_runs = 0

    mode_descriptions = ["one socket accesses local data",
                         "one socket accesses remote data",
                         "both sockets access local data",
                         "both sockets access remote data"]
    for bandwidth in list(data.values()):
        worksheet.write(row + num_runs, col, "mode: " + mode_descriptions[num_runs], cell_format)
        bw_stats = "bandwidth: " + str("{:.1f}".format(bandwidth)) + " GB/s"
        worksheet.write(row + num_runs, col + 1, str(bw_stats), cell_format)
        num_runs += 1
    return row + num_runs


def gather_ker_L1_latency(worksheet, row, col, name, data, arch_name):
    global first_gather_L1_row
    global last_gather_L1_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + len(data) - 1, 0, "gather kernel L1 latency", cell_format)
    worksheet.merge_range(row, 1, row + len(data) - 1, 1, "memory -> bandwidth -> L1 latency", cell_format)

    shift = 0
    for size, bw in data.items():
        worksheet.write(row + shift, col, str(size), cell_format)
        worksheet.write(row + shift, col + 1, int(bw), cell_format)
        shift += 1

    first_gather_L1_row = row
    last_gather_L1_row = row + len(data) - 1

    return row + len(data)


def gather_ker_LLC_latency(worksheet, row, col, name, data, arch_name):
    global first_gather_LLC_row
    global last_gather_LLC_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + len(data) - 1, 0, "gather kernel LLC latency", cell_format)
    worksheet.merge_range(row, 1, row + len(data) - 1, 1, "memory -> bandwidth -> LLC latency", cell_format)

    shift = 0
    for size, bw in data.items():
        worksheet.write(row + shift, col, str(size), cell_format)
        worksheet.write(row + shift, col + 1, int(bw), cell_format)
        shift += 1

    first_gather_LLC_row = row
    last_gather_LLC_row = row + len(data) - 1

    return row + len(data)


def gather_ker_DRAM_latency(worksheet, row, col, name, data, arch_name):
    global first_gather_DRAM_row
    global last_gather_DRAM_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + len(data) - 1, 0, "gather kernel DRAM latency", cell_format)
    worksheet.merge_range(row, 1, row + len(data) - 1, 1, "memory -> bandwidth -> DRAM latency", cell_format)

    shift = 0
    for size, bw in data.items():
        worksheet.write(row + shift, col, str(size), cell_format)
        worksheet.write(row + shift, col + 1, int(bw), cell_format)
        shift += 1

    first_gather_DRAM_row = row
    last_gather_DRAM_row = row + len(data) - 1

    return row + len(data)


def scatter_ker(worksheet, row, col, name, data, arch_name):
    global first_scatter_row
    global last_scatter_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    worksheet.merge_range(row, 0, row + len(data) - 1, 0, "scatter kernel", cell_format)
    worksheet.merge_range(row, 1, row + len(data) - 1, 1, "aimed to benchmark ...", cell_format)

    shift = 0
    for size, bw in data.items():
        worksheet.write(row + shift, col, str(size), cell_format)
        worksheet.write(row + shift, col + 1, int(bw), cell_format)
        shift += 1

    first_scatter_row = row
    last_scatter_row = row + len(data) - 1

    return row + len(data)


def create_diagrams(worksheet, arch_name, first_row, last_row, col, diag_row, diag_col, name):
    # Create a new chart object.
    chart = workbook.add_chart({'type': 'line'})

    # Add a series to the chart.
    chart.add_series({
        'categories': [SHEET_NAME, first_row, col, last_row, col],
        'values':     [SHEET_NAME, first_row, col + 1, last_row, col + 1],
        'line':       {'color': 'red'},
        'name': arch_name})

    chart.set_title({'name': name})

    # Insert the chart into the worksheet.
    worksheet.insert_chart(diag_row, diag_col, chart)


def add_stats_to_table(testing_results, worksheet, position):
    print(json.dumps(testing_results, indent=4))

    arch_name = testing_results["arch_name"]
    worksheet.write(0, position, arch_name)

    row = 1
    for bench_name in ordered_benchmarks:

        bench_data = testing_results[bench_name]
        if bench_name == "arch_name":
            print("processing arch " + bench_data)
            row += 1
        elif bench_name in globals():
            row = globals()[bench_name](worksheet, row, position, bench_name, bench_data, arch_name) + 1
    return worksheet


if __name__ == "__main__":
    # create .csv files
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-f', '--files',
                      action="store", dest="files",
                      help="specify file names with performance stats ", default=None)

    options, args = parser.parse_args()

    list_of_files = []
    if options.files is not None:
        list_of_files = options.files.split(",")
    else:
        print("Error! At least one file name should be specified")

    workbook = xlsxwriter.Workbook('./output/arch_comparison.xlsx')
    worksheet = workbook.add_worksheet(SHEET_NAME)

    worksheet.set_column(0, 0, 25)
    worksheet.set_column(1, 1, 25)

    for index, file_name in enumerate(list_of_files):
        with open(file_name, 'r') as f:
            data = f.read()
        testing_results = json.loads(data)
        arch_name = testing_results["arch_name"]

        worksheet.set_column(3*index + 1, 3*index + 3, 25)
        add_stats_to_table(testing_results, worksheet, 3*index + 2)
        create_diagrams(worksheet, arch_name, first_gather_L1_row, last_gather_L1_row, 3*index + 2, 1, (index + 1)*10,
                        "L1 latency")
        create_diagrams(worksheet, arch_name, first_gather_LLC_row, last_gather_LLC_row, 3*index + 2, 20,
                        (index + 1)*10, "LLC latency")
        create_diagrams(worksheet, arch_name, first_gather_DRAM_row, last_gather_DRAM_row, 3*index + 2, 40,
                        (index + 1)*10, "DRAM latency")
    workbook.close()

