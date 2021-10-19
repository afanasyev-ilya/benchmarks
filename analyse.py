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
import sys
import random


SHEET_NAME = "main_stats"
FIRST_COL = 4

first_gather_L1_row = 100000
last_gather_L1_row = 0
first_gather_LLC_row = 100000
last_gather_LLC_row = 0
first_gather_DRAM_row = 100000
last_gather_DRAM_row = 0

first_scatter_row = 0
last_scatter_row = 0

SHIFT = 100


ordered_benchmarks = ["fma_ker", "gemm_alg",  # compute -> vector -> unit
                      "compute_latency_ker",  # compute -> vector -> latency

                      "fib_ker",  # compute -> scalar -> unit
                      "lehmer_ker", "primes_alg",  # compute -> scalar -> latency

                      "L1_bandwidth_ker", "stencil_1D_alg",  # memory -> bandwidth -> L1
                      "LLC_bandwidth_ker", "prefix_sum_alg",  # memory -> bandwidth -> LLC
                      "dense_vec_ker", "norm_alg",  # memory -> bandwidth -> DRAM
                      "interconnect_band_ker",  # memory -> bandwidth -> interconnect

                      "gather_ker_L1_latency",  # memory -> bandwidth -> L1
                      "gather_ker_LLC_latency",  # memory -> bandwidth -> LLC
                      "gather_ker_DRAM_latency", "naive_transpose_alg",  # memory -> bandwidth -> DRAM
                      "interconnect_latency_ker",  # memory -> latency -> interconnect
]


def add_generic_compute_header(worksheet, row, col, test_name, test_category, modes_info):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 2*len(modes_info) - 1, 0, test_name, cell_format)
        worksheet.merge_range(row, 1, row + 2*len(modes_info) - 1, 1, test_category, cell_format)

        i = 0
        for test_info in modes_info:
            worksheet.merge_range(row + i, 2, row + i + 1, 2, test_info, cell_format)
            worksheet.write(row + i + 0, 3, "performance:", cell_format)
            worksheet.write(row + i + 1, 3, "efficiency:", cell_format)
            i += 2


def add_generic_memory_header(worksheet, row, col, test_name, test_category, modes_info):
    if col == FIRST_COL:
        cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
        worksheet.merge_range(row, 0, row + 2*len(modes_info) - 1, 0, test_name, cell_format)
        worksheet.merge_range(row, 1, row + 2*len(modes_info) - 1, 1, test_category, cell_format)

        i = 0
        for test_info in modes_info:
            worksheet.merge_range(row + i, 2, row + i + 1, 2, test_info, cell_format)
            worksheet.write(row + i + 0, 3, "bandwidth:", cell_format)
            worksheet.write(row + i + 1, 3, "efficiency:", cell_format)
            i += 2


def add_generic_compute_content(worksheet, row, col, modes_data):
    i = 0
    for data in modes_data:
        cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
        worksheet.write(row + i + 0, col, str("{:.1f}".format(data["performance"])) + " GFLOP/s", cell_format)
        worksheet.write(row + i + 1, col, str("{:.1f}".format(data["efficiency"])) + "%", cell_format)
        i += 2


def add_generic_memory_content(worksheet, row, col, modes_data):
    i = 0
    for data in modes_data:
        cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})
        worksheet.write(row + i + 0, col, str("{:.1f}".format(data["bandwidth"])) + " GB/s", cell_format)
        worksheet.write(row + i + 1, col, str("{:.1f}".format(data["efficiency"])) + "%", cell_format)
        i += 2


def fma_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 3, 0,  "FMA kernel", cell_format)
        worksheet.merge_range(row, 1, row + 3, 1, "cpu-bound -> vector-bound -> unit-bound", cell_format)
        worksheet.merge_range(row, 2, row + 3, 2, "Performs a sequence of FMA operations.", cell_format)

    worksheet.write(row, 3, "float perf:", cell_format)
    worksheet.write(row, col, str("{:.1f}".format(data["flt_perf"])) + " GFLOP/s", cell_format)

    worksheet.write(row + 1, 3, "float efficiency:", cell_format)
    worksheet.write(row + 1, col, str("{:.1f}".format(data["flt_efficiency"])) + "%", cell_format)

    worksheet.write(row + 2, 3, "double perf:", cell_format)
    worksheet.write(row + 2, col, str("{:.1f}".format(data["dbl_perf"])) + " GFLOP/s", cell_format)

    worksheet.write(row + 3, 3, "double efficiency:", cell_format)
    worksheet.write(row + 3, col, str("{:.1f}".format(data["dbl_efficiency"])) + "%", cell_format)

    return row + 4


def gemm_alg(worksheet, row, col, name, data, arch_name):
    description = ["Performs dense matrix-matrix multiplication in single preciesion."]
    add_generic_compute_header(worksheet, row, col, "GEMM algorithm", "cpu-bound -> vector-bound -> unit-bound", description)
    add_generic_compute_content(worksheet, row, col, list(data.values()))
    return row + 2


def fib_ker(worksheet, row, col, name, data, arch_name):
    description = ["Each core calculates it's onw Fibonachi sequence."]
    add_generic_compute_header(worksheet, row, col, "fibonachi kernel", "cpu-bound -> scalar-bound -> unit-bound", description)
    add_generic_compute_content(worksheet, row, col, list(data.values()))
    return row + 2


def compute_latency_ker(worksheet, row, col, name, data, arch_name):
    description = ["Performs a sequence of sqrt and fma operations."]
    add_generic_compute_header(worksheet, row, col, "compute latency kernel", "cpu-bound -> vector-bound -> latency-bound", description)
    add_generic_compute_content(worksheet, row, col, list(data.values()))
    return row + 2


def primes_alg(worksheet, row, col, name, data, arch_name):
    description = ["Detects first N prime numbers. Unlike lemher kernel is also affected by workload imbalance issues."]
    add_generic_compute_header(worksheet, row, col, "prime numbers detection algorithms", "cpu-bound -> scalar-bound -> latency-bound", description)
    add_generic_compute_content(worksheet, row, col, list(data.values()))
    return row + 2


def lehmer_ker(worksheet, row, col, name, data, arch_name):
    description = ["X_k+1= a* X_k mod m, used for random numbers generation, for example in rand() gcc implementation."]
    add_generic_compute_header(worksheet, row, col, "lehmer kernel", "cpu-bound -> scalar-bound -> latency-bound", description)
    add_generic_compute_content(worksheet, row, col, list(data.values()))
    return row + 2


def L1_bandwidth_ker(worksheet, row, col, name, data, arch_name):
    descriptions = ["reads-only, instruction level parallelism available",
                   "reads and writes, instruction level parallelism available",
                   "reads-only, no instruction level parallelism available",
                   "reads-only, array is accessed with random offsets"]
    add_generic_memory_header(worksheet, row, col, "L1 bandwidth kernel", "memory-bound -> bandwidth-bound -> L1-bound", descriptions)
    add_generic_memory_content(worksheet, row, col, list(data.values()))

    return row + 2*len(descriptions)


def dense_vec_ker(worksheet, row, col, name, data, arch_name):
    descriptions = ["2 vectors, copy",
                    "2 vectors, scale",
                    "3 vectors, add",
                    "3 vectors, triada",
                    "4 vectors, add",
                    "5 vectors, add"]

    add_generic_memory_header(worksheet, row, col, "dense vectors kernel", "memory-bound -> bandwidth-bound -> DRAM-bound",
                              descriptions)
    add_generic_memory_content(worksheet, row, col, list(data.values()))

    return row + 2*len(descriptions)


def norm_alg(worksheet, row, col, name, data, arch_name):
    description = ["Computing euclidean norm between 2 large vectors (4GB)."]

    add_generic_memory_header(worksheet, row, col, "computing norm algorithm", "memory-bound -> bandwidth-bound -> DRAM-bound",
                              description)
    add_generic_memory_content(worksheet, row, col, list(data.values()))

    return row + 2*len(description)


def interconnect_band_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 4, 0, "interconnect kernel", cell_format)
        worksheet.merge_range(row, 1, row + 4, 1, "memory-bound -> bandwidth-bound -> interconnect-bound", cell_format)
    num_runs = 0

    mode_descriptions = ["one socket accesses local data",
                         "one socket accesses remote data",
                         "both sockets access local data",
                         "both sockets access remote data"]
    for bandwidth in list(data.values()):
        worksheet.write(row + num_runs, 2, "mode: " + mode_descriptions[num_runs], cell_format)
        bw_stats = str("{:.1f}".format(bandwidth)) + " GB/s"
        worksheet.write(row + num_runs, 3, "bandwidth", cell_format)
        worksheet.write(row + num_runs, col, str(bw_stats), cell_format)
        num_runs += 1
    return row + num_runs


def gather_ker_L1_latency(worksheet, row, col, name, data, arch_name):
    global first_gather_L1_row
    global last_gather_L1_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 3, 0, "gather kernel L1 latency", cell_format)
        worksheet.merge_range(row, 1, row + 3, 1, "memory -> bandwidth -> L1 latency", cell_format)
        worksheet.merge_range(row, 2, row + 3, 2, "Indirectly accessed array is in [1KB, L1 size] range", cell_format)

    gather_common(row, col, worksheet, cell_format, data, 0)

    first_gather_L1_row = min(row, first_gather_L1_row)
    last_gather_L1_row = max(row + len(data) - 1, last_gather_L1_row)

    return row + 4


def gather_ker_LLC_latency(worksheet, row, col, name, data, arch_name):
    global first_gather_LLC_row
    global last_gather_LLC_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 3, 0, "gather kernel LLC latency", cell_format)
        worksheet.merge_range(row, 1, row + 3, 1, "memory -> bandwidth -> LLC latency", cell_format)
        worksheet.merge_range(row, 2, row + 3, 2, "Indirectly accessed array is in [L1/L2 size, LLC size] range", cell_format)

    gather_common(row, col, worksheet, cell_format, data, 1)

    first_gather_LLC_row = min(row, first_gather_LLC_row)
    last_gather_LLC_row = max(row + len(data) - 1, last_gather_LLC_row)

    return row + 4


def gather_ker_DRAM_latency(worksheet, row, col, name, data, arch_name):
    global first_gather_DRAM_row
    global last_gather_DRAM_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 3, 0, "gather kernel DRAM latency", cell_format)
        worksheet.merge_range(row, 1, row + 3, 1, "memory -> bandwidth -> DRAM latency", cell_format)
        worksheet.merge_range(row, 2, row + 3, 2, "Indirectly accessed array is in [LLC size, 2GB] range", cell_format)

    gather_common(row, col, worksheet, cell_format, data, 2)

    first_gather_DRAM_row = min(row, first_gather_DRAM_row)
    last_gather_DRAM_row = max(row + len(data) - 1, last_gather_DRAM_row)

    return row + 4


def gather_common(row, col, worksheet, cell_format, data, col_shift):
    max_bw = 0
    max_pos = 0
    min_bw = sys.float_info.max
    min_pos = 0
    for cur_size, cur_band in data.items():
        if cur_band > max_bw:
            max_bw = cur_band
            max_pos = cur_size
        if cur_band < min_bw:
            min_bw = cur_band
            min_pos = cur_size

    worksheet.write(row + 0, 3, "min bandwidth", cell_format)
    worksheet.write(row + 1, 3, "size at which min bandwidth achieved", cell_format)
    worksheet.write(row + 2, 3, "max bandwidth", cell_format)
    worksheet.write(row + 3, 3, "size at which max bandwidth achieved", cell_format)

    worksheet.write(row + 0, col, str(min_bw) + " GB/s ", cell_format)
    worksheet.write(row + 1, col, str(min_pos), cell_format)
    worksheet.write(row + 2, col, str(max_bw) + " GB/s ", cell_format)
    worksheet.write(row + 3, col, str(max_pos), cell_format)

    shift = 0
    for size, bw in data.items():
        worksheet.write(row + shift, SHIFT + 2*col_shift + col * 6, str(size), cell_format)
        worksheet.write(row + shift, SHIFT + 2*col_shift + col * 6 + 1, int(bw), cell_format)
        shift += 1


def scatter_ker(worksheet, row, col, name, data, arch_name):
    global first_scatter_row
    global last_scatter_row
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
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


def LLC_bandwidth_ker(worksheet, row, col, name, data, arch_name):
    cell_format = workbook.add_format({'text_wrap': True, 'valign': "top"})

    if col == FIRST_COL:
        worksheet.merge_range(row, 0, row + 3, 0, "LLC bandwidth", cell_format)
        worksheet.merge_range(row, 1, row + 3, 1, "memory-bound -> bandwidth-bound -> LLC-bound", cell_format)
        worksheet.merge_range(row, 2, row + 3, 2, "multiple operations over ", cell_format)

    worksheet.write(row, 3, "max bandwidth", cell_format)
    worksheet.write(row, col, str("{:.1f}".format(data["max_bandwidth"])) + " GB/s", cell_format)
    worksheet.write(row + 1, 3, "max efficiency", cell_format)
    worksheet.write(row + 1, col, str("{:.1f}".format(data["efficiency"])) + "%", cell_format)
    worksheet.write(row + 2, 3, "at size", cell_format)
    worksheet.write(row + 2, col, str(data["max_size"]), cell_format)

    return row + 3


def prefix_sum_alg(worksheet, row, col, name, data, arch_name):
    descriptions = ["Inclusive inplace parallel prefix sum algorithm."]
    add_generic_memory_header(worksheet, row, col, "prefix sum algorithm",
                              "memory-bound -> bandwidth-bound -> LLC-bound", descriptions)
    add_generic_memory_content(worksheet, row, col, list(data.values()))

    return row + 2*len(descriptions)


def naive_transpose_alg(worksheet, row, col, name, data, arch_name):
    descriptions = ["Simple matrix transpose algorithm. Sequential stores, strided loads.",
                    "Simple matrix transpose algorithm. Strided stores, sequential loads."]
    add_generic_memory_header(worksheet, row, col, "naive matrix transpose algorithm",
                              "memory-bound -> latency-bound -> DRAM-bound", descriptions)
    add_generic_memory_content(worksheet, row, col, list(data.values()))

    return row + 2*len(descriptions)


def create_diagrams(worksheet, arch_names, first_row, last_row, first_col, diag_row, diag_col, name, col_shift):
    # Create a new chart object.
    chart = workbook.add_chart({'type': 'line'})

    # Add a series to the chart.
    arch_col = 4

    for arch_name in arch_names:
        number_of_colors = 1
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
        chart.add_series({
            'categories': [SHEET_NAME, first_row, first_col + 6*arch_col, last_row, first_col + 6*arch_col],
            'values':     [SHEET_NAME, first_row, first_col + 6*arch_col + 1, last_row, first_col + 6*arch_col + 1],
            'line':       {'color': color[0]},
            'name': arch_name})
        arch_col += 1

    chart.set_title({'name': name})
    chart.set_size({'width': 420, 'height': 320})

    # Insert the chart into the worksheet.
    worksheet.insert_chart(diag_row, diag_col, chart)


def add_stats_to_table(testing_results, worksheet, position):
    #print(json.dumps(testing_results, indent=4))

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
    worksheet.set_column(1, 1, 20)
    worksheet.set_column(2, 2, 25)
    worksheet.set_column(3, 3, 15)

    arch_names = []
    for index, file_name in enumerate(list_of_files):
        with open(file_name, 'r') as f:
            data = f.read()
        testing_results = json.loads(data)
        arch_name = testing_results["arch_name"]

        worksheet.set_column(index + 4, index + 4, 20)
        add_stats_to_table(testing_results, worksheet, index + 4)
        arch_names.append(arch_name)

    create_diagrams(worksheet, arch_names, first_gather_L1_row, last_gather_L1_row, SHIFT + 2*0,
                    1, 10, "L1 latency", 0)
    create_diagrams(worksheet, arch_names, first_gather_LLC_row, last_gather_LLC_row, SHIFT + 2*1,
                    20, 10, "LLC latency", 1)
    create_diagrams(worksheet, arch_names, first_gather_DRAM_row, last_gather_DRAM_row, SHIFT + 2*2,
                    40, 10, "DRAM latency", 2)
    workbook.close()

