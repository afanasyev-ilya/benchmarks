def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{int(num)}{unit}{suffix}"
        num /= 1024.0
    return f"{int(num)}Yi{suffix}"


def get_timing_from_file_line(line, timings):
    for key, val in timings.items():
        if key in line:
            timings[key] = float(line.split(" ")[1])

    #timings["mem_efficiency"] = 100.0 * timings["avg_bw"]/dram_bandwidth
    return timings


def parse_timings(output):  # collect time, perf and BW values
    lines = output.splitlines()
    timings = {"avg_time": 0, "avg_bw": 0, "avg_flops": 0, "flops_per_byte": 0, "mem_efficiency": 0}
    for line in lines:
        timings = get_timing_from_file_line(line, timings)
    return timings
