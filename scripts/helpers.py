import subprocess


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


def get_cores_count():  # returns number of sockets of target architecture
    output = subprocess.check_output(["lscpu"])
    cores = -1
    for item in output.decode().split("\n"):
        if "Core(s) per socket:" in item:
            cores_line = item.strip()
            cores = int(cores_line.split(":")[1])
        if "Ядер на сокет:" in item:
            cores_line = item.strip()
            cores = int(cores_line.split(":")[1])
    if cores == -1:
        raise NameError('Can not detect number of cores of target architecture')
    return cores


def get_sockets_count():  # returns number of sockets of target architecture
    output = subprocess.check_output(["lscpu"])
    cores = -1
    for item in output.decode().split("\n"):
        if "Socket(s)" in item:
            sockets_line = item.strip()
            sockets = int(sockets_line.split(":")[1])
        if "Сокетов:" in item:
            sockets_line = item.strip()
            sockets = int(sockets_line.split(":")[1])
    if sockets == -1:
        raise NameError('Can not detect number of cores of target architecture')
    return sockets


def get_threads_count():
    return get_sockets_count()*get_cores_count()


def get_arch():  # returns architecture, eigher kunpeng or intel_xeon
    architecture = "unknown"
    output = subprocess.check_output(["lscpu"])
    arch_line = ""
    vendor_line = ""
    for item in output.decode().split("\n"):
        if "Architecture" in item:
            arch_line = item.strip()
        if "Vendor" in item or "ID" in item:
            vendor_line = item.strip()

    if "aarch64" in arch_line:
        architecture = "kunpeng"
    if "x86_64" in arch_line:
        if "Intel" in vendor_line:
            architecture = "intel_xeon"
        if "AMD" in vendor_line:
            architecture = "amd_epyc"
    return architecture
