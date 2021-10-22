import subprocess
import re
from .roofline import platform_specs


def b2t(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{int(num)}{unit}{suffix}"
        num /= 1024.0
    return f"{int(num)}Yi{suffix}"


def t2b(value):
    match = re.match(r"(\d+)(\w+)", value, re.I)
    if match:
        items = match.groups()
        num = int(items[0])
        suffix = items[1]
        multiplier = {"B": 1, "KB": 1024, "MB": 1024*1024, "GB": 1024*1024*1024}
        return num * multiplier[suffix]
    return 1


def get_LLC_name(arch_name):
    roof_name = "LLC"
    output = subprocess.check_output(["lscpu"]).decode()
    if "L4" in output:
        roof_name = "L4"
    elif "L3" in output:
        roof_name = "L3"
    elif "L2" in output:
        roof_name = "L2"
    return roof_name


def get_prev_LLC_name(arch_name):
    roof_name = "LLC"
    output = subprocess.check_output(["lscpu"]).decode()
    if "L4" in output:
        roof_name = "L3"
    elif "L3" in output:
        roof_name = "L2"
    elif "L2" in output:
        roof_name = "L1"
    return roof_name


def get_L1_size(arch_name):
    return platform_specs[arch_name]["cache_sizes"]["L1"]


def get_L2_size(arch_name):
    return platform_specs[arch_name]["cache_sizes"]["L2"]


def get_L3_size(arch_name):
    return platform_specs[arch_name]["cache_sizes"]["L3"]


def get_LLC_size(arch_name):
    if arch_name == "a64fx":
        return platform_specs[arch_name]["cache_sizes"]["L2"]
    else:
        return platform_specs[arch_name]["cache_sizes"]["L3"]


def get_prev_LLC_size(arch_name):
    if arch_name == "a64fx":
        return platform_specs[arch_name]["cache_sizes"]["L1"]
    else:
        return platform_specs[arch_name]["cache_sizes"]["L2"]


def adjust_scale(min_size, max_size):
    cur_scale = 1
    while True:
        array_size = 4 * pow(2, cur_scale)
        if min_size <= array_size < max_size:
            print("selected scale " + str(cur_scale) + " is " + b2t(array_size))
            return cur_scale
        if max_size < array_size:
            return 1
        cur_scale += 1


def get_timing_from_file_line(line, timings):
    for key, val in timings.items():
        if key in line:
            timings[key] = float(line.split(" ")[1])

    #timings["mem_efficiency"] = 100.0 * timings["avg_bw"]/dram_bandwidth
    return timings


def parse_timings(output):  # collect time, perf and BW values
    lines = output.splitlines()
    timings = {"avg_time": 0, "avg_bw": 0, "avg_flops": 0, "flops_per_byte": 0}
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
        if get_cores_count() == 64:
            architecture = "kunpeng_920_64_core"
        if get_cores_count() == 48:
            architecture = "kunpeng_920_48_core"
    if "x86_64" in arch_line:
        if "Intel" in vendor_line:
            architecture = "intel_xeon_6140"
        if "AMD" in vendor_line:
            architecture = "amd_epyc"
    return architecture


def make_binaries(arch):
    arch_params = ""
    if "intel" in arch:
        arch_params = "CXX=icpc ARCH=intel"
    elif "kunpeng" in arch:
        arch_params = "CXX=g++ ARCH=kunpeng"
    elif "a64fx" in arch:
        arch_params = "CXX=FCCpx ARCH=a64fx"
    else:
        raise Exception("Unsupported architecture for compilation")

    cmd = "make " + arch_params
    p = subprocess.Popen(cmd, shell=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.wait()
