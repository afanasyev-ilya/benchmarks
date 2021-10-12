import matplotlib.pyplot as plt
import numpy as np


def plot_gather_or_scatter(benchmark_name, sizes, bandwidths):
    x = np.array(list(range(0, len(sizes))))
    for i in range(0, len(sizes)):
        x[i] = x[i] * 1000
    y = np.array(bandwidths)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    plt.xticks(x, sizes, rotation='vertical')
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set(xlabel='size (bytes)', ylabel='bandwidth (GB/s)',
           title=benchmark_name)
    ax.grid()

    fig.savefig("./output/" + benchmark_name + ".png")