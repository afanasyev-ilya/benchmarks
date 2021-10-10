import matplotlib.pyplot as plt
import numpy as np


def plot_gather_or_scatter(benchmark_name, sizes, bandwidths):
    x = np.array(list(range(0, len(sizes))))
    y = np.array(bandwidths)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set_xticklabels(sizes)

    ax.set(xlabel='size (bytes)', ylabel='bandwidth (GB/s)',
           title=benchmark_name)
    ax.grid()

    fig.savefig(benchmark_name + ".png")