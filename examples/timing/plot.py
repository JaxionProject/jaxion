import numpy as np
import matplotlib.pyplot as plt

resolutions = np.array([32, 64, 128, 256, 512])
cpu_times = np.array([1.55, 9.83, 75.19, 878.68, 13897.47]) / 100  # seconds
gpu_times = np.array([1.70, 2.30, 3.86, 17.85, 52.42]) / 100  # seconds


def main():
    num_cells = resolutions**3

    # Compute million cell updates per second
    cpu_mcups = num_cells / cpu_times / 1e6
    gpu_mcups = num_cells / gpu_times / 1e6

    plt.figure(figsize=(4, 4))
    plt.plot(resolutions, cpu_mcups, label="Apple M3 Max (cpu)", marker="o")
    plt.plot(resolutions, gpu_mcups, label="H100 (gpu)", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("resolution (N)")
    plt.ylabel("million cell updates per second")
    plt.xticks(resolutions, labels=[str(r) for r in resolutions])
    plt.legend()
    plt.tight_layout()
    plt.savefig("timing.eps", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
