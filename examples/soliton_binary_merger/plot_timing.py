import numpy as np
import matplotlib.pyplot as plt

# Timings on rusty
resolutions = np.array(
    [
        32,
        64,
        128,
        256,
        256,
        256,
        256,
        256,
        512,
        512,
        512,
        512,
        512,
        1024,
        1024,
        1024,
        1024,
    ]
)
mcups = np.array(
    [
        1.0,
        27.1,
        626.6,
        2201.1,
        184.6,
        216.1,
        203.4,
        200.4,
        2616.5,
        801.7,
        820.6,
        1103.2,
        1381.8,
        850.2,
        1095.3,
        1743.3,
        2219.5,
    ]
)
num_nodes = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2])
num_gpus = np.array([1, 1, 1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 2, 4, 8, 16])

yy = np.array([0.01, 0.001, 0.1, 1, 10])


def main():
    # Compute billion cell updates per second
    bcups = mcups / 1000.0

    plt.figure(figsize=(4, 4))
    plt.plot(
        resolutions[num_gpus == 1],
        bcups[num_gpus == 1],
        marker="s",
        label="H100 (1 gpu)",
    )
    plt.plot(
        resolutions[num_gpus == 2],
        bcups[num_gpus == 2],
        marker="s",
        label="H100 (2 gpus)",
    )
    plt.plot(
        resolutions[num_gpus == 4],
        bcups[num_gpus == 4],
        marker="s",
        label="H100 (4 gpus)",
    )
    plt.plot(
        resolutions[num_gpus == 8],
        bcups[num_gpus == 8],
        marker="s",
        label="H100 (8 gpus)",
    )
    plt.plot(
        resolutions[num_gpus == 16],
        bcups[num_gpus == 16],
        marker="s",
        label="H100 (16 gpus, 2 nodes)",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("resolution")
    plt.ylabel("billion cell updates per second")
    plt.xticks(resolutions, labels=[f"{r}³" for r in resolutions])
    plt.yticks(yy, labels=[f"{y}" for y in yy])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("timing.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
