import numpy as np
import matplotlib.pyplot as plt

# Timings on Frontier (MI250X) & Rusty (H100 & H200)

resolutions = np.array(
    [
        512,
        1024,
        2048,
        4096,
        #        8192,
    ]
)
# run time for 10 steps
runtime = np.array(
    [
        3.8,
        6.3,
        136.1,
        174.3,
        #        999,
    ]
)

runtime_rusty_h100 = np.array([2.2, 13.2, np.nan, np.nan])  # , np.nan])
runtime_rusty_h200 = np.array([2.3, 7.0, 10.3, np.nan])  # , np.nan])

num_nodes = np.array([1, 1, 8, 64])  # 512])
num_gpus = np.array([1, 8, 64, 512])  # 4096])  # really # GCDs (=2 * # gpus)

yy = np.array([0.1, 1, 10])


def main():
    # Compute billion cell updates per second
    nsteps = 10
    bcups = nsteps * (resolutions**3) / (runtime * 1.0e9)
    # bcups_no_awsofirccl = 10 * (resolutions**3) / (runtime_no_awsofirccl * 1.0e9)
    bcups_rusty_h100 = nsteps * (resolutions**3) / (runtime_rusty_h100 * 1.0e9)
    bcups_rusty_h200 = nsteps * (resolutions**3) / (runtime_rusty_h200 * 1.0e9)

    plt.figure(figsize=(4, 4))
    plt.plot(
        resolutions,
        bcups_rusty_h200,
        marker="^",
        label="H200 (Rusty)",
    )
    plt.plot(
        resolutions,
        bcups_rusty_h100,
        marker="o",
        label="H100 (Rusty)",
    )
    plt.plot(
        resolutions,
        bcups,
        marker="s",
        label="MI250X (Frontier)",
    )
    for i, (res, bcup, nodes, gpus) in enumerate(
        zip(resolutions, bcups, num_nodes, num_gpus)
    ):
        plt.text(res * 0.85, bcup / 1.22, f"# nodes={nodes}", fontsize=7, va="center")
        plt.text(
            res * 0.85, bcup / (1.22) ** 2, f"# gpus={gpus}", fontsize=7, va="center"
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("resolution")
    plt.ylabel("billion cell updates per second")
    plt.xticks(resolutions, labels=[f"{r}³" for r in resolutions])
    plt.gca().set_xticks(resolutions, minor=False)
    plt.gca().set_xticks([], minor=True)
    plt.yticks(yy, labels=[f"{y}" for y in yy])
    plt.xlim(resolutions[0] / 1.3, resolutions[-1] * 1.3)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("timing.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
