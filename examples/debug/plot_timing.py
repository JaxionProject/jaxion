import numpy as np
import matplotlib.pyplot as plt

# Timings on Frontier

resolutions = np.array(
    [
        512,
        1024,
        2048,
        #        4096,
        #        8192,
    ]
)
runtime = np.array(
    [
        4.5,
        12,
        27,
        #        999,
        #        999,
    ]
)
runtime_no_awsofirccl = np.array(
    [
        4.5,
        78,
        160,
        #        999,
        #        999,
    ]
)
num_nodes = np.array([1, 2, 16])  # , 128, 1024])
num_gpus = np.array([1, 8, 64])  # , 512, 4096])

yy = np.array([0.01, 0.1, 1])


def main():
    # Compute billion cell updates per second
    bcups = (resolutions**3) / (runtime * 1.0e9)
    bcups_no_awsofirccl = (resolutions**3) / (runtime_no_awsofirccl * 1.0e9)

    plt.figure(figsize=(4, 4))
    plt.plot(
        resolutions,
        bcups,
        marker="s",
        label="MI250X",
    )
    plt.plot(
        resolutions,
        bcups_no_awsofirccl,
        marker="o",
        label="MI250X (no aws-ofi-rccl)",
    )
    for i, (res, bcup, nodes, gpus) in enumerate(
        zip(resolutions, bcups, num_nodes, num_gpus)
    ):
        plt.text(res * 0.9, bcup / 1.3, f"# nodes={nodes}", fontsize=7, va="center")
        plt.text(
            res * 0.9, bcup / (1.3) ** 2, f"# gpus={gpus}", fontsize=7, va="center"
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
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("timing.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
