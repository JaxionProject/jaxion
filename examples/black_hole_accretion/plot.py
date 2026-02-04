import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxion


def main():
    # plot m_bh vs time
    for res in [1, 2, 3, 4, 6]:
        checkpoint_dir = f"./checkpoints{res}/"
        t = []
        m_bh = []
        for i in range(101):
            sim = jaxion.Simulation(checkpoint_dir, checkpoint_number=i)
            t.append(sim.state["t"])
            m_bh.append(sim.state["mass"][0])
        t = jnp.array(t)
        m_bh = jnp.array(m_bh)
        plt.plot(t, m_bh, label=f"res={res}")
    plt.xlabel("time [kpc/(km/s)]")
    plt.ylabel(r"mass [$M_\odot$]")
    plt.yscale("log")
    plt.legend()
    plt.savefig("bh_mass_vs_time.png")


if __name__ == "__main__":
    main()
