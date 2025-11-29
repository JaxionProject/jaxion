Tutorial 6: Multiple GPUs
=========================

This tutorial describes how to run Jaxion simulations on distributed GPUs.
An example is provided in `examples/soliton_binary_merger <https://github.com/JaxionProject/jaxion/tree/main/examples/soliton_binary_merger>`_

First, ensure that JAX is initialized for distributed GPU use, by calling:

.. code-block:: python

    import jax

    jax.distributed.initialize()

On some machines, you may need to specify additional parameters to the ``initialize()`` function.
For most SLURM clusters, you will not need to specify anything.

If you'd like your Python script to print info, it should by guarded by:

.. code-block:: python

    if jax.process_index() == 0:
        print("Using distributed GPU mode")

to prevent multiple processes from printing simultaneously.


Sharding
--------

We need to set up sharding for the simulation state arrays.
Sharding splits the arrays across multiple devices for distributed computation.
This can be done as follows:

.. code-block:: python

    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, PartitionSpec, NamedSharding

    # Create mesh and sharding for distributed computation
    n_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((1, n_devices))
    mesh = Mesh(devices, axis_names=("x", "y"))
    sharding = NamedSharding(mesh, PartitionSpec("x", "y"))

In the example above, we create a virtual device mesh with one row and ``n_devices`` columns.
Arrays (2D and 3D) are split along the "y" axis across the devices.

When creating the simulation object, we need to pass the sharding to it:

.. code-block:: python

    sim = jaxion.Simulation(params, sharding=sharding)

And that's it! Jaxion will now run the simulation on multiple GPUs.

If you grab arrays from the simulation, such as the grid (``sim.grid``) or spectral grid (``sim.kgrid``),
these will be sharded arrays too.


Slurm Example
-------------

An example SLURM submission script for running our example on the Flatiron Rusty cluster
is provided below:

.. literalinclude:: ../../examples/soliton_binary_merger/sbatch_rusty_distributed.sh
  :language: bash
