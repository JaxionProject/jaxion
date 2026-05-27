Quickstart
==========

To quickly try out Jaxion, install the package via pip:

.. code-block:: bash

    pip install jaxion

If pip needs to build ``jaxdecomp`` from source, install the build prerequisites listed
on the :doc:`Installation <installation>` page first.

And run a simple example simulation (``examples/soliton_binary_merger/soliton_binary_merger.py``):

.. literalinclude:: ../../examples/soliton_binary_merger/soliton_binary_merger.py
  :language: python

with a resolution boost factor of 2 as:

.. code-block:: bash

    python soliton_binary_merger.py --res=2

Running this should take under a minute and produce output (in ``checkpoints2/``) that look something like:

.. image:: ../../examples/soliton_binary_merger/movie.gif
   :width: 480px
   :align: center
   :alt: soliton binary merger

You can also run the example in the cloud: `Colab notebook <https://colab.research.google.com/github/JaxionProject/jaxion/blob/main/notebooks/soliton_binary_merger/soliton_binary_merger.ipynb>`__.


For more info
-------------

For info on how to install Jaxion with GPU support, see the :doc:`Installation <installation>` page.

For more examples of simulations, see the :doc:`Examples <examples>` page.
