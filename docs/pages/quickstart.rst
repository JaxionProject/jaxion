Quickstart
==========

To quickly try out Jaxion, install the package via pip:

.. code-block:: bash

    pip install jaxion

And run a a simple example simulation (``soliton_binary_merger.py``):

.. literalinclude:: ../../examples/soliton_binary_merger/soliton_binary_merger.py
  :language: python

with a resolution boost of 2 as:

.. code-block:: bash

    python soliton_binary_merger.py --res=2

Running this should take under a minute and produce output (in ``checkpoints2/``) that look like:

.. image:: ../../examples/soliton_binary_merger/movie.gif
   :width: 480px
   :align: center
   :alt: soliton binary merger

For info on how to install Jaxion with GPU support, see the :ref:`installation <installation>` page.

For more examples of simulations that can be run with Jaxion, see the :ref:`examples <examples>` page.
