Quickstart
==========

To quickly try out Jaxion, install the package via pip:

.. code-block:: bash

    pip install jaxion

And run a a simple example simulation (``examples/soliton_binary_merger/soliton_binary_merger.py``):

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


For more info
-------------

For info on how to install Jaxion with GPU support, see the :doc:`Installation <installation>` page.

For more examples of simulations, see the :doc:`Examples <examples>` page.
