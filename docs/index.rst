jaxion
======

.. grid:: 1
    :class-container: color-cards

    .. grid-item-card:: Differentiable Astrophysics for Fuzzy Dark Matter
      :columns: 12 12 12 12
      :class-card: jaxion-summary

      .. image:: _static/jaxion.png
         :alt: Jaxion logo
         :width: 96px
         :align: left

      **Jaxion** is a scientific research software for conducting astrophysical and cosmological simulations of Fuzzy Dark Matter, implemented in JAX and utilizing automatic differentiation and multi-GPU acceleration.


.. grid:: 3
   :class-container: product-offerings
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Differentiable Physics
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Solvers in Jaxion are fully differentiable, enabling gradient-based inference, parameter optimization, and coupling to ML models directly through the simulation.

   .. grid-item-card:: Scalable Performance
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Built with JAX, Jaxion seamlessly scales across CPUs, GPUs, and TPUs, allowing large-scale simulations to run efficiently on modern accelerators.

   .. grid-item-card:: Modular & Composable
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Jaxion’s physics modules and processing tools are composable, making it easy to extend or embed into larger workflows.


.. grid:: 3
    :class-container: color-cards

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Installation
      :columns: 12 6 6 4
      :link: pages/installation
      :link-type: doc
      :class-card: installation

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Quickstart
      :columns: 12 6 6 4
      :link: pages/quickstart
      :link-type: doc
      :class-card: quickstart

    .. grid-item-card:: :material-regular:`library_books;2em` Examples
      :columns: 12 6 6 4
      :link: pages/examples
      :link-type: doc
      :class-card: examples


.. list-table::
   :widths: 32 32 32
   :header-rows: 0

   * - .. figure:: ../examples/dynamical_friction/movie.gif
         :width: 300px
         :align: center
         :alt: dynamical friction
         :target: pages/examples.html#dynamical-friction

     - .. figure:: ../examples/heating_gas/movie.gif
         :width: 300px
         :align: center
         :alt: gas heating
         :target: pages/examples.html#heating-gas

     - .. figure:: ../examples/tidal_stripping/movie.gif
         :width: 300px
         :align: center
         :alt: tidal stripping
         :target: pages/examples.html#tidal-stripping


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    pages/installation
    pages/quickstart

.. toctree::
    :maxdepth: 1
    :caption: Tutorials & Examples

    pages/examples

.. toctree::
    :maxdepth: 1
    :caption: References

    pages/parameters
    pages/api
    pages/about
