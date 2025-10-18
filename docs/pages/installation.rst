Installation
============

From PyPI
---------

To install the latest release version of the Jaxion package, run the following command:

.. code-block:: bash

    pip install jaxion

For GPU support, use the following command instead:

.. code-block:: bash

    pip install jaxion[cuda12]

Build from Source
-----------------

Check out the repository:

.. code-block:: bash

    git clone git@github.com:JaxionProject/jaxion.git

Navigate to the project directory:

.. code-block:: bash

    cd jaxion

Install the package using pip (CPU version):

.. code-block:: bash

    pip install .

For GPU support, use the following command instead:

.. code-block:: bash

    pip install .[cuda12]

Verify the installation by running the test suite:

.. code-block:: bash

    pytest
