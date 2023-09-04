Installation
==================================================

.. code-block:: bash

    git clone https://github.com/lishenghui/fllib
    cd fllib
    pip install -v -e .
    # "-v" means verbose, or more output
    # "-e" means installing a project in editable mode,
    # thus any local modifications made to the code will take effect without reinstallation.


.. code-block:: bash

    cd fllib/blades
    python train.py file ./tuned_examples/fedavg_cnn_mnist.yaml

