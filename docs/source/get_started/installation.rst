Installation
============

:blades:`Blades` is available for :python:`Python 3.9` to :python:`Python 3.11`.

.. note::
   We do not recommend installation as a root user on your system :python:`Python`.
   Please setup a virtual environment, *e.g.*, via :conda:`null` `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_, or create a `Docker image <https://www.docker.com/>`_.


Installation via PyPi
---------------------

You can install and use :blades:`Blades` **without any external library** required except for :pytorch:`null` `PyTorch <https://pytorch.org>`_
and :ray:`null` `Ray <https://ray.io>`_.

For this, simply run:

.. code-block:: none

   pip install blades



Installation from Source
-------------------------

In case a specific version is not supported by `our wheels <https://data.pyg.org/whl>`_, you can alternatively install them from source:

#. Ensure that your CUDA is setup correctly (optional):

   #. Check if :pytorch:`PyTorch` is installed with CUDA support:

      .. code-block:: none

         python -c "import torch; print(torch.cuda.is_available())"
         >>> True

   #. Add CUDA to :obj:`$PATH` and :obj:`$CPATH` (note that your actual CUDA path may vary from :obj:`/usr/local/cuda`):

      .. code-block:: none

         export PATH=/usr/local/cuda/bin:$PATH
         echo $PATH
         >>> /usr/local/cuda/bin:...

         export CPATH=/usr/local/cuda/include:$CPATH
         echo $CPATH
         >>> /usr/local/cuda/include:...

   #. Add CUDA to :obj:`$LD_LIBRARY_PATH` on Linux and to :obj:`$DYLD_LIBRARY_PATH` on macOS (note that your actual CUDA path may vary from :obj:`/usr/local/cuda`):

      .. code-block:: none

         export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
         echo $LD_LIBRARY_PATH
         >>> /usr/local/cuda/lib64:...

         export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
         echo $DYLD_LIBRARY_PATH
         >>> /usr/local/cuda/lib:...

   #. Verify that :obj:`nvcc` is accessible from terminal:

      .. code-block:: none

         nvcc --version
         >>> 11.8

   #. Ensure that :pytorch:`PyTorch` and system CUDA versions match:

      .. code-block:: none

         python -c "import torch; print(torch.version.cuda)"
         >>> 11.8

         nvcc --version
         >>> 11.8

#. Install the relevant packages:

   .. code-block:: none

      pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
      pip install --verbose torch_scatter
      pip install --verbose torch_sparse
      pip install --verbose torch_cluster
      pip install --verbose torch_spline_conv

In rare cases, CUDA or :python:`Python` path problems can prevent a successful installation.
:obj:`pip` may even signal a successful installation, but execution simply crashes with :obj:`Segmentation fault (core dumped)`.
We collected common installation errors in the `Frequently Asked Questions <installation.html#frequently-asked-questions>`__ subsection.
In case the FAQ does not help you in solving your problem, please create an `issue <https://github.com/pyg-team/pytorch_geometric/issues>`_.
Before, please verify that your CUDA is set up correctly by following the official `installation guide <https://docs.nvidia.com/cuda>`_.
