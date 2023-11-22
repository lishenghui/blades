
.. .. raw:: html

..     <div style="text-align: center;">

.. container:: badges

    .. image:: https://img.shields.io/badge/arXiv-2206.05359-red?logo=arxiv&style=flat-square&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2206.05359.pdf
        :alt: Static Badge
        :target: https://arxiv.org/pdf/2206.05359.pdf

    .. image:: https://img.shields.io/github/last-commit/lishenghui/blades/master?logo=Github
        :alt: GitHub last commit (branch)
        :target: https://github.com/lishenghui/blades

    .. image:: https://img.shields.io/github/actions/workflow/status/lishenghui/blades/.github%2Fworkflows%2Funit-tests.yml?logo=Pytest&logoColor=hsl&label=Unit%20Testing
       :alt: GitHub Workflow Status (with event)

    .. image:: https://img.shields.io/badge/Pytorch-2.0-brightgreen?logo=pytorch&logoColor=red
       :alt: Static Badge
       :target: https://pytorch.org/get-started/pytorch-2.0/

    .. image:: https://img.shields.io/badge/Ray-2.8-brightgreen?logo=ray&logoColor=blue
       :alt: Static Badge
       :target: https://docs.ray.io/en/releases-2.8.0/

    .. image:: https://readthedocs.org/projects/blades/badge/?version=latest
        :target: https://blades.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

    .. image:: https://img.shields.io/github/license/lishenghui/blades?logo=apache&logoColor=red
        :alt: GitHub
        :target: https://github.com/lishenghui/blades/blob/master/LICENSE


.. .. raw:: html

..     <p align=center>
..         <img src="https://github.com/lishenghui/blades/blob/master/docs/source/images/arch.png" width="1000" alt="Blades Logo">
..     </p>

image:: https://github.com/lishenghui/blades/blob/master/docs/source/images/arch.png



Installation
==================================================

.. code-block:: bash

    git clone https://github.com/lishenghui/blades
    cd blades
    pip install -v -e .
    # "-v" means verbose, or more output
    # "-e" means installing a project in editable mode,
    # thus any local modifications made to the code will take effect without reinstallation.


.. code-block:: bash

    cd blades/blades
    python train.py file ./tuned_examples/fedsgd_cnn_fashion_mnist.yaml


**Blades** internally calls `ray.tune <https://docs.ray.io/en/latest/tune/tutorials/tune-output.html>`_; therefore, the experimental results are output to its default directory: ``~/ray_results``.

Experiment Results
==================================================

.. image:: https://github.com/lishenghui/blades/blob/master/docs/source/images/fashion_mnist.png

.. image:: https://github.com/lishenghui/blades/blob/master/docs/source/images/cifar10.png




Cluster Deployment
===================

To run **blades** on a cluster, you only need to deploy ``Ray cluster`` according to the `official guide <https://docs.ray.io/en/latest/cluster/user-guide.html>`_.


Built-in Implementations
==================================================
In detail, the following strategies are currently implemented:



Attacks
---------

General Attacks
^^^^^^^^^^^^^^^^^
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| Strategy           | Description                                                                                                                                                                                              | Sourse                                                                                                    |
+====================+==========================================================================================================================================================================================================+===========================================================================================================+
| **Noise**          |  Put random noise to the updates.                                                                                                                                                                        | `Sourse <https://github.com/lishenghui/blades/blob/master/blades/adversaries/noise_adversary.py>`_        |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **Labelflipping**  | `Fang et al. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning <https://www.usenix.org/conference/usenixsecurity20/presentation/fang>`_, *USENIX Security' 20*                        | `Sourse <https://github.com/lishenghui/blades/blob/master/blades/adversaries/labelflip_adversary.py>`_    |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **Signflipping**   | `Li et al. RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets <https://ojs.aaai.org/index.php/AAAI/article/view/3968>`_, *AAAI' 19*               | `Sourse <https://github.com/lishenghui/blades/blob/master/blades/adversaries/signflip_adversary.py>`_     |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **ALIE**           | `Baruch et al. A little is enough: Circumventing defenses for distributed learning <https://proceedings.neurips.cc/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html>`_ *NeurIPS' 19*       | `Sourse <https://github.com/lishenghui/blades/blob/master/blades/adversaries/alie_adversary.py>`_         |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **IPM**            | `Xie et al. Fall of empires: Breaking byzantine- tolerant sgd by inner product manipulation <https://arxiv.org/abs/1903.03936>`_, *UAI' 20*                                                              | `Sourse <https://github.com/lishenghui/blades/blob/master/blades/adversaries/ipm_adversary.py>`_          |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+

Adaptive Attacks
^^^^^^^^^^^^^^^^^
+--------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Strategy                 | Description                                                                                                                                                                         | Sourse                                                                                                          |
+==========================+=====================================================================================================================================================================================+=================================================================================================================+
| **DistanceMaximization** |  `Shejwalkar et al. Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning <https://par.nsf.gov/servlets/purl/10286354>`_, *NDSS' 21*   | `Sourse <https://github.com/lishenghui/blades/blob/master/blades/adversaries/minmax_adversary.py>`_             |
+--------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+


.. | **FangAttack**           |  `Fang et al. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning <https://www.usenix.org/conference/usenixsecurity20/presentation/fang>`_, *USENIX Security' 20*  | `Sourse <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/fangattackclient.py>`_           |
.. +--------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+


Defenses
---------

Robust Aggregation
^^^^^^^^^^^^^^^^^^^

+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| Strategy              | Descriptions                                                                                                                                                                                                                                                | Source                                                                                                   |
+=======================+=============================================================================================================================================================================================================================================================+==========================================================================================================+
| **MultiKrum**         | `Blanchard et al. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent <https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html>`_, *NIPS'17*                                                       | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/multikrum.py>`_              |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **GeoMed**            | `Chen et al. Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent <https://arxiv.org/abs/1705.05491>`_, *POMACS'18*                                                                                                 | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/aggregators.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Median**            | `Yin et al. Byzantine-robust distributed learning: Towards optimal statistical rates <https://proceedings.mlr.press/v80/yin18a>`_, *ICML'18*                                                                                                                | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/aggregators.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **TrimmedMean**       | `Yin et al. Byzantine-robust distributed learning: Towards optimal statistical rates <https://proceedings.mlr.press/v80/yin18a>`_, *ICML'18*                                                                                                                | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/aggregators.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **CenteredClipping**  | `Karimireddy et al. Learning from History for Byzantine Robust Optimization <http://proceedings.mlr.press/v139/karimireddy21a.html>`_, *ICML'21*                                                                                                            | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/centeredclipping.py>`_       |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Clustering**        | `Sattler et al. On the byzantine robustness of clustered federated learning <https://ieeexplore.ieee.org/abstract/document/9054676>`_, *ICASSP'20*                                                                                                          | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/clippedclustering.py>`_      |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **ClippedClustering** | `Li et al. An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning <https://ieeexplore.ieee.org/abstract/document/10018261>`_, *IEEE TBD'23*                                                                                    | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/clippedclustering.py>`_      |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **DnC**               | `Shejwalkar et al. Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning <https://par.nsf.gov/servlets/purl/10286354>`_, *NDSS'21*                                                                             | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/aggregators.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **SignGuard**         | `Xu et al. SignGuard: Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering <https://arxiv.org/abs/2109.05872>`_, *ICDCS'22*                                                                                               | `Source <https://github.com/lishenghui/blades/blob/master/fllib/aggregators/signguard.py>`_              |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+


Data Partitioners:
==================================================

Dirichlet Partitioner
----------------------

.. image:: https://github.com/lishenghui/blades/blob/master/docs/source/images/dirichlet_partition.png

Sharding Partitioner
----------------------

.. image:: https://github.com/lishenghui/blades/blob/master/docs/source/images/shard_partition.png


Citation
=========

Please cite our `paper <https://arxiv.org/abs/2206.05359>`_ (and the respective papers of the methods used) if you use this code in your own work:

::

   @article{li2023blades,
     title={Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning},
     author= {Li, Shenghui and Ju, Li and Zhang, Tianru and Ngai, Edith and Voigt, Thiemo},
     journal={arXiv preprint arXiv:2206.05359},
     year={2023}
   }
