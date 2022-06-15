
**Blades**: A simulator for **B**\ yzantine-robust federated **L**\ earning with **A**\ ttacks and **D**\ efenses **E**\ xperimental **S**\ imulation
    | Know thy self, know thy enemy. A thousand battles, a thousand victories. --Sun Tzu

    | 知己知彼，百战百胜 ——孙武


.. raw:: html

    <p align=center>
      <a href="https://www.python.org/downloads/release/python-397/">
        <img src="https://img.shields.io/badge/Python->=3.9-3776AB?logo=python" alt="Python">
      </a>
      <a href="https://github.com/pytorch/pytorch">
        <img src="https://img.shields.io/badge/PyTorch->=1.8-FF6F00?logo=pytorch" alt="pytorch">
      </a>
      <!-- <a href="https://pypi.org/project/graphwar/">
        <img src="https://badge.fury.io/py/graphwar.png" alt="pypi">
      </a>        -->
      <a href="https://github.com/bladesteam/blades/blob/master/LICENSE.md">
        <img src="https://img.shields.io/github/license/bladesteam/blades?style=plastic" alt="license">
        <img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/>
      </a>
    </p>


NOTE: More features are under development and the APIs are subject to change.
If you are interested in this project, don't hesitate to contact me or make a PR directly.



`Documentation <https://bladesteam.github.io/>`_
==================================================



Installation
-------------

>>> pip install blades

Get Started
-------------

How fast can we simulate attack and defense in federated learning?
Take `ALIE Attack <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/alieclient.py>`_ and
`Trimmedmean Aggregation <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/trimmedmean.py>`_ as an `example <https://github.com/bladesteam/blades/blob/master/src/blades/examples/mini_example.py>`_.


.. include:: ../../src/blades/examples/mini_example.py
   :code: python


Illustration of Blades
--------------------------
<p align="center">
  <img width = "700" src="https://github.com/bladesteam/blades/blob/master/docs/source/_static/blade_architecture.pdf" alt="banner"/>
  <br/>
</p>



Build-in Implementations
--------------------------
In detail, the following strategies are currently implemented:



Attacks
---------

Untargeted Attack
^^^^^^^^^^^^^^^^^
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| Strategy           | Description                                                                                                                                                                                              | Sourse                                                                                                    |
+====================+==========================================================================================================================================================================================================+===========================================================================================================+
| **Noise**          |  Put random noise to the updates.                                                                                                                                                                        | `Sourse <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/noiseclient.py>`_          |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **Labelflipping**  | `Fang et al. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning <https://www.usenix.org/conference/usenixsecurity20/presentation/fang>`_, *USENIX Security' 20*                        | `Sourse <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/labelflippingclient.py>`_  |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **Signflipping**   | `Li et al. RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets <https://ojs.aaai.org/index.php/AAAI/article/view/3968>`_, *AAAI' 19*               | `Sourse <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/signflippingclient.py>`_   |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **ALIE**           | `Baruch et al. A little is enough: Circumventing defenses for distributed learning <https://proceedings.neurips.cc/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html>`_ *NeurIPS' 19*       | `Sourse <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/alieclient.py>`_           |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **IPM**            | `Xie et al. Fall of empires: Breaking byzantine- tolerant sgd by inner product manipulation <https://arxiv.org/abs/1903.03936>`_, *UAI' 20*                                                              | `Sourse <https://github.com/bladesteam/blades/blob/master/src/blades/attackers/ipmclient.py>`_            |
+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+



Defenses
---------

Robust Aggregation
^^^^^^^^^^^^^^^^^^^

+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| Strategy              | Descriptions                                                                                                                                                                                                                                                | Source                                                                                                   |
+=======================+=============================================================================================================================================================================================================================================================+==========================================================================================================+
| **Krum**              | `Blanchard et al. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent <https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html>`_, *NIPS'17*                                                       | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/krum.py>`_              |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **GeoMed**            | `Chen et al. Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent <https://arxiv.org/abs/1705.05491>`_, *POMACS'18*                                                                                                 | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/geomed.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **AutoGM**            | `Li et al. Byzantine-Robust Aggregation in Federated Learning Empowered Industrial IoT <https://ieeexplore.ieee.org/abstract/document/9614992>`_, *IEEE TII'22*                                                                                             | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/autogm.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Median**            | `Yin et al. Byzantine-robust distributed learning: Towards optimal statistical rates <https://proceedings.mlr.press/v80/yin18a>`_, *ICML'18*                                                                                                                | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/median.py>`_            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **TrimmedMean**       | `Yin et al. Byzantine-robust distributed learning: Towards optimal statistical rates <https://proceedings.mlr.press/v80/yin18a>`_, *ICML'18*                                                                                                                | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/trimmedmean.py>`_       |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **CenteredClipping**  | `Karimireddy et al. Learning from History for Byzantine Robust Optimization <http://proceedings.mlr.press/v139/karimireddy21a.html>`_, *ICML'21*                                                                                                            | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/centeredclipping.py>`_  |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Clustering**        | `Sattler et al. On the byzantine robustness of clustered federated learning <https://ieeexplore.ieee.org/abstract/document/9054676>`_, *ICASSP'20*                                                                                                          | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/clustering.py>`_        |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **ClippedClustering** | `Li et al. An Experimental Study of Byzantine-Robust sAggregation Schemes in Federated Learning <https://www.techrxiv.org/articles/preprint/An_Experimental_Study_of_Byzantine-Robust_Aggregation_Schemes_in_Federated_Learning/19560325>`_, *TechRxiv'22*  | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/clippedclustering.py>`_ |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+


Trust-based Strategies
^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| Strategy              | Description                                                                                                                                   | Source                                                                                                   |
+=======================+===============================================================================================================================================+==========================================================================================================+
| **FLTrust**           | `Cao et al. FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping <https://arxiv.org/abs/2012.13995>`_, NDSS'21                | `Source <https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/fltrust.py>`_           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+



Cluster Deployment
===================
To run **blades** on a cluster, you only need to deploy ``Ray cluster`` according to the `official guide <https://docs.ray.io/en/latest/cluster/user-guide.html>`_.


Citation
========

Please cite our `paper <https://arxiv.org/abs/2206.05359>`_ (and the respective papers of the methods used) if you use this code in your own work:

::

   @article{2022arXiv220605359L,
     title={Blades: A Simulator for Attacks and Defenses in Federated Learning},
     author= {Li, Shenghui and Ju, Li and Zhang, Tianru and Ngai, Edith and Voigt, Thiemo},
     journal={arXiv preprint arXiv:2206.05359},
     year={2022}
   }

Reference
==========

* Part of the code is from *Karimireddy*'s `repository <https://github.com/epfml/byzantine-robust-optimizer>`_. *Paper:* `Karimireddy et al. Learning from History for Byzantine Robust Optimization <http://proceedings.mlr.press/v139/karimireddy21a.html>`_.
