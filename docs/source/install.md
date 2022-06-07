---
html_meta:
  description: |
    Learn how to install the Awesome Theme for your documentation project.
---

(sec:install)=

# Installation

## Single Machine Simulation

To use ``Blades`` on a single machine, you only need to install the simulator using ``pip``:

   ```terminal
   pip install blades
   ```

## Ray Cluster Simulation

Borrowing the power of [Ray](https://docs.ray.io/en/latest/index.html#), ``Blades`` can be easily adapted to clusters without 
changing a single line of the code.

```{seealso}
   See how to deploy `Ray cluster` according to the [official guide](https://docs.ray.io/en/latest/cluster/user-guide.html).
   Make sure that `Ray` on all machines has the same version.
```