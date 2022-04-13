"""
Allen-Zhu Z, Ebrahimian F, Li J, et al. Byzantine-Resilient Non-Convex Stochastic Gradient Descent[J].
    arXiv preprint arXiv:2012.14368, 2020.
"""
import torch
import random
from .base import _BaseAggregator
from ..utils import log


class Safeguard(_BaseAggregator):
    """[summary]

    Args:
        _BaseAggregator ([type]): [description]
    """

    def __init__(self, T0, T1, th0, th1, nu, tuningThreshold=False, reset=False):
        assert T1 >= T0 >= 1
        assert th1 > th0 > 0

        self.T0 = T0
        self.T1 = T1
        self.th0 = th0
        self.th1 = th1

        self.nu = nu
        # reset good set every T1 step
        self.reset = reset

        self.good = None

        # The length of histA should be less than  T1
        # The element of histA is a list
        self.histA = []

        # The length of histB should be less than  T0
        self.histB = []

        self.histIndices = []

        self.tuningThreshold = tuningThreshold
        self.thresholdHistory = {}

    def add_to_threshold_hist(self, threshold_name, value):
        if threshold_name not in self.thresholdHistory:
            self.thresholdHistory[threshold_name] = {}

        reformatted_value = "{:.1f}".format(value)
        self.thresholdHistory[threshold_name][reformatted_value] = (
            self.thresholdHistory[threshold_name].get(reformatted_value, 0) + 1
        )
        sorted_values = sorted(
            self.thresholdHistory[threshold_name].items(), key=lambda x: -x[1]
        )
        log("=> {} {}".format(threshold_name, sorted_values[:3]))

    def adjust_threshold(self, d2m, good, threshold, threshold_name):
        """
        Args:
            d2m (dict): index -> distance to median. length of all workers.
            good (list): The list of current good worker indices.
            threshold_name (str): name of the threshold
        """
        m = len(d2m)
        sorted_values = sorted(d2m.values())
        # print(
        #     "adjust_threshold {}".format(
        #         ["{:.3f}".format(i.item()) for i in sorted_values]
        #     )
        # )

        # Requirement 1: At least half of the workers satisfies d2m[i] <= threshold
        candidate_threshold = sorted_values[m // 2] + 0.001
        # print(
        #     "==> {:.1f} {}".format(
        #         candidate_threshold, ["{:.1f}".format(i.item()) for i in sorted_values]
        #     )
        # )

        # Requirement 2: At least one worker in good set is 2 times greater than the threshold
        # if any(d2m[i] > 2 * candidate_threshold for i in good):
        #     # Round to first decimal point
        #     value = torch.ceil(candidate_threshold * 10) / 10
        #     self.add_to_threshold_hist(threshold_name, value)
        #     print(
        #         "!!!=> {} {:.1f} | {:.1f}".format(
        #             threshold_name, candidate_threshold, candidate_threshold
        #         )
        #     )
        #     return candidate_threshold
        # else:
        #     print(
        #         "!!!=> {} {:.1f} | {:.1f}".format(
        #             threshold_name, candidate_threshold, threshold
        #         )
        #     )
        #     return threshold

        # Round to first decimal point
        value = torch.ceil(candidate_threshold * 10) / 10
        self.add_to_threshold_hist(threshold_name, value)
        return candidate_threshold

    def compute_distance(self, v1, v2):
        return (v1 - v2).norm()

    def __str__(self):
        return "Safeguard(T0={},T1={},th0={},th1={},nu={})".format(
            self.T0, self.T1, self.th0, self.th1, self.nu
        )

    def find_median_grad(self, grads, threshold, m):
        """[summary]

        Args:
            grads (dict): node_idx -> gradient
            threshold (float): threshold
            m (int): number of total nodes
        """
        indices = list(grads.keys())

        # Since in the experiment we assume the workers indices [0, n-f) are good
        # and [n-f, n) are Byzantine. Shuffling removes the bias.
        random.shuffle(indices)

        distances = {}
        counts = {}
        for i in indices:
            count = 0
            for j in indices:
                idx = tuple(sorted([i, j]))
                distance = self.compute_distance(grads[i], grads[j]).item()
                distances[idx] = distances.get(idx, distance)
                if distances[idx] <= threshold:
                    count += 1
                if count >= m / 2:
                    print(
                        "\nhistA={} | Find median {} among indices={} threshold={} distances={}\n".format(
                            len(self.histA), i, indices, threshold, distances
                        )
                    )
                    return grads[i]
            counts[i] = count
        # If no one over m / 2
        print(f"counts={counts}")
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[0]
        max_count_indices = []
        for k, v in counts.items():
            if v == sorted_counts[1]:
                max_count_indices.append(k)
        random.shuffle(max_count_indices)
        print(
            "\nhistA={} | (Not Found) Find median {}  indices={} threshold={} distances={}  max_count_indices={}\n".format(
                len(self.histA),
                max_count_indices[0],
                indices,
                threshold,
                distances,
                max_count_indices,
            )
        )
        print(f"max_count_indices[0]={max_count_indices[0]}")
        return grads[max_count_indices[0]]

    def __call__(self, inputs):
        if self.good is None:
            self.good = list(range(len(inputs)))
        log(self.good)

        self.histA.append(inputs)
        self.histB.append(inputs)
        self.histIndices.append(self.good.copy())

        # Note that A_all and B_all are for tuning threshold.
        A = {}
        B = {}
        A_all = {}
        B_all = {}
        for node_idx in range(len(inputs)):
            Ai = 0
            for j in range(1, len(self.histA) + 1):
                grad = self.histA[-j][node_idx]
                Ai += grad / len(self.histIndices[-j])

            Bi = 0
            for j in range(1, len(self.histB) + 1):
                grad = self.histB[-j][node_idx]
                Bi += grad / len(self.histIndices[-j])

            A_all[node_idx] = Ai
            B_all[node_idx] = Bi

            if node_idx in self.good:
                A[node_idx] = Ai
                B[node_idx] = Bi

        # Find the median among the good
        A_med = self.find_median_grad(A, self.th1, len(inputs))
        B_med = self.find_median_grad(B, self.th0, len(inputs))

        # Update good sets
        new_goodset = []
        d2m_A = {}
        d2m_B = {}
        for i in range(len(inputs)):
            d2m_A[i] = self.compute_distance(A_all[i], A_med)
            d2m_B[i] = self.compute_distance(B_all[i], B_med)
            # if i in self.good and d2m_A[i] <= 2 * self.th1 and d2m_B[i] <= 2 * self.th0:
            if i in self.good and d2m_A[i] <= self.th1 and d2m_B[i] <= self.th0:
                new_goodset.append(i)
                print(
                    f"i={i} d2m_A[i]={d2m_A[i]:.3f} d2m_B[i]={d2m_B[i]:.3f} | i in good"
                )
            else:
                print(
                    f"i={i} d2m_A[i]={d2m_A[i]:.3f} d2m_B[i]={d2m_B[i]:.3f} | i not in good"
                )

        # if len(new_goodset) < len(inputs) / 2:
        #     new_goodset = list(range(len(inputs)))

        if self.tuningThreshold and len(self.histA) >= self.T1:
            self.th1 = self.adjust_threshold(d2m_A, self.good, self.th1, "th1")

        if self.tuningThreshold and len(self.histB) >= self.T0:
            self.th0 = self.adjust_threshold(d2m_B, self.good, self.th0, "th0")

        noise = torch.randn_like(A_med) * self.nu
        output = noise + sum(inputs[i] for i in self.good) / len(self.good)

        # if not algorithm;
        self.good = new_goodset

        if len(self.histA) >= self.T1:
            self.histA = []
            if self.reset:
                self.good = list(range(len(inputs)))

        if len(self.histB) >= self.T0:
            self.histB = []

        return output
