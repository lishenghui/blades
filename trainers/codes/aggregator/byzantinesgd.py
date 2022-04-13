import statistics
import torch
from .base import _BaseAggregator


def _get_vectorized_parameters(optimizer):
    c = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            c.append(p.data.view(-1))
    return torch.cat(c)


class ByzantineSGD(_BaseAggregator):
    def __init__(self, m, th_A, th_B, th_V, optimizer):
        # m is the number of workers
        self.m = m
        self.th_A = th_A
        self.th_B = th_B
        self.th_V = th_V

        # TODO:vectorize
        self.optimizer = optimizer

        # TODO: vectorize the model
        self.init_model = _get_vectorized_parameters(optimizer).clone()

        self.A = [0] * m
        self.B = [0] * m
        self.good = list(range(self.m))
        self.debug_message = ""

    def vector_median(self, vs, threshold):
        for i in range(self.m):
            count = 0
            for j in range(self.m):
                distance = (vs[i] - vs[j]).norm()
                count += distance <= threshold
                self.debug_message += f"(i, j, d)=({i}, {j}, {distance.item():.3f})\n"
                if count > self.m / 2:
                    return i, vs[i]
        raise NotImplementedError("No median found")

    def __call__(self, inputs):
        model_diff = _get_vectorized_parameters(self.optimizer) - self.init_model
        for i in range(self.m):
            self.A[i] += inputs[i].dot(model_diff)
            self.B[i] += inputs[i]
        self.debug_message = "\n=== Aggregation begins ===\n"
        self.debug_message += f"=> model_diff={model_diff.norm().item():.3f} \n"
        self.debug_message += (
            "=> A=[" + (" ".join([f"{a.item():.3f}" for a in self.A])) + "]\n"
        )

        A_med = statistics.median(self.A)
        self.debug_message += f"=> A_med={A_med.item():.3f}\n"

        self.debug_message += "=> Find median of B\n"
        B_med_index, B_med = self.vector_median(self.B, self.th_B)
        self.debug_message += f"=> Select median from worker {B_med_index}\n"

        self.debug_message += "=> Find median of current gradients\n"
        Grad_med_index, grad_median = self.vector_median(inputs, 2 * self.th_V)
        self.debug_message += f"=> Select median from worker {Grad_med_index}\n"

        self.debug_message += "\n=> start filtering\n"
        candidate = []
        for i in self.good:
            ai2med = abs(self.A[i] - A_med).item()
            bi2med = (self.B[i] - B_med).norm().item()
            g2med = (inputs[i] - grad_median).norm().item()
            self.debug_message += f"i={i} (a2m, th)=({ai2med:.3f}, {self.th_A}) (b2m, th)=({bi2med:.3f}, {self.th_B}) (g2m, th)=({g2med:.3f},{4*self.th_V})\n"
            if all([ai2med <= self.th_A, bi2med <= self.th_B, g2med <= 4 * self.th_V]):
                candidate.append(i)

        self.good = candidate

        return sum(inputs[i] for i in self.good) / len(self.good)
