import torch

from blades.client import ByzantineClient


class MediantailoredClient(ByzantineClient):
    r"""Uploads random noise as local update. The noise is drawn from a
    ``normal`` distribution.  The ``means`` and ``standard deviation`` are shared among all drawn elements.

    :param mean: the mean for all distributions
    :param std: the standard deviation for all distributions
    """
    
    def __init__(self, num_byzantine: int, dev_type='std', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.num_byzantine = num_byzantine
    
    def omniscient_callback(self, simulator):
        # compression = 'none'
        # q_level = 2
        # norm = 'inf'
        benign_update = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        model_re = torch.mean(benign_update, 0)

        if self.dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif self.dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif self.dev_type == 'std':
            deviation = torch.std(benign_update, 0)

        lamda = torch.Tensor([10.0])  # compute_lambda_our(all_updates, model_re, n_attackers)

        threshold_diff = 1e-5
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0
        iters = 0
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * self.num_byzantine)
            mal_updates = torch.cat((mal_updates, benign_update), 0)
    
            agg_grads = torch.median(mal_updates, 0)[0]
    
            loss = torch.norm(agg_grads - model_re)
    
            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
    
            lamda_fail = lamda_fail / 2
            prev_loss = loss

        mal_update = (model_re - lamda_succ * deviation)
        
        self._state['saved_update'] = mal_update


