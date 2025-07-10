import numpy as np


def linear_kl_annealing_func(step: int, x0: float, **kwargs):
    return min(1, step/x0)

def logistic_kl_annealing_func(step: int, k: float, x0: float, **kwargs):
    return float(1/(1 + np.exp(-k * (step - x0))))

def scaled_logistic_kl_annealing_func(step: int, k: float, x0: float, **kwargs):
    return float(1/(1 + np.exp(-k * (step - x0))) * 10)