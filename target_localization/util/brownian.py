from math import sqrt
from scipy.stats import norm
import numpy as np
import torch 

def brownian_one_pass(x: torch.tensor, dt=1., delta=0.1):
    res = norm.rvs(size=x.shape, scale=delta*sqrt(dt))
    res = torch.from_numpy(res).float()

    x_next = (x+res)
    return x_next
