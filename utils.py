"""
❤Description: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-06-13 21:03:10
❤LastEditTime: 2023-07-18 15:37:51
❤Github: https://github.com/MilknoCandy
"""
import torch as pt
from torch import Tensor


##################################### TOOLS #####################################
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def do_nothing(x: Tensor, *args, **kwargs):
    return x

################################# TOKEN ADAPTER COMPUTATION #################################
def compute_relation_norm(q, k, mode="cosine"):
    # [B, N, C]
    if mode == "cosine":
        q_ = q / q.norm(dim=-1, keepdim=True)
        k_ = k / k.norm(dim=-1, keepdim=True)
        return q_ @ k_.transpose(-2, -1)

# mask generator
def mask_gen(x: Tensor, cdim=-1, keep_values=False, mode='top', gate=None):
    """_summary_

    Args:
    - x (Tensor): [B, N, N]
    - keep_values (bool, optional): _description_. Defaults to False.
    - mode(str): top for topk, need to set k; max for the maximum value
    """
    # keep only the most similar features
    mask = pt.zeros_like(x)
    if mode == 'max':
        max_values, _ = x.max(dim=cdim, keepdim=True)
        mask[x == max_values] = 1. if not keep_values else x[x == max_values]
    elif mode == 'threshold':
        max_vR, _ = x.max(dim=cdim, keepdim=True)
        mask[x == max_vR] = 1. if not keep_values else x[x == max_vR]
        
        max_vC, _ = x.max(dim=-2, keepdim=True)
        mask[x == max_vC] = 1. if not keep_values else x[x == max_vC]
        # x > threshold will be merge
        # mask[x >= gate] = 1. if not keep_values else x[x >= gate]

    values_count = mask.sum(dim=-1, keepdim=True)
    return mask, values_count