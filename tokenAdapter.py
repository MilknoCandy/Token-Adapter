"""
❤Description: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-06-13 19:43:35
❤LastEditTime: 2023-07-23 10:36:26
❤Github: https://github.com/MilknoCandy
"""
import torch
import torch as pt
import torch.nn.functional as F
from torch import Tensor, _assert
from torch.autograd import Variable
from typing import Any, Callable, Dict, Tuple, Type

from .utils import compute_relation_norm, mask_gen


def compute_toad(x: Tensor, attn: Tensor, x_size: Tuple, 
                 toad_info: Dict[str, Any], ejector_args) -> Tuple[Callable, ...]:
    H_ratio, W_ratio = (toad_info['H_ratio'], toad_info['W_ratio']) if toad_info['HW_ratio'] is None else toad_info['HW_ratio']
    H, W = x_size
    toph = H - round(H * H_ratio)
    topw = W - round(W * W_ratio)
    toad_info['x_merge_size'] = (toph, topw)
    toad_info['x_size'] = x_size
    inj, eje = token_adapter_compute(x, attn, x_size, class_token=toad_info['cls_token'],
                                mode1=toad_info['mode1'], mode2=toad_info['mode2'],
                                toph=toph, topw=topw, sort_ascending=toad_info['sort_ascending'],
                                h_r=toad_info['H_sr'], w_r=toad_info['W_sr'],
                                ejector_args=ejector_args)
    m_t, u_t = inj, eje
    return m_t, u_t

######################################## TOKEN INJECTOR ########################################
def get_dist_range(x: Tensor, cdim=-1, keepdim=False):
    x_mx, _ = x.max(dim=cdim, keepdim=keepdim)
    x_mn, _ = x.min(dim=cdim, keepdim=keepdim)
    return x_mx - x_mn

def cross_attn_q_kv(q, kv, thres_value=None, keep_values=False):
    # get relative weight
    attn = compute_relation_norm(q=q, k=kv, mode='cosine')
    # attn, attn_counts = mask_gen(attn, cdim=-2, keep_values=False, mode='threshold', gate=0.75)
    attn_, attn_counts = mask_gen(attn, cdim=-2, keep_values=keep_values, mode='max')
    attn_n, _ = mask_gen(attn, cdim=-1, keep_values=keep_values, mode='max')
        
    feats = torch.einsum("bnm, bmc -> bnc", attn_, kv)
    # feats = attn @ kv
    return feats, attn_counts, attn_n

def cross_attn_ref(kv, attn, thres_value=None, keep_values=False):
    if thres_value is not None:
        attn_, attn_counts = mask_gen(attn, cdim=-1, keep_values=keep_values, mode='threshold', gate=thres_value)
    else:
        attn_, attn_counts = mask_gen(attn, cdim=-1, keep_values=keep_values, mode='max')
        # attn_c, _ = mask_gen(attn, cdim=-1, keep_values=keep_values, mode='max')
        # attn_ = attn_ + attn_c
        # attn_[attn_==2] = 1

    if keep_values:
        attn_[attn_ == 0] = -100
        attn_ = attn_.softmax(dim=-2)
        attn_counts = attn_.sum(dim=-1, keepdim=True)
    feats = torch.einsum("bnm, bmc -> bnc", attn_, kv)
    # feats = attn @ kv
    return feats, attn_counts


def token_adapter_compute(x: Tensor, attn: Tensor, x_size: tuple,
                  weit_ref='feat', mode1='max', mode2='max',
                  toph: int=1, topw: int=1, sort_ascending: bool=False,
                  h_r=1.0, w_r=1.0, h_mode=1, w_mode=1,
                  class_token: bool=False,
                  ejector_args: dict=None):
    """_summary_

    Args:
        x (Tensor): [B, N, C] or [B, H, W, C]
        attn (Tensor): [B, N, N]
    """
    x_shape = x.size()
    protected = 0
    if class_token:
        protected += 1
    if len(x_shape) == 3:
        b, n, c = x_shape
        h, w = x_size
        _assert((n - protected) == h*w, "input faeture has wrong size")

    with pt.no_grad():
        if weit_ref == 'feat':
            x_ = x[:, protected:].reshape(b, h, w, c)
        # 1. [B, N, N+protected] -> [B, N] -> [B, H, W] mode: mean
            if mode1 == 'max':
                x_, _ = x_.max(dim=-1)  # [B, H, W]
            elif mode1 == 'min':
                x_, _ = x_.min(dim=-1)  # [B, H, W]
            elif mode1 == 'sum':
                x_ = x_.sum(dim=-1)
            elif mode1 == 'mean':
                x_ = x_.mean(dim=-1)
            elif mode1 == 'range':
                x_ = get_dist_range(x_, cdim=-1, keepdim=False)
        elif weit_ref == 'attn':
            if mode1 == 'max':
                x_, _ = attn[:, protected:].max(dim=-1, keepdim=True)
            elif mode1 == 'mean':
                x_ = attn[:, protected:].mean(dim=-1, keepdim=True)
            elif mode1 == 'sum':
                x_ = attn[:, protected:].sum(dim=-1, keepdim=True)
            elif mode1 == 'range':
                x_ = get_dist_range(attn[:, protected:], cdim=-1, keepdim=True)

            x_ = x_.view(b, h, w)


        # 2. [B, H, W] -> [B, H, 1] and [B, 1, W]
        if mode2 == 'max':
            x_H, _ = x_.max(dim=-1, keepdim=True)  # [B, H, 1]
            x_W, _ = x_.max(dim=-2, keepdim=True)  # [B, 1, W]
        elif mode2 == 'min':
            x_H, _ = x_.min(dim=-1, keepdim=True)  # [B, H, 1]
            x_W, _ = x_.min(dim=-2, keepdim=True)  # [B, 1, W]
        elif mode2 == 'mean':
            x_H = x_.mean(dim=-1, keepdim=True)
            x_W = x_.mean(dim=-2, keepdim=True)
        elif mode2 == 'sum':
            x_H = x_.sum(dim=-1, keepdim=True)
            x_W = x_.sum(dim=-2, keepdim=True)
        elif mode2 == 'range':
            x_H = get_dist_range(x_, cdim=-1, keepdim=True)
            x_W = get_dist_range(x_, cdim=-2, keepdim=True)

        # 3. sort by measurement along H and W dimension, respectively
        H_idx = x_H.argsort(dim=-2, descending=sort_ascending)
        W_idx = x_W.argsort(dim=-1, descending=sort_ascending)
        # 4. toph and topw as the index of merged tokens
        x_labs = pt.zeros_like(x_)   # [B, H, W]
        x_idx = pt.arange(h*w, device=x_.device).reshape(1, h, w).expand(b, -1, -1)   # [B, H, W]
        # toph rows
        th = round(toph*h_r)
        lh = toph - th
        
        if h_mode == 1:
            # unique token: 0-lh presentative token: lh-(lh+th) normal token: (lh+th)-ed NOTE: mode1
            H_u = H_idx[:, :lh]
            H_p = H_idx[:, lh:lh+th]
            H_n = H_idx[:, lh+th:]
        else:
            # presentative token: 0-th 
            H_p = H_idx[:, :th]
            if h_mode == 2:
            # unique token: th-(th+lh) normal token: (lh+th)-ed NOTE: mode2
                H_u = H_idx[:, th:th+lh] if lh != 0 else H_idx[:, -lh:-lh]
                H_n = H_idx[:, th+lh:] if lh != 0 else H_idx[:, th:]
            elif h_mode == 3:
            # normal token: th-(-lh) unique token: -lh-ed NOTE: mode3
                H_u = H_idx[:, -lh:] if lh != 0 else H_idx[:, -lh:-lh]
                H_n = H_idx[:, th:-lh] if lh != 0 else H_idx[:, th:]

        # topw columns
        tw = round(topw*w_r)
        lw = topw - tw
        
        W_u = W_idx[:, :, :lw]
        W_p = W_idx[:, :, lw:lw+tw]
        W_n = W_idx[:, :, lw+tw:]


        srH = pt.ones(size=(1,1,1), dtype=x_labs.dtype, device=x_labs.device)
        srW = pt.ones(size=(1,1,1), dtype=x_labs.dtype, device=x_labs.device)
        # 1: unique tokens
        x_labs.scatter_(dim=-2, index=H_u.expand(-1, -1, w), src=srH.expand(b, H_u.shape[1], w))    # H dimension
        x_labs.scatter_(dim=-1, index=W_u.expand(-1, h, -1), src=srW.expand(b, h, W_u.shape[2]))    # W dimension
        # 2: normal tokens
        x_labs.scatter_(dim=-2, index=H_n.expand(-1, -1, w), src=2*srH.expand(b, H_n.shape[1], w))    # H dimension
        x_labs.scatter_(dim=-1, index=W_n.expand(-1, h, -1), src=2*srW.expand(b, h, W_n.shape[2]))    # W dimension
        # 0: presentative tokens

        hwp = x_idx[x_labs==0].view(b, -1)       # 0: presentative tokens [B, np]
        hwu = x_idx[x_labs==1].view(b, -1)       # 1: unique tokens [B, nu]
        hwn = x_idx[x_labs==2].view(b, -1)       # 2: normal tokens [B, nn]
        # release memory
        # pt.cuda.empty_cache()

    def token_injector(x: Tensor):
        # x: [B, N, C]
        x_ = x[:, protected:]
        protected_token = x[:, :protected]

        x_p = x_.gather(dim=-2, index=hwp.unsqueeze(-1).expand(-1, -1, c))
        x_u = x_.gather(dim=-2, index=hwu.unsqueeze(-1).expand(-1, -1, c))
        x_n = x_.gather(dim=-2, index=hwn.unsqueeze(-1).expand(-1, -1, c))
        
        # injector
        values, counts, attn_n = cross_attn_q_kv(x_p, x_n)
        value = (x_p + values) / (counts + 1)    # average
        value = pt.cat((value, x_u), dim=1)

        restore_attn = compute_relation_norm(x_, value, mode='cosine')  # x_p to all

        # protect cls token
        value = pt.cat((protected_token, value), dim=1)     # [B, n1+protected, C]
        # value = pt.cat((protected_token, value, x_u), dim=1)     # [B, n1+protected, C]
        return value, restore_attn

    def token_ejector(x:Tensor, attn_mask,
                      resotre_seg=True,
                      thres_value=None, keep_values=False):
        """_summary_

        Args:
            x (Tensor): merged tokens [B, n1, C]
            x_o (_type_): origianl tokens [B, N, C]
            attn_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        thres_value, keep_values = ejector_args.values()

        if resotre_seg:
            # x_restore, counts = cross_attn_ref(x, attn_mask, 0.9, keep_values=None)
            x_restore, counts = cross_attn_ref(x, attn_mask, thres_value, keep_values=keep_values)  # restore all
            return x_restore / (counts + 1e-10)

        else:
            x_ = x[:, protected:]
            protected_token = x[:, :protected]
            x_restore, counts = cross_attn_ref(x_, attn_mask, thres_value, keep_values=keep_values)  # restore all
            
            return pt.cat((protected_token, x_restore / (counts + 1e-10)), dim=1)

    
    return token_injector, token_ejector