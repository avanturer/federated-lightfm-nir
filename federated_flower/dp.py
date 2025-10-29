from typing import List
import torch


def clip_by_global_l2(params: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    if max_norm is None or max_norm <= 0:
        return params
    vec = torch.cat([p.view(-1) for p in params])
    l2 = torch.linalg.vector_norm(vec, ord=2)
    if l2 == 0 or l2 <= max_norm:
        return params
    scale = (max_norm / l2).to(vec.device)
    return [p * scale for p in params]


def add_gaussian_noise(params: List[torch.Tensor], sigma: float) -> List[torch.Tensor]:
    if sigma is None or sigma <= 0:
        return params
    return [p + torch.randn_like(p) * sigma for p in params]


