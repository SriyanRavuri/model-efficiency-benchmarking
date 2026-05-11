"""Three model variants: full / compact / quantised — torch-based.

Imported only on the torch path; fallback.py provides a torch-free alternative.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelStats:
    name: str
    param_count: int
    param_bytes: int
    flops_per_inference: int
    is_quantised: bool


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, depth, output_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_full(input_dim, output_dim):
    return MLP(input_dim, hidden=512, depth=4, output_dim=output_dim)


def build_compact(input_dim, output_dim):
    return MLP(input_dim, hidden=128, depth=4, output_dim=output_dim)


def build_quantised(compact_model):
    compact_model.eval()
    return torch.quantization.quantize_dynamic(compact_model, {nn.Linear}, dtype=torch.qint8)


def compute_stats(model, name, input_dim, is_quantised=False) -> ModelStats:
    params = 0
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            params += in_f * out_f + out_f
            flops += 2 * in_f * out_f + out_f
    bytes_per_param = 1 if is_quantised else 4
    return ModelStats(name=name, param_count=params,
                      param_bytes=params * bytes_per_param,
                      flops_per_inference=flops, is_quantised=is_quantised)
