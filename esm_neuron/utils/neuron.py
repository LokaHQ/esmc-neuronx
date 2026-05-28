"""Neuron compilation and device utilities for ESM models."""

import torch
import torch.nn as nn


def compile_esm_model(model: nn.Module, backend: str = "neuron") -> nn.Module:
    """Wrap a model with torch.compile for the given backend.

    Uses fullgraph=False to allow Python control flow inside forward() (required
    for ESM3's conditional track handling) and dynamic=False for static shapes
    (required by the Neuron compiler).
    """
    return torch.compile(model, backend=backend, fullgraph=False, dynamic=False)
