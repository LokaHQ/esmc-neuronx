"""Neuron-compatible ESM3-open model factory.

ESM3-open uses standard (non-Flash) attention by default, so no P1 patch is needed.
The P2 and P3 patches are already applied inline in esm/utils/misc.py and
esm/utils/structure/affine3d.py respectively.
"""

import torch
import torch.nn as nn

from esm_neuron.utils.neuron import compile_esm_model


def NeuronESM3(device: str = "neuron", compile: bool = True) -> nn.Module:
    """Load and optionally compile ESM3-open for Neuron inference.

    Sequence-only mode: structure coordinates default to NaN (black-hole affine).
    Geometric attention runs on layer 0 using identity rotation / zero translation.

    Args:
        device: Target device ("neuron", "cuda", or "cpu").
        compile: If True on Neuron, wrap with torch.compile(backend="neuron").
            CPU/CUDA run eager PyTorch.

    Returns:
        Model in eval mode on the requested device.
    """
    from esm.pretrained import ESM3_sm_open_v0

    model: nn.Module = ESM3_sm_open_v0("cpu")
    if device != "cpu":
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()

    if compile and device == "neuron":
        model = compile_esm_model(model, backend=device)

    return model
