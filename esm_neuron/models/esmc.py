"""Neuron-compatible ESMC model factory.

Applies P1 (disable Flash Attention) and wraps with torch.compile for Trainium.
Sequence length and batch size must be fixed at compile time (static shapes).
"""

import torch
import torch.nn as nn

from esm_neuron.utils.neuron import compile_esm_model

_BUILDERS: dict = {}


def _get_builders() -> dict:
    global _BUILDERS
    if not _BUILDERS:
        from esm.pretrained import ESMC_300M_202412, ESMC_600M_202412

        _BUILDERS = {"esmc_300m": ESMC_300M_202412, "esmc_600m": ESMC_600M_202412}
    return _BUILDERS


def NeuronESMC(model_name: str, device: str = "neuron", compile: bool = True) -> nn.Module:
    """Load and optionally compile an ESMC model for Neuron inference.

    Args:
        model_name: One of "esmc_300m" or "esmc_600m".
        device: Target device ("neuron", "cuda", or "cpu").
        compile: If True on Neuron, wrap with torch.compile(backend="neuron").
            CPU/CUDA run eager PyTorch.

    Returns:
        Model in eval mode on the requested device.
    """
    builders = _get_builders()
    if model_name not in builders:
        raise ValueError(f"Unknown ESMC model '{model_name}'. Choose from: {list(builders)}")

    model: nn.Module = builders[model_name]("cpu", use_flash_attn=False)
    if device != "cpu":
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()

    if compile and device == "neuron":
        model = compile_esm_model(model, backend=device)

    return model
