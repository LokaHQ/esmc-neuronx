"""ESM Neuron adaptation layer — Trainium/Inferentia-compatible wrappers for ESM3 and ESMC."""

# Apply compatibility patches before any esm imports.
# Patches EsmSequenceTokenizer for transformers 4.46.3+ (property setters).
import esm_neuron.patch  # noqa: F401

from esm_neuron.models.esm3 import NeuronESM3
from esm_neuron.models.esmc import NeuronESMC
from esm_neuron.utils.neuron import compile_esm_model

__all__ = ["NeuronESM3", "NeuronESMC", "compile_esm_model"]
