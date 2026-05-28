"""Compatibility patches for esm library dependencies.

Applies patches at import time that are required for the esm_neuron package
to function correctly. Currently:

  - EsmSequenceTokenizer property setters: transformers 4.46.3 changed
    PreTrainedTokenizerBase.__init__ to set special tokens via setattr().
    EsmSequenceTokenizer defines cls_token, eos_token, mask_token, and
    pad_token as read-only properties backed by the Rust tokenizer, so
    setattr() raises AttributeError on init. This patch adds no-op setters
    that satisfy the new transformers init path; the Rust backend still
    handles all reads through __getattr__ (behaviour is unchanged).
"""

from __future__ import annotations


def _patch_esm_sequence_tokenizer() -> None:
    try:
        from esm.tokenization import EsmSequenceTokenizer
    except ImportError:
        return

    needs_patch = False
    for name in ("cls_token", "eos_token", "mask_token", "pad_token"):
        prop = getattr(type(EsmSequenceTokenizer), name, None)
        if isinstance(prop, property) and prop.fset is None:
            needs_patch = True
            break

    if not needs_patch:
        return

    for name in ("cls_token", "eos_token", "mask_token", "pad_token"):
        prop = getattr(type(EsmSequenceTokenizer), name, None)
        if not isinstance(prop, property):
            continue
        backing = f"_{name}"

        def _make_setter(attr: str):
            def _setter(self, value: object) -> None:
                object.__setattr__(self, attr, value)
            return _setter

        patched = prop.setter(_make_setter(backing))
        setattr(type(EsmSequenceTokenizer), name, patched)


_patch_esm_sequence_tokenizer()
