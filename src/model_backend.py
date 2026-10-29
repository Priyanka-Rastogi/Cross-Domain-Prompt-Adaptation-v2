"""Frozen-model loading. No gradients, no training, zero trainable parameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    model_id: str
    revision: str


def load_frozen_model(model_id: str, revision: str | None = None, dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to("cuda")
    model.eval()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable_after == 0, f"expected 0 trainable params, found {n_trainable_after}"

    resolved_revision = revision
    if resolved_revision is None:
        # resolve the actual commit sha actually loaded, for the manifest
        from huggingface_hub import HfApi

        resolved_revision = HfApi().model_info(model_id).sha

    return LoadedModel(model=model, tokenizer=tokenizer, model_id=model_id, revision=resolved_revision)
