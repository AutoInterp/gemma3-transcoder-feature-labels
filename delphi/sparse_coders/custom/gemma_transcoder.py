"""Loader for GemmaScope-2 affine MLP transcoders (circuit-tracer packaging).

Target repo layout (e.g. ``mwhanna/gemma-scope-2-4b-it``)::

    <subfolder>/layer_{L}.safetensors     # one file per layer

Keys per safetensors (all float32)::

    W_enc                          [d_sae, d_model]
    W_dec                          [d_sae, d_model]
    W_skip                         [d_model, d_model]   # affine skip (Ameisen 2025)
    b_enc                          [d_sae]
    b_dec                          [d_model]
    activation_function.threshold  [d_sae]              # per-feature JumpReLU

Forward (pre-residual-add MLP output prediction)::

    pre   = x @ W_enc.T + b_enc
    acts  = pre * (pre > threshold)                     # JumpReLU
    recon = acts @ W_dec + b_dec + x @ W_skip.T

Delphi only needs ``encode(x) -> dense [..., num_latents]`` for caching + labelling,
so the decoder and skip are kept for completeness but never used by the pipeline.

The transcoder reads at ``mlp.hook_in`` — equivalently the input to ``layers.{L}.mlp``
in HF transformers. We expose hookpoints as ``layers.{L}.mlp`` and rely on delphi's
``transcode=True`` branch in ``collect_activations`` to capture module *input*.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


@dataclass
class _TranscoderCfg:
    """Duck-typed stand-in for ``sparsify.SparseCoderConfig``.

    Only the two flags delphi reads downstream (``load_sparsify.py`` line 182-187
    and the ``transcode`` propagation in ``cache.py``) need to be set.
    """

    transcode: bool = True
    skip_connection: bool = True


class GemmaScopeAffineTranscoder(nn.Module):
    """JumpReLU MLP transcoder with an affine skip connection."""

    def __init__(self, d_model: int, d_sae: int, has_skip: bool = True):
        super().__init__()
        self.d_in = d_model
        self.num_latents = d_sae
        self.cfg = _TranscoderCfg(transcode=True, skip_connection=has_skip)

        # W_enc is stored in [d_sae, d_model] orientation (applied via F.linear).
        self.W_enc = nn.Parameter(torch.zeros(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        if has_skip:
            self.W_skip = nn.Parameter(torch.zeros(d_model, d_model))
        else:
            self.register_parameter("W_skip", None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.linear(x, self.W_enc, self.b_enc)
        return torch.where(pre > self.threshold, pre, pre.new_zeros(()))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.decode(self.encode(x))
        if self.W_skip is not None:
            recon = recon + F.linear(x, self.W_skip)
        return recon

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "GemmaScopeAffineTranscoder":
        state = load_file(str(path), device="cpu")
        d_sae, d_model = state["W_enc"].shape
        has_skip = "W_skip" in state
        sae = cls(d_model=d_model, d_sae=d_sae, has_skip=has_skip)
        with torch.no_grad():
            sae.W_enc.copy_(state["W_enc"])
            sae.W_dec.copy_(state["W_dec"])
            sae.b_enc.copy_(state["b_enc"])
            sae.b_dec.copy_(state["b_dec"])
            sae.threshold.copy_(state["activation_function.threshold"])
            if has_skip:
                sae.W_skip.copy_(state["W_skip"])
        return sae.to(device)


def _parse_repo_and_subfolder(sparse_model: str) -> tuple[str, str]:
    """Split ``<org>/<repo>/<subfolder...>`` into (repo_id, subfolder)."""
    parts = sparse_model.strip("/").split("/")
    if len(parts) < 3:
        raise ValueError(
            "Expected sparse_model of the form `<org>/<repo>/<subfolder>` for "
            f"GemmaScope transcoders, got: {sparse_model!r}"
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def _parse_layer_indices(hookpoints: list[str]) -> list[int]:
    """Accept ``layer_13`` / ``layers.13`` / ``13`` → ``13``."""
    out = []
    for h in hookpoints:
        token = h.split("/")[0]  # tolerate any trailing segments
        if token.startswith("layer_"):
            out.append(int(token[len("layer_") :]))
        elif token.startswith("layers."):
            out.append(int(token.split(".")[1]))
        else:
            out.append(int(token))
    return out


def is_gemma_transcoder_path(sparse_model: str) -> bool:
    """Detect the mwhanna/circuit-tracer-packaged transcoder layout."""
    return "transcoder_all" in sparse_model or sparse_model.startswith("mwhanna/")


def load_gemma_transcoder_autoencoders(
    sparse_model: str,
    hookpoints: list[str],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, nn.Module]:
    """Return ``{hookpoint: module}`` for each requested layer.

    ``sparse_model`` is ``<org>/<repo>/<subfolder>`` (e.g.
    ``mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine``).
    ``hookpoints`` is a list of layer specifiers (``layer_13``, ``13``, ...).
    """
    repo_id, subfolder = _parse_repo_and_subfolder(sparse_model)
    layers = _parse_layer_indices(hookpoints)

    saes: dict[str, nn.Module] = {}
    for layer in layers:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/layer_{layer}.safetensors",
        )
        sae = GemmaScopeAffineTranscoder.from_safetensors(local_path, device=device)
        sae.to(dtype=dtype)
        saes[f"language_model.layers.{layer}.mlp"] = sae
    return saes


def load_gemma_transcoder_hooks(
    sparse_model: str,
    hookpoints: list[str],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile: bool = False,
) -> tuple[dict[str, Callable], bool]:
    """Return ``({hookpoint: encode_fn}, transcode=True)``.

    Each ``encode_fn`` takes MLP-input activations ``[..., d_model]`` and returns
    dense JumpReLU activations ``[..., num_latents]``, matching the shape contract
    the rest of delphi's cache expects.
    """
    saes = load_gemma_transcoder_autoencoders(
        sparse_model, hookpoints, device=device, dtype=dtype,
    )
    hooks: dict[str, Callable] = {}
    for hookpoint, sae in saes.items():
        if compile:
            sae = torch.compile(sae)

        def _encode(x: torch.Tensor, sae=sae) -> torch.Tensor:
            return sae.encode(x)

        hooks[hookpoint] = partial(_encode)
    return hooks, True
