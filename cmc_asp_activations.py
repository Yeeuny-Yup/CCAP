# cmc_asp_activations.py
"""
CMC-ASP: Hook-based activation statistics collection.

This module collects per-channel statistics needed by CMC-ASP scoring.

--------------------------
For a target layer F and channel C, given a sample x:

  ||O_{F,C}(x)||_1  :=  L1 norm over all non-channel dimensions.

We estimate E[||O_{F,C}(x)||_1] over a calibration loader by:
  - computing sample-wise L1 per channel for each batch,
  - summing over all samples,
  - dividing by total number of samples.

This matches the expectation form used in Eqs. (3)-(4).

----------------------
To keep this repo model-agnostic, you must provide:
  - forward_fn(model, noisy_acm, bcm) -> Any
  - batch_to_modalities(batch, device) -> (noisy_acm, bcm)

Modes
-----
  - mode="multi"      : x = [noisy_acm, bcm]     (unmasked)
  - mode="noisy_only" : x = [noisy_acm, 0]       (BCM zero-masked)
  - mode="bcm_only"   : x = [0, bcm]             (Noisy ACM zero-masked)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

Tensor = torch.Tensor
Target = Union[str, nn.Module]


@dataclass(frozen=True)
class ActivationCollectConfig:
    """
    output_index:
      If a hooked module returns (tensor, ...), use out[output_index] as the activation tensor.
    max_batches:
      Limit number of calibration batches (None = use all).
    """
    output_index: int = 0
    max_batches: Optional[int] = None


def _to_tensor_output(out: Any, output_index: int) -> Optional[Tensor]:
    """Extract a Tensor from module output."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        x = out[output_index]
        return x if torch.is_tensor(x) else None
    return None


def _samplewise_channel_l1(out: Tensor) -> Optional[Tensor]:
    """
    Compute sample-wise channel L1 norms.

    out: [B, C, ...] -> l1: [B, C]
      l1[b, c] = sum_{all non-channel dims} |out[b, c, ...]|
    """
    if (not torch.is_tensor(out)) or out.dim() < 2:
        return None
    b, c = out.shape[0], out.shape[1]
    x = out.detach().abs().reshape(b, c, -1).sum(dim=-1)  # [B, C]
    return x


def _resolve_targets(model: nn.Module, targets: Iterable[Target]) -> List[Tuple[str, nn.Module]]:
    """
    Resolve targets into (name, module) pairs.

    - If target is str: must match a key in model.named_modules()
    - If target is nn.Module: tries to find its name by identity, otherwise assigns "target_{i}"
    """
    name_to_module = dict(model.named_modules())
    resolved: List[Tuple[str, nn.Module]] = []

    for i, t in enumerate(targets):
        if isinstance(t, str):
            if t not in name_to_module:
                raise ValueError(f"[CMC-ASP] Target name not found: '{t}'")
            resolved.append((t, name_to_module[t]))
        elif isinstance(t, nn.Module):
            found_name = None
            for n, m in name_to_module.items():
                if m is t:
                    found_name = n
                    break
            resolved.append((found_name or f"target_{i}", t))
        else:
            raise TypeError(f"[CMC-ASP] Unsupported target type: {type(t)}")

    # Ensure unique names
    seen = set()
    uniq: List[Tuple[str, nn.Module]] = []
    for name, module in resolved:
        if name in seen:
            k = 2
            new_name = f"{name}__{k}"
            while new_name in seen:
                k += 1
                new_name = f"{name}__{k}"
            name = new_name
        seen.add(name)
        uniq.append((name, module))

    return uniq


def _mask_inputs(noisy: Tensor, bcm: Tensor, mode: str) -> Tuple[Tensor, Tensor]:
    """Apply modality-wise zero masking consistent with the paper."""
    if mode == "multi":
        return noisy, bcm
    if mode == "noisy_only":
        return noisy, torch.zeros_like(bcm)
    if mode == "bcm_only":
        return torch.zeros_like(noisy), bcm
    raise ValueError(f"[CMC-ASP] Unknown mode: {mode}")


@torch.no_grad()
def collect_channel_l1_expectation(
    model: nn.Module,
    calib_loader: DataLoader,
    *,
    targets: Iterable[Target],
    forward_fn: Callable[[nn.Module, Tensor, Tensor], Any],
    batch_to_modalities: Callable[[Any, torch.device], Tuple[Tensor, Tensor]],
    device: torch.device,
    mode: str,  # "multi" | "noisy_only" | "bcm_only"
    cfg: ActivationCollectConfig = ActivationCollectConfig(),
) -> Dict[str, Tensor]:
    """
    Collect E[||O_{F,C}(x)||_1] for each target layer F.

    Returns:
      acts[name] = Tensor[C] on CPU
    """
    resolved = _resolve_targets(model, targets)

    # Accumulate sum over samples for each channel: sum_x ||O_{F,C}(x)||_1
    acts_sum: Dict[str, Optional[Tensor]] = {name: None for name, _ in resolved}
    total_samples: int = 0
    hooks: List[Any] = []

    def make_hook(name: str):
        def hook(_m, _inp, out):
            out_t = _to_tensor_output(out, cfg.output_index)
            if out_t is None:
                return
            l1 = _samplewise_channel_l1(out_t)  # [B, C]
            if l1 is None:
                return
            v = l1.sum(dim=0)                   # [C] sum over batch samples
            v_cpu = v.to(dtype=torch.float32).cpu()
            if acts_sum[name] is None:
                acts_sum[name] = v_cpu.clone()
            else:
                acts_sum[name] = acts_sum[name] + v_cpu
        return hook

    # Register hooks
    for name, module in resolved:
        hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()

    try:
        for bi, batch in enumerate(calib_loader):
            noisy, bcm = batch_to_modalities(batch, device)
            noisy_in, bcm_in = _mask_inputs(noisy, bcm, mode)

            # Forward
            _ = forward_fn(model, noisy_in, bcm_in)

            # Update sample count using input batch size
            total_samples += int(noisy.shape[0])

            if cfg.max_batches is not None and (bi + 1) >= cfg.max_batches:
                break

    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        model.train(was_training)

    if total_samples <= 0:
        raise RuntimeError("[CMC-ASP] No samples processed (empty loader or max_batches=0).")

    # Convert sums to expectations
    acts: Dict[str, Tensor] = {}
    denom = float(total_samples)
    for name, s in acts_sum.items():
        if s is None:
            continue
        acts[name] = (s / denom).detach().cpu()

    if len(acts) == 0:
        raise RuntimeError("[CMC-ASP] No activations collected. Check targets / hook outputs / forward_fn.")

    return acts


__all__ = [
    "ActivationCollectConfig",
    "Target",
    "collect_channel_l1_expectation",
]
