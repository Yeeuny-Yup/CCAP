"""
CMC-ASP: Sensitivity & importance scoring.

Implements Eqs. (3)-(5):

  S^{Noisy}_{F,C} = E[||O^{Noisy}_{F,C}(x)||_1] / E[||O^{Multi}_{F,C}(x)||_1]
  S^{BCM}_{F,C}   = E[||O^{BCM}_{F,C}(x)||_1]   / E[||O^{Multi}_{F,C}(x)||_1]

  IS_{F,C} = 0.5 * ( S^{Noisy}_{F,C} + S^{BCM}_{F,C} )

Returns channel scores for hooked target layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cmc_asp_activations import ActivationCollectConfig, Target, collect_channel_l1_expectation

Tensor = torch.Tensor


@dataclass(frozen=True)
class CMCASPConfig:
    """
    eps:
      Numerical stability for division by E[||O^{Multi}||_1].
      (The paper does not require eps; this is a safe implementation detail.)
    collect_cfg:
      Forwarded to activation collection (output_index, max_batches).
    """
    eps: float = 1e-8
    collect_cfg: ActivationCollectConfig = ActivationCollectConfig()


def compute_cmc_asp_sensitivities(
    acts_multi: Dict[str, Tensor],
    acts_noisy: Dict[str, Tensor],
    acts_bcm: Dict[str, Tensor],
    *,
    eps: float = 1e-8,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Compute normalized sensitivities per target layer.

    Returns:
      sens_noisy[name] = S^{Noisy}_{F,:}  (Tensor[C])
      sens_bcm[name]   = S^{BCM}_{F,:}    (Tensor[C])
    """
    common = set(acts_multi.keys()) & set(acts_noisy.keys()) & set(acts_bcm.keys())
    if len(common) == 0:
        raise RuntimeError("[CMC-ASP] No common target keys across multi/noisy/bcm activations.")

    sens_noisy: Dict[str, Tensor] = {}
    sens_bcm: Dict[str, Tensor] = {}

    for name in sorted(common):
        multi = acts_multi[name].float()
        noisy = acts_noisy[name].float()
        bcm = acts_bcm[name].float()

        denom = multi + eps
        s_noisy = noisy / denom
        s_bcm = bcm / denom

        s_noisy = torch.nan_to_num(s_noisy, nan=0.0, posinf=0.0, neginf=0.0).cpu()
        s_bcm = torch.nan_to_num(s_bcm, nan=0.0, posinf=0.0, neginf=0.0).cpu()

        sens_noisy[name] = s_noisy
        sens_bcm[name] = s_bcm

    return sens_noisy, sens_bcm


def compute_cmc_asp_importance(
    sens_noisy: Dict[str, Tensor],
    sens_bcm: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """
    Compute importance scores per target layer:

      IS_{F,C} = 0.5 * ( S^{Noisy}_{F,C} + S^{BCM}_{F,C} )

    Returns:
      scores[name] = Tensor[C]
    """
    common = set(sens_noisy.keys()) & set(sens_bcm.keys())
    if len(common) == 0:
        raise RuntimeError("[CMC-ASP] No common keys across sensitivities.")

    scores: Dict[str, Tensor] = {}
    for name in sorted(common):
        iscore = 0.5 * (sens_noisy[name].float() + sens_bcm[name].float())
        iscore = torch.nan_to_num(iscore, nan=0.0, posinf=0.0, neginf=0.0).cpu()
        scores[name] = iscore

    return scores


@torch.no_grad()
def cmc_asp_score_channels(
    model: nn.Module,
    calib_loader: DataLoader,
    *,
    targets: Iterable[Target],
    forward_fn: Callable[[nn.Module, Tensor, Tensor], Any],
    batch_to_modalities: Callable[[Any, torch.device], Tuple[Tensor, Tensor]],
    device: torch.device,
    cfg: CMCASPConfig = CMCASPConfig(),
    return_intermediates: bool = False,
):
    """
    One-call CMC-ASP scoring pipeline (paper-consistent):

      1) Collect E[||O^{Multi}||_1]  with mode="multi"
      2) Collect E[||O^{Noisy}||_1]  with mode="noisy_only" (BCM masked)
      3) Collect E[||O^{BCM}||_1]    with mode="bcm_only"   (Noisy masked)
      4) Compute sensitivities by ratio to Multi (Eqs. 3-4)
      5) Compute importance as 0.5 average (Eq. 5)

    Returns:
      If return_intermediates == False:
        scores: Dict[str, Tensor[C]]
      Else:
        (scores, sens_noisy, sens_bcm, acts_multi, acts_noisy, acts_bcm)
    """
    acts_multi = collect_channel_l1_expectation(
        model,
        calib_loader,
        targets=targets,
        forward_fn=forward_fn,
        batch_to_modalities=batch_to_modalities,
        device=device,
        mode="multi",
        cfg=cfg.collect_cfg,
    )
    acts_noisy = collect_channel_l1_expectation(
        model,
        calib_loader,
        targets=targets,
        forward_fn=forward_fn,
        batch_to_modalities=batch_to_modalities,
        device=device,
        mode="noisy_only",
        cfg=cfg.collect_cfg,
    )
    acts_bcm = collect_channel_l1_expectation(
        model,
        calib_loader,
        targets=targets,
        forward_fn=forward_fn,
        batch_to_modalities=batch_to_modalities,
        device=device,
        mode="bcm_only",
        cfg=cfg.collect_cfg,
    )

    sens_noisy, sens_bcm = compute_cmc_asp_sensitivities(
        acts_multi, acts_noisy, acts_bcm, eps=cfg.eps
    )
    scores = compute_cmc_asp_importance(sens_noisy, sens_bcm)

    if return_intermediates:
        return scores, sens_noisy, sens_bcm, acts_multi, acts_noisy, acts_bcm
    return scores


__all__ = [
    "CMCASPConfig",
    "compute_cmc_asp_sensitivities",
    "compute_cmc_asp_importance",
    "cmc_asp_score_channels",
]
