# CMC-ASP

**Cross-Modal Consistency–Aware Structured Pruning for Efficient Speech Enhancement with Air- and Bone-Conduction Microphones**

This repository provides the **channel scoring utilities** of **CMC-ASP** (Cross-Modal Consistency–Aware Structured Pruning).
Specifically, it implements:
- **Modality-wise zero-masking** (Noisy-ACM-only / BCM-only) at the input,
- **Hook-based activation extraction** at target layers,
- **CMC-ASP sensitivities** and **importance scores** defined in the paper.

> ⚠️ Scope: This repo focuses on **scoring only** (activation collection + sensitivity/importance computation).  
> Actual pruning / model surgery / fine-tuning pipelines are intentionally **not** included.

<p align="center">
  <img src="fig1.png" width="950" alt="CMC-ASP overview"/>
</p>

---

## Method Overview (CMC-ASP Scoring)

Given a paired multimodal input
\[
\mathbf{x} = [x_{\text{Noisy}}, x_{\text{BCM}}],
\]
CMC-ASP evaluates three input conditions:
1. **Multimodal reference**: \(\mathbf{x}^{\text{Multi}} = [x_{\text{Noisy}}, x_{\text{BCM}}]\)
2. **Noisy-only (BCM masked)**: \(\mathbf{x}^{\text{Noisy}} = [x_{\text{Noisy}}, 0]\)
3. **BCM-only (Noisy masked)**: \(\mathbf{x}^{\text{BCM}} = [0, x_{\text{BCM}}]\)

For a target convolutional layer/filter \(F\) and output channel \(C\), let \(O_{F,C}(\cdot)\) be the channel activation.
CMC-ASP summarizes activation magnitude using the **L1 norm** and computes dataset-level expectations over a calibration set \(D_{\text{cal}}\):

- Normalized sensitivities:
\[
S^{\text{Noisy}}_{F,C} =
\frac{\mathbb{E}_{x \sim D_{\text{cal}}}\big[\| O^{\text{Noisy}}_{F,C}(x)\|_1\big]}
{\mathbb{E}_{x \sim D_{\text{cal}}}\big[\| O^{\text{Multi}}_{F,C}(x)\|_1\big] + \epsilon},
\qquad
S^{\text{BCM}}_{F,C} =
\frac{\mathbb{E}_{x \sim D_{\text{cal}}}\big[\| O^{\text{BCM}}_{F,C}(x)\|_1\big]}
{\mathbb{E}_{x \sim D_{\text{cal}}}\big[\| O^{\text{Multi}}_{F,C}(x)\|_1\big] + \epsilon}.
\]

- Final importance score (symmetric aggregation):
\[
IS_{F,C} = 0.5 \cdot S^{\text{Noisy}}_{F,C} + 0.5 \cdot S^{\text{BCM}}_{F,C}.
\]

A high \(IS_{F,C}\) indicates that a channel responds **consistently** under both zero-masked conditions relative to the multimodal reference, suggesting **modality-shared / fusion-relevant** behavior.

---

## Repository Structure

- `cmc_asp_activations.py`  
  Hook-based activation extraction and estimation of
  \(\mathbb{E}[\|O_{F,C}(x)\|_1]\) for each target layer/channel.

- `cmc_asp_scoring.py`  
  CMC-ASP sensitivity and importance computation:
  \(S^{\text{Noisy}}, S^{\text{BCM}}, IS\).

---

## Installation

This code is lightweight and only requires PyTorch.

```bash
pip install torch
