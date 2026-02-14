# CMC-ASP

**Cross-Modal Consistency–Aware Structured Pruning for Efficient Speech Enhancement with Air- and Bone-Conduction Microphones**

This repository provides the **channel scoring utilities** of **CMC-ASP** (Cross-Modal Consistency–Aware Structured Pruning).
Specifically, it implements:
- **Modality-wise zero-masking** (Noisy-ACM-only / BCM-only) at the input,
- **Hook-based activation extraction** at target layers,
- **CMC-ASP sensitivities** and **importance scores** defined in the paper.

This repo focuses on **scoring only** (activation collection + sensitivity/importance computation).  


<p align="center">
  <img src="fig1.png" width="950" alt="CMC-ASP overview"/>
</p>

---

## Method Overview (CMC-ASP Scoring)

CMC-ASP evaluates channel importance by comparing channel responses under **modality-wise zero-masking** against a **multimodal reference**.

### 1) Paired input and three masking conditions

<p align="center">
  <img src="eq3.png" width="400" alt="Channel response statistic with L1 norm"/>
</p>

### 2) Channel response statistic

For a target layer (or filter) $F$ and output channel $C$, let $O_{F,C}(\cdot)$ denote the channel activation.  
CMC-ASP summarizes activation magnitude using the L1 norm and computes dataset-level expectations over a calibration set $D_{\mathrm{cal}}$.

<p align="center">
  <img src="eq4_5.png" width="350" alt="Normalized sensitivities for Noisy and BCM"/>
</p>

### 3) Normalized sensitivities and final importance

CMC-ASP computes normalized sensitivities by measuring how well each channel response is preserved under each masked condition relative to the multimodal reference.
The final importance score is obtained by a symmetric aggregation of the two sensitivities:

<p align="center">
  <img src="eq6.png" width="350" alt="Final importance score"/>
</p>

A high $IS_{F,C}$ indicates that a channel responds **consistently** under both zero-masked conditions relative to the multimodal reference, suggesting **modality-shared / fusion-relevant** behavior.

---

## Implementation Notes (Paper-faithful details)

- **Exact statistic used:** for each sample $x$, we compute channel-wise $\|O_{F,C}(x)\|_1$ by summing absolute activations over all non-channel dimensions. We then estimate $\mathbb{E}[\|O_{F,C}(x)\|_1]$ over $D_{\mathrm{cal}}$.

- **Unbiased expectation over samples:** accumulation is performed by summing $\|O_{F,C}(x)\|_1$ over **all samples** and dividing by the **total number of samples**. This ensures correctness even when the last batch has a different size.

- **Zero-masking location:** modality-wise zero-masking is applied **at the input** (Noisy-only / BCM-only), matching the paper definition.

- **Numerical stability:** $\epsilon$ is included only to avoid division-by-zero when $\mathbb{E}[\|O^{\mathrm{Multi}}_{F,C}(x)\|_1]$ is very small; it does not change the intended scoring behavior.

---
