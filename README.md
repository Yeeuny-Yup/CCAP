# CMC-ASP

**Cross-Modal Consistency–Aware Structured Pruning for Efficient Speech Enhancement with Air- and Bone-Conduction Microphones**

This repository provides the **channel scoring utilities** of **CMC-ASP** (Cross-Modal Consistency–Aware Structured Pruning).
Specifically, it implements:
- **Modality-wise zero-masking** (Noisy-ACM-only / BCM-only) at the input,
- **Hook-based activation extraction** at target layers,
- **CMC-ASP sensitivities** and **importance scores** defined in the paper.

> Scope: This repo focuses on **scoring only** (activation collection + sensitivity/importance computation).  


<p align="center">
  <img src="fig1.png" width="950" alt="CMC-ASP overview"/>
</p>

---
## Method Overview (CMC-ASP Scoring)

Given a paired multimodal input:

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bx%7D%20%3D%20%5Bx_%7B%5Cmathrm%7BNoisy%7D%7D%2C%20x_%7B%5Cmathrm%7BBCM%7D%7D%5D" />

CMC-ASP evaluates three input conditions:

**(1) Multimodal reference**

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bx%7D%5E%7B%5Cmathrm%7BMulti%7D%7D%20%3D%20%5Bx_%7B%5Cmathrm%7BNoisy%7D%7D%2C%20x_%7B%5Cmathrm%7BBCM%7D%7D%5D" />

**(2) Noisy-only (BCM masked)**

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bx%7D%5E%7B%5Cmathrm%7BNoisy%7D%7D%20%3D%20%5Bx_%7B%5Cmathrm%7BNoisy%7D%7D%2C%200%5D" />

**(3) BCM-only (Noisy masked)**

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bx%7D%5E%7B%5Cmathrm%7BBCM%7D%7D%20%3D%20%5B0%2C%20x_%7B%5Cmathrm%7BBCM%7D%7D%5D" />

For a target layer (or filter) $F$ and output channel $C$, let $O_%7BF%2CC%7D(%5Ccdot)$ be the channel activation.
CMC-ASP summarizes activation magnitude using the L1 norm and computes dataset-level expectations over a calibration set $D_%7B%5Cmathrm%7Bcal%7D%7D$.

**Normalized sensitivities**

<img src="https://render.githubusercontent.com/render/math?math=S_%7BF%2CC%7D%5E%7B%5Cmathrm%7BNoisy%7D%7D%20%3D%20%5Cfrac%7B%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20D_%7B%5Cmathrm%7Bcal%7D%7D%7D%5B%5C%7Cl%20O_%7BF%2CC%7D%5E%7B%5Cmathrm%7BNoisy%7D%7D(x)%20%5C%7Cr%7C_1%5D%7D%7B%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20D_%7B%5Cmathrm%7Bcal%7D%7D%7D%5B%5C%7Cl%20O_%7BF%2CC%7D%5E%7B%5Cmathrm%7BMulti%7D%7D(x)%20%5C%7Cr%7C_1%5D%20%2B%20%5Cepsilon%7D" />

<img src="https://render.githubusercontent.com/render/math?math=S_%7BF%2CC%7D%5E%7B%5Cmathrm%7BBCM%7D%7D%20%3D%20%5Cfrac%7B%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20D_%7B%5Cmathrm%7Bcal%7D%7D%7D%5B%5C%7Cl%20O_%7BF%2CC%7D%5E%7B%5Cmathrm%7BBCM%7D%7D(x)%20%5C%7Cr%7C_1%5D%7D%7B%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20D_%7B%5Cmathrm%7Bcal%7D%7D%7D%5B%5C%7Cl%20O_%7BF%2CC%7D%5E%7B%5Cmathrm%7BMulti%7D%7D(x)%20%5C%7Cr%7C_1%5D%20%2B%20%5Cepsilon%7D" />

**Final importance score (symmetric aggregation)**

<img src="https://render.githubusercontent.com/render/math?math=IS_%7BF%2CC%7D%20%3D%200.5%20%5Ccdot%20S_%7BF%2CC%7D%5E%7B%5Cmathrm%7BNoisy%7D%7D%20%2B%200.5%20%5Ccdot%20S_%7BF%2CC%7D%5E%7B%5Cmathrm%7BBCM%7D%7D" />

---

## Repository Structure

- `cmc_asp_activations.py`  
  Hook-based activation extraction and estimation of
  \(\mathbb{E}[\|O_{F,C}(x)\|_1]\) for each target layer/channel.

- `cmc_asp_scoring.py`  
  CMC-ASP sensitivity and importance computation:
  \(S^{\text{Noisy}}, S^{\text{BCM}}, IS\).


