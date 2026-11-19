# Noise-Resilient ClassiFIM Ablation Under Simulated NISQ Readout Noise

**Author:** Surat Sai Venkata Nagendra Babu, B.Tech CSE, Amrita Vishwa Vidyapeetham

A 3-seed ablation study inspired by the `ClassiFIM` framework from Prof. Daniel Lidar's group at USC. It uses synthetic noisy 1D spin-chain snapshots generated in the notebook (not the public datasets from the paper cited below).

## 1. Motivation
In **Kasatkin et al. (2024)** ([arXiv:2408.03323](https://arxiv.org/abs/2408.03323)), the authors estimate the Fisher Information Metric (FIM) $\chi_F(\lambda)$ by training a binary classifier to distinguish projective measurement bitstrings from neighboring parameter values $\lambda - \delta\lambda$ and $\lambda + \delta\lambda$, then computing the variance of the predicted log-odds:

<img width="472" height="83" alt="image" src="https://github.com/user-attachments/assets/36a7223a-0000-4345-aa14-8a3f7622384c" />


When bitstrings are corrupted by hardware readout (SPAM) bit-flips, unregularized MLPs can overfit to random noise in the disordered phase, pushing predicted probabilities toward $0$ or $1$ and causing log-odds variance explosion. This repository tests whether convolutional inductive bias and label smoothing (from our Springer LNEE medical-imaging work) stabilize FIM estimation under **8% simulated readout bit-flip noise**.

## 2. Simulated Dataset (Generated in Notebook)
- **System:** Synthetic 1D spin chain with $L = 20$ spins and an analytical susceptibility peak at $\lambda_c = 1.000$.
- **Parameter sweep:** 31 grid points $\lambda \in [0.25, 1.75]$ ($\delta\lambda = 0.05$), $800$ shots per $\lambda$, 75/25 train/test split per adjacent pair.
- **Readout noise:** Independent 8% bit-flip corruption applied to every spin measurement.
- **Statistical rigor:** Repeated across 3 random seeds (`seeds = [42, 100, 2026]`), with 3-point local window smoothing `[0.25, 0.50, 0.25]` applied identically to all models before curve normalization.

## 3. Ablation Models
1. **Unregularized MLP Baseline:** `Flatten` $\to$ `Dense(64)` $\to$ `Dense(64)` $\to$ `Sigmoid`, standard binary cross-entropy (`label_smoothing = 0.0`).
2. **Standard 1D-CNN Ablation (No Label Smoothing):** Single-branch `Conv1D(32, k=3)` $\to$ `GlobalAveragePooling1D` $\to$ `Dense(32)` $\to$ `Sigmoid`, standard binary cross-entropy (`label_smoothing = 0.0`).
3. **Proposed Dual-Branch + Label Smoothing + Consensus:** Local (`dilation=1`) + dilated (`dilation=2`) `Conv1D` branches with BatchNorm, sigmoid attention gating, $L_2$ decay ($10^{-3}$), binary cross-entropy with `label_smoothing = 0.08`, and a 2-model consensus average.

## 4. Results (3-Seed Mean ± Std, 20 Spins, 8% Readout Noise)

| Model | Critical Point Error $\|\hat{\lambda}_c - 1.00\|$ (↓) | Curve Fidelity $R^2$ (↑) | Curve MSE (↓) |
| :--- | :---: | :---: | :---: |
| **Unregularized MLP Baseline** | `0.283 ± 0.125` | `-0.8059 ± 0.1871` | `0.2059 ± 0.0213` |
| **Standard 1D-CNN (No Label Smoothing)** | **`0.067 ± 0.047`** | **`+0.8780 ± 0.0250`** | **`0.0139 ± 0.0028`** |
| **Proposed Dual-Branch + Label Smoothing (0.08)** | `0.100 ± 0.041` | `+0.7513 ± 0.0879` | `0.0284 ± 0.0100` |

<img width="1200" height="725" alt="image" src="https://github.com/user-attachments/assets/abe6be16-bd64-46a9-868b-5ab603539345" />

### Key Scientific Takeaway
Both convolutional architectures dramatically outperform the unregularized MLP baseline, which suffers severe log-odds inflation in the noisy disordered tail ($\lambda > 1.2$). Crucially, the ablation reveals that **the simpler translation-invariant 1D-CNN without label smoothing outperforms the more complex label-smoothed dual-branch model on every metric** ($R^2 = +0.8780$ vs. $+0.7513$; MSE $0.0139$ vs. $0.0284$). Because `ClassiFIM` relies on log-odds curvature, spatial averaging via `Conv1D + GlobalAveragePooling1D` effectively filters uncorrelated bit-flip noise, whereas aggressive label smoothing ($\epsilon = 0.08$) caps extreme log-odds and slightly broadens the estimated susceptibility peak.

## 5. Limitations and Next Steps
1. Evaluate on the official `USCqserver/classifim` benchmark datasets (e.g., 2D Ising and Rydberg atom arrays) using the unified conditional classifier $P(y=1 \mid x, \lambda)$ from Kasatkin et al. (2024).
2. Sweep smaller label-smoothing values ($\epsilon \in [0.005, 0.02]$) and $L_2$ penalties to quantify the exact bias-variance trade-off in log-odds FIM estimation.

## 6. References
1. Kasatkin, V., Mozgunov, E., Ezzell, N., Mishra, U., Hen, I., & Lidar, D. A. (2024). *ClassiFIM: An Unsupervised Method To Detect Phase Transitions.* arXiv:2408.03323. GitHub: `USCqserver/classifim`.
2. Harshith, Y. S., Menariya, N., Nagendra Babu, S. S. V., Venkateswaran, S., Batchu, R. K., & Kumar, C. (2026). *A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification.* Springer LNEE, Proceedings of Confluence 2026 (accepted, in production).
