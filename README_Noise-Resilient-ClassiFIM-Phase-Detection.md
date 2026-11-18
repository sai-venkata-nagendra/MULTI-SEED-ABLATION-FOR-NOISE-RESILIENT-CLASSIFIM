# Label-Smoothed Dual-Branch Classifier for a ClassiFIM-Style Estimate (Synthetic Toy Test)

**Author:** Surat Sai Venkata Nagendra Babu
**Affiliation:** B.Tech in Computer Science and Engineering, Amrita Vishwa Vidyapeetham

> **Status: exploratory learning exercise.** The data is synthetic and the comparison baseline is simple. This is not a test of the published ClassiFIM method on real quantum data. See [Limitations](#5-limitations).

---

## 1. Background

**ClassiFIM** (Kasatkin, Mozgunov, Ezzell, Mishra, Hen, & Lidar, [arXiv:2408.03323](https://arxiv.org/abs/2408.03323)) finds phase transitions without labels. The idea:
- Take data at two nearby parameter values (lambda - delta and lambda + delta).
- Train a classifier to tell which one a sample came from.
- If the classifier can tell them apart easily, the data is changing fast there, which is measured by the **Fisher information metric (FIM)**. A peak in the FIM marks a likely phase transition.

The paper notes that an overconfident classifier can inflate the log-odds and bias the estimate, and addresses it with small L2 regularization.

**My question:** does a regularizer I used in my own work (label smoothing, dual branches, averaging two models) keep this estimate stable when the data has 6% random bit-flip "readout noise"?

---

## 2. Setup (everything is in the notebook)

**Synthetic data (my own formula, not real TFIM simulations):**
- 16-spin chains, 31 values of lambda from 0.25 to 1.75, 600 samples each.
- A hand-written sigmoid defines how ordered the chain is, with the transition placed at lambda = 1.0.
- 6% of bits are flipped to mimic readout noise.

**Estimate:** for each lambda, train a classifier on samples from lambda +/- delta and use the variance of its log-odds divided by (2 * delta)^2. This is a simplified version of the paper's estimator, which uses log-odds gradients.

**Comparison against a "true" curve:** the squared derivative of the same hand-written sigmoid, normalized.

**Models:**
- *Baseline:* plain MLP (two dense layers of 64), no regularization.
- *Proposed:* two 1D convolution branches (normal and dilated), a sigmoid attention gate, dropout, L2 (1e-3), label smoothing (0.08), and the average of 2 independently initialized models.

---

## 3. Results (one run, seed 42, with 3-point smoothing across neighboring lambda)

| Metric vs. synthetic reference curve | Baseline MLP | Proposed |
| :--- | ---: | ---: |
| MSE | 0.1866 | **0.0353** |
| R^2 | -0.6364 | **+0.6907** |
| Estimated peak position | 1.10 | 1.10 |

The proposed model follows the reference curve more closely, but **both models put the peak at lambda = 1.10**, so peak localization did not improve in this run.

![ClassiFIM results](ClassiFIM_Phase_Transition_Results.png)

---

## 4. What this shows

Label smoothing, dropout and model averaging make the classifier's log-odds less extreme, which makes the estimated curve smoother and closer to the reference under bit-flip noise.

## 5. Limitations

- **Synthetic data.** The data comes from my own formula. The transition location is built in, and the "true" curve comes from the same formula. This is not a real TFIM or hardware dataset.
- **Weaker baseline than the paper.** The baseline has no regularization at all, while the ClassiFIM authors already use small L2 regularization.
- **Simplified estimator** (log-odds variance, not the paper's gradient-based formula).
- **One run, one seed.** No error bars.
- The notebook clones the official ClassiFIM repository for reference but does **not** use its code or its public datasets.

## 6. Planned next steps
1. Repeat over several seeds and noise levels, with error bars.
2. Add a baseline with L2 regularization only, to separate the effect of label smoothing.
3. Use the public ClassiFIM datasets and the paper's own estimator.
4. Try exact transverse-field Ising snapshots.

## 7. References
1. Kasatkin, V., Mozgunov, E., Ezzell, N., Mishra, U., Hen, I., & Lidar, D. A. (2024). *ClassiFIM: An Unsupervised Method To Detect Phase Transitions.* arXiv:2408.03323. Code: github.com/USCqserver/classifim.
2. Harshith, Y. S., Menariya, N., Nagendra Babu, S. S. V., Venkateswaran, S., Batchu, R. K., & Kumar, C. (2026). *A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification.* Springer LNEE, Confluence 2026 proceedings (accepted, in production).
