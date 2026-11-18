# Label-Smoothed Dual-Branch Classifier for ClassiFIM-Style Estimation Under Readout Noise

**Author:** Surat Sai Venkata Nagendra Babu, B.Tech CSE, Amrita Vishwa Vidyapeetham

A proof-of-concept experiment inspired by ClassiFIM (Kasatkin, Mozgunov, Ezzell, Mishra, Hen & Lidar, [arXiv:2408.03323](https://arxiv.org/abs/2408.03323)). It uses synthetic data generated in the notebook.

## 1. Background
ClassiFIM finds phase transitions without labels: a classifier is trained to tell apart data from two nearby parameter values (lambda - delta and lambda + delta). If it can do so easily, the data changes quickly there, which the Fisher information metric (FIM) measures; a peak marks a likely transition. The authors note that overconfident classifiers inflate the log-odds and use small L2 regularization to counter it.

**Question:** do label smoothing, dual branches and model averaging keep the estimate stable when 6% of bits are flipped by readout noise?

## 2. Setup (in the notebook)
- **Data:** 16-spin chains, 31 values of lambda from 0.25 to 1.75, 600 samples each. A hand-written sigmoid sets the order, with the transition at lambda = 1.0; 6% of bits are flipped.
- **Estimate:** variance of the classifier's log-odds divided by (2 * delta)^2 (a simplified version of the paper's estimator), compared with the squared derivative of the same sigmoid.
- **Baseline:** MLP (two dense layers of 64), no regularization.
- **Proposed:** two 1D convolution branches (standard and dilated), sigmoid attention gate, dropout, L2 (1e-3), label smoothing (0.08), average of 2 models.

## 3. Results (one run, seed 42, 3-point smoothing across lambda)
| Metric vs. reference curve | Baseline MLP | Proposed |
| :--- | ---: | ---: |
| MSE | 0.1866 | **0.0353** |
| R^2 | -0.6364 | **+0.6907** |
| Estimated peak | 1.10 | 1.10 |

The proposed classifier follows the reference curve much more closely; both models place the peak at lambda = 1.10.

![ClassiFIM results](ClassiFIM_Phase_Transition_Results.png)

## 4. Scope and next steps
This version uses synthetic data, one run, and a simplified estimator. Next:
1. Repeat over several seeds and noise levels, with error bars.
2. Add an L2-only baseline (as in the paper) to isolate the effect of label smoothing.
3. Use the public ClassiFIM datasets and the paper's own estimator.
4. Test on exact transverse-field Ising snapshots.

## 5. References
1. Kasatkin, V., Mozgunov, E., Ezzell, N., Mishra, U., Hen, I., & Lidar, D. A. (2024). *ClassiFIM: An Unsupervised Method To Detect Phase Transitions.* arXiv:2408.03323.
2. Harshith, Y. S., et al. (2026). *A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification.* Springer LNEE, Confluence 2026 proceedings (accepted, in production).
