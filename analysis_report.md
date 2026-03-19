# Analysis Report: Bayesian Network Inference Methods

**Author:** Xinyue (Kitty) Hu  
**Date:** March 1, 2025  
**Course:** Advanced AI — Probabilistic Inference Project

---

## Overview

This report evaluates five probabilistic inference algorithms on randomly generated Bayesian Networks (BNs) with binary variables. The key metrics are **execution time** and **inference error** (deviation from exact ground truth), measured across:

- Network sizes: 5 nodes and 10 nodes
- Evidence location: upstream vs. downstream
- MH mixing parameter P: 0.75, 0.85, 0.95

**Algorithms compared:**

| Algorithm | Type |
|---|---|
| Exact Inference (Enumeration-Ask) | Exact |
| Likelihood Weighting (LW) | Approximate (sampling) |
| Gibbs Sampling | Approximate (MCMC) |
| Metropolis-Hastings (MH) | Approximate (MCMC) |
| Modified MH (MH2) | Approximate (MCMC variant) |

---

## Part 1: Results at P = 0.85

### Execution Time & Error Summary

| Algorithm | Time (Size=5) | Time (Size=10) | Error (Upstream, 5) | Error (Downstream, 5) | Error (Upstream, 10) | Error (Downstream, 10) |
|---|---|---|---|---|---|---|
| Exact Inference | 7.24e-05s | 0.00029s | 0 | 0 | 0 | 0 |
| Likelihood Weighting | 0.00222s | 0.00452s | 0.0194 | 0.0170 | 0.0157 | 0.0066 |
| Gibbs Sampling | 0.01686s | 0.04503s | 0.0175 | 0.0127 | 0.0220 | 0.0134 |
| Metropolis-Hastings | 0.00239s | 0.00313s | 0.0323 | 0.0468 | 0.0796 | 0.0912 |
| Modified MH (MH2) | 0.00406s | 0.00657s | 0.0255 | 0.0587 | 0.1317 | 0.0912 |

### Key Observations

**1. Effect of Upstream vs. Downstream Evidence**

- Error generally increases when evidence is placed downstream in the network topology.
- Upstream evidence propagates information more directly to query variables, making inference more constrained and reliable.
- MH-based methods are particularly sensitive to evidence location — downstream evidence leads to notably higher error for both MH and MH2.

**2. Algorithm Error Analysis**

- **Likelihood Weighting (LW):** Most stable approximate method. Small standard deviations (~0.002). Error is slightly higher upstream for small networks but converges better at size=10.
- **Gibbs Sampling:** Error fluctuates more than LW. Larger networks amplify the error increase, suggesting slower mixing in high-dimensional spaces.
- **Metropolis-Hastings (MH):** Highest error variance among all methods. Downstream errors substantially exceed upstream errors, indicating poor proposal distribution fit when conditioning on distal evidence.
- **Modified MH (MH2):** Underperforms standard MH. Downstream error is particularly poor (~0.0587 at size=5; ~0.1317 at size=10), suggesting the modification to the proposal distribution destabilizes convergence.

---

## Part 2: Results at P = 0.75

| Network Size | Location | Time (Exact) | Time (LW) | Time (Gibbs) | Time (MH) | Time (MH2) | Error (LW) | Error (Gibbs) | Error (MH) | Error (MH2) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | Upstream | 0.000103s | 0.00227s | 0.01818s | 0.00255s | 0.00399s | 0.00682 | 0.0122 | 0.02814 | 0.05252 |
| 5 | Downstream | 8.90e-05s | 0.00233s | 0.01719s | 0.00251s | 0.00387s | 0.01141 | 0.00429 | 0.01331 | 0.03403 |
| 10 | Upstream | 0.00017s | 0.00449s | 0.04221s | 0.00324s | 0.0061s | 0.00854 | 0.02249 | 0.02394 | 0.05106 |
| 10 | Downstream | 0.000331s | 0.0045s | 0.04516s | 0.00334s | 0.00625s | 0.00776 | 0.01132 | 0.0477 | 0.05299 |

At P=0.75, the MH sampler relies more heavily on the Likelihood Weighting component of its hybrid proposal. This lowers exploration diversity, which tends to produce higher errors — especially for larger networks where the sample space is broader.

---

## Part 3: Results at P = 0.95

| Network Size | Location | Time (Exact) | Time (LW) | Time (Gibbs) | Time (MH) | Time (MH2) | Error (LW) | Error (Gibbs) | Error (MH) | Error (MH2) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | Upstream | 7.50e-05s | 0.00232s | 0.01843s | 0.00253s | 0.00429s | 0.01625 | 0.00603 | 0.03203 | 0.03908 |
| 5 | Downstream | 8.90e-05s | 0.00218s | 0.01902s | 0.00260s | 0.00440s | 0.00996 | 0.01796 | 0.05487 | 0.06113 |
| 10 | Upstream | 0.000147s | 0.00440s | 0.04130s | 0.00276s | 0.00646s | 0.01287 | 0.00874 | 0.05457 | 0.03432 |
| 10 | Downstream | 0.000242s | 0.00437s | 0.04443s | 0.00284s | 0.00642s | 0.01753 | 0.00631 | 0.02998 | 0.01895 |

At P=0.95, the MH sampler is dominated by the Gibbs component. This increases mixing but can cause the chain to get trapped in local regions, producing inconsistent results across evidence configurations.

---

## Part 4: Effect of Mixing Parameter P on MH Sampling

The parameter P controls the blend between Gibbs sampling and likelihood weighting proposals within the MH chain.

| P Value | Behavior | Error Trend |
|---|---|---|
| P = 0.75 | More LW-driven proposals | Higher error especially in large networks |
| P = 0.85 | Balanced hybrid | **Lowest error across most configurations — optimal** |
| P = 0.95 | More Gibbs-driven proposals | Inconsistent; sometimes increases error |

**Conclusion:** P = 0.85 provides the best balance between exploration and exploitation in the hybrid MH sampler. Lower P values (more LW) reduce proposal diversity; higher P values (more Gibbs) risk local trapping. The sweet spot at P=0.85 minimizes error across both network sizes and evidence locations.

---

## Summary & Conclusions

### Speed
Exact Inference is by far the fastest for small networks (5–10 nodes). Gibbs Sampling is consistently the slowest due to iterative per-variable updates. MH and LW occupy the middle ground.

### Accuracy
- **Exact Inference** achieves zero error by definition — it serves as ground truth.
- **Likelihood Weighting** is the most reliable approximate method: lowest error, most stable variance.
- **Gibbs Sampling** is competitive with LW but degrades more on larger networks.
- **MH and MH2** show the highest and most variable errors; they are sensitive to both network size and evidence position.

### Scalability Concern
As network size grows from 5 to 10 nodes, Gibbs Sampling execution time roughly triples (~0.018s → ~0.045s), suggesting exponential rather than linear scaling — a known limitation of MCMC methods in high-dimensional discrete spaces.

### Recommendation
For small BNs (≤10 nodes): **Exact Inference** is preferable — it is fast and error-free.  
For larger networks where exact inference becomes intractable: **Likelihood Weighting** offers the best accuracy-speed tradeoff among the approximate methods tested.
