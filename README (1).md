# Bayesian Inference & Robot Localization

Implementation of probabilistic inference algorithms including Bayesian Networks, Hidden Markov Models, and Particle Filters, applied to a robot localization problem in a dynamic maze environment.

---

## Overview

This project covers three interconnected areas of probabilistic AI:

1. **Exact Inference in Bayesian Networks** — enumeration-based exact inference with multiprocessing support
2. **HMM Temporal Inference** — Forward-Backward smoothing, Fixed-lag smoothing, Viterbi algorithm
3. **Robot Localization** — Particle Filter and DBN-based localization in a stochastic maze environment

---

## Part 1: Bayesian Network Exact Inference

Implements the **Enumeration-Ask** algorithm for exact inference in Bayesian Networks with binary random variables.

**Key features:**
- Loads arbitrary BN structures from JSON
- Topological sort for correct variable elimination order
- Optional multiprocessing to parallelize query evaluation
- Random BN generator supporting DAG and polytree structures

**Usage:**
```bash
# Generate a random Bayesian network
python bayesian_network_generator.py dag 10 --max-parents 3 --output-file bn.json

# Run exact inference
python exact_inference.py -f bn.json -q 5
```

**Benchmark results (from experiments):**

| Network Size | Evidence Location | Exact Inference Time | Error |
|-------------|-------------------|---------------------|-------|
| 5 nodes | Upstream | 7.24e-05s | 0 |
| 5 nodes | Downstream | 7.24e-05s | 0 |
| 10 nodes | Upstream | 0.00029s | 0 |
| 10 nodes | Downstream | 0.00029s | 0 |

Exact inference has zero error by definition — it serves as the ground truth for comparing approximate methods.

---

## Part 2: HMM Temporal Inference

Implements multiple inference algorithms for Hidden Markov Models applied to a sleep-deprivation monitoring example (hidden state: rested/not-rested; observations: red eyes, sleeping in class).

**Algorithms implemented:**
- **Forward algorithm** — filtering: P(Xt | e1:t)
- **Forward-Backward algorithm** — smoothing: P(Xt | e1:T)
- **Fixed-lag smoothing** — online smoothing with bounded memory
- **Viterbi algorithm** — most likely state sequence
- **Constant-space smoothing** — memory-efficient backward pass

**Usage:**
```python
from hmm_temporal_inference import forward, backward, forward_backward_smoothing, viterbi

fwd      = forward(prior, transition, emission, observations)
smoothed = forward_backward_smoothing(prior, transition, emission, observations)
path     = viterbi(prior, transition, emission, observations)
```

---

## Part 3: Robot Localization with Particle Filter

Locates a robot in a stochastic maze environment modeled as a Dynamic Bayesian Network. The robot's hidden state is (x, y, heading) and observations are noisy sensor readings.

**Key features:**
- 200-particle filter with systematic resampling
- Transition model samples from environment's stochastic movement tables
- Observation model uses environment's pre-computed observation probability tables
- Real-time pygame visualization of particle distribution
- Results exported to CSV

**Sample results (first 4 steps):**

| Time Step | Most Likely State | Probability |
|-----------|------------------|-------------|
| 1 | (12,4,W), (11,4,W) | 0.026 |
| 2 | (1,3,W) | 0.040 |
| 3 | (2,3,W), (4,3,E) | 0.040 |
| 4 | (12,4,E) | 0.052 |

**Usage:**
```bash
python particle_filter_localization.py
python robot_localization_dbn.py
```

---

## File Structure

```
├── exact_inference.py              # Bayesian network exact inference (Enumeration-Ask)
├── bayesian_network_generator.py   # Random BN generator (DAG / polytree)
├── hmm_temporal_inference.py       # HMM Forward, Backward, Viterbi, Fixed-lag
├── hmm_smoothing.py                # Forward-Backward & Fixed-lag smoothing variants
├── particle_filter_localization.py # HMM + Particle Filter robot localization
├── robot_localization_dbn.py       # DBN-based localization with CSV output
├── localization_env.py             # Stochastic maze environment
├── visualization.py                # Pygame visualization framework
├── maze.py                         # Random maze generator (depth-first search)
└── localization_results.csv        # Sample particle filter output
```

---

## Libraries

`networkx` · `numpy` · `pygame` · `ujson` · `tabulate` · `collections` · `multiprocessing`
