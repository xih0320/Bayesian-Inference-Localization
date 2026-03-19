"""
HMM Temporal Inference
======================
Implements exact temporal inference algorithms for Hidden Markov Models:
  - Forward algorithm           (filtering:  P(X_t | e_{1:t}))
  - Backward algorithm          (backward messages)
  - Forward-Backward smoothing  (smoothing:  P(X_t | e_{1:T}))
  - Fixed-lag smoothing         (online smoothing with bounded memory)
  - Viterbi algorithm           (most likely state sequence)

Example domain: sleep-deprivation monitoring
  Hidden states  : 'enough' (rested), 'not' (sleep-deprived)
  Observations   : 'red_eyes', 'sleeping_in_class', 'none'

Usage
-----
    from hmm_temporal_inference import forward, backward, forward_backward_smoothing
    from hmm_temporal_inference import fixed_lag_smoothing, viterbi

    fwd      = forward(prior, transition, emission, observations)
    bwd      = backward(transition, emission, observations)
    smoothed = forward_backward_smoothing(prior, transition, emission, observations)
    online   = fixed_lag_smoothing(prior, transition, emission, observations, lag=3)
    path     = viterbi(prior, transition, emission, observations)
"""

from collections import deque


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def normalize(distribution: dict) -> None:
    """Normalise a dict distribution in-place so values sum to 1."""
    total = sum(distribution.values())
    if total == 0:
        raise ValueError("Cannot normalise a zero distribution.")
    for key in distribution:
        distribution[key] /= total


# ---------------------------------------------------------------------------
# Forward algorithm — filtering: P(X_t | e_{1:t})
# ---------------------------------------------------------------------------

def forward(prior: dict, transition: dict, emission: dict,
            observations: list) -> list:
    """
    Compute forward messages for each time step.

    Parameters
    ----------
    prior        : {state: P(X_0 = state)}
    transition   : {state_from: {state_to: prob}}
    emission     : {state: {observation: prob}}
    observations : sequence of observed values e_{1:T}

    Returns
    -------
    fwd : list of dicts — fwd[t] = P(X_t | e_{1:t})
    """
    states = list(prior.keys())
    T = len(observations)
    fwd = [{} for _ in range(T)]

    for s in states:
        fwd[0][s] = prior[s] * emission[s][observations[0]]
    normalize(fwd[0])

    for t in range(1, T):
        for s in states:
            fwd[t][s] = (
                sum(fwd[t-1][sp] * transition[sp][s] for sp in states)
                * emission[s][observations[t]]
            )
        normalize(fwd[t])

    return fwd


# ---------------------------------------------------------------------------
# Backward algorithm — backward messages b_{t:T}
# ---------------------------------------------------------------------------

def backward(transition: dict, emission: dict, observations: list) -> list:
    """
    Compute backward messages for each time step.

    Returns
    -------
    bwd : list of dicts — bwd[t] = P(e_{t+1:T} | X_t)
    """
    states = list(transition.keys())
    T = len(observations)
    bwd = [{} for _ in range(T)]

    for s in states:
        bwd[T-1][s] = 1.0

    for t in reversed(range(T - 1)):
        for s in states:
            bwd[t][s] = sum(
                transition[s][sp] * emission[sp][observations[t+1]] * bwd[t+1][sp]
                for sp in states
            )

    return bwd


# ---------------------------------------------------------------------------
# Forward-Backward smoothing — P(X_t | e_{1:T})
# ---------------------------------------------------------------------------

def forward_backward_smoothing(prior: dict, transition: dict, emission: dict,
                                observations: list) -> list:
    """
    Full offline smoothing: combines forward and backward passes.

    Returns
    -------
    smoothed : list of dicts — smoothed[t] = P(X_t | e_{1:T})
    """
    states = list(prior.keys())
    T = len(observations)

    # Forward pass
    fwd = [{} for _ in range(T)]
    for s in states:
        fwd[0][s] = prior[s] * emission[s][observations[0]]
    normalize(fwd[0])
    for t in range(1, T):
        for s in states:
            fwd[t][s] = (
                sum(fwd[t-1][sp] * transition[sp][s] for sp in states)
                * emission[s][observations[t]]
            )
        normalize(fwd[t])

    # Backward pass (single running message)
    bwd = {s: 1.0 for s in states}
    smoothed = [None] * T

    for t in reversed(range(T)):
        smoothed[t] = {s: fwd[t][s] * bwd[s] for s in states}
        normalize(smoothed[t])
        bwd = {
            s: sum(
                transition[s][sp] * emission[sp][observations[t]] * bwd[sp]
                for sp in states
            )
            for s in states
        }

    return smoothed


# ---------------------------------------------------------------------------
# Fixed-lag smoothing — online, O(lag) memory
# ---------------------------------------------------------------------------

def fixed_lag_smoothing(prior: dict, transition: dict, emission: dict,
                         observations: list, lag: int = 3) -> list:
    """
    Online smoothing estimating P(X_{t-lag} | e_{1:t}) at each step.
    Uses a sliding window of (lag+1) forward messages — constant memory.

    Returns
    -------
    results : list of length T.
              results[t] = smoothed distribution at time (t - lag),
              or None for t < lag (insufficient history).
    """
    states = list(prior.keys())
    T = len(observations)

    fwd_window = deque()
    f0 = {s: prior[s] * emission[s][observations[0]] for s in states}
    normalize(f0)
    fwd_window.append(f0)

    results = [None]

    for t in range(1, T):
        f_prev = fwd_window[-1]
        f_new = {}
        for s in states:
            f_new[s] = (
                sum(f_prev[sp] * transition[sp][s] for sp in states)
                * emission[s][observations[t]]
            )
        normalize(f_new)
        fwd_window.append(f_new)

        if len(fwd_window) > lag + 1:
            fwd_window.popleft()

        if t >= lag:
            bwd = {s: 1.0 for s in states}
            window_list = list(fwd_window)
            for i in reversed(range(len(window_list))):
                obs_idx = t - (len(window_list) - 1 - i)
                smoothed = {s: window_list[i][s] * bwd[s] for s in states}
                normalize(smoothed)
                if i > 0:
                    bwd = {
                        s: sum(
                            transition[s][sp] * emission[sp][observations[obs_idx]] * bwd[sp]
                            for sp in states
                        )
                        for s in states
                    }
            results.append(smoothed)
        else:
            results.append(None)

    return results


# ---------------------------------------------------------------------------
# Viterbi algorithm — most likely state sequence
# ---------------------------------------------------------------------------

def viterbi(prior: dict, transition: dict, emission: dict,
            observations: list) -> list:
    """
    Find the most likely hidden state sequence via dynamic programming.

    Returns
    -------
    path : list of states — most probable sequence X_{1:T}
    """
    states = list(prior.keys())
    T = len(observations)

    delta = [{} for _ in range(T)]
    psi   = [{} for _ in range(T)]

    for s in states:
        delta[0][s] = prior[s] * emission[s][observations[0]]
        psi[0][s]   = None

    for t in range(1, T):
        for s in states:
            best_prob, best_prev = max(
                (delta[t-1][sp] * transition[sp][s], sp)
                for sp in states
            )
            delta[t][s] = best_prob * emission[s][observations[t]]
            psi[t][s]   = best_prev

    path = [None] * T
    path[T-1] = max(states, key=lambda s: delta[T-1][s])
    for t in reversed(range(T - 1)):
        path[t] = psi[t+1][path[t+1]]

    return path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prior = {'enough': 0.5, 'not': 0.5}

    transition = {
        'enough': {'enough': 0.7, 'not': 0.3},
        'not':    {'enough': 0.2, 'not': 0.8},
    }

    emission = {
        'enough': {'red_eyes': 0.2, 'sleeping_in_class': 0.1, 'none': 0.7},
        'not':    {'red_eyes': 0.7, 'sleeping_in_class': 0.6, 'none': 0.1},
    }

    observations = ['red_eyes', 'sleeping_in_class', 'none', 'red_eyes', 'red_eyes']

    print("=== Forward (filtering) ===")
    fwd = forward(prior, transition, emission, observations)
    for t, f in enumerate(fwd):
        print(f"  t={t}: { {k: round(v,4) for k,v in f.items()} }")

    print("\n=== Forward-Backward Smoothing ===")
    smoothed = forward_backward_smoothing(prior, transition, emission, observations)
    for t, s in enumerate(smoothed):
        print(f"  t={t}: { {k: round(v,4) for k,v in s.items()} }")

    print("\n=== Fixed-Lag Smoothing (lag=2) ===")
    online = fixed_lag_smoothing(prior, transition, emission, observations, lag=2)
    for t, s in enumerate(online):
        print(f"  t={t}: {s if s is None else {k: round(v,4) for k,v in s.items()}}")

    print("\n=== Viterbi (most likely path) ===")
    path = viterbi(prior, transition, emission, observations)
    print(f"  path: {path}")
