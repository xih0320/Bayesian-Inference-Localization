def forward_backward_smoothing(prior, transition, emission, observations):
    T = len(observations)
    states = ['enough', 'not']


    fwd = [{} for _ in range(T)]
    for s in states:
        fwd[0][s] = prior[s] * emission[s][observations[0]]
    normalize(fwd[0])

    for t in range(1, T):
        for s in states:
            fwd[t][s] = sum(fwd[t-1][sp] * transition[sp][s] for sp in states) * emission[s][observations[t]]
        normalize(fwd[t])


    bwd = {s: 1.0 for s in states}
    smoothed = [None] * T
    for t in reversed(range(T)):
        smoothed[t] = {
            s: fwd[t][s] * bwd[s] for s in states
        }
        normalize(smoothed[t])
        # update backward message
        bwd = {
            s: sum(transition[s][sp] * emission[sp][observations[t]] * bwd[sp] for sp in states)
            for s in states
        }
    return smoothed

def fixed_lag_smoothing(prior, transition, emission, observations, lag=3):
    states = ['enough', 'not']
    T = len(observations)


    fwd = [{}]
    for s in states:
        fwd[0][s] = prior[s] * emission[s][observations[0]]
    normalize(fwd[0])

    results = []
    window = deque()

    for t in range(1, T):
        fwd_t = {}
        for s in states:
            fwd_t[s] = sum(fwd[-1][sp] * transition[sp][s] for sp in states) * emission[s][observations[t]]
        normalize(fwd_t)
        fwd.append(fwd_t)

        # Store only (lag+1) messages
        if len(fwd) > lag + 1:
            fwd.popleft()

        if t >= lag:
            smoothed = {}
            t_lag = lag
            # backward pass: one step
            bwd = {s: 1.0 for s in states}
            for i in reversed(range(t_lag+1)):
                smoothed = {s: fwd[i][s] * bwd[s] for s in states}
                normalize(smoothed)
                bwd = {s: sum(transition[s][sp] * emission[sp][observations[t - t_lag + i]] * bwd[sp] for sp in states) for s in states}
            results.append(smoothed)
        else:
            results.append(None)
    return results

from collections import deque
