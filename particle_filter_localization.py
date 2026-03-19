import numpy as np
from localization_env import Environment, Headings
import random
from collections import Counter
import pygame

# 1. 初始化环境
env = Environment(
    action_bias=0.0,
    observation_noise=0.2,
    action_noise=0.0,
    dimensions=(10, 4),
    seed=42,
    window_size=[750, 300]
)

# HMM Implementation
class HMM:
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

    def forward(self, obs_seq):
        N = len(self.states)
        T = len(obs_seq)
        fwd = np.zeros((T, N))
        for i in range(N):
            fwd[0, i] = self.start_prob[i] * self.emit_prob[i, obs_seq[0]]
        for t in range(1, T):
            for j in range(N):
                fwd[t, j] = np.sum(fwd[t-1] * self.trans_prob[:, j]) * self.emit_prob[j, obs_seq[t]]
        return fwd

    def backward(self, obs_seq):
        N = len(self.states)
        T = len(obs_seq)
        bwd = np.zeros((T, N))
        bwd[T-1] = 1
        for t in reversed(range(T-1)):
            for i in range(N):
                bwd[t, i] = np.sum(self.trans_prob[i, :] * self.emit_prob[:, obs_seq[t+1]] * bwd[t+1])
        return bwd

    def smoothing(self, obs_seq):
        fwd = self.forward(obs_seq)
        bwd = self.backward(obs_seq)
        posterior = fwd * bwd
        posterior /= posterior.sum(axis=1, keepdims=True)
        return posterior

    def viterbi(self, obs_seq):
        N = len(self.states)
        T = len(obs_seq)
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        for i in range(N):
            delta[0, i] = self.start_prob[i] * self.emit_prob[i, obs_seq[0]]
        for t in range(1, T):
            for j in range(N):
                seq_probs = delta[t-1] * self.trans_prob[:, j]
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) * self.emit_prob[j, obs_seq[t]]
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        for t in reversed(range(1, T)):
            path[t-1] = psi[t, path[t]]
        return path

    def fixed_lag_smoothing(self, obs_seq, lag):
        N = len(self.states)
        T = len(obs_seq)
        fwd = np.zeros((T, N))
        bwd = np.ones((lag+1, N))
        smoothed = []
        for t in range(T):
            if t == 0:
                fwd[t] = self.start_prob * self.emit_prob[:, obs_seq[t]]
            else:
                fwd[t] = np.dot(fwd[t-1], self.trans_prob) * self.emit_prob[:, obs_seq[t]]
            fwd[t] /= np.sum(fwd[t])
            if t >= lag:
                bwd[-1] = 1
                for k in range(lag, 0, -1):
                    bwd[k-1] = np.dot(self.trans_prob, self.emit_prob[:, obs_seq[t-k+1]] * bwd[k])
                    bwd[k-1] /= np.sum(bwd[k-1])
                smoothed_prob = fwd[t-lag] * bwd[0]
                smoothed_prob /= np.sum(smoothed_prob)
                smoothed.append(smoothed_prob)
        return np.array(smoothed)

# Particle Filter Implementation for DBN
class ParticleFilter:
    def __init__(self, num_particles, state_space, transition_fn, observation_fn):
        self.num_particles = num_particles
        self.state_space = state_space
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.particles = random.choices(state_space, k=num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def update(self, observation):
        self.particles = np.array([self.transition_fn(p) for p in self.particles])
        self.weights = np.array([self.observation_fn(obs=observation, state=p) for p in self.particles])
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        counter = Counter([tuple(p) for p in self.particles])  # 修正后的写法

        max_count = max(counter.values())
        best_states = [state for state, count in counter.items() if count == max_count]
        prob = max_count / self.num_particles
        return best_states, prob

#
state_space = []
for x in range(env.dimensions[0]):
    for y in range(env.dimensions[1]):
        if env.map[x][y] == 0:
            for heading in Headings:
                state_space.append((x, y, heading))

def transition_fn(state):
    x, y, heading = state
    # 采样方向
    direction_probs = env.location_transitions[x][y][heading]
    directions, probs = zip(*direction_probs.items())
    direction = random.choices(directions, probs)[0]
    new_x, new_y = x + direction.value[0], y + direction.value[1]
    # 采样新heading
    heading_probs = env.headings_transitions[new_x][new_y][heading]
    headings, h_probs = zip(*heading_probs.items())
    new_heading = random.choices(headings, h_probs)[0]
    return (new_x, new_y, new_heading)

def observation_fn(obs, state):
    x, y, heading = state
    if env.map[x][y] == 1:
        return 0.0
    return env.observation_tables[x][y].get(tuple(obs), 0.0)

pf = ParticleFilter(num_particles=200, state_space=state_space, transition_fn=transition_fn, observation_fn=observation_fn)

max_steps = 20
for t in range(max_steps):
    obs = env.observe()
    pf.update(obs)
    best_states, prob = pf.get_estimate()
    print(f"Step {t}, Best state(s): {best_states}, Probability: {prob}")

    loc_probs = np.zeros(env.dimensions)
    heading_probs = {h: 0.0 for h in Headings}
    for p in pf.particles:
        x, y, h = p
        loc_probs[x, y] += 1
        heading_probs[h] += 1
    loc_probs /= pf.num_particles
    for h in heading_probs:
        heading_probs[h] /= pf.num_particles

    print("loc_probs max:", np.max(loc_probs), "min:", np.min(loc_probs))  # 

    env.update(loc_probs.tolist(), heading_probs)
    env.move()
    pygame.image.save(env.game.screen, f"step_{t}.png")
