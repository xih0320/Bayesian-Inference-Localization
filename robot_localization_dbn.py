"""
    Robot Localization in a 4x16 Grid (Figure 14.7f)
    Using a DBN Formulation with Particle Filtering
"""

import localization_env as le
import pygame
import numpy as np
import time
import argparse
from typing import List, Tuple, Dict
from enum import Enum

import tabulate  # For printing nice tables
import csv  # For CSV file output

# Helper functions for particle filtering
def encode_state(x: int, y: int, h: int) -> int:
    """
    Encode (x, y, h) state into a single index
    x, y are coordinates, h is heading index
    """
    # Calculate the maximum number of cells
    max_cells = 18 * 6
    
    # Calculate the cell index
    cell_idx = x + 18 * y
    
    # Bound check to avoid index errors
    if cell_idx >= max_cells:
        cell_idx = max_cells - 1  # Use the last valid cell
    
    # Calculate the state index
    state_id = cell_idx * 4 + h
    
    # Ensure state_id is within bounds (168 = 42 cells * 4 headings)
    if state_id >= 168:
        state_id = 167  # Use the last valid state
        
    return state_id

def decode_state(state_id: int) -> Tuple[int, int, int]:
    """
    Decode a state_id back to (x, y, h)
    """
    h = state_id // (18*6)
    remainder = state_id % (18*6)
    y = remainder // 18
    x = remainder % 18
    return (x, y, h)

def heading_to_index(heading: le.Headings) -> int:
    """
    Convert a Heading enum to an index (0-3)
    Assuming order is E, N, W, S
    """
    if heading == le.Headings.E:
        return 0
    elif heading == le.Headings.N:
        return 1
    elif heading == le.Headings.W:
        return 2
    elif heading == le.Headings.S:
        return 3
    raise ValueError(f"Unknown heading: {heading}")

def index_to_heading(index: int) -> le.Headings:
    """
    Convert an index (0-3) to a Heading enum
    Assuming order is E, N, W, S
    """
    if index == 0:
        return le.Headings.E
    elif index == 1:
        return le.Headings.N
    elif index == 2:
        return le.Headings.W
    elif index == 3:
        return le.Headings.S
    raise ValueError(f"Invalid heading index: {index}")

def index_to_heading_str(index: int) -> str:
    """
    Convert an index (0-3) to a heading string
    """
    headings = ['E', 'N', 'W', 'S']
    return headings[index]

def build_transition_alias_tables(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build transition tables for fast sampling
    Returns:
        next_ids: For each state_id, the possible next state ids
        next_cdf: Cumulative distribution for each transition
    """
    n_states = 168  # 42 cells * 4 headings
    max_transitions = 5  # Max number of possible transitions (4 directions + stationary)
    
    # Initialize arrays with -1 (invalid transition)
    next_ids = np.full((n_states, max_transitions), -1, dtype=int)
    next_cdf = np.zeros((n_states, max_transitions), dtype=float)
    
    # For each free cell
    for cell_idx, (x, y) in enumerate(env.free_cells):
        # Skip if out of bounds
        if cell_idx >= 42 or x >= 18 or y >= 6:
            continue
            
        for h_idx in range(4):  # 0=E, 1=N, 2=W, 3=S
            state_id = encode_state(x, y, h_idx)
            
            # Skip if invalid state
            if state_id >= n_states:
                continue
                
            # Fill in transitions
            idx = 0
            cumulative_prob = 0.0
            
            # Default transition: stay in place with same heading
            next_ids[state_id, 0] = state_id
            next_cdf[state_id, 0] = 1.0
            
            # Add possible transitions with equal probability
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # S, E, N, W
            
            for dir_idx, (dx, dy) in enumerate(directions):
                if idx >= max_transitions - 1:
                    break  # Don't exceed array bounds
                    
                nx, ny = x + dx, y + dy
                
                # Check if next position is valid
                if 0 <= nx < env.dimensions[0] and 0 <= ny < env.dimensions[1] and env.map[nx][ny] != 1:
                    # Add transition to next position with same heading
                    next_state_id = encode_state(nx, ny, h_idx)
                    idx += 1
                    next_ids[state_id, idx] = next_state_id
                    cumulative_prob += 0.25  # Equal probability for each direction
                    next_cdf[state_id, idx] = cumulative_prob
            
            # Normalize CDF
            if idx > 0:
                next_cdf[state_id, 1:idx+1] /= cumulative_prob
    
    return next_ids, next_cdf

def build_sensor_table(env) -> np.ndarray:
    """
    Build sensor likelihood table
    Shape: (168, 16) where 
    - 168 is number of states (42 cells * 4 headings)
    - 16 is number of possible observations (2^4)
    
    Each entry [s, o] is P(e=o | X=s)
    """
    n_states = 168
    n_observations = 16
    
    # Initialize likelihood table
    sensor_like = np.zeros((n_states, n_observations), dtype=float)
    
    # For each valid state
    for cell_idx, (x, y) in enumerate(env.free_cells):
        if 0 <= x < env.dimensions[0] and 0 <= y < env.dimensions[1]:
            for h_idx in range(4):  # 0=E, 1=N, 2=W, 3=S
                state_id = encode_state(x, y, h_idx)
                
                # Generate the expected observation at this position
                # [S, E, N, W] = 1 for wall, 0 for free space
                observation = [0, 0, 0, 0]
                
                # Check each direction (S, E, N, W)
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # S, E, N, W
                for i, (dx, dy) in enumerate(directions):
                    nx, ny = x + dx, y + dy
                    # Check if neighbor is a wall or out of bounds
                    if (nx < 0 or nx >= env.dimensions[0] or 
                        ny < 0 or ny >= env.dimensions[1] or 
                        env.map[nx][ny] == 1):
                        observation[i] = 1  # Wall
                
                # Calculate the expected observation ID
                expected_obs_id = observation[0] + 2*observation[1] + 4*observation[2] + 8*observation[3]
                
                # For each possible observation
                for obs_id in range(n_observations):
                    # Count bits that differ between expected and actual observation
                    obs_bits = [obs_id & 1, (obs_id >> 1) & 1, (obs_id >> 2) & 1, (obs_id >> 3) & 1]
                    diff_count = sum(o != e for o, e in zip(obs_bits, observation))
                    
                    # Calculate probability based on observation noise
                    if diff_count == 0:  # Perfect match
                        prob = (1 - env.observation_noise) ** 4
                    else:
                        # Each differing bit reduces probability
                        prob = (1 - env.observation_noise) ** (4 - diff_count) * env.observation_noise ** diff_count
                    
                    sensor_like[state_id, obs_id] = prob
    
    return sensor_like

def init_particles(env, N: int, seed: int) -> np.ndarray:
    """
    Return N i.i.d. samples from the prior P(X0)
    Using uniform distribution over valid cells and headings
    """
    rng = np.random.default_rng(seed)
    
    # Get valid cells
    free_cells = env.free_cells
    n_cells = len(free_cells)
    
    # Ensure we have valid cells
    if n_cells == 0:
        # Fallback to random positions
        particles = np.zeros((N, 3), dtype=int)
        for i in range(N):
            particles[i] = [rng.integers(1, 17), rng.integers(1, 5), rng.integers(0, 4)]
        return particles
    
    # Initialize particles array
    particles = np.zeros((N, 3), dtype=int)
    
    # Sample N particles
    for i in range(N):
        # Sample cell uniformly (with bounds checking)
        cell_idx = rng.integers(0, min(n_cells, 42))
        if cell_idx < len(free_cells):
            x, y = free_cells[cell_idx]
            # Ensure x,y are in bounds
            x = min(x, 17)
            y = min(y, 5)
            # Sample heading uniformly
            h = rng.integers(0, 4)  # 0=E, 1=N, 2=W, 3=S
            particles[i] = [x, y, h]
        else:
            # Fallback
            particles[i] = [1, 1, 0]
    
    return particles

def sample_next_state(particle: np.ndarray, next_ids, next_cdf, rng) -> np.ndarray:
    """
    Sample next state according to transition model P(X'|X)
    """
    x, y, h = particle
    state_id = encode_state(x, y, h)
    
    # Get possible next states and their CDFs
    possible_next_ids = next_ids[state_id]
    cdf = next_cdf[state_id]
    
    # If no valid transitions (shouldn't happen in well-formed model)
    if possible_next_ids[0] == -1:
        return particle.copy()  # Stay in place
    
    # Sample using inverse CDF
    u = rng.random()
    
    # Find the index where u falls in the CDF
    idx = np.searchsorted(cdf, u)
    if idx >= len(possible_next_ids) or possible_next_ids[idx] == -1:
        idx = 0  # Fallback to first valid transition
    
    # Get the sampled next state ID
    next_state_id = possible_next_ids[idx]
    
    # Decode back to (x,y,h)
    if next_state_id == -1:
        return particle.copy()  # Stay in place if invalid
    
    nx, ny, nh = decode_state(next_state_id)
    return np.array([nx, ny, nh])

def weight_particle(particle, obs_id: int, sensor_like) -> float:
    """
    Return P(e_t | X_t = particle) from the likelihood table
    """
    x, y, h = particle
    state_id = encode_state(x, y, h)
    return sensor_like[state_id, obs_id]

def systematic_resample(particles, weights, rng) -> np.ndarray:
    """
    O(N) stratified resampling; returns unweighted particle set
    """
    N = len(weights)
    
    # Handle edge cases
    if N == 0:
        return particles
        
    if np.sum(weights) == 0:
        # If all weights are zero, return uniform sampling
        indices = rng.integers(0, N, size=N)
        return particles[indices]
    
    # Normalize weights if they don't sum to 1
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)
    
    # Simple implementation of systematic resampling
    positions = (np.arange(N) + rng.random()) / N
    indices = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
            if j >= N:  # Handle case where cumulative sum doesn't reach 1.0
                indices[i:] = N-1
                break
    
    return particles[indices]

def most_likely_states(particles) -> Tuple[List[Tuple[Tuple[int, int], str]], float]:
    """
    Find the most likely state(s) from particle counts
    Returns list of ((x,y), heading_str) tuples and their probability
    """
    N = len(particles)
    
    # Count occurrences of each state
    state_counts = {}
    for p in particles:
        x, y, h = p
        key = ((x, y), index_to_heading_str(h))
        if key in state_counts:
            state_counts[key] += 1
        else:
            state_counts[key] = 1
    
    # Find max frequency
    max_count = max(state_counts.values())
    max_prob = max_count / N
    
    # Find all states with this frequency (for ties)
    max_states = [state for state, count in state_counts.items() 
                 if np.isclose(count/N, max_prob)]
    
    return max_states, max_prob

def particle_filter(observations, env, N=1000, seed=42):
    """
    Run particle filter algorithm on observation sequence
    Returns history of most likely states for each timestep
    """
    rng = np.random.default_rng(seed)
    
    # 0. Pre-compute tables for fast inner loop
    print("Building transition tables...")
    next_ids, next_cdf = build_transition_alias_tables(env)
    print("Building sensor likelihood tables...")
    sensor_like = build_sensor_table(env)
    
    # 1. Initialize
    print(f"Initializing {N} particles...")
    P = init_particles(env, N, seed)  # shape (N, 3)
    T = len(observations)
    hist = []  # to store answers per step
    
    # 2. Filtering loop
    print(f"Running particle filter for {T} steps...")
    for t in range(1, T+1):
        e_t = observations[t-1]
        # Compute observation ID from tuple (S,E,N,W)
        obs_id = e_t[0] + 2*e_t[1] + 4*e_t[2] + 8*e_t[3]
        
        # (a) PROPAGATE - Sample next state for each particle
        P_next = np.zeros_like(P)
        for i in range(N):
            P_next[i] = sample_next_state(P[i], next_ids, next_cdf, rng)
        P = P_next
        
        # (b) WEIGHT - Calculate importance weights
        W = np.zeros(N)
        for i in range(N):
            x, y, h = P[i]
            # Check for valid state
            if 0 <= x < env.dimensions[0] and 0 <= y < env.dimensions[1] and 0 <= h < 4:
                state_id = encode_state(x, y, h)
                if state_id < sensor_like.shape[0]:
                    W[i] = sensor_like[state_id, obs_id]
        
        # Handle zero weights (all particles incompatible)
        if np.sum(W) < 1e-10:
            print(f"Warning: Zero total weight at step {t}. Reinitializing particles.")
            P = init_particles(env, N, seed + t)
            W = np.ones(N) / N
        else:
            # Normalize weights
            W = W / np.sum(W)
        
        # (c) RESAMPLE - Create unweighted particle set
        try:
            P = systematic_resample(P, W, rng)
        except Exception as e:
            print(f"Resampling error at step {t}: {e}")
            # Fall back to simple random resampling
            indices = rng.choice(N, size=N, p=W)
            P = P[indices]
        
        # 3. Estimate posterior & store answer
        argmax_states, p_star = most_likely_states(P)
        hist.append((t, argmax_states, p_star))
        
        # Print current step results
        print(f"{t}: p* = {p_star:.3f}   arg-max state(s) = {argmax_states}")
    
    return hist

class ParticleFilter:
    """
    Particle filter implementation for robot localization
    """
    def __init__(self, env, n_particles=1000, seed=42):
        """
        Initialize the particle filter
        """
        self.env = env
        self.n_particles = n_particles
        self.rng = np.random.RandomState(seed)
        
        # Initialize particles
        self.initialize_particles()
        
    def initialize_particles(self):
        """
        Initialize particles with uniform distribution over valid states
        """
        # Get free cells
        free_cells = self.env.free_cells
        n_free_cells = len(free_cells)
        
        # Initialize particles with uniform distribution
        self.particles = np.zeros((self.n_particles, 3), dtype=int)
        
        # Sample particles
        for i in range(self.n_particles):
            # Random free cell
            cell_idx = self.rng.randint(0, n_free_cells)
            x, y = free_cells[cell_idx]
            
            # Random heading (0=E, 1=N, 2=W, 3=S)
            h = self.rng.randint(0, 4)
            
            self.particles[i] = [x, y, h]
        
        # Equal weights
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def update(self, observation):
        """
        Update particle filter with new observation
        Steps: prediction, weighting, resampling
        """
        # 1. Prediction step - move particles according to motion model
        self.predict()
        
        # 2. Update step - weight particles based on observation
        self.weight_particles(observation)
        
        # 3. Resample step - resample particles based on weights
        self.resample()
        
        # Return most likely state
        return self.get_most_likely_state()
    
    def predict(self):
        """
        Move particles according to motion model
        """
        # For each particle
        for i in range(self.n_particles):
            x, y, h = self.particles[i]
            
            # Possible movements: forward, left, right, or stay
            directions = [
                (0, 0),  # Stay
                self.get_direction_vector(h),  # Forward
                self.get_direction_vector((h+1) % 4),  # Left
                self.get_direction_vector((h-1) % 4)   # Right
            ]
            
            # Choose direction with some randomness
            # Higher probability to continue in same direction
            probs = [0.1, 0.6, 0.15, 0.15]
            direction_idx = self.rng.choice(4, p=probs)
            dx, dy = directions[direction_idx]
            
            # Update position
            new_x, new_y = x + dx, y + dy
            
            # Check if new position is valid (not a wall or out of bounds)
            if (0 <= new_x < self.env.dimensions[0] and 
                0 <= new_y < self.env.dimensions[1] and 
                self.env.map[new_x][new_y] != 1):
                x, y = new_x, new_y
            
            # Possibly change heading when hitting a wall
            if direction_idx == 1 and (x, y) == (self.particles[i][0], self.particles[i][1]):
                # If we tried to move forward but couldn't (hit a wall)
                # Choose a new heading
                h = self.rng.randint(0, 4)
            
            self.particles[i] = [x, y, h]
    
    def get_direction_vector(self, heading):
        """
        Convert heading index to direction vector
        0=E, 1=N, 2=W, 3=S
        """
        directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # E, N, W, S
        return directions[heading]
    
    def weight_particles(self, observation):
        """
        Weight particles based on observation
        """
        # observation: [S, E, N, W] - 1 if wall, 0 if free
        
        # For each particle
        for i in range(self.n_particles):
            x, y, h = self.particles[i]
            
            # Calculate expected observation at this position
            expected_obs = self.calculate_expected_observation(x, y)
            
            # Compare with actual observation
            match_count = sum(1 for a, b in zip(observation, expected_obs) if a == b)
            
            # Update weight based on how well observation matches expectation
            # More matches = higher weight
            self.weights[i] = (match_count / 4) ** 2
        
        # Normalize weights
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights /= sum_weights
        else:
            # If all weights are zero, reinitialize
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def calculate_expected_observation(self, x, y):
        """
        Calculate expected observation at position (x, y)
        Returns [S, E, N, W] - 1 if wall, 0 if free
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # S, E, N, W
        observation = []
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if neighbor is wall or out of bounds
            if (nx < 0 or nx >= self.env.dimensions[0] or 
                ny < 0 or ny >= self.env.dimensions[1] or 
                self.env.map[nx][ny] == 1):
                observation.append(1)  # Wall
            else:
                observation.append(0)  # Free
        
        return observation
    
    def resample(self):
        """
        Resample particles based on weights
        """
        # Systematic resampling
        cumsum = np.cumsum(self.weights)
        step = 1.0 / self.n_particles
        u = self.rng.random() * step
        
        indices = []
        i = 0
        
        for j in range(self.n_particles):
            while u > cumsum[i]:
                i += 1
                if i >= self.n_particles:
                    i = self.n_particles - 1
                    break
            indices.append(i)
            u += step
        
        # Create new particles
        self.particles = self.particles[indices]
        
        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_most_likely_state(self):
        """
        Get the most likely state based on particle counts
        Returns ((x, y), heading_str) and probability
        """
        # Count occurrences of each state
        state_counts = {}
        
        for p in self.particles:
            x, y, h = p
            heading_str = ['E', 'N', 'W', 'S'][h]
            key = ((x, y), heading_str)
            
            if key in state_counts:
                state_counts[key] += 1
            else:
                state_counts[key] = 1
        
        # Find max frequency
        if not state_counts:
            return [], 0.0
            
        max_count = max(state_counts.values())
        max_prob = max_count / self.n_particles
        
        # Find all states with this frequency (for ties)
        max_states = [state for state, count in state_counts.items() 
                     if np.isclose(count/self.n_particles, max_prob)]
        
        return max_states, max_prob

class LocalizationFilter:
    """
    Class to handle the DBN filtering for robot localization
    """
    def __init__(self, env):
        """
        Initialize the filter with the environment
        """
        self.env = env
        self.rows = 6
        self.cols = 18
        
        # Initialize our belief state with the prior probabilities
        self.location_probs = self.initialize_location_probs()
        self.heading_probs = self.initialize_heading_probs()
        
    def initialize_location_probs(self):
        """
        Initialize location probabilities based on the prior (uniform over traversable cells)
        """
        location_probs = [[0.0 for _ in range(self.rows)] for _ in range(self.cols)]
        
        # Count traversable cells
        free_cells = 0
        for x in range(self.cols):
            for y in range(self.rows):
                if self.env.map[x][y] != 1:
                    free_cells += 1
        
        # Set uniform probability for each traversable cell
        uniform_prob = 1.0 / free_cells
        for x in range(self.cols):
            for y in range(self.rows):
                if self.env.map[x][y] != 1:
                    location_probs[x][y] = uniform_prob
        
        return location_probs
    
    def initialize_heading_probs(self):
        """
        Initialize heading probabilities (uniform over all headings)
        """
        heading_probs = {}
        for heading in le.Headings:
            heading_probs[heading] = 0.25
        return heading_probs
    
    def update(self, observation):
        """
        Update the belief state based on the observation
        """
        # Filter step
        self.filter_step(observation)
        
    def filter_step(self, observation):
        """
        Perform forward filtering for a single time step
        """
        # Normalize the belief state as needed
        self.normalize_beliefs()
        
        # Here in a full implementation you would:
        # 1. Perform prediction step (time update)
        # 2. Perform correction step (observation update)
        
        # For demonstration, we'll use random probabilities
        # In a real implementation, you would update based on:
        # - Environment's transition models
        # - Observation model
        # - Current belief state
        
        # This is a placeholder - the real implementation would calculate exact probabilities
        for x in range(self.cols):
            for y in range(self.rows):
                if self.env.map[x][y] != 1:
                    # Update location probability based on observation
                    self.location_probs[x][y] = max(0.01, self.location_probs[x][y])
        
        # Add some randomness
        for heading in le.Headings:
            self.heading_probs[heading] = max(0.1, self.heading_probs[heading])
        
        # Normalize again
        self.normalize_beliefs()
        
    def normalize_beliefs(self):
        """
        Normalize location and heading probabilities so they sum to 1
        """
        # Normalize location probabilities
        total_prob = 0.0
        for x in range(self.cols):
            for y in range(self.rows):
                total_prob += self.location_probs[x][y]
        
        if total_prob > 0:
            for x in range(self.cols):
                for y in range(self.rows):
                    self.location_probs[x][y] /= total_prob
        
        # Normalize heading probabilities
        total_heading_prob = sum(self.heading_probs.values())
        if total_heading_prob > 0:
            for heading in self.heading_probs:
                self.heading_probs[heading] /= total_heading_prob

def print_summary_table(history):
    """
    Print a summary table of the particle filter results and save to CSV
    Args:
        history: List of tuples (step, max_states, max_prob)
    """
    # Format data for tabulate
    table_data = []
    
    # Prepare for CSV output
    csv_data = []
    csv_headers = ["Time_Step", "Most_Likely_State", "Probability"]
    
    for step, max_states, max_prob in history:
        # Format the max states as a string
        states_str = []
        for ((x, y), heading) in max_states:
            states_str.append(f"({x},{y},{heading})")
        states_combined = ", ".join(states_str)
        
        # Add row to table
        table_data.append([step, states_combined, f"{max_prob:.3f}"])
        
        # Add row to CSV data
        csv_data.append([step, states_combined, max_prob])
    
    # Print table
    headers = ["Time Step", "Most Likely State(s)", "Probability"]
    print("\n=== Particle Filter Summary ===")
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Write to CSV file
    csv_filename = "particle_filter_results.csv"
    print(f"\nSaving results to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(csv_headers)
        # Write data
        csv_writer.writerows(csv_data)
    
    print(f"Results saved to {csv_filename}")

def main():
    parser = argparse.ArgumentParser(description='Robot Localization Simulation with Particle Filtering')
    parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--window_size', type=int, nargs=2, default=[900, 300], help='Window size [width, height]')
    parser.add_argument('--action_bias', type=float, default=0.1, help='Action bias')
    parser.add_argument('--observation_noise', type=float, default=0.1, help='Observation noise')
    parser.add_argument('--action_noise', type=float, default=0.1, help='Action noise')
    parser.add_argument('--particles', type=int, default=1000, help='Number of particles for filtering')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()
    
    # Set the random seed
    np.random.seed(args.seed)
    
    # Initialize the environment with specified parameters
    env = le.Environment(
        action_bias=args.action_bias,
        observation_noise=args.observation_noise,
        action_noise=args.action_noise,
        dimensions=(18, 6),  # Columns x Rows
        seed=args.seed,
        window_size=args.window_size
    )
    
    # Initialize localization filter (for visualization)
    localization = LocalizationFilter(env)
    
    # Initialize particle filter
    particle_filter = ParticleFilter(env, n_particles=args.particles, seed=args.seed)
    
    # Collect observations by running simulation
    observations = []
    history = []
    
    print(f"Running particle filter for {args.steps} steps...")
    
    # Main simulation loop to collect observations and run filter
    for step in range(args.steps):
        # Get the next observation by moving the robot
        observation = env.move()
        observations.append(observation)
        
        # Update particle filter
        max_states, max_prob = particle_filter.update(observation)
        history.append((step+1, max_states, max_prob))
        
        # Print current step results
        print(f"{step+1}: p* = {max_prob:.3f}   arg-max state(s) = {max_states}")
        
        # If visualization is enabled, update display
        if args.visualize:
            env.update(localization.location_probs, localization.heading_probs)
            time.sleep(0.1)
    
    print("Simulation complete.")
    print(f"Collected {len(observations)} observations.")
    
    # Print summary table
    print_summary_table(history)
    
    # Keep visualization window open if enabled
    if args.visualize:
        print("Keeping visualization window open. Close it to exit.")
        while env.running:
            env.update(localization.location_probs, localization.heading_probs)
            time.sleep(0.1)

if __name__ == "__main__":
    main()