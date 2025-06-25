import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.pos = 0
        self.size = 0

        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, transition, td_error=None):
        priority = (abs(td_error) + 1e-5) ** self.alpha if td_error is not None else 1.0
        if self.size < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return [], [], []

        scaled_priorities = self.priorities[:self.size]
        probs = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.memory[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return self.size
