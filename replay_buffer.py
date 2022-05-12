import random
from collections import namedtuple, deque, defaultdict

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple(
            'Experience',
            field_names=['states', 'actions', 'rewards', 'next_states', 'dones'])

    def add(self, e):
        """Add a new experience to memory."""
        self.memory.append(self.experience(*e))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        results = defaultdict(list)
        for experience in experiences:
            for name, value in experience._asdict().items():
                results[name].append(getattr(experience, name))
        for name, value in results.items():
            results[name] = np.asarray(value)
        return results

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
