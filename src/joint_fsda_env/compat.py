"""Compatibility layer: use real Gymnasium / PettingZoo when installed, otherwise
minimal API-compatible shims so the environment can be developed and tested offline."""
from types import SimpleNamespace
import numpy as np

try:  # pragma: no cover - exercised only when the libs are installed
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

    class _Space:
        def __init__(self, shape=None, dtype=None, seed=None):
            self.shape, self.dtype = shape, dtype
            self.np_random = np.random.default_rng(seed)

        def seed(self, seed=None):
            self.np_random = np.random.default_rng(seed)

    class Discrete(_Space):
        def __init__(self, n, seed=None, start=0):
            super().__init__((), np.int64, seed)
            self.n, self.start = int(n), int(start)

        def sample(self):
            return int(self.np_random.integers(self.n)) + self.start

        def contains(self, x):
            try:
                x = int(x)
            except (TypeError, ValueError):
                return False
            return self.start <= x < self.start + self.n

    class MultiBinary(_Space):
        def __init__(self, n, seed=None):
            super().__init__((int(n),), np.int8, seed)
            self.n = int(n)

        def sample(self):
            return self.np_random.integers(0, 2, size=self.shape).astype(np.int8)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.isin(x, [0, 1]).all())

    class Box(_Space):
        def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
            shape = tuple(shape) if shape is not None else np.shape(low)
            super().__init__(shape, dtype, seed)
            self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()

        def sample(self):
            return self.np_random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return (x.shape == self.shape and bool(np.all(x >= self.low))
                    and bool(np.all(x <= self.high)))

    class Dict(_Space):
        def __init__(self, spaces_dict, seed=None):
            super().__init__(None, None, seed)
            self.spaces = dict(spaces_dict)

        def sample(self):
            return {k: s.sample() for k, s in self.spaces.items()}

        def contains(self, x):
            return (isinstance(x, dict) and set(x) == set(self.spaces)
                    and all(s.contains(x[k]) for k, s in self.spaces.items()))

    spaces = SimpleNamespace(Discrete=Discrete, MultiBinary=MultiBinary, Box=Box, Dict=Dict)

try:  # pragma: no cover
    from pettingzoo import ParallelEnv
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

    class ParallelEnv:  # minimal stand-in mirroring pettingzoo.utils.env.ParallelEnv
        metadata = {}
        possible_agents = []
        agents = []
