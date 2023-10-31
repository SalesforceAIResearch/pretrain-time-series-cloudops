import numpy as np
from gluonts.transform import InstanceSampler


class ValidationRegionSampler(InstanceSampler):
    num_instances: int

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        assert b - a + 1 >= self.num_instances
        return np.arange(self.num_instances) + b - self.num_instances + 1


class RandomSampler(InstanceSampler):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        assert window_size >= 1

        return a + np.random.randint(window_size, size=1)
