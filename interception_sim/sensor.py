"""
Onboard sensor model: noisy position-only observation of the evader.

Samples at a fixed rate (>= 20 Hz per the spec) independent of the
simulation's inner control-loop rate (>= 50 Hz). Supports Gaussian
position noise and optional dropout windows (bonus extension:
intermittent dropout lasting 0.5-2s).
"""

import numpy as np
from .base import Observation


class NoisySensor:
    def __init__(self, rate_hz: float = 20.0, noise_std: float = 0.3,
                 dropout_enabled: bool = False, dropout_prob_per_sec: float = 0.02,
                 dropout_duration_range=(0.5, 2.0), rng: np.random.Generator | None = None):
        self.dt = 1.0 / rate_hz
        self.noise_std = noise_std
        self.dropout_enabled = dropout_enabled
        self.dropout_prob_per_sec = dropout_prob_per_sec
        self.dropout_duration_range = dropout_duration_range
        self.rng = rng or np.random.default_rng()

        self._next_sample_t = 0.0
        self._dropout_until = -1.0
        self._last_obs: Observation | None = None

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self._next_sample_t = 0.0
        self._dropout_until = -1.0
        self._last_obs = None

    def maybe_sample(self, t: float, true_evader_position: np.ndarray) -> Observation | None:
        """
        Call every simulation tick. Returns a new Observation only when a
        sensor sample is actually due (per `rate_hz`), else returns None
        (meaning: no new data this tick, keep using the previous estimate).
        """
        if t < self._next_sample_t:
            return None
        self._next_sample_t = t + self.dt

        # Start a new dropout window stochastically.
        if self.dropout_enabled and t > self._dropout_until:
            if self.rng.uniform() < self.dropout_prob_per_sec * self.dt:
                duration = self.rng.uniform(*self.dropout_duration_range)
                self._dropout_until = t + duration

        if self.dropout_enabled and t <= self._dropout_until:
            obs = Observation(position=None, t=t, valid=False)
        else:
            noisy_pos = true_evader_position + self.rng.normal(0, self.noise_std, size=3)
            obs = Observation(position=noisy_pos, t=t, valid=True)

        self._last_obs = obs
        return obs
