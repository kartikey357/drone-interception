"""
Evader behavior generators.

Each behavior is a pure function of (t, state, rng) -> commanded
acceleration for the evader. These are intentionally NOT visible to the
pursuer's code path — only `fast_sim.py` calls these internally to move
the evader; the pursuer only ever sees noisy position observations via
`get_noisy_evader_observation()`.

Adding a new profile = adding a new class here + registering it in
BEHAVIOR_REGISTRY at the bottom. Nothing about the evaluation harness
needs to know the internals of a profile, which is what lets the grader
swap in hidden/unannounced profiles without touching your code.
"""

from abc import ABC, abstractmethod
import numpy as np


class EvaderBehavior(ABC):
    max_speed: float = 4.0
    max_accel: float = 3.0

    @abstractmethod
    def command(self, t: float, position: np.ndarray, velocity: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
        """Return desired acceleration (3,) for this instant."""
        raise NotImplementedError

    def reset(self, rng: np.random.Generator):
        """Override to (re)initialize any internal random parameters."""
        pass


class NonManeuvering(EvaderBehavior):
    """Straight-line / slowly varying flight — the 'easy' profile."""
    max_speed = 3.0
    max_accel = 0.5

    def reset(self, rng):
        heading = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(1.5, self.max_speed)
        climb = rng.uniform(-0.2, 0.2)
        self._cruise_vel = np.array([speed * np.cos(heading),
                                      speed * np.sin(heading), climb])
        # Very slow random-walk drift so it's not perfectly linear.
        self._drift_phase = rng.uniform(0, 2 * np.pi, size=3)

    def command(self, t, position, velocity, rng):
        drift = 0.15 * np.sin(0.1 * t + self._drift_phase)
        target_vel = self._cruise_vel + drift
        return np.clip((target_vel - velocity) * 0.8, -self.max_accel, self.max_accel)


class Maneuvering(EvaderBehavior):
    """Aggressive turns / accelerations / nonlinear trajectory."""
    max_speed = 6.0
    max_accel = 4.0

    def reset(self, rng):
        self._omega = rng.uniform(0.5, 1.2)      # turn rate rad/s
        self._radius = rng.uniform(4.0, 10.0)
        self._phase = rng.uniform(0, 2 * np.pi)
        self._vertical_omega = rng.uniform(0.3, 0.8)
        self._t0 = None

    def command(self, t, position, velocity, rng):
        if self._t0 is None:
            self._t0 = t
        tau = t - self._t0
        # Target a tight, fast, weaving circular/S-curve path with
        # occasional random jinks layered on top.
        target_speed = self._radius * self._omega
        tangent = np.array([-np.sin(self._omega * tau + self._phase),
                             np.cos(self._omega * tau + self._phase), 0.0])
        target_vel_xy = target_speed * tangent
        target_vz = 0.8 * np.sin(self._vertical_omega * tau)
        target_vel = np.array([target_vel_xy[0], target_vel_xy[1], target_vz])

        jink = np.zeros(3)
        if rng.uniform() < 0.02:  # occasional sharp random jink
            jink = rng.normal(0, 1.5, size=3)

        accel = (target_vel - velocity) * 1.5 + jink
        return np.clip(accel, -self.max_accel, self.max_accel)


class BehaviorSwitching(EvaderBehavior):
    """Switches between multiple behaviors mid-episode."""
    max_speed = 6.0
    max_accel = 4.0

    def __init__(self, modes=None, min_hold=6.0, max_hold=15.0):
        self._modes = modes or [NonManeuvering(), Maneuvering()]
        self._min_hold, self._max_hold = min_hold, max_hold

    def reset(self, rng):
        for m in self._modes:
            m.reset(rng)
        self._active_idx = 0
        self._next_switch_t = rng.uniform(self._min_hold, self._max_hold)
        self._t0 = None

    @property
    def active_mode_index(self) -> int:
        """Evaluation-only accessor: which behavior is currently active."""
        return self._active_idx

    def command(self, t, position, velocity, rng):
        if self._t0 is None:
            self._t0 = t
        tau = t - self._t0
        if tau >= self._next_switch_t:
            self._active_idx = (self._active_idx + 1) % len(self._modes)
            self._next_switch_t = tau + self.__dict__.get("_min_hold", 6.0) + \
                rng.uniform(0, self._max_hold - self._min_hold)
            self._modes[self._active_idx].reset(rng)
        return self._modes[self._active_idx].command(t, position, velocity, rng)


BEHAVIOR_REGISTRY = {
    "non_maneuvering": NonManeuvering,
    "maneuvering": Maneuvering,
    "behavior_switching": BehaviorSwitching,
    # Shorter hold times so a typical (few-second) episode actually
    # contains at least one switch -- useful for specifically
    # stress-testing switch-detection latency rather than the default
    # profile, where most episodes end via interception before any
    # switch occurs at all (worth reporting both, honestly).
    "behavior_switching_fast": lambda: BehaviorSwitching(min_hold=2.0, max_hold=4.0),
}


def make_behavior(name: str) -> EvaderBehavior:
    if name not in BEHAVIOR_REGISTRY:
        raise ValueError(f"Unknown evader profile '{name}'. "
                          f"Available: {list(BEHAVIOR_REGISTRY)}")
    return BEHAVIOR_REGISTRY[name]()
