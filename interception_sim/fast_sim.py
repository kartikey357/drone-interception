"""
Fast, headless, pure-Python simulation backend.

Implements InterceptionSimulator so it's a drop-in replacement for the
PX4/Gazebo backend during the hundreds of episodes needed for the
evaluation suite. No rendering, no network, no Docker — just numeric
integration, which is why this can run orders of magnitude faster than
real time.
"""

import numpy as np
from .base import InterceptionSimulator, AgentState, Observation, StepResult, EpisodeStatus
from .dynamics import PointMassQuad, QuadrotorLimits
from .evader_behaviors import make_behavior
from .sensor import NoisySensor


class FastInterceptionSim(InterceptionSimulator):
    def __init__(self,
                 arena_size=(50.0, 50.0, 30.0),   # x, y, z half-extents ARE NOT halved: full bounds [0, size]
                 capture_radius: float = 0.5,
                 time_limit: float = 60.0,
                 pursuer_limits: QuadrotorLimits | None = None,
                 evader_limits_scale: float = 1.0,
                 sensor_rate_hz: float = 20.0,
                 sensor_noise_std: float = 0.3,
                 dropout_enabled: bool = False):
        self.arena_size = np.array(arena_size)
        self.capture_radius = capture_radius
        self.time_limit = time_limit

        self.pursuer_limits = pursuer_limits or QuadrotorLimits()
        self.pursuer = PointMassQuad(self.pursuer_limits)

        self.evader_limits_scale = evader_limits_scale
        self.sensor = NoisySensor(rate_hz=sensor_rate_hz, noise_std=sensor_noise_std,
                                   dropout_enabled=dropout_enabled)

        self._t = 0.0
        self._status = EpisodeStatus.RUNNING
        self._evader_behavior = None
        self._evader_pos = np.zeros(3)
        self._evader_vel = np.zeros(3)
        self._rng = np.random.default_rng()
        self._last_obs: Observation | None = None

    # ---------------------------------------------------------------

    def reset(self, evader_profile: str, seed: int | None = None) -> AgentState:
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._status = EpisodeStatus.RUNNING

        center = self.arena_size / 2.0
        pursuer_start = center + self._rng.uniform(-5, 5, size=3)
        pursuer_start[2] = np.clip(pursuer_start[2], 2.0, self.arena_size[2] - 2.0)
        self.pursuer.reset(pursuer_start)

        evader_start = center + self._rng.uniform(-10, 10, size=3)
        evader_start[2] = np.clip(evader_start[2], 2.0, self.arena_size[2] - 2.0)
        self._evader_pos = evader_start
        self._evader_vel = np.zeros(3)

        self._evader_behavior = make_behavior(evader_profile)
        self._evader_behavior.reset(self._rng)
        self._evader_behavior.max_accel *= self.evader_limits_scale

        self.sensor.reset(self._rng)
        self._last_obs = None

        return self._pursuer_state()

    def step(self, pursuer_accel_cmd: np.ndarray, dt: float) -> StepResult:
        # --- evader moves according to its (hidden) behavior ---
        evader_accel = self._evader_behavior.command(
            self._t, self._evader_pos, self._evader_vel, self._rng)
        evader_accel = np.clip(evader_accel,
                                -self._evader_behavior.max_accel,
                                self._evader_behavior.max_accel)
        new_evader_vel = self._evader_vel + evader_accel * dt
        speed = np.linalg.norm(new_evader_vel)
        if speed > self._evader_behavior.max_speed:
            new_evader_vel *= self._evader_behavior.max_speed / speed
        new_evader_pos = self._evader_pos + new_evader_vel * dt

        # Soft containment: reflect velocity component if crossing a wall,
        # so the evader (like the pursuer) stays inside the bounded arena
        # instead of flying off to infinity.
        margin = 2.0
        for axis in range(3):
            if new_evader_pos[axis] < margin and new_evader_vel[axis] < 0:
                new_evader_vel[axis] *= -0.5
            elif new_evader_pos[axis] > self.arena_size[axis] - margin and new_evader_vel[axis] > 0:
                new_evader_vel[axis] *= -0.5
        new_evader_pos = np.clip(new_evader_pos, 0.0, self.arena_size)

        self._evader_pos = new_evader_pos
        self._evader_vel = new_evader_vel

        # --- pursuer moves according to commanded acceleration ---
        self.pursuer.step(pursuer_accel_cmd, dt)

        self._t += dt

        # --- sensor sampling (independent of control rate) ---
        obs = self.sensor.maybe_sample(self._t, self._evader_pos)
        if obs is not None:
            self._last_obs = obs

        # --- episode termination checks ---
        miss = float(np.linalg.norm(self.pursuer.position - self._evader_pos))
        if miss <= self.capture_radius:
            self._status = EpisodeStatus.INTERCEPTED
        elif self._t >= self.time_limit:
            self._status = EpisodeStatus.TIMEOUT
        elif np.any(self.pursuer.position < 0) or np.any(self.pursuer.position > self.arena_size):
            self._status = EpisodeStatus.OUT_OF_BOUNDS

        return StepResult(
            pursuer_state=self._pursuer_state(),
            status=self._status,
            miss_distance=miss,
        )

    # ---------------------------------------------------------------

    def get_pursuer_state(self) -> AgentState:
        return self._pursuer_state()

    def get_noisy_evader_observation(self) -> Observation:
        if self._last_obs is None:
            # Before the first sensor tick, return an invalid observation
            # rather than leaking ground truth.
            return Observation(position=None, t=self._t, valid=False)
        return self._last_obs

    def get_ground_truth(self) -> dict:
        active_mode = getattr(self._evader_behavior, "active_mode_index", None)
        return {
            "t": self._t,
            "pursuer_position": self.pursuer.position.copy(),
            "pursuer_velocity": self.pursuer.velocity.copy(),
            "evader_position": self._evader_pos.copy(),
            "evader_velocity": self._evader_vel.copy(),
            "evader_active_mode": active_mode,  # evaluation-only; e.g. for behavior_switching profile
        }

    def is_done(self) -> tuple[bool, EpisodeStatus]:
        return self._status != EpisodeStatus.RUNNING, self._status

    # ---------------------------------------------------------------

    def _pursuer_state(self) -> AgentState:
        return AgentState(
            position=self.pursuer.position.copy(),
            velocity=self.pursuer.velocity.copy(),
            acceleration=self.pursuer.acceleration.copy(),
            t=self._t,
        )
