"""
Simplified quadrotor dynamics used by the fast backend.

Rather than a full 6-DOF rigid body model, this uses a widely-accepted
simplification for interception/planning research: a point-mass model
with velocity and acceleration saturation (acceleration limit derived
from a max tilt angle, i.e. thrust saturation), plus a first-order lag
on achieved acceleration to emulate attitude-response delay. This is
the same simplification used in the "gym-pybullet-drones" style
quadrotor benchmarks referenced in the project brief.

A `LowLevelController` sits between the planner (which outputs a
desired acceleration or velocity) and the raw dynamics — this is the
"inner-loop controller" the PS explicitly asks for as a separate
component from the planner.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class QuadrotorLimits:
    max_speed: float = 8.0          # m/s
    max_horizontal_accel: float = 6.0   # m/s^2, derived from max tilt angle
    max_vertical_accel: float = 4.0     # m/s^2, thrust saturation
    attitude_lag_tau: float = 0.15      # s, first-order lag time constant


class LowLevelController:
    """
    Tracks a commanded acceleration (from the planner) subject to the
    vehicle's real dynamic limits. This models motor/attitude response
    lag and saturation — i.e., "stabilise the pursuer's attitude and
    altitude" from the PS, without a full attitude-dynamics simulation.
    """

    def __init__(self, limits: QuadrotorLimits):
        self.limits = limits
        self._achieved_accel = np.zeros(3)

    def reset(self):
        self._achieved_accel[:] = 0.0

    def track(self, commanded_accel: np.ndarray, dt: float) -> np.ndarray:
        # Saturate the *commanded* acceleration to the vehicle's limits first.
        cmd = commanded_accel.copy()
        horiz_mag = np.linalg.norm(cmd[:2])
        if horiz_mag > self.limits.max_horizontal_accel:
            cmd[:2] *= self.limits.max_horizontal_accel / horiz_mag
        cmd[2] = np.clip(cmd[2], -self.limits.max_vertical_accel, self.limits.max_vertical_accel)

        # First-order lag toward the saturated command (attitude/motor response delay).
        alpha = dt / (self.limits.attitude_lag_tau + dt)
        self._achieved_accel += alpha * (cmd - self._achieved_accel)
        return self._achieved_accel.copy()


class PointMassQuad:
    """Double-integrator dynamics with velocity saturation."""

    def __init__(self, limits: QuadrotorLimits):
        self.limits = limits
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.controller = LowLevelController(limits)

    def reset(self, position: np.ndarray, velocity: np.ndarray | None = None):
        self.position = position.copy()
        self.velocity = np.zeros(3) if velocity is None else velocity.copy()
        self.acceleration = np.zeros(3)
        self.controller.reset()

    def step(self, commanded_accel: np.ndarray, dt: float):
        achieved = self.controller.track(commanded_accel, dt)

        # Semi-implicit Euler integration.
        new_velocity = self.velocity + achieved * dt
        speed = np.linalg.norm(new_velocity)
        if speed > self.limits.max_speed:
            new_velocity *= self.limits.max_speed / speed

        self.position = self.position + new_velocity * dt
        self.acceleration = (new_velocity - self.velocity) / dt
        self.velocity = new_velocity
