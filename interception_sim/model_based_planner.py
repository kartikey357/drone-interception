"""
Week 5: model-based intercept planner.

Given the pursuer's current state and the online evader model's
predicted future trajectory, this:

  1. Searches the model's predicted trajectory for the earliest time at
     which the pursuer could plausibly reach the evader's predicted
     position (a simple, conservative average-speed feasibility check
     against the pursuer's max speed).
  2. Generates a minimum-jerk (quintic polynomial) trajectory from the
     pursuer's current [position, velocity, acceleration] to the
     predicted [position, velocity, acceleration] at that intercept
     time -- a genuine time-parameterized path, not just a heading.
  3. Exposes `evaluate(t)` so a tracking controller can sample the
     planned position/velocity/acceleration at any elapsed time since
     the plan was made.

Replanning: the calling script re-invokes `plan_intercept` periodically
(a fixed interval is used here, e.g. every 0.3-0.5s, which is a simple
and defensible way to satisfy "replan whenever the evader model changes
significantly" -- since the model updates continuously, periodic
replanning at a rate faster than the evader's behavior changes
approximates continuous replanning without needing a separate
change-detection trigger layered on top).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class InterceptPlan:
    t_start: float                 # simulation time this plan was created
    intercept_time: float          # planned duration from t_start to intercept
    poly_coeffs: np.ndarray        # shape (3, 6): quintic coefficients per axis
    target_position: np.ndarray
    target_velocity: np.ndarray

    def evaluate(self, elapsed: float):
        """Return (position, velocity, acceleration) at `elapsed` seconds
        into this plan (clamped to the plan's total duration)."""
        tau = np.clip(elapsed, 0.0, self.intercept_time)
        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        powers = np.array([tau**k for k in range(6)])
        vel_powers = np.array([0, 1, 2 * tau, 3 * tau**2, 4 * tau**3, 5 * tau**4])
        acc_powers = np.array([0, 0, 2, 6 * tau, 12 * tau**2, 20 * tau**3])
        for axis in range(3):
            c = self.poly_coeffs[axis]
            pos[axis] = c @ powers
            vel[axis] = c @ vel_powers
            acc[axis] = c @ acc_powers
        return pos, vel, acc


def _quintic_coeffs(p0, v0, a0, pf, vf, af, T):
    """
    Solve the 6 boundary conditions [p0,v0,a0,pf,vf,af] for a single
    axis's quintic polynomial p(t) = c0 + c1 t + ... + c5 t^5 over
    t in [0, T]. Standard closed-form minimum-jerk boundary solve.
    """
    if T < 1e-6:
        # Degenerate: just hold position.
        return np.array([p0, 0, 0, 0, 0, 0])

    c0, c1, c2 = p0, v0, a0 / 2.0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    A = np.array([
        [T3,      T4,       T5],
        [3*T2,    4*T3,     5*T4],
        [6*T,     12*T2,    20*T3],
    ])
    b = np.array([
        pf - (c0 + c1*T + c2*T2),
        vf - (c1 + 2*c2*T),
        af - 2*c2,
    ])
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5])


def find_intercept_time(pursuer_pos: np.ndarray, max_speed: float,
                         predicted_states, speed_margin: float = 0.85,
                         current_evader_pos: np.ndarray = None,
                         current_evader_speed_estimate: float = None):
    """
    predicted_states: output of IMMEvaderModel.predict_future_state(...),
    i.e. a list of (t, pos, vel, acc, cov) tuples.

    Returns the earliest (t, pos, vel, acc) at which the required
    average speed to reach the predicted evader position is within
    `speed_margin` of the pursuer's max speed (leaving headroom for
    maneuvering, not just a straight-line dash). Falls back to the
    furthest predicted point if nothing in the horizon is feasible.

    Long-horizon constant-acceleration extrapolation diverges badly for
    curved (e.g. circular/maneuvering) trajectories -- the compounding
    0.5*a*t^2 term assumes the evader keeps accelerating the same way
    for the whole horizon, which is never true for a turning target.
    To keep the planner from chasing physically implausible points, any
    candidate predicted position farther than a generous
    speed-based radius from the evader's LAST KNOWN position is
    clamped back to that radius (direction preserved) -- this uses only
    the model's own current speed belief, not the evader's true
    (unknown) max speed, so it doesn't violate the "pursuer cannot
    assume evader capabilities" constraint.
    """
    best = None
    MIN_INTERCEPT_TIME = 0.5  # avoid ill-conditioned quintic solves at very small T
    for t, pos, vel, acc, cov in predicted_states:
        if t < MIN_INTERCEPT_TIME:
            continue
        if current_evader_pos is not None and current_evader_speed_estimate is not None:
            plausible_radius = (current_evader_speed_estimate * 1.6 + 2.0) * t
            offset = pos - current_evader_pos
            dist_from_last_known = np.linalg.norm(offset)
            if dist_from_last_known > plausible_radius and dist_from_last_known > 1e-6:
                pos = current_evader_pos + offset * (plausible_radius / dist_from_last_known)

        dist = np.linalg.norm(pos - pursuer_pos)
        required_avg_speed = dist / t if t > 1e-6 else np.inf
        if required_avg_speed <= speed_margin * max_speed:
            best = (t, pos, vel, acc)
            break
    if best is None:
        t, pos, vel, acc, cov = predicted_states[-1]
        best = (t, pos, vel, acc)
    return best


def plan_intercept(pursuer_pos: np.ndarray, pursuer_vel: np.ndarray,
                    pursuer_acc: np.ndarray, model, max_speed: float,
                    t_start: float, horizon: float = 2.0, n_points: int = 20) -> InterceptPlan:
    predicted_states = model.predict_future_state(horizon, n_points=n_points)
    current_evader_pos = model.get_position()
    current_evader_speed_estimate = float(np.linalg.norm(model.get_velocity()))
    t_intercept, target_pos, target_vel, target_acc = find_intercept_time(
        pursuer_pos, max_speed, predicted_states,
        current_evader_pos=current_evader_pos,
        current_evader_speed_estimate=current_evader_speed_estimate)

    poly_coeffs = np.zeros((3, 6))
    for axis in range(3):
        poly_coeffs[axis] = _quintic_coeffs(
            pursuer_pos[axis], pursuer_vel[axis], pursuer_acc[axis],
            target_pos[axis], target_vel[axis], target_acc[axis],
            t_intercept)

    return InterceptPlan(t_start=t_start, intercept_time=t_intercept,
                          poly_coeffs=poly_coeffs,
                          target_position=target_pos, target_velocity=target_vel)
