"""
Week 3 baseline intercept strategies.

Both strategies consume ONLY the estimator's output (position + velocity
estimate), never ground truth. They are deliberately simple, well-known
guidance laws used as the baseline the later model-based planner (Week 5)
must beat.
"""

import numpy as np


def pure_pursuit(pursuer_pos: np.ndarray, pursuer_vel: np.ndarray,
                  target_pos: np.ndarray, max_accel: float,
                  closing_speed: float = None, k_p: float = 2.0) -> np.ndarray:
    """
    Proportional pursuit: rather than always thrusting at max magnitude
    straight at the target (which overshoots and oscillates once the
    pursuer is faster than the target and gets close), this computes a
    desired velocity proportional to the position error, capped at
    `closing_speed`, and accelerates to track THAT velocity. This
    naturally decelerates on final approach instead of blowing past
    the target at full speed.
    """
    if closing_speed is None:
        closing_speed = max_accel * 2.0  # a reasonable default cap
    direction = target_pos - pursuer_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return -pursuer_vel * k_p  # sitting on target: kill residual velocity
    desired_speed = min(k_p * dist, closing_speed)
    desired_vel = (direction / dist) * desired_speed
    accel = (desired_vel - pursuer_vel) * k_p
    mag = np.linalg.norm(accel)
    if mag > max_accel:
        accel = accel * (max_accel / mag)
    return accel


def boundary_repulsion(pursuer_pos: np.ndarray, arena_size: np.ndarray,
                        margin: float = 5.0, max_repulsion_accel: float = 6.0) -> np.ndarray:
    """
    Simple safety layer: adds an inward-pushing acceleration as the
    pursuer approaches any arena wall, so a naive pursuit law doesn't
    fly it straight out of bounds while chasing a target near the edge.
    Zero effect away from walls; ramps up linearly inside `margin`.
    """
    accel = np.zeros(3)
    for axis in range(3):
        dist_to_low = pursuer_pos[axis]
        dist_to_high = arena_size[axis] - pursuer_pos[axis]
        if dist_to_low < margin:
            accel[axis] += max_repulsion_accel * (1 - dist_to_low / margin)
        if dist_to_high < margin:
            accel[axis] -= max_repulsion_accel * (1 - dist_to_high / margin)
    return accel


def intercept_point_aiming(pursuer_pos: np.ndarray, pursuer_vel: np.ndarray,
                            target_pos: np.ndarray, target_vel: np.ndarray,
                            pursuer_speed: float, max_accel: float,
                            max_time_horizon: float = 10.0) -> np.ndarray:
    """
    Lead pursuit / "firing solution": assumes the target continues at
    constant velocity, and solves for the future rendezvous point the
    pursuer can just barely reach at its own max speed, then steers
    toward THAT point instead of the target's current position. This is
    the standard closed-form solution to the classic pursuit-intercept
    geometry problem (quadratic in intercept time).
    """
    rel_pos = target_pos - pursuer_pos
    a = np.dot(target_vel, target_vel) - pursuer_speed ** 2
    b = 2 * np.dot(rel_pos, target_vel)
    c = np.dot(rel_pos, rel_pos)

    t_intercept = None
    if abs(a) < 1e-9:
        # Degenerate (relative speeds ~equal): fall back to linear solve.
        if abs(b) > 1e-9:
            t_candidate = -c / b
            if t_candidate > 0:
                t_intercept = t_candidate
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            candidates = [t for t in (t1, t2) if t > 0]
            if candidates:
                t_intercept = min(candidates)

    if t_intercept is None or t_intercept > max_time_horizon:
        # No feasible intercept point found (target too fast / diverging):
        # fall back to pure pursuit toward current position.
        return pure_pursuit(pursuer_pos, pursuer_vel, target_pos, max_accel)

    intercept_point = target_pos + target_vel * t_intercept
    return pure_pursuit(pursuer_pos, pursuer_vel, intercept_point, max_accel)
