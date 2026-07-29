"""
Proportional Navigation (PN) guidance.

This REPLACES the Week 5 quintic-trajectory-with-periodic-replan
approach as the primary intercept strategy, after a documented failure
analysis showed the quintic approach plateaued around 26-32% against
the maneuvering evader regardless of tuning (replan interval, lookahead
time, tracking gains).

ROOT CAUSE FOUND: the pursuer's minimum turn radius at full speed
(v^2/a_max = 8^2/6 = 10.67m) exceeds the evader's actual turn radius
(4-10m) -- the pursuer cannot physically match the evader's tightest
turns without first slowing down, and none of the quintic/PD tuning
variants accounted for this speed-vs-turn-radius tradeoff.

THE FIX: True Proportional Navigation, the standard missile-guidance
law for intercepting maneuvering targets under a finite lateral
acceleration budget. PN commands lateral acceleration proportional to
the closing speed and the line-of-sight (LOS) rotation rate:

    a_lateral = N * Vc * (omega_LOS x LOS_unit)

This is derived from the geometry of a *collision course* (constant LOS
bearing = intercept), and crucially it naturally reduces commanded
lateral effort when the geometry doesn't demand a sharp turn, rather
than always trying to out-turn the target at maximum speed -- which is
exactly the mismatch the quintic approach didn't handle.

An axial (closing) term is added on top since real PN assumes a
pre-established closing velocity (true for a rocket motor with fixed
thrust; not true for a hovering multirotor that must also decide how
fast to close), tuned empirically (see week8 ablation notes: N=1.5,
axial_gain=1.0, axial_speed_gain=3.0 -- found via grid search over
N in [1.0, 6.0] and axial_speed_gain in [1.0, 4.0]).

RESULT (100 episodes each, vs. quintic / vs. Week 3 baseline):
    non_maneuvering:     95%  (quintic: 88%,  baseline: 97-99%)
    maneuvering:         42%  (quintic: 26%,  baseline: 32%)
    behavior_switching:  97%  (quintic: 84%,  baseline: n/a)
"""

import numpy as np


def pn_guidance(pursuer_pos: np.ndarray, pursuer_vel: np.ndarray,
                 target_pos: np.ndarray, target_vel: np.ndarray,
                 max_accel: float, max_speed: float,
                 N: float = 1.5, axial_gain: float = 1.0,
                 axial_speed_gain: float = 3.0) -> np.ndarray:
    """
    N: navigation constant (typically 3-5 in classical missile guidance;
        empirically 1.5 worked best here, likely because our pursuer's
        acceleration budget relative to its speed is much lower than a
        missile's, so a gentler lateral gain avoids overshoot/oscillation).
    axial_gain: how aggressively to correct toward the desired closing
        velocity.
    axial_speed_gain: how quickly desired closing speed ramps up with
        distance (capped at max_speed).

    NOTE (ablation finding, kept for the record): a close-range damping
    term on the lateral component was tried here, hypothesising that
    the 1/distance sensitivity near intercept was causing observed
    overshoot-and-recover cycles. Empirically this was WRONG -- damping
    made both interception rate and time-to-intercept worse at every
    setting tested (damping_dist 2/4/6/8m all underperformed no damping).
    The lateral term is apparently necessary right up to intercept, not
    harmful; reverted. The actual overshoot cause is still open --
    likely candidates for a real fix: N itself may need to be
    distance-adaptive (higher when far, to establish a good collision
    course, lower when close, to avoid overcorrection) rather than a
    single constant, or the axial term's `axial_speed_gain * dist`
    formula may be commanding too much closing speed at exactly the
    range where a gentler final approach is needed.
    """
    r = target_pos - pursuer_pos
    dist = np.linalg.norm(r)
    if dist < 1e-6:
        return -pursuer_vel * 2.0  # sitting on target: kill residual velocity

    v_rel = target_vel - pursuer_vel
    Vc = -np.dot(r, v_rel) / dist                      # closing speed (positive = closing)
    omega_los = np.cross(r, v_rel) / np.dot(r, r)       # LOS angular rate vector
    lateral_accel = N * Vc * np.cross(omega_los, r) / dist

    r_hat = r / dist
    desired_speed = min(axial_speed_gain * dist, max_speed)
    desired_vel_along_los = r_hat * desired_speed
    axial_accel = (desired_vel_along_los - pursuer_vel) * axial_gain

    accel = lateral_accel + axial_accel
    mag = np.linalg.norm(accel)
    if mag > max_accel:
        accel = accel * (max_accel / mag)
    return accel
