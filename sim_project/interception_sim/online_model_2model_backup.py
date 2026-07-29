"""
Online evader motion model -- Interacting Multiple Model (IMM) filter.

IMPORTANT DESIGN NOTE (kept here because it was a real, measured bug
during development, not just a design choice): an earlier version of
this module used two Constant-Acceleration (CA) filters differing only
in process-noise magnitude. Empirically, this FAILED to discriminate
behaviour at all (mode probability stayed pinned near 0.5 regardless of
whether the evader was maneuvering) because with 20 Hz measurements
(~0.05s between updates), position variance injected by process noise
scales as dt^5 and is negligible at that timescale for any reasonable
q -- so both models predicted almost identical positions and produced
almost identical innovations. There was nothing to discriminate.

The fix: use two STRUCTURALLY different models instead, which is the
classical, textbook IMM configuration:

  - Model 0: Constant Velocity (CV) -- assumes zero acceleration.
  - Model 1: Constant Acceleration (CA) -- tracks a moving acceleration.

When the evader actually accelerates/turns, the CV model's position
prediction develops a systematic, growing lag (it structurally cannot
represent the true motion), producing a large and CONSISTENT innovation
that the CA model does not share. This gives real, measurable
discrimination, unlike the noise-level-only variant above.

Both models are kept in the same 9-dimensional augmented state space
[p(3), v(3), a(3)] so IMM mixing works uniformly across them; the CV
model's transition matrix simply never lets its acceleration block
influence velocity/position, and its own acceleration state is pinned
near zero with tight process noise.
"""

import numpy as np


def _ca_F(dt):
    """Constant-acceleration per-axis transition."""
    return np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ])


def _ca_Q(dt, q):
    dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
    return q * np.array([
        [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
        [dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0],
        [dt3 / 6.0,  dt2 / 2.0, dt],
    ])


def _cv_F(dt):
    """
    Constant-velocity per-axis transition, in the same 3-component
    [p, v, a] layout: acceleration does NOT feed into velocity/position,
    and the acceleration state does not propagate itself (pinned near 0
    by tight process noise below, not by dynamics).
    """
    return np.array([
        [1.0, dt, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])


def _cv_Q(dt, q_vel, q_acc_pin=1e-4):
    """
    Standard 2nd-order (white-noise-acceleration) process noise on the
    [p, v] block, plus a tiny fixed variance injected into the unused
    acceleration slot so it doesn't collapse to a singular covariance.
    """
    dt2, dt3 = dt**2, dt**3
    Q = np.zeros((3, 3))
    Q[0, 0] = q_vel * dt3 / 3.0
    Q[0, 1] = Q[1, 0] = q_vel * dt2 / 2.0
    Q[1, 1] = q_vel * dt
    Q[2, 2] = q_acc_pin
    return Q


def _assemble(f_axis, q_axis):
    n = 9
    F = np.zeros((n, n))
    Q = np.zeros((n, n))
    idx = {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5, 8]}
    for axis in range(3):
        ids = idx[axis]
        for i in range(3):
            for j in range(3):
                F[ids[i], ids[j]] = f_axis[i, j]
                Q[ids[i], ids[j]] = q_axis[i, j]
    return F, Q


class IMMEvaderModel:
    def __init__(self, initial_position: np.ndarray,
                 q_cv_vel: float = 0.02, q_ca: float = 6.0,
                 measurement_noise_std: float = 0.3,
                 mode_transition_stay_prob: float = 0.999,
                 initial_mode_prob_ca: float = 0.3):
        """
        q_cv_vel: process-noise PSD for the CV model's velocity
            (higher = CV tolerates faster speed changes before its
            innovations blow up -- keep this moderate; it's meant to
            represent gentle real-world drift, not maneuvers).
        q_ca: process-noise PSD for the CA model's acceleration/jerk.
        """
        self.n = 9
        self.r = measurement_noise_std ** 2
        self.q_cv_vel = q_cv_vel
        self.q_ca = q_ca

        self.H = np.zeros((3, self.n))
        self.H[0:3, 0:3] = np.eye(3)

        x0 = np.zeros(self.n)
        x0[0:3] = initial_position
        P0 = np.eye(self.n)
        P0[0:3, 0:3] *= 1.0
        P0[3:6, 3:6] *= 5.0 ** 2
        P0[6:9, 6:9] *= 5.0 ** 2

        # index 0 = CV, index 1 = CA
        self.x = [x0.copy(), x0.copy()]
        self.P = [P0.copy(), P0.copy()]

        p_stay = mode_transition_stay_prob
        self.Pi = np.array([[p_stay, 1 - p_stay],
                             [1 - p_stay, p_stay]])
        self.mu = np.array([1 - initial_mode_prob_ca, initial_mode_prob_ca])
        self._c_bar = self.mu.copy()

    # ------------------------------------------------------------------

    def _model_F_Q(self, model_idx: int, dt: float):
        if model_idx == 0:  # CV
            f_axis = _cv_F(dt)
            q_axis = _cv_Q(dt, self.q_cv_vel)
        else:  # CA
            f_axis = _ca_F(dt)
            q_axis = _ca_Q(dt, self.q_ca)
        return _assemble(f_axis, q_axis)

    def _mix(self):
        c_bar = self.Pi.T @ self.mu
        mix_prob = np.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                mix_prob[i, j] = self.Pi[i, j] * self.mu[i] / (c_bar[j] + 1e-12)

        mixed_x = [np.zeros(self.n), np.zeros(self.n)]
        mixed_P = [np.zeros((self.n, self.n)), np.zeros((self.n, self.n))]
        for j in range(2):
            for i in range(2):
                mixed_x[j] += mix_prob[i, j] * self.x[i]
            for i in range(2):
                dx = self.x[i] - mixed_x[j]
                mixed_P[j] += mix_prob[i, j] * (self.P[i] + np.outer(dx, dx))
        return mixed_x, mixed_P, c_bar

    def predict(self, dt: float):
        mixed_x, mixed_P, self._c_bar = self._mix()
        for m in range(2):
            F, Q = self._model_F_Q(m, dt)
            self.x[m] = F @ mixed_x[m]
            self.P[m] = F @ mixed_P[m] @ F.T + Q

    def update(self, measured_position: np.ndarray):
        likelihoods = np.zeros(2)
        for m in range(2):
            y = measured_position - self.H @ self.x[m]
            S = self.H @ self.P[m] @ self.H.T + self.r * np.eye(3)
            K = self.P[m] @ self.H.T @ np.linalg.inv(S)
            self.x[m] = self.x[m] + K @ y
            self.P[m] = (np.eye(self.n) - K @ self.H) @ self.P[m]

            det_S = np.linalg.det(S)
            inv_S = np.linalg.inv(S)
            likelihoods[m] = np.exp(-0.5 * y @ inv_S @ y) / np.sqrt((2 * np.pi) ** 3 * max(det_S, 1e-12))

        posterior = likelihoods * self._c_bar
        total = posterior.sum()
        if total > 1e-300:
            self.mu = posterior / total

    # ------------------------------------------------------------------

    def _combined_state(self) -> np.ndarray:
        return self.mu[0] * self.x[0] + self.mu[1] * self.x[1]

    def get_position(self) -> np.ndarray:
        return self._combined_state()[0:3]

    def get_velocity(self) -> np.ndarray:
        return self._combined_state()[3:6]

    def get_acceleration(self) -> np.ndarray:
        # Only the CA model has a meaningful acceleration estimate.
        return self.x[1][6:9].copy()

    def get_mode_probability_high_maneuver(self) -> float:
        """Probability mass on the CA (maneuvering-capable) model."""
        return float(self.mu[1])

    def is_maneuvering(self, threshold: float = 0.5) -> bool:
        return self.mu[1] > threshold

    def predict_future_position(self, horizon: float, n_points: int = 1):
        dominant = 1 if self.mu[1] > self.mu[0] else 0
        x, P = self.x[dominant].copy(), self.P[dominant].copy()
        dt = horizon / n_points
        results = []
        for i in range(1, n_points + 1):
            F, Q = self._model_F_Q(dominant, dt)
            x = F @ x
            P = F @ P @ F.T + Q
            results.append((i * dt, x[0:3].copy(), P[0:3, 0:3].copy()))
        return results

    def predict_future_state(self, horizon: float, n_points: int = 1):
        """
        Like `predict_future_position`, but returns full (position,
        velocity, acceleration, position-covariance) tuples -- needed by
        the intercept planner to set a smooth-arrival velocity target,
        not just a target point.
        """
        dominant = 1 if self.mu[1] > self.mu[0] else 0
        x, P = self.x[dominant].copy(), self.P[dominant].copy()
        dt = horizon / n_points
        results = []
        for i in range(1, n_points + 1):
            F, Q = self._model_F_Q(dominant, dt)
            x = F @ x
            P = F @ P @ F.T + Q
            results.append((i * dt, x[0:3].copy(), x[3:6].copy(), x[6:9].copy(), P[0:3, 0:3].copy()))
        return results
