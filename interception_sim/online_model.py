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

The fix: use STRUCTURALLY different models instead, which is the
classical, textbook IMM configuration. This version uses THREE:

  - Model 0: Constant Velocity (CV) -- assumes zero acceleration.
  - Model 1: Constant Acceleration (CA) -- tracks a moving acceleration.
  - Model 2: Coordinated Turn (CT) -- assumes near-constant-speed
    circular motion in the horizontal plane, with turn rate estimated
    from the model's own current velocity/acceleration each predict
    step (a linearized, re-estimated-every-step turn rate, rather than
    a full EKF with turn rate as an explicit filtered state -- a
    deliberate simplification to keep this a linear-KF-style IMM where
    all three models share the same 9-dim [p,v,a] state and the
    standard IMM mixing/combination equations apply unchanged).

WHY CT WAS ADDED: Week 5/7 evaluation showed the CV/CA pair alone
badly under-predicts the evader's future position for a tightly
circling "maneuvering" profile -- extrapolating a constant acceleration
vector forward for 1-2s assumes the evader keeps thrusting the same
direction, which is exactly wrong for something turning in a circle.
The CT model instead assumes the CURRENT turn rate persists, which is
the correct structural assumption for exactly this failure case.

Both CV/CA models are kept in the same 9-dimensional augmented state
space [p(3), v(3), a(3)] so IMM mixing works uniformly across them; the
CV model's transition matrix simply never lets its acceleration block
influence velocity/position, and its own acceleration state is pinned
near zero with tight process noise. The CT model shares this same
9-dim layout (so mixing/combination works identically across all three
models with no special-casing needed there) but uses a genuinely
different, cross-coupled x/y transition for position and velocity.
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


# State layout everywhere in this module: [px,py,pz, vx,vy,vz, ax,ay,az]
PX, PY, PZ = 0, 1, 2
VX, VY, VZ = 3, 4, 5
AX, AY, AZ = 6, 7, 8

MAX_TURN_RATE = 4.0  # rad/s clamp, avoids blow-up when estimated speed ~ 0


def _estimate_turn_rate(state: np.ndarray) -> float:
    """
    Instantaneous turn-rate estimate from a state's own horizontal
    velocity/acceleration, assuming the acceleration is (approximately)
    centripetal: omega = (vx*ay - vy*ax) / |v_xy|^2. This is exact for
    pure circular motion and a reasonable local estimate otherwise.
    """
    vx, vy, ax, ay = state[VX], state[VY], state[AX], state[AY]
    speed_sq = vx * vx + vy * vy
    if speed_sq < 1e-4:
        return 0.0
    omega = (vx * ay - vy * ax) / speed_sq
    return float(np.clip(omega, -MAX_TURN_RATE, MAX_TURN_RATE))


def _ct_F_Q(dt: float, omega: float, q_ct_horizontal: float, q_ca_vertical: float,
            q_acc_pin: float = 1e-3):
    """
    Coordinated-turn transition for the full 9-dim state: horizontal
    (px,py,vx,vy) uses the standard cross-coupled circular-motion
    discretization at the given (locally estimated) turn rate; vertical
    (pz,vz,az) uses the same CA propagation as the CA model, since
    turning is assumed to happen in the horizontal plane only; the
    horizontal acceleration slots (ax,ay) are not used for propagation
    in this model (the turn IS the acceleration) and are simply carried
    forward with injected noise so covariance doesn't collapse.
    """
    F = np.zeros((9, 9))
    Q = np.zeros((9, 9))

    if abs(omega) < 1e-4:
        # Straight-line limit: falls back to constant velocity in x,y.
        F[PX, PX] = 1.0; F[PX, VX] = dt
        F[PY, PY] = 1.0; F[PY, VY] = dt
        F[VX, VX] = 1.0
        F[VY, VY] = 1.0
        q_pos_block = _cv_Q(dt, q_ct_horizontal)
        Q[PX, PX] = q_pos_block[0, 0]; Q[PX, VX] = Q[VX, PX] = q_pos_block[0, 1]
        Q[VX, VX] = q_pos_block[1, 1]
        Q[PY, PY] = q_pos_block[0, 0]; Q[PY, VY] = Q[VY, PY] = q_pos_block[0, 1]
        Q[VY, VY] = q_pos_block[1, 1]
    else:
        sin_wt, cos_wt = np.sin(omega * dt), np.cos(omega * dt)
        s_w, c_w = sin_wt / omega, (1 - cos_wt) / omega

        F[PX, PX] = 1.0; F[PX, VX] = s_w;      F[PX, VY] = -c_w
        F[PY, PY] = 1.0; F[PY, VX] = c_w;      F[PY, VY] = s_w
        F[VX, VX] = cos_wt; F[VX, VY] = -sin_wt
        F[VY, VX] = sin_wt; F[VY, VY] = cos_wt

        # Process noise on the horizontal block: moderate, isotropic,
        # representing uncertainty in the turn-rate estimate itself.
        q = q_ct_horizontal
        for p_idx, v_idx in ((PX, VX), (PY, VY)):
            Q[p_idx, p_idx] = q * dt**3 / 3.0
            Q[p_idx, v_idx] = Q[v_idx, p_idx] = q * dt**2 / 2.0
            Q[v_idx, v_idx] = q * dt

    # ax, ay: not used for propagation here; carry forward with injected noise.
    F[AX, AX] = 1.0
    F[AY, AY] = 1.0
    Q[AX, AX] = q_acc_pin
    Q[AY, AY] = q_acc_pin

    # Vertical: same CA treatment as the CA model.
    f_z = _ca_F(dt)
    q_z = _ca_Q(dt, q_ca_vertical)
    z_ids = [PZ, VZ, AZ]
    for i in range(3):
        for j in range(3):
            F[z_ids[i], z_ids[j]] = f_z[i, j]
            Q[z_ids[i], z_ids[j]] = q_z[i, j]

    return F, Q


class IMMEvaderModel:
    N_MODELS = 3  # CV, CA, CT

    def __init__(self, initial_position: np.ndarray,
                 q_cv_vel: float = 0.02, q_ca: float = 6.0, q_ct_horizontal: float = 6.0,
                 measurement_noise_std: float = 0.3,
                 mode_transition_stay_prob: float = 0.999,
                 initial_mode_probs=(0.6, 0.2, 0.2)):
        """
        q_cv_vel: process-noise PSD for the CV model's velocity.
        q_ca: process-noise PSD for the CA model's acceleration/jerk.
        q_ct_horizontal: process-noise PSD for the CT model's horizontal
            position/velocity block (accounts for turn-rate estimation
            error between updates).
        """
        self.n = 9
        self.r = measurement_noise_std ** 2
        self.q_cv_vel = q_cv_vel
        self.q_ca = q_ca
        self.q_ct_horizontal = q_ct_horizontal

        self.H = np.zeros((3, self.n))
        self.H[0:3, 0:3] = np.eye(3)

        x0 = np.zeros(self.n)
        x0[0:3] = initial_position
        P0 = np.eye(self.n)
        P0[0:3, 0:3] *= 1.0
        P0[3:6, 3:6] *= 5.0 ** 2
        P0[6:9, 6:9] *= 5.0 ** 2

        # index 0 = CV, index 1 = CA, index 2 = CT
        self.x = [x0.copy() for _ in range(self.N_MODELS)]
        self.P = [P0.copy() for _ in range(self.N_MODELS)]

        p_stay = mode_transition_stay_prob
        p_other = (1 - p_stay) / (self.N_MODELS - 1)
        self.Pi = np.full((self.N_MODELS, self.N_MODELS), p_other)
        np.fill_diagonal(self.Pi, p_stay)

        self.mu = np.array(initial_mode_probs, dtype=float)
        self.mu /= self.mu.sum()
        self._c_bar = self.mu.copy()

    # ------------------------------------------------------------------

    def _model_F_Q(self, model_idx: int, dt: float, omega_hat: float = None):
        if model_idx == 0:  # CV
            f_axis = _cv_F(dt)
            q_axis = _cv_Q(dt, self.q_cv_vel)
            return _assemble(f_axis, q_axis)
        elif model_idx == 1:  # CA
            f_axis = _ca_F(dt)
            q_axis = _ca_Q(dt, self.q_ca)
            return _assemble(f_axis, q_axis)
        else:  # CT
            # Turn rate is estimated from the CA model's state, not
            # CT's own -- CT's transition never drives its ax,ay slots
            # from real dynamics, so they'd never learn a meaningful
            # turn rate on their own. CA's acceleration estimate, by
            # contrast, genuinely fits the observed curvature.
            if omega_hat is None:
                omega_hat = _estimate_turn_rate(self.x[1])
            return _ct_F_Q(dt, omega_hat, self.q_ct_horizontal, self.q_ca)

    def _mix(self):
        n = self.N_MODELS
        c_bar = self.Pi.T @ self.mu
        mix_prob = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                mix_prob[i, j] = self.Pi[i, j] * self.mu[i] / (c_bar[j] + 1e-12)

        mixed_x = [np.zeros(self.n) for _ in range(n)]
        mixed_P = [np.zeros((self.n, self.n)) for _ in range(n)]
        for j in range(n):
            for i in range(n):
                mixed_x[j] += mix_prob[i, j] * self.x[i]
            for i in range(n):
                dx = self.x[i] - mixed_x[j]
                mixed_P[j] += mix_prob[i, j] * (self.P[i] + np.outer(dx, dx))
        return mixed_x, mixed_P, c_bar

    def predict(self, dt: float):
        mixed_x, mixed_P, self._c_bar = self._mix()
        omega_hat = _estimate_turn_rate(mixed_x[1])  # from CA's mixed state
        for m in range(self.N_MODELS):
            F, Q = self._model_F_Q(m, dt, omega_hat=omega_hat)
            self.x[m] = F @ mixed_x[m]
            self.P[m] = F @ mixed_P[m] @ F.T + Q

    def update(self, measured_position: np.ndarray):
        likelihoods = np.zeros(self.N_MODELS)
        for m in range(self.N_MODELS):
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
        result = np.zeros(self.n)
        for m in range(self.N_MODELS):
            result += self.mu[m] * self.x[m]
        return result

    def get_position(self) -> np.ndarray:
        return self._combined_state()[0:3]

    def get_velocity(self) -> np.ndarray:
        return self._combined_state()[3:6]

    def get_acceleration(self) -> np.ndarray:
        # The CA model has the most physically meaningful acceleration estimate.
        return self.x[1][6:9].copy()

    def get_mode_probability_high_maneuver(self) -> float:
        """Probability mass on the maneuvering-capable models (CA + CT)."""
        return float(self.mu[1] + self.mu[2])

    def get_mode_probabilities(self) -> dict:
        return {"cv": float(self.mu[0]), "ca": float(self.mu[1]), "ct": float(self.mu[2])}

    def is_maneuvering(self, threshold: float = 0.5) -> bool:
        return (self.mu[1] + self.mu[2]) > threshold

    def _dominant_model(self) -> int:
        return int(np.argmax(self.mu))

    def predict_future_position(self, horizon: float, n_points: int = 1):
        dominant = self._dominant_model()
        x, P = self.x[dominant].copy(), self.P[dominant].copy()
        omega_hat = _estimate_turn_rate(self.x[1])  # from CA's current state, held fixed forward
        dt = horizon / n_points
        results = []
        for i in range(1, n_points + 1):
            F, Q = self._model_F_Q(dominant, dt, omega_hat=omega_hat)
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
        dominant = self._dominant_model()
        x, P = self.x[dominant].copy(), self.P[dominant].copy()
        omega_hat = _estimate_turn_rate(self.x[1])  # from CA's current state, held fixed forward
        dt = horizon / n_points
        results = []
        for i in range(1, n_points + 1):
            F, Q = self._model_F_Q(dominant, dt, omega_hat=omega_hat)
            x = F @ x
            P = F @ P @ F.T + Q
            results.append((i * dt, x[0:3].copy(), x[3:6].copy(), x[6:9].copy(), P[0:3, 0:3].copy()))
        return results
