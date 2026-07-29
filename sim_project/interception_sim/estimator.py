"""
Kalman filter state estimator for the evader.

Model: constant-acceleration (CA) kinematic model per axis, i.e. state
    x = [px, py, pz, vx, vy, vz, ax, ay, az]   (9-dimensional)
with a continuous white-noise-on-jerk process model (the standard
"DWNA"/third-order kinematic filter from Bar-Shalom, Estimation with
Applications to Tracking and Navigation). Measurements are position-only
(from the noisy sensor), which is exactly what `get_noisy_evader_observation()`
provides.

This runs as a strict predict/update loop:
  - `predict(dt)` is called every control tick (>= 50 Hz), even when no
    new sensor observation has arrived.
  - `update(z)` is called only when a new noisy position observation is
    available (>= 20 Hz), correcting the predicted state.

This satisfies the PS requirement that the estimator run at the full
control-loop rate while the sensor itself samples more slowly.
"""

import numpy as np


class ConstantAccelerationKF:
    def __init__(self,
                 initial_position: np.ndarray,
                 process_noise_psd: float = 4.0,
                 measurement_noise_std: float = 0.3,
                 initial_pos_std: float = 1.0,
                 initial_vel_std: float = 5.0,
                 initial_acc_std: float = 5.0):
        """
        process_noise_psd: continuous-time power spectral density of the
            jerk noise (higher = filter trusts new measurements more,
            tracks maneuvers faster but is noisier).
        measurement_noise_std: standard deviation of the position sensor
            noise (should match, or be tuned near, the sensor's actual
            noise_std for a well-calibrated filter).
        """
        self.q = process_noise_psd
        self.r = measurement_noise_std ** 2

        self.n = 9
        self.x = np.zeros(self.n)
        self.x[0:3] = initial_position

        self.P = np.eye(self.n)
        self.P[0:3, 0:3] *= initial_pos_std ** 2
        self.P[3:6, 3:6] *= initial_vel_std ** 2
        self.P[6:9, 6:9] *= initial_acc_std ** 2

        # Measurement matrix: observe position only.
        self.H = np.zeros((3, self.n))
        self.H[0:3, 0:3] = np.eye(3)

        self._last_update_t = None

    # ------------------------------------------------------------------

    @staticmethod
    def _per_axis_F(dt: float) -> np.ndarray:
        return np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _per_axis_Q(dt: float, q: float) -> np.ndarray:
        # Standard discretized continuous white-noise-jerk model
        # (Bar-Shalom eq. 6.2.3-8, third-order kinematic model).
        dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
        return q * np.array([
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0,  dt2 / 2.0, dt],
        ])

    def _build_F_Q(self, dt: float):
        f_axis = self._per_axis_F(dt)
        q_axis = self._per_axis_Q(dt, self.q)

        F = np.zeros((self.n, self.n))
        Q = np.zeros((self.n, self.n))
        idx = {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5, 8]}  # per-axis [p, v, a] indices
        for axis in range(3):
            ids = idx[axis]
            for i in range(3):
                for j in range(3):
                    F[ids[i], ids[j]] = f_axis[i, j]
                    Q[ids[i], ids[j]] = q_axis[i, j]
        return F, Q

    # ------------------------------------------------------------------

    def predict(self, dt: float):
        if dt <= 0:
            return
        F, Q = self._build_F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_position: np.ndarray):
        z = measured_position
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.r * np.eye(3)
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        return y, S  # returned for behavior-change / innovation monitoring later

    # ------------------------------------------------------------------

    def get_position(self) -> np.ndarray:
        return self.x[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    def get_acceleration(self) -> np.ndarray:
        return self.x[6:9].copy()

    def get_position_covariance(self) -> np.ndarray:
        return self.P[0:3, 0:3].copy()

    def predict_future_position(self, horizon: float, n_points: int = 1):
        """
        Probabilistic prediction of future evader position: propagates
        the current state/covariance forward `horizon` seconds using the
        same CA model (no new measurements), returning a list of
        (t_offset, mean_position, covariance) tuples. This is what
        satisfies the ">= 2s probabilistic prediction" requirement.
        """
        x, P = self.x.copy(), self.P.copy()
        dt = horizon / n_points
        results = []
        for i in range(1, n_points + 1):
            F, Q = self._build_F_Q(dt)
            x = F @ x
            P = F @ P @ F.T + Q
            results.append((i * dt, x[0:3].copy(), P[0:3, 0:3].copy()))
        return results
