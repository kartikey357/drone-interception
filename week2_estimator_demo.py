"""
Week 2 deliverable: validate the Kalman filter estimator.

Runs episodes feeding ONLY noisy sensor observations into the
ConstantAccelerationKF (the pursuer's actual information channel),
while separately recording ground truth for offline comparison only
(never fed back into the filter or the controller).

Produces:
  1. estimation_error_vs_time.png — filter position error over the
     course of a single episode (should settle/shrink after the first
     few observations as the filter converges).
  2. estimation_error_vs_noise.png — mean steady-state error across a
     sweep of sensor noise levels, to quantify robustness to noise
     (a required deliverable per the PS).

Also uses the estimator's own output (not raw noisy readings, and
NOT ground truth) to drive a simple pursuit controller, replacing the
Week-1 smoke test's "chase the raw noisy reading" placeholder with
something closer to the real system.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interception_sim import FastInterceptionSim, ConstantAccelerationKF, EpisodeStatus

DT = 1.0 / 50.0  # 50 Hz control loop
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def estimator_accel_cmd(pursuer_state, estimated_evader_pos, max_accel=6.0):
    direction = estimated_evader_pos - pursuer_state.position
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.zeros(3)
    return (direction / dist) * max_accel


def run_episode_with_estimator(profile: str, seed: int, sensor_noise_std: float = 0.3,
                                process_noise_psd: float = 4.0, max_steps: int = 3000):
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=sensor_noise_std)
    sim.reset(evader_profile=profile, seed=seed)

    kf = None
    t_hist, err_hist = [], []

    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()

        if kf is None:
            if obs.valid and obs.position is not None:
                kf = ConstantAccelerationKF(obs.position,
                                             process_noise_psd=process_noise_psd,
                                             measurement_noise_std=sensor_noise_std)
            # No estimate yet: hold position (can't safely chase nothing).
            pursuer_state = sim.get_pursuer_state()
            result = sim.step(np.zeros(3), DT)
        else:
            kf.predict(DT)
            if obs.valid and obs.position is not None:
                kf.update(obs.position)

            pursuer_state = sim.get_pursuer_state()
            accel_cmd = estimator_accel_cmd(pursuer_state, kf.get_position())
            result = sim.step(accel_cmd, DT)

            gt = sim.get_ground_truth()
            true_evader_pos = gt["evader_position"]
            err = float(np.linalg.norm(kf.get_position() - true_evader_pos))
            t_hist.append(gt["t"])
            err_hist.append(err)

        done, status = sim.is_done()
        if done:
            return status, np.array(t_hist), np.array(err_hist)

    return EpisodeStatus.TIMEOUT, np.array(t_hist), np.array(err_hist)


def plot_error_vs_time():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, profile in zip(axes, ("non_maneuvering", "maneuvering", "behavior_switching")):
        status, t, err = run_episode_with_estimator(profile, seed=42)
        ax.plot(t, err, lw=1.2)
        ax.set_title(f"{profile}\n(final status: {status.value})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("estimator position error (m)")
        ax.grid(alpha=0.3)
    fig.suptitle("Kalman filter estimation error over a single episode")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "estimation_error_vs_time.png")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")


def plot_error_vs_noise():
    noise_levels = [0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    mean_errors, std_errors = [], []

    for sigma in noise_levels:
        errs_this_level = []
        for seed in range(10):
            status, t, err = run_episode_with_estimator(
                "maneuvering", seed=seed, sensor_noise_std=sigma)
            if len(err) > 50:
                # steady-state error: average over the second half of the run
                errs_this_level.append(np.mean(err[len(err) // 2:]))
        mean_errors.append(np.mean(errs_this_level))
        std_errors.append(np.std(errs_this_level))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(noise_levels, mean_errors, yerr=std_errors, marker="o", capsize=3)
    ax.set_xlabel("sensor noise std (m)")
    ax.set_ylabel("steady-state estimator position error (m)")
    ax.set_title("Estimator robustness vs. sensor noise\n(maneuvering evader, 10 seeds/level)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "estimation_error_vs_noise.png")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_error_vs_time()
    plot_error_vs_noise()
