"""
Week 4 deliverable: validate the online (IMM) evader model.

Two required pieces of evidence, both explicit PS deliverables:
  1. Prediction error decreases with more observation time within an
     episode (using the maneuvering profile, since it's the harder case).
  2. Behavior-switch detection: when the evader flips between modes,
     how long until the model's mode probability reflects the change
     (target: < 2s per the PS).

Ground truth (`get_ground_truth()["evader_active_mode"]`) is used ONLY
here, for offline evaluation -- never fed into the model itself.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interception_sim import FastInterceptionSim, IMMEvaderModel

DT = 1.0 / 50.0
PREDICTION_HORIZON = 2.0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_prediction_error_experiment(profile="maneuvering", seed=7, max_steps=3000):
    """
    At every control tick, snapshot the model's current 2s-ahead
    prediction. After the episode, compare each snapshot's prediction
    against the actual evader position 2s later (ground truth, offline
    only) to see whether prediction error trends down as the model sees
    more data.
    """
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3)
    sim.reset(evader_profile=profile, seed=seed)

    model = None
    snapshot_times, snapshot_preds = [], []
    true_pos_log_t, true_pos_log_pos = [], []

    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        gt = sim.get_ground_truth()
        true_pos_log_t.append(gt["t"])
        true_pos_log_pos.append(gt["evader_position"].copy())

        if model is None:
            if obs.valid and obs.position is not None:
                model = IMMEvaderModel(obs.position)
            sim.step(np.zeros(3), DT)
        else:
            model.predict(DT)
            if obs.valid and obs.position is not None:
                model.update(obs.position)

            preds = model.predict_future_position(PREDICTION_HORIZON, n_points=1)
            _, pred_pos, _ = preds[0]
            snapshot_times.append(gt["t"])
            snapshot_preds.append(pred_pos)

            direction = model.get_position() - sim.get_pursuer_state().position
            dist = np.linalg.norm(direction)
            accel = (direction / dist) * 4.0 if dist > 1e-6 else np.zeros(3)
            sim.step(accel, DT)

        done, status = sim.is_done()
        if done:
            break

    true_pos_log_t = np.array(true_pos_log_t)
    true_pos_log_pos = np.array(true_pos_log_pos)

    errors = []
    for t_snap, pred_pos in zip(snapshot_times, snapshot_preds):
        target_t = t_snap + PREDICTION_HORIZON
        if target_t > true_pos_log_t[-1]:
            continue
        idx = np.searchsorted(true_pos_log_t, target_t)
        idx = min(idx, len(true_pos_log_t) - 1)
        true_future_pos = true_pos_log_pos[idx]
        errors.append((t_snap, float(np.linalg.norm(pred_pos - true_future_pos))))

    return np.array(errors)


def plot_prediction_error_vs_observation_time():
    all_errors = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed in range(8):
        errs = run_prediction_error_experiment(seed=seed)
        if len(errs) > 0:
            ax.plot(errs[:, 0], errs[:, 1], alpha=0.35, color="tab:blue")
            all_errors.append(errs)

    pooled = np.concatenate(all_errors, axis=0)
    bins = np.linspace(0, pooled[:, 0].max(), 15)
    bin_idx = np.digitize(pooled[:, 0], bins)
    bin_centers, bin_means = [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() > 3:
            bin_centers.append((bins[b - 1] + bins[b]) / 2)
            bin_means.append(pooled[mask, 1].mean())
    ax.plot(bin_centers, bin_means, color="tab:red", lw=2.5, label="binned mean (8 episodes)")

    ax.set_xlabel("observation time within episode (s)")
    ax.set_ylabel(f"{PREDICTION_HORIZON}s-ahead prediction error (m)")
    ax.set_title("IMM model: prediction error vs. observation time\n(maneuvering evader, 8 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "prediction_error_vs_observation_time.png")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")


def run_behavior_switch_experiment(seed=6, max_steps=4000):
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3)
    sim.reset(evader_profile="behavior_switching", seed=seed)

    model = None
    t_log, mode_prob_log, true_mode_log = [], [], []

    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        gt = sim.get_ground_truth()

        if model is None:
            if obs.valid and obs.position is not None:
                model = IMMEvaderModel(obs.position)
            sim.step(np.zeros(3), DT)
        else:
            model.predict(DT)
            if obs.valid and obs.position is not None:
                model.update(obs.position)

            t_log.append(gt["t"])
            mode_prob_log.append(model.get_mode_probability_high_maneuver())
            true_mode_log.append(gt["evader_active_mode"])

            direction = model.get_position() - sim.get_pursuer_state().position
            dist = np.linalg.norm(direction)
            accel = (direction / dist) * 4.0 if dist > 1e-6 else np.zeros(3)
            sim.step(accel, DT)

        done, status = sim.is_done()
        if done:
            break

    return np.array(t_log), np.array(mode_prob_log), np.array(true_mode_log)


def plot_behavior_switch_detection():
    t, mode_prob, true_mode = run_behavior_switch_experiment()

    switch_indices = np.where(np.diff(true_mode) != 0)[0]
    switch_times = t[switch_indices]

    latencies = []
    for i, switch_t in zip(switch_indices, switch_times):
        expected_mode_after = true_mode[i + 1]
        after = t > switch_t
        detected_t = None
        for tt, mp in zip(t[after], mode_prob[after]):
            detected = (mp > 0.5) if expected_mode_after == 1 else (mp < 0.5)
            if detected:
                detected_t = tt
                break
        if detected_t is not None:
            latencies.append(detected_t - switch_t)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, mode_prob, label="P(maneuvering mode)", color="tab:blue")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    for switch_t in switch_times:
        ax.axvline(switch_t, color="tab:red", ls="--", alpha=0.6)
    ax.fill_between(t, 0, 1, where=(true_mode == 1), alpha=0.08, color="tab:red",
                     label="true mode = maneuvering")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(maneuvering mode)")
    ax.set_ylim(-0.05, 1.05)
    title = "Behavior-switch detection\n"
    if latencies:
        title += f"(mean detection latency: {np.mean(latencies):.2f}s over {len(latencies)} switches)"
    else:
        title += "(no switches captured this run)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "behavior_switch_detection.png")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")
    if latencies:
        print(f"Detection latencies (s): {[round(l, 2) for l in latencies]}")
        print(f"Mean latency: {np.mean(latencies):.2f}s  (target: < 2.0s)")


if __name__ == "__main__":
    plot_prediction_error_vs_observation_time()
    plot_behavior_switch_detection()
