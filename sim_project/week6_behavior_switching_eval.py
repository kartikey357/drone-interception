"""
Week 6 deliverable: behaviour-switching robustness.

Runs the full Week 5 closed loop (sensor -> IMM model -> model-based
planner -> tracking controller) against the behavior_switching evader
profile over N episodes, reporting:
  - interception rate / miss distance / time-to-intercept (same metrics
    as Weeks 3 and 5, now on the hardest profile)
  - detection + recovery latency for each true behaviour switch that
    occurs during an episode (target: < 2s per the PS), using
    `get_ground_truth()["evader_active_mode"]` for evaluation ONLY --
    never fed into the model or planner.
"""

import numpy as np
from interception_sim import FastInterceptionSim, IMMEvaderModel, EpisodeStatus, QuadrotorLimits, plan_intercept
from interception_sim.baseline_planner import boundary_repulsion

DT = 1.0 / 50.0
N_EPISODES = 100
REPLAN_INTERVAL = 0.3
ARENA_SIZE = np.array([50.0, 50.0, 30.0])
KP, KV = 3.0, 2.5


def run_episode(seed: int, evader_profile: str = "behavior_switching", max_steps: int = 3000):
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3, pursuer_limits=limits)
    sim.reset(evader_profile=evader_profile, seed=seed)

    model = None
    plan = None
    last_plan_t = -np.inf

    t_log, mode_prob_log, true_mode_log = [], [], []

    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        pursuer_state = sim.get_pursuer_state()
        t = pursuer_state.t
        gt = sim.get_ground_truth()

        if model is None:
            if obs.valid and obs.position is not None:
                model = IMMEvaderModel(obs.position)
            result = sim.step(np.zeros(3), DT)
        else:
            model.predict(DT)
            if obs.valid and obs.position is not None:
                model.update(obs.position)

            t_log.append(t)
            mode_prob_log.append(model.get_mode_probability_high_maneuver())
            true_mode_log.append(gt["evader_active_mode"])

            if plan is None or (t - last_plan_t) >= REPLAN_INTERVAL:
                plan = plan_intercept(pursuer_state.position, pursuer_state.velocity,
                                       pursuer_state.acceleration, model,
                                       max_speed=limits.max_speed, t_start=t)
                last_plan_t = t

            target_pos, target_vel, target_acc = plan.evaluate(t - last_plan_t)
            accel_cmd = (target_acc
                         + KP * (target_pos - pursuer_state.position)
                         + KV * (target_vel - pursuer_state.velocity))
            repulsion = boundary_repulsion(pursuer_state.position, ARENA_SIZE, margin=2.0,
                                            max_repulsion_accel=limits.max_horizontal_accel)
            if np.linalg.norm(repulsion) > 1e-6:
                accel_cmd = repulsion
            mag = np.linalg.norm(accel_cmd)
            if mag > limits.max_horizontal_accel:
                accel_cmd = accel_cmd * (limits.max_horizontal_accel / mag)

            result = sim.step(accel_cmd, DT)

        done, status = sim.is_done()
        if done:
            latencies = _compute_latencies(np.array(t_log), np.array(mode_prob_log),
                                            np.array(true_mode_log))
            return status, sim.get_ground_truth()["t"], result.miss_distance, latencies

    latencies = _compute_latencies(np.array(t_log), np.array(mode_prob_log), np.array(true_mode_log))
    return EpisodeStatus.TIMEOUT, max_steps * DT, None, latencies


def _compute_latencies(t, mode_prob, true_mode):
    if len(true_mode) < 2:
        return []
    switch_indices = np.where(np.diff(true_mode) != 0)[0]
    latencies = []
    for i in switch_indices:
        switch_t = t[i]
        expected_after = true_mode[i + 1]
        after = t > switch_t
        for tt, mp in zip(t[after], mode_prob[after]):
            detected = (mp > 0.5) if expected_after == 1 else (mp < 0.5)
            if detected:
                latencies.append(tt - switch_t)
                break
    return latencies


def evaluate(evader_profile: str = "behavior_switching", n_episodes=N_EPISODES):
    outcomes = {s: 0 for s in EpisodeStatus}
    intercept_times, miss_distances, all_latencies = [], [], []

    for seed in range(n_episodes):
        status, t, miss, latencies = run_episode(seed, evader_profile=evader_profile)
        outcomes[status] += 1
        if status == EpisodeStatus.INTERCEPTED:
            intercept_times.append(t)
        if miss is not None:
            miss_distances.append(miss)
        all_latencies.extend(latencies)

    rate = outcomes[EpisodeStatus.INTERCEPTED] / n_episodes
    print(f"\n=== Week 6: {evader_profile}, full closed loop (n={n_episodes}) ===")
    for status in EpisodeStatus:
        print(f"  {status.value:15s}: {outcomes[status]:4d}  ({100*outcomes[status]/n_episodes:.1f}%)")
    print(f"  interception rate : {100*rate:.1f}%  (PS minimum: 30%, target: 55%)")
    if intercept_times:
        print(f"  mean time-to-intercept (intercepted only): {np.mean(intercept_times):.2f}s")
    if miss_distances:
        print(f"  mean miss distance at episode end: {np.mean(miss_distances):.2f}m")

    if all_latencies:
        all_latencies = np.array(all_latencies)
        under_2s = np.mean(all_latencies < 2.0) * 100
        print(f"\n  behaviour-switch detections observed: {len(all_latencies)}")
        print(f"  mean recovery latency: {all_latencies.mean():.2f}s")
        print(f"  max recovery latency:  {all_latencies.max():.2f}s")
        print(f"  fraction under 2.0s target: {under_2s:.1f}%")
    else:
        print("  (no behaviour switches captured across these episodes)")

    return rate, all_latencies


if __name__ == "__main__":
    print("### Default profile (min_hold=6-15s) -- realistic episode-length exposure ###")
    evaluate("behavior_switching")
    print("\n### Fast-switching stress test (min_hold=2-4s) -- more switch events per episode ###")
    evaluate("behavior_switching_fast")
