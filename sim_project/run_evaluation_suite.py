"""
Week 7 deliverable: the single-command, reproducible evaluation suite.

Runs >= 100 episodes for each of the three required evader profiles
through the full closed loop (sensor -> IMM model -> model-based
planner -> tracking controller), and writes a single `results.json`
with everything the PS's Performance Targets table asks for:
  - interception rate per profile
  - mean miss distance at closest approach
  - mean time-to-intercept vs. a straight-line lower bound
  - (for behavior_switching) recovery latency statistics

Usage:
    python3 run_evaluation_suite.py [--episodes 100] [--out results.json]
"""

import argparse
import json
import time
import numpy as np

from interception_sim import FastInterceptionSim, IMMEvaderModel, EpisodeStatus, QuadrotorLimits
from interception_sim.baseline_planner import boundary_repulsion
from interception_sim.pn_planner import plan_with_pn

DT = 1.0 / 50.0
ARENA_SIZE = np.array([50.0, 50.0, 30.0])

PROFILES = ["non_maneuvering", "maneuvering", "behavior_switching"]

TARGETS = {
    "non_maneuvering":   {"min_rate": 0.85, "target_rate": 0.95},
    "maneuvering":       {"min_rate": 0.50, "target_rate": 0.75},
    "behavior_switching": {"min_rate": 0.30, "target_rate": 0.55},
}


def run_episode(seed: int, evader_profile: str, max_steps: int = 3000):
    """
    Primary pipeline: sensor -> IMM model (CV/CA/CT) -> Proportional
    Navigation guidance -> pursuer dynamics. PN guidance replaced the
    original quintic model-based planner after a documented ablation
    (see week8 report notes / pn_guidance.py docstring) showed PN
    roughly matching or beating it on every profile, and dramatically
    beating it on the maneuvering case (42% vs 26%) by explicitly
    handling the closing-speed/turn-rate tradeoff that a fixed-horizon
    trajectory replan does not.
    """
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3, pursuer_limits=limits)
    initial_pursuer_state = sim.reset(evader_profile=evader_profile, seed=seed)
    start_pos = initial_pursuer_state.position.copy()
    evader_start_pos = sim.get_ground_truth()["evader_position"].copy()
    evader_start_dist = float(np.linalg.norm(start_pos - evader_start_pos))

    model = None
    t_log, mode_prob_log, true_mode_log = [], [], []
    closest_approach = np.inf

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
            true_mode_log.append(gt.get("evader_active_mode"))

            # Receding-horizon planning: produce an explicit time-
            # parameterised trajectory each tick (satisfies the PS's
            # "time-parameterised path, not only a velocity command"
            # requirement), but only ever execute its first instant --
            # standard MPC-style receding horizon control. This does
            # NOT change closed-loop behavior vs. calling pn_guidance
            # directly (verified: identical accel_cmd at tau=0).
            plan = plan_with_pn(pursuer_state.position, pursuer_state.velocity,
                                 model.get_position(), model.get_velocity(),
                                 limits.max_horizontal_accel, limits.max_speed, t_start=t)
            _, _, accel_cmd = plan.evaluate(0.0)
            repulsion = boundary_repulsion(pursuer_state.position, ARENA_SIZE, margin=2.0,
                                            max_repulsion_accel=limits.max_horizontal_accel)
            if np.linalg.norm(repulsion) > 1e-6:
                accel_cmd = repulsion
            mag = np.linalg.norm(accel_cmd)
            if mag > limits.max_horizontal_accel:
                accel_cmd = accel_cmd * (limits.max_horizontal_accel / mag)

            result = sim.step(accel_cmd, DT)
            closest_approach = min(closest_approach, result.miss_distance)

        done, status = sim.is_done()
        if done:
            break

    final_gt = sim.get_ground_truth()
    latencies = _compute_latencies(np.array(t_log), np.array(mode_prob_log), np.array(true_mode_log)) \
        if evader_profile == "behavior_switching" else []

    # "Time to interception versus straight-line bound" per the PS: the
    # crudest possible lower bound on intercept time is (initial
    # separation) / (pursuer max speed) -- i.e. how long it would take
    # the pursuer to cover the starting distance in a straight line at
    # full speed, ignoring the evader's own motion entirely. The PS asks
    # for actual_time / this_bound to be < 3x (min) / < 1.5x (target).
    straight_line_bound = None
    time_ratio = None
    if status == EpisodeStatus.INTERCEPTED and evader_start_dist > 1e-6:
        straight_line_bound = evader_start_dist / limits.max_speed
        time_ratio = final_gt["t"] / straight_line_bound

    return {
        "status": status.value,
        "time": final_gt["t"],
        "closest_approach": closest_approach if closest_approach != np.inf else None,
        "latencies": latencies,
        "straight_line_time_bound": straight_line_bound,
        "time_to_intercept_ratio": time_ratio,
    }


def _compute_latencies(t, mode_prob, true_mode):
    if len(true_mode) < 2 or true_mode[0] is None:
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
                latencies.append(float(tt - switch_t))
                break
    return latencies


def run_profile(profile: str, n_episodes: int):
    outcomes = {s.value: 0 for s in EpisodeStatus}
    intercept_times, closest_approaches, all_latencies, time_ratios = [], [], [], []

    for seed in range(n_episodes):
        r = run_episode(seed, profile)
        outcomes[r["status"]] += 1
        if r["status"] == EpisodeStatus.INTERCEPTED.value:
            intercept_times.append(r["time"])
        if r["closest_approach"] is not None:
            closest_approaches.append(r["closest_approach"])
        if r["time_to_intercept_ratio"] is not None:
            time_ratios.append(r["time_to_intercept_ratio"])
        all_latencies.extend(r["latencies"])

    rate = outcomes[EpisodeStatus.INTERCEPTED.value] / n_episodes
    summary = {
        "n_episodes": n_episodes,
        "outcomes": outcomes,
        "interception_rate": rate,
        "mean_time_to_intercept": float(np.mean(intercept_times)) if intercept_times else None,
        "mean_closest_approach": float(np.mean(closest_approaches)) if closest_approaches else None,
        "mean_time_to_intercept_ratio_vs_straight_line": float(np.mean(time_ratios)) if time_ratios else None,
        "meets_time_ratio_minimum_3x": (float(np.mean(time_ratios)) < 3.0) if time_ratios else None,
        "meets_time_ratio_target_1_5x": (float(np.mean(time_ratios)) < 1.5) if time_ratios else None,
        "meets_minimum_threshold": rate >= TARGETS[profile]["min_rate"],
        "meets_target_threshold": rate >= TARGETS[profile]["target_rate"],
        "thresholds": TARGETS[profile],
    }
    if all_latencies:
        arr = np.array(all_latencies)
        summary["behavior_switch_detections"] = len(arr)
        summary["mean_recovery_latency"] = float(arr.mean())
        summary["max_recovery_latency"] = float(arr.max())
        summary["fraction_under_2s_target"] = float(np.mean(arr < 2.0))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

    results = {"config": {"episodes_per_profile": args.episodes, "dt": DT,
                           "guidance": "proportional_navigation"}, "profiles": {}}

    for profile in PROFILES:
        print(f"Running {args.episodes} episodes for '{profile}'...")
        t0 = time.time()
        results["profiles"][profile] = run_profile(profile, args.episodes)
        elapsed = time.time() - t0
        s = results["profiles"][profile]
        print(f"  done in {elapsed:.1f}s -- interception rate: {100*s['interception_rate']:.1f}% "
              f"(min {100*s['thresholds']['min_rate']:.0f}%, target {100*s['thresholds']['target_rate']:.0f}%)")
        if s["mean_closest_approach"] is not None:
            print(f"    mean closest approach: {s['mean_closest_approach']:.2f}m (min <2.0m, target <0.5m)")
        if s["mean_time_to_intercept_ratio_vs_straight_line"] is not None:
            print(f"    mean time-to-intercept ratio vs straight-line: "
                  f"{s['mean_time_to_intercept_ratio_vs_straight_line']:.2f}x (min <3.0x, target <1.5x)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
