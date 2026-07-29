"""
Week 5 deliverable: model-based intercept planner in the closed loop.

Pipeline per control tick:
  sensor -> IMM model (predict+update) -> periodic replan (quintic
  intercept trajectory from the model's predicted future state) ->
  trajectory-tracking controller (feedforward accel + PD feedback on
  the planned position/velocity) -> pursuer dynamics.

Evaluated against the maneuvering evader profile and compared directly
to Week 3's baseline (intercept-point aiming against a raw constant-
velocity extrapolation) to quantify the improvement from using the
online model's predictions instead of a fixed-velocity assumption.
"""

import numpy as np
from interception_sim import (FastInterceptionSim, IMMEvaderModel, EpisodeStatus,
                               QuadrotorLimits, plan_intercept)
from interception_sim.baseline_planner import intercept_point_aiming, boundary_repulsion

DT = 1.0 / 50.0
N_EPISODES = 100
REPLAN_INTERVAL = 0.3  # seconds
ARENA_SIZE = np.array([50.0, 50.0, 30.0])

# Trajectory-tracking gains (feedforward acceleration + PD feedback).
KP = 3.0
KV = 2.5


def run_model_based_episode(seed: int, evader_profile: str = "maneuvering", max_steps: int = 3000):
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3, pursuer_limits=limits)
    sim.reset(evader_profile=evader_profile, seed=seed)

    model = None
    plan = None
    last_plan_t = -np.inf

    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        pursuer_state = sim.get_pursuer_state()
        t = pursuer_state.t

        if model is None:
            if obs.valid and obs.position is not None:
                model = IMMEvaderModel(obs.position)
            result = sim.step(np.zeros(3), DT)
        else:
            model.predict(DT)
            if obs.valid and obs.position is not None:
                model.update(obs.position)

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
            return status, sim.get_ground_truth()["t"], result.miss_distance

    return EpisodeStatus.TIMEOUT, max_steps * DT, None


def run_baseline_episode(seed: int, evader_profile: str = "maneuvering", max_steps: int = 3000):
    """Week 3 baseline (intercept-point aiming, constant-velocity assumption) for comparison."""
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3, pursuer_limits=limits)
    sim.reset(evader_profile=evader_profile, seed=seed)

    from interception_sim import ConstantAccelerationKF
    kf = None
    for _ in range(max_steps):
        obs = sim.get_noisy_evader_observation()
        pursuer_state = sim.get_pursuer_state()

        if kf is None:
            if obs.valid and obs.position is not None:
                kf = ConstantAccelerationKF(obs.position, measurement_noise_std=0.3)
            result = sim.step(np.zeros(3), DT)
        else:
            kf.predict(DT)
            if obs.valid and obs.position is not None:
                kf.update(obs.position)
            accel_cmd = intercept_point_aiming(
                pursuer_state.position, pursuer_state.velocity,
                kf.get_position(), kf.get_velocity(),
                pursuer_speed=limits.max_speed, max_accel=limits.max_horizontal_accel)

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
            return status, sim.get_ground_truth()["t"], result.miss_distance

    return EpisodeStatus.TIMEOUT, max_steps * DT, None


def evaluate(run_fn, label, n_episodes=N_EPISODES):
    outcomes = {s: 0 for s in EpisodeStatus}
    intercept_times, miss_distances = [], []
    for seed in range(n_episodes):
        status, t, miss = run_fn(seed)
        outcomes[status] += 1
        if status == EpisodeStatus.INTERCEPTED:
            intercept_times.append(t)
        if miss is not None:
            miss_distances.append(miss)

    rate = outcomes[EpisodeStatus.INTERCEPTED] / n_episodes
    print(f"\n=== {label} (n={n_episodes}, evader=maneuvering) ===")
    for status in EpisodeStatus:
        print(f"  {status.value:15s}: {outcomes[status]:4d}  ({100*outcomes[status]/n_episodes:.1f}%)")
    print(f"  interception rate : {100*rate:.1f}%")
    if intercept_times:
        print(f"  mean time-to-intercept (intercepted only): {np.mean(intercept_times):.2f}s")
    if miss_distances:
        print(f"  mean miss distance at episode end: {np.mean(miss_distances):.2f}m")
    return rate


if __name__ == "__main__":
    evaluate(run_baseline_episode, "Week 3 baseline (intercept-point aiming)")
    evaluate(run_model_based_episode, "Week 5 model-based planner (IMM + quintic intercept)")
