"""
Week 3 deliverable: get the first reproducible interceptions.

Runs both baseline strategies (pure pursuit, intercept-point aiming)
against the non-maneuvering evader over N episodes, reporting
interception rate, mean miss distance, and mean time-to-intercept —
the same metrics the PS's final evaluation suite requires, just scoped
to one profile for now as a first checkpoint.
"""

import numpy as np
from interception_sim import FastInterceptionSim, ConstantAccelerationKF, EpisodeStatus, QuadrotorLimits
from interception_sim.baseline_planner import pure_pursuit, intercept_point_aiming, boundary_repulsion

DT = 1.0 / 50.0
N_EPISODES = 100
ARENA_SIZE = np.array([50.0, 50.0, 30.0])


def run_episode(strategy: str, seed: int, evader_profile: str = "non_maneuvering",
                 max_steps: int = 3000):
    limits = QuadrotorLimits()
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3, pursuer_limits=limits)
    sim.reset(evader_profile=evader_profile, seed=seed)

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

            est_pos = kf.get_position()
            est_vel = kf.get_velocity()

            if strategy == "pure_pursuit":
                accel_cmd = pure_pursuit(pursuer_state.position, pursuer_state.velocity,
                                          est_pos, limits.max_horizontal_accel)
            elif strategy == "intercept_point":
                accel_cmd = intercept_point_aiming(
                    pursuer_state.position, pursuer_state.velocity,
                    est_pos, est_vel,
                    pursuer_speed=limits.max_speed,
                    max_accel=limits.max_horizontal_accel)
            else:
                raise ValueError(strategy)

            repulsion = boundary_repulsion(pursuer_state.position, ARENA_SIZE, margin=2.0,
                                            max_repulsion_accel=limits.max_horizontal_accel)
            if np.linalg.norm(repulsion) > 1e-6:
                # Near a wall: braking/repulsion takes full priority over
                # pursuit so the pursuer can actually stop in time given
                # its real acceleration limit (stopping distance at max
                # speed ~= v^2 / (2*max_accel) ~ 5.3m here).
                accel_cmd = repulsion
            mag = np.linalg.norm(accel_cmd)
            if mag > limits.max_horizontal_accel:
                accel_cmd = accel_cmd * (limits.max_horizontal_accel / mag)

            result = sim.step(accel_cmd, DT)

        done, status = sim.is_done()
        if done:
            return status, sim.get_ground_truth()["t"], result.miss_distance

    return EpisodeStatus.TIMEOUT, max_steps * DT, None


def evaluate_strategy(strategy: str, n_episodes: int = N_EPISODES):
    outcomes = {s: 0 for s in EpisodeStatus}
    intercept_times, miss_distances = [], []

    for seed in range(n_episodes):
        status, t, miss = run_episode(strategy, seed)
        outcomes[status] += 1
        if status == EpisodeStatus.INTERCEPTED:
            intercept_times.append(t)
        if miss is not None:
            miss_distances.append(miss)

    rate = outcomes[EpisodeStatus.INTERCEPTED] / n_episodes
    print(f"\n=== {strategy} (n={n_episodes}, evader=non_maneuvering) ===")
    for status in EpisodeStatus:
        print(f"  {status.value:15s}: {outcomes[status]:4d}  ({100*outcomes[status]/n_episodes:.1f}%)")
    print(f"  interception rate : {100*rate:.1f}%")
    if intercept_times:
        print(f"  mean time-to-intercept (intercepted only): {np.mean(intercept_times):.2f}s")
    if miss_distances:
        print(f"  mean miss distance at episode end: {np.mean(miss_distances):.2f}m")
    return rate


if __name__ == "__main__":
    evaluate_strategy("pure_pursuit")
    evaluate_strategy("intercept_point")
