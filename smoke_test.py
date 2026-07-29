"""
Quick sanity check for the fast simulation backend.

Uses a placeholder pure-pursuit controller (accelerate toward the
LAST NOISY OBSERVATION of the evader, with zero estimation/prediction)
just to prove the simulation loop, dynamics, sensor, and episode
termination logic all work end-to-end. This is NOT the real estimator/
planner (those come in Weeks 2-5) -- it's deliberately dumb so that any
bugs found are in the simulator itself, not in future algorithm code.
"""

import numpy as np
from interception_sim import FastInterceptionSim, EpisodeStatus

DT = 1.0 / 50.0  # 50 Hz control loop, per PS minimum


def naive_pursuit_accel(pursuer_state, last_obs_position, max_accel=6.0):
    if last_obs_position is None:
        return np.zeros(3)
    direction = last_obs_position - pursuer_state.position
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.zeros(3)
    return (direction / dist) * max_accel


def run_episode(profile: str, seed: int, verbose=True):
    sim = FastInterceptionSim(sensor_rate_hz=20.0, sensor_noise_std=0.3)
    sim.reset(evader_profile=profile, seed=seed)

    last_known_evader_pos = None
    steps = 0
    while True:
        obs = sim.get_noisy_evader_observation()
        if obs.valid and obs.position is not None:
            last_known_evader_pos = obs.position

        pursuer_state = sim.get_pursuer_state()
        accel_cmd = naive_pursuit_accel(pursuer_state, last_known_evader_pos)
        result = sim.step(accel_cmd, DT)
        steps += 1

        done, status = sim.is_done()
        if done:
            if verbose:
                gt = sim.get_ground_truth()
                print(f"[{profile:20s}] seed={seed:3d}  status={status.value:14s}  "
                      f"t={gt['t']:5.1f}s  steps={steps:5d}  final_miss={result.miss_distance:.2f}m")
            return status


if __name__ == "__main__":
    for profile in ("non_maneuvering", "maneuvering", "behavior_switching"):
        for seed in range(5):
            run_episode(profile, seed)
