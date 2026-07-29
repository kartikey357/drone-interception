"""
Week 7 (extended): Dynamic Window Approach (DWA) intercept controller.

Motivation (found via ablation, see project docs): the quintic-plan
model-based planner and the simple proportional guidance laws both
plateau around 26-32% interception against the maneuvering evader,
because neither explicitly reasons about the pursuer's OWN kinematic
constraint: minimum turn radius r_min = v^2 / max_accel. At the
pursuer's own top speed (8 m/s), r_min = 10.67m -- larger than the
evader's entire turn-radius range (4-10m). Any guidance law that always
commands near-max speed toward the target will therefore periodically
be physically unable to turn sharply enough to stay with the evader.

DWA handles this WITHOUT hand-tuned heuristics: at every control tick,
sample a set of dynamically-feasible candidate accelerations, forward-
simulate each one (using the pursuer's real velocity/acceleration
limits) over a short horizon, and score each candidate by how close its
simulated endpoint gets to the evader model's OWN predicted trajectory
over that same horizon. A candidate that commands too much speed to
turn sharply enough will simply simulate to a poor position and be
naturally deprioritized -- no explicit "turn severity" formula needed.
"""

import numpy as np


def _simulate_candidate(pos, vel, accel_cmd, max_speed, horizon, n_steps):
    """Simple point-mass forward simulation (velocity-saturated double
    integrator) for scoring one candidate acceleration command."""
    dt = horizon / n_steps
    p, v = pos.copy(), vel.copy()
    trajectory = []
    for _ in range(n_steps):
        v = v + accel_cmd * dt
        speed = np.linalg.norm(v)
        if speed > max_speed:
            v = v * (max_speed / speed)
        p = p + v * dt
        trajectory.append(p.copy())
    return trajectory


def _candidate_directions(n_azimuth=12, n_elevation=3):
    """A fixed set of unit vectors spread over the sphere, used to seed
    candidate acceleration directions each tick."""
    dirs = []
    elevations = np.linspace(-0.4, 0.4, n_elevation)  # mostly horizontal, some vertical
    for elev in elevations:
        for i in range(n_azimuth):
            az = 2 * np.pi * i / n_azimuth
            horiz = np.cos(elev)
            dirs.append(np.array([horiz * np.cos(az), horiz * np.sin(az), np.sin(elev)]))
    dirs.append(np.zeros(3))  # "coast" option: no acceleration
    return dirs


_DIRS = _candidate_directions()


def dwa_step(pursuer_pos, pursuer_vel, model, max_speed, max_accel,
             horizon=1.0, n_steps=5, speed_fractions=(0.4, 0.7, 1.0)):
    """
    Returns the best-scoring acceleration command for THIS tick.
    `model` must support `predict_future_state(horizon, n_points)`.
    """
    predicted = model.predict_future_state(horizon, n_points=n_steps)
    evader_trajectory = [p[1] for p in predicted]  # positions only, at each sample time

    best_score = np.inf
    best_accel = np.zeros(3)

    for direction in _DIRS:
        for frac in speed_fractions:
            accel_cmd = direction * max_accel * frac
            sim_traj = _simulate_candidate(pursuer_pos, pursuer_vel, accel_cmd,
                                            max_speed, horizon, n_steps)
            # Score: weighted sum of distance-to-predicted-evader at
            # each sampled time, weighted more heavily toward the end
            # of the horizon (favor trajectories that converge).
            score = 0.0
            for k, (sim_p, ev_p) in enumerate(zip(sim_traj, evader_trajectory)):
                weight = (k + 1) / len(sim_traj)
                score += weight * np.linalg.norm(sim_p - ev_p)
            if score < best_score:
                best_score = score
                best_accel = accel_cmd

    return best_accel
