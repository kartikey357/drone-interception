"""
Shared simulation interface.

Any backend (the fast headless Python simulator, or a PX4/Gazebo/MAVROS
wrapper) implements this interface so that the estimator, evader model,
planner, and controller code can be written once and run against either.

Design rule enforced by this interface: the pursuer-facing methods
(`get_pursuer_state`, `get_noisy_evader_observation`) never expose the
evader's true state. Ground truth is only available via
`get_ground_truth`, which must be used for logging/evaluation only —
never fed into the estimator or planner.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class EpisodeStatus(Enum):
    RUNNING = "running"
    INTERCEPTED = "intercepted"
    TIMEOUT = "timeout"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass
class AgentState:
    """True kinematic state of a single agent (pursuer or evader)."""
    position: np.ndarray      # shape (3,) meters, world frame
    velocity: np.ndarray      # shape (3,) m/s
    acceleration: np.ndarray  # shape (3,) m/s^2
    yaw: float = 0.0          # radians
    t: float = 0.0            # simulation time, seconds


@dataclass
class Observation:
    """A single noisy sensor reading of the evader, or None if dropped out."""
    position: np.ndarray | None   # shape (3,), noisy evader position, or None if dropout
    t: float = 0.0
    valid: bool = True


@dataclass
class StepResult:
    pursuer_state: AgentState
    status: EpisodeStatus
    miss_distance: float          # current true distance pursuer<->evader (logging only)
    info: dict = field(default_factory=dict)


class InterceptionSimulator(ABC):
    """
    Common contract for all simulation backends.

    Units: meters, seconds, m/s, m/s^2, radians. World frame is ENU-like
    (x east, y north, z up), consistent with MAVROS local_position conventions
    so the PX4/Gazebo backend and the fast backend agree.
    """

    # ---- lifecycle ---------------------------------------------------

    @abstractmethod
    def reset(self, evader_profile: str, seed: int | None = None) -> AgentState:
        """
        Start a new episode. `evader_profile` selects the hidden evader
        behavior ("non_maneuvering", "maneuvering", "behavior_switching",
        or a custom registered profile name). Returns the pursuer's
        initial true state (the pursuer is always allowed to know its
        own state).
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, pursuer_accel_cmd: np.ndarray, dt: float) -> StepResult:
        """
        Advance the simulation by `dt` seconds. `pursuer_accel_cmd` is a
        commanded acceleration vector (m/s^2) in world frame, which the
        backend's low-level controller/dynamics will saturate according
        to the pursuer's velocity/acceleration/tilt limits.
        """
        raise NotImplementedError

    # ---- pursuer-facing (allowed) ------------------------------------

    @abstractmethod
    def get_pursuer_state(self) -> AgentState:
        """True pursuer state — always allowed, the pursuer knows itself."""
        raise NotImplementedError

    @abstractmethod
    def get_noisy_evader_observation(self) -> Observation:
        """
        The ONLY evader information the pursuer's estimator/model/planner
        may use during operation. Returns None position on a dropout tick.
        """
        raise NotImplementedError

    # ---- evaluation-only (never feed into the pursuer's logic) -------

    @abstractmethod
    def get_ground_truth(self) -> dict:
        """
        Full true state of both agents, for offline logging/plotting/
        evaluation ONLY. Must never be passed to the estimator or planner.
        """
        raise NotImplementedError

    # ---- bookkeeping --------------------------------------------------

    @abstractmethod
    def is_done(self) -> tuple[bool, EpisodeStatus]:
        raise NotImplementedError
