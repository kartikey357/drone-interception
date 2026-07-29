"""
live_dashboard.py
--------------------------------------------------------------
Thread-safe live matplotlib dashboard for the Gazebo pursuit demo.

Usage pattern (see the small patch in pursuit_gazebo_demo.py):
    dash = PursuitDashboard()
    ...inside the ROS timer callback, after model.update()/model.predict()...
    dash.push(t=now, own_pos=self.own_pos, evader_true_pos=self.evader_true_pos,
               model=self.model)
    ...in main(), on the MAIN thread (not the rclpy spin thread)...
    dash.run()   # blocks, opens the matplotlib window, pops up next to Gazebo

Design notes:
- rclpy.spin() and matplotlib's GUI loop both want the main thread. We run
  rclpy.spin() on a background thread and let matplotlib own the main thread,
  since only Tk/Qt event loops are picky about that.
- push() is called from the ROS thread, run()'s animation callback reads from
  the ROS thread -- so all shared state goes through a simple lock.
"""

import threading
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # switch to "Qt5Agg" if your container has Qt instead of Tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HISTORY_LEN = 600          # ~ how many samples to keep on the time-series panels
REFRESH_MS = 150


class PursuitDashboard:
    def __init__(self, title="Live Pursuit Dashboard"):
        self.title = title
        self._lock = threading.Lock()

        self.t = deque(maxlen=HISTORY_LEN)
        self.dist_true = deque(maxlen=HISTORY_LEN)     # ||own - evader_true||  (display only)
        self.dist_est = deque(maxlen=HISTORY_LEN)      # ||own - model_estimate||  (what pursuer "believes")
        self.mode_cv = deque(maxlen=HISTORY_LEN)
        self.mode_ca = deque(maxlen=HISTORY_LEN)
        self.mode_ct = deque(maxlen=HISTORY_LEN)

        self.own_path = deque(maxlen=HISTORY_LEN)
        self.evader_true_path = deque(maxlen=HISTORY_LEN)
        self.evader_est_path = deque(maxlen=HISTORY_LEN)
        self.forecast_xy = None

        self.captured = False
        self.t0 = None

        # display-only behaviour-switch-detection tracking (never fed to model/planner)
        self.true_mode = None
        self._last_true_mode = None
        self._switch_t0 = None
        self._latency = None
        self.latency_since_switch = None
        self.last_latency = None

    def push(self, t, own_pos, evader_true_pos, model, true_mode=None):
        """Call this once per control loop tick from the ROS thread.
        true_mode is DISPLAY-ONLY (never fed to the model/planner) -- used
        purely to prove the <2s behaviour-change-detection requirement
        visually during the demo video.
        """
        if model is None or own_pos is None:
            return
        if self.t0 is None:
            self.t0 = t

        est_pos = model.get_position()
        mp = model.get_mode_probabilities()

        try:
            # predict_future_position returns a LIST of (t, pos, cov) tuples
            fut = model.predict_future_position(horizon=2.0, n_points=15)
            forecast_xy = np.array([pos[:2] for (_t, pos, _cov) in fut])
        except Exception:
            forecast_xy = None

        with self._lock:
            self.t.append(t - self.t0)
            self.dist_est.append(float(np.linalg.norm(est_pos - own_pos)))
            if evader_true_pos is not None:
                self.dist_true.append(float(np.linalg.norm(evader_true_pos - own_pos)))
                self.evader_true_path.append(evader_true_pos[:2].copy())
            self.mode_cv.append(mp.get("cv", 0.0))
            self.mode_ca.append(mp.get("ca", 0.0))
            self.mode_ct.append(mp.get("ct", 0.0))
            self.own_path.append(own_pos[:2].copy())
            self.evader_est_path.append(est_pos[:2].copy())
            self.forecast_xy = forecast_xy
            if self.dist_true and self.dist_true[-1] < 0.5:
                self.captured = True

            # --- display-only: behaviour-switch detection latency ---
            is_maneuvering_true = true_mode in ("aggressive", "erratic")
            is_maneuvering_est = (mp.get("ca", 0.0) + mp.get("ct", 0.0)) > 0.5
            if true_mode is not None and true_mode != self._last_true_mode:
                self._switch_t0 = t
                self._latency = None
                self._last_true_mode = true_mode
            if self._switch_t0 is not None and self._latency is None:
                if is_maneuvering_est == is_maneuvering_true:
                    self._latency = t - self._switch_t0
            self.true_mode = true_mode
            self.latency_since_switch = (t - self._switch_t0) if self._switch_t0 is not None else None
            self.last_latency = self._latency

    # -------------------------------------------------------------
    def run(self):
        """Blocking call -- run this on the MAIN thread only."""
        fig = plt.figure(figsize=(13, 6))
        ax_xy = fig.add_subplot(1, 3, 1)
        ax_dist = fig.add_subplot(1, 3, 2)
        ax_mode = fig.add_subplot(1, 3, 3)
        fig.suptitle(self.title)

        def update(_frame):
            with self._lock:
                t = list(self.t)
                d_true = list(self.dist_true)
                d_est = list(self.dist_est)
                cv, ca, ct = list(self.mode_cv), list(self.mode_ca), list(self.mode_ct)
                own = np.array(self.own_path) if self.own_path else None
                ev_true = np.array(self.evader_true_path) if self.evader_true_path else None
                ev_est = np.array(self.evader_est_path) if self.evader_est_path else None
                forecast = self.forecast_xy
                captured = self.captured
                true_mode = self.true_mode
                latency_since_switch = self.latency_since_switch
                last_latency = self.last_latency

            # ---- Panel 1: top-down XY trajectory ----
            ax_xy.cla()
            ax_xy.set_title("Top-down trajectory (XY)")
            if ev_true is not None and len(ev_true):
                ax_xy.plot(ev_true[:, 0], ev_true[:, 1], color="green", label="Evader (true)")
                ax_xy.scatter(*ev_true[-1], color="green", s=60, zorder=5)
            if ev_est is not None and len(ev_est):
                ax_xy.plot(ev_est[:, 0], ev_est[:, 1], color="blue", linestyle="--", label="IMM estimate")
            if forecast is not None and len(forecast):
                ax_xy.plot(forecast[:, 0], forecast[:, 1], color="orange", linestyle=":", label="2s forecast")
            if own is not None and len(own):
                ax_xy.plot(own[:, 0], own[:, 1], color="red", label="Pursuer")
                ax_xy.scatter(*own[-1], color="red", s=60, zorder=5, marker="^")
            if ax_xy.get_legend_handles_labels()[0]:
                ax_xy.legend(loc="upper right", fontsize=7)
            ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
            ax_xy.set_aspect("equal", adjustable="datalim")

            # ---- Panel 2: distance to evader over time ----
            ax_dist.cla()
            ax_dist.set_title("Distance to evader")
            if t and d_est:
                ax_dist.plot(t, d_est, color="blue", label="via model estimate")
            if t and d_true:
                ax_dist.plot(t, d_true, color="green", linestyle="--", label="true (display only)")
            ax_dist.axhline(0.5, color="black", linestyle=":", linewidth=1, label="capture radius")
            ax_dist.set_xlabel("time (s)"); ax_dist.set_ylabel("distance (m)")
            if ax_dist.get_legend_handles_labels()[0]:
                ax_dist.legend(loc="upper right", fontsize=7)
            if d_est or d_true:
                latest = d_true[-1] if d_true else d_est[-1]
                colour = "darkgreen" if latest < 0.5 else "black"
                label = "CAPTURED!" if captured else f"{latest:.2f} m"
                ax_dist.text(0.98, 0.95, label, transform=ax_dist.transAxes,
                             ha="right", va="top", fontsize=14, fontweight="bold", color=colour)

            # ---- Panel 3: IMM mode probabilities ----
            ax_mode.cla()
            ax_mode.set_title("IMM mode probabilities")
            if t and cv:
                ax_mode.stackplot(t, cv, ca, ct,
                                   labels=["CV (constant vel.)", "CA (constant accel.)", "CT (coord. turn)"],
                                   colors=["#8ecae6", "#ffb703", "#fb8500"])
                ax_mode.set_ylim(0, 1)
                dominant = ["CV", "CA", "CT"][int(np.argmax([cv[-1], ca[-1], ct[-1]]))]
                ax_mode.text(0.02, 0.92, f"model mode: {dominant}", transform=ax_mode.transAxes,
                             fontsize=12, fontweight="bold")
            if true_mode is not None:
                if last_latency is not None:
                    lat_str = f"detected in {last_latency:.2f}s  {'OK' if last_latency < 2.0 else 'SLOW'}"
                    lat_color = "darkgreen" if last_latency < 2.0 else "darkred"
                elif latency_since_switch is not None:
                    lat_str = f"waiting... {latency_since_switch:.2f}s since switch"
                    lat_color = "black"
                else:
                    lat_str = ""
                    lat_color = "black"
                ax_mode.text(0.02, 0.80, f"true mode: {true_mode}", transform=ax_mode.transAxes,
                             fontsize=11, color="dimgray")
                ax_mode.text(0.02, 0.70, lat_str, transform=ax_mode.transAxes,
                             fontsize=10, color=lat_color, fontweight="bold")
            ax_mode.set_xlabel("time (s)"); ax_mode.set_ylabel("probability")
            if ax_mode.get_legend_handles_labels()[0]:
                ax_mode.legend(loc="lower left", fontsize=6)

            fig.tight_layout(rect=[0, 0, 1, 0.94])

        self._ani = FuncAnimation(fig, update, interval=REFRESH_MS, cache_frame_data=False)
        plt.show()
