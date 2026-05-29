"""
callbacks/training_callbacks.py — Custom SB3 callbacks for monitoring training.

Three callbacks:
  1. SuccessRateCallback: logs per-episode success metrics to TensorBoard
  2. VideoRecorderCallback: saves MP4 renders every N steps (visual progress)
  3. CurriculumCallback: optionally increases difficulty as agent improves
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt


class SuccessRateCallback(BaseCallback):
    """
    Logs reward component breakdowns and success rate to TensorBoard.
    Attaches to the training env (not eval env) — runs every episode end.

    Tracked metrics:
      - success_rate: fraction of episodes where all 5 objects were placed
      - objects_placed_mean: average number of objects placed per episode
      - Episode-level reward components from info dict
    """

    def __init__(self, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._successes = []
        self._objects_placed = []
        # Track reward components
        self._reward_components = {
            "r_reach": [], "r_grasp": [], "r_lift": [],
            "r_place": [], "r_penalties": [],
        }

    def _on_step(self) -> bool:
        """Called after every env step."""
        # SB3 passes 'infos' as a list (one per parallel env)
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                # Episode ended — log stats
                ep_info = info["episode"]
                self._episode_rewards.append(ep_info["r"])
                self._episode_lengths.append(ep_info["l"])
                self._successes.append(float(info.get("is_success", False)))
                self._objects_placed.append(info.get("objects_placed", 0))

                # Reward component tracking
                for key in self._reward_components:
                    if key in info:
                        self._reward_components[key].append(info[key])

        # Periodic logging
        if self.n_calls % self.eval_freq == 0 and self._successes:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Write accumulated stats to TensorBoard."""
        if not self._successes:
            return

        success_rate = np.mean(self._successes[-100:])
        mean_objects = np.mean(self._objects_placed[-100:])
        mean_reward = np.mean(self._episode_rewards[-100:])
        mean_length = np.mean(self._episode_lengths[-100:])

        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/objects_placed_mean", mean_objects)
        self.logger.record("rollout/ep_rew_mean_100", mean_reward)
        self.logger.record("rollout/ep_len_mean_100", mean_length)

        for key, vals in self._reward_components.items():
            if vals:
                self.logger.record(f"reward/{key}", np.mean(vals[-100:]))

        if self.verbose:
            print(f"\n[Step {self.num_timesteps}] "
                  f"Success: {success_rate:.2%} | "
                  f"Objects/ep: {mean_objects:.2f} | "
                  f"Reward: {mean_reward:.1f}")


class VideoRecorderCallback(BaseCallback):
    """
    Records MP4 videos of the agent at regular intervals.
    Saves to log_dir/videos/step_{N}.mp4

    Uses the eval environment (single env, render_mode='rgb_array').
    """

    def __init__(
        self,
        eval_env,
        record_freq: int = 50_000,
        n_eval_episodes: int = 1,
        log_dir: str = "./logs/",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_dir = os.path.join(log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        try:
            import imageio
        except ImportError:
            print("Install imageio for video recording: pip install imageio[ffmpeg]")
            return

        frames = []
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)
                done = terminated or truncated

        if frames:
            path = os.path.join(self.video_dir, f"step_{self.num_timesteps:08d}.mp4")
            imageio.mimsave(path, frames, fps=30)
            if self.verbose:
                print(f"[VideoRecorder] Saved {len(frames)} frames → {path}")


class RewardPlotCallback(BaseCallback):
    """
    Saves a PNG chart of rolling reward and success rate every N steps.
    Useful for quick visual inspection without TensorBoard.
    """

    def __init__(self, plot_freq: int = 20_000, log_dir: str = "./logs/", verbose: int = 0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self._steps = []
        self._rewards = []
        self._successes = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                self._steps.append(self.num_timesteps)
                self._rewards.append(info["episode"]["r"])
                self._successes.append(float(info.get("is_success", False)))

        if self.n_calls % self.plot_freq == 0 and len(self._rewards) > 10:
            self._save_plot()
        return True

    def _save_plot(self):
        window = min(200, len(self._rewards))
        rolling_reward = np.convolve(
            self._rewards, np.ones(window) / window, mode="valid"
        )
        rolling_success = np.convolve(
            self._successes, np.ones(window) / window, mode="valid"
        )
        steps_trimmed = self._steps[window - 1:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"Training Progress — Step {self.num_timesteps:,}", fontsize=14)

        ax1.plot(steps_trimmed, rolling_reward, color="#2196F3", linewidth=1.5)
        ax1.set_ylabel("Rolling Reward (ep)")
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(steps_trimmed, rolling_reward, alpha=0.15, color="#2196F3")

        ax2.plot(steps_trimmed, rolling_success, color="#4CAF50", linewidth=1.5)
        ax2.set_ylabel("Success Rate")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(steps_trimmed, rolling_success, alpha=0.15, color="#4CAF50")

        plt.tight_layout()
        path = os.path.join(self.plot_dir, f"progress_{self.num_timesteps:08d}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        if self.verbose:
            print(f"[PlotCallback] Saved plot → {path}")


class CurriculumCallback(BaseCallback):
    """
    Optional curriculum: starts with 1 object and adds more as success rate improves.
    Modify env.n_objects dynamically during training.

    Thresholds: unlock next object when 80% success on current count.
    """

    def __init__(self, success_threshold: float = 0.8, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.check_freq = check_freq
        self._successes = []
        self._current_n_objects = 1

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                # Count success as placing AT LEAST current_n_objects
                placed = info.get("objects_placed", 0)
                self._successes.append(float(placed >= self._current_n_objects))

        if self.n_calls % self.check_freq == 0:
            self._maybe_advance()
        return True

    def _maybe_advance(self):
        if len(self._successes) < 50:
            return
        recent = np.mean(self._successes[-100:])
        if recent >= self.success_threshold and self._current_n_objects < 5:
            self._current_n_objects += 1
            # Update all parallel envs
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method(
                    "set_attr", "n_objects", self._current_n_objects
                )
            if self.verbose:
                print(f"\n[Curriculum] Success {recent:.0%} → "
                      f"advancing to {self._current_n_objects} objects!")
            self._successes = []  # Reset tracking for new difficulty
