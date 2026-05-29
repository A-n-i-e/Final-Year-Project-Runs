"""
evaluate.py — Load a trained PPO model and evaluate / render it.

Usage:
    python evaluate.py --model ./models/best/best_model.zip  # Evaluate
    python evaluate.py --model ./models/best/best_model.zip --render  # Visual
    python evaluate.py --model ./models/best/best_model.zip --record  # Save MP4
    python evaluate.py --model ./models/best/best_model.zip --plot-rewards  # Stats
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(__file__))
from first.envs.sawyer_env import SawyerPickPlaceEnv
from first.config import ENV_CONFIG, TRAIN_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sawyer PPO")
    parser.add_argument("--model", required=True, help="Path to .zip model checkpoint")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="Path to VecNormalize .pkl (auto-detected if not given)")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer")
    parser.add_argument("--record", action="store_true", help="Record MP4 video")
    parser.add_argument("--plot-rewards", action="store_true",
                        help="Plot per-episode reward breakdown")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_eval_env(render: bool = False):
    def _init():
        return SawyerPickPlaceEnv(
            render_mode="human" if render else "rgb_array",
            n_objects=ENV_CONFIG["n_objects"],
            max_episode_steps=ENV_CONFIG["max_episode_steps"],
        )
    return DummyVecEnv([_init])


def load_model_and_env(args):
    """Load PPO model + VecNormalize wrapper."""
    env = make_eval_env(render=args.render)

    # Auto-detect vecnorm path
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        candidate = args.model.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(candidate):
            vecnorm_path = candidate

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False        # Don't update running stats
        env.norm_reward = False     # Show un-normalized rewards
    else:
        print("⚠ No VecNormalize found — results may differ from training")

    model = PPO.load(args.model, env=env, device="cpu")
    return model, env


def run_evaluation(model, env, n_episodes: int, record: bool = False):
    """
    Run evaluation episodes, collecting per-episode metrics.
    Returns a dict of lists: rewards, lengths, objects_placed, is_success.
    """
    try:
        import imageio
        CAN_RECORD = True
    except ImportError:
        CAN_RECORD = False
        if record:
            print("Install imageio for recording: pip install imageio[ffmpeg]")

    metrics = {
        "rewards": [], "lengths": [], "objects_placed": [], "is_success": [],
        "r_reach": [], "r_grasp": [], "r_lift": [], "r_place": [], "r_penalties": [],
    }
    all_frames = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_info = {}
        ep_frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1
            ep_info = info_arr[0]

            if record and CAN_RECORD:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, list):
                        frame = frame[0]
                    ep_frames.append(frame)

            done = bool(done_arr[0])

        metrics["rewards"].append(ep_reward)
        metrics["lengths"].append(ep_len)
        metrics["objects_placed"].append(ep_info.get("objects_placed", 0))
        metrics["is_success"].append(float(ep_info.get("is_success", False)))

        for key in ["r_reach", "r_grasp", "r_lift", "r_place", "r_penalties"]:
            metrics[key].append(ep_info.get(key, 0.0))

        if ep_frames:
            all_frames.extend(ep_frames)

        print(f"  Episode {ep+1:3d}/{n_episodes} | "
              f"Reward: {ep_reward:7.1f} | "
              f"Objects: {ep_info.get('objects_placed', 0)}/5 | "
              f"{'✓ SUCCESS' if ep_info.get('is_success') else '✗'}")

    if record and all_frames and CAN_RECORD:
        video_path = "evaluation_video.mp4"
        imageio.mimsave(video_path, all_frames, fps=30)
        print(f"\nSaved video → {video_path}")

    return metrics


def print_summary(metrics):
    """Print a formatted evaluation summary."""
    print("\n" + "═" * 50)
    print("  EVALUATION SUMMARY")
    print("═" * 50)
    print(f"  Episodes:         {len(metrics['rewards'])}")
    print(f"  Success rate:     {np.mean(metrics['is_success']):.1%}")
    print(f"  Avg objects/ep:   {np.mean(metrics['objects_placed']):.2f} / 5")
    print(f"  Avg reward:       {np.mean(metrics['rewards']):.1f} ± {np.std(metrics['rewards']):.1f}")
    print(f"  Avg ep length:    {np.mean(metrics['lengths']):.1f} steps")
    print(f"\n  Reward breakdown (mean per episode):")
    print(f"    Reach:          {np.mean(metrics['r_reach']):.2f}")
    print(f"    Grasp:          {np.mean(metrics['r_grasp']):.2f}")
    print(f"    Lift:           {np.mean(metrics['r_lift']):.2f}")
    print(f"    Place:          {np.mean(metrics['r_place']):.2f}")
    print(f"    Penalties:      {np.mean(metrics['r_penalties']):.2f}")
    print("═" * 50)


def plot_evaluation(metrics, save_path: str = "evaluation_analysis.png"):
    """Multi-panel analysis plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Evaluation Analysis — Sawyer PPO", fontsize=16, fontweight="bold")

    episodes = range(1, len(metrics["rewards"]) + 1)
    colors = {"reach": "#2196F3", "grasp": "#FF5722", "lift": "#4CAF50",
              "place": "#9C27B0", "penalties": "#F44336"}

    # 1. Per-episode reward
    ax = axes[0, 0]
    ax.bar(episodes, metrics["rewards"], color="#42A5F5", alpha=0.8)
    ax.axhline(np.mean(metrics["rewards"]), color="red", linestyle="--", label="Mean")
    ax.set_title("Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()

    # 2. Objects placed distribution
    ax = axes[0, 1]
    counts = [metrics["objects_placed"].count(i) for i in range(6)]
    ax.bar(range(6), counts, color="#66BB6A", alpha=0.8)
    ax.set_title("Objects Placed per Episode")
    ax.set_xlabel("Objects Placed")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(6))

    # 3. Success rate (rolling)
    ax = axes[0, 2]
    window = min(10, len(metrics["is_success"]))
    rolling = np.convolve(metrics["is_success"],
                          np.ones(window) / window, mode="valid")
    ax.plot(rolling, color="#AB47BC", linewidth=2)
    ax.fill_between(range(len(rolling)), rolling, alpha=0.2, color="#AB47BC")
    ax.set_ylim(0, 1)
    ax.set_title(f"Rolling Success Rate (window={window})")
    ax.set_xlabel("Episode")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # 4. Reward components stacked area
    ax = axes[1, 0]
    components = ["r_reach", "r_grasp", "r_lift", "r_place"]
    comp_labels = ["Reach", "Grasp", "Lift", "Place"]
    comp_colors = [colors["reach"], colors["grasp"], colors["lift"], colors["place"]]
    data = [metrics[c] for c in components]
    ax.stackplot(episodes, *data, labels=comp_labels, colors=comp_colors, alpha=0.7)
    ax.set_title("Reward Components (stacked)")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper left", fontsize=8)

    # 5. Episode lengths
    ax = axes[1, 1]
    ax.plot(episodes, metrics["lengths"], color="#FF7043", alpha=0.7)
    ax.axhline(np.mean(metrics["lengths"]), color="red", linestyle="--", label="Mean")
    ax.set_title("Episode Lengths")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    # 6. Reward vs objects placed scatter
    ax = axes[1, 2]
    sc = ax.scatter(metrics["objects_placed"], metrics["rewards"],
                    c=metrics["is_success"], cmap="RdYlGn", alpha=0.7, s=50)
    plt.colorbar(sc, ax=ax, label="Success")
    ax.set_title("Reward vs Objects Placed")
    ax.set_xlabel("Objects Placed")
    ax.set_ylabel("Total Reward")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved analysis plot → {save_path}")
    plt.close()


def main():
    args = parse_args()
    print(f"\nLoading model from: {args.model}")

    model, env = load_model_and_env(args)

    print(f"\nRunning {args.episodes} evaluation episodes...\n")
    metrics = run_evaluation(model, env, args.episodes, record=args.record)

    print_summary(metrics)

    if args.plot_rewards:
        plot_evaluation(metrics)

    env.close()


if __name__ == "__main__":
    main()
