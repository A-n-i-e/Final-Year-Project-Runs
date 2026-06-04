"""
evaluate.py
===========
Evaluate and compare trained PPO, SAC, and TD3 agents on PandaPickEnv.

Produces
--------
  results/
  ├── metrics_summary.csv          ← tabulated metrics for all 3 algorithms
  ├── graph1_training_reward.png   ← Training Reward vs Timesteps
  ├── graph2_success_rate.png      ← Success Rate bar chart
  ├── graph3_episode_length.png    ← Episode Length bar chart
  └── graph4_learning_curve.png    ← Success Rate vs Timesteps

Usage
-----
  # Evaluate all 3 algorithms (default)
  python evaluate.py

  # Evaluate specific algorithms only
  python evaluate.py --algos PPO SAC

  # More evaluation episodes for more accurate metrics
  python evaluate.py --n-episodes 50

Requirements
------------
  Trained models must exist at:
    models/panda_ppo/best_model.zip
    models/panda_sac/best_model.zip
    models/panda_td3/best_model.zip

  VecNormalize stats (if training used normalisation):
    models/panda_ppo/vec_normalize.pkl
    models/panda_sac/vec_normalize.pkl
    models/panda_td3/vec_normalize.pkl

  Training curves (from EvalCallback):
    logs/panda_ppo/evaluations.npz
    logs/panda_sac/evaluations.npz
    logs/panda_td3/evaluations.npz
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import csv

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.panda_pick_env import PandaPickEnv

warnings.filterwarnings("ignore")


# Config
ALGO_CLASS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}

EXP_NAMES = {
    "PPO": "panda_ppo",
    "SAC": "panda_sac",
    "TD3": "panda_td3",
}

# Colours used consistently across all graphs
ALGO_COLORS = {
    "PPO": "#2196F3",   # blue
    "SAC": "#4CAF50",   # green
    "TD3": "#FF5722",   # deep orange
}

RESULTS_DIR = "results"


# Model loading

def load_model_and_env(algo: str):
    """
    Load a trained model and its evaluation environment.
    Handles both normalised and non-normalised setups automatically.

    Returns
    -------
    model  : loaded StableBaselines3 model (PPO / SAC / TD3)
    vec_env: wrapped evaluation environment
    normalised: bool — whether VecNormalize was applied
    """
    exp_name  = EXP_NAMES[algo]
    model_dir = os.path.join("models", exp_name)
    model_path = os.path.join(model_dir, "best_model.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}\n"
            f"  → Train {algo} first:  python train.py --algo {algo}"
        )

    # Build raw env
    raw_env = DummyVecEnv([
        lambda: PandaPickEnv(max_episode_steps=200, render_mode=None)
    ])

    # Wrap with VecNormalize if stats exist
    normalised = os.path.exists(stats_path)
    if normalised:
        vec_env = VecNormalize.load(stats_path, raw_env)
        vec_env.training    = False   # freeze stats
        vec_env.norm_reward = False   # no reward normalisation during eval
    else:
        vec_env = raw_env

    # Load model
    AlgoClass = ALGO_CLASS[algo]
    model = AlgoClass.load(model_path, env=vec_env)

    return model, vec_env, normalised



# Live evaluation

def evaluate_model(model, vec_env, n_episodes: int = 30, algo: str = "?"):
    """
    Run n_episodes evaluation episodes and collect per-episode stats.

    Returns a dict with:
      rewards       : list of total episode rewards
      ep_lengths    : list of episode step counts
      successes     : list of bools (did the episode succeed?)
      object_results: dict mapping object_name → list of success bools
    """
    rewards        = []
    ep_lengths     = []
    successes      = []
    object_results = {name: [] for name in PandaPickEnv.OBJECT_NAMES}

    print(f"\n  Evaluating {algo} over {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs   = vec_env.reset()
        done  = False
        total_reward = 0.0
        steps        = 0
        ep_success   = False
        obj_name     = "unknown"

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            total_reward += float(reward[0])
            steps        += 1

            if info[0].get("is_success", False):
                ep_success = True
            obj_name = info[0].get("object_name", "unknown")

            if done[0]:
                break

        rewards.append(total_reward)
        ep_lengths.append(steps)
        successes.append(ep_success)

        if obj_name in object_results:
            object_results[obj_name].append(ep_success)

        # Progress dots
        print(f"    ep {ep+1:>3}/{n_episodes}  "
              f"reward={total_reward:+7.2f}  "
              f"steps={steps:>4}  "
              f"obj={obj_name:<10}  "
              f"{'✓' if ep_success else '✗'}")

    return {
        "rewards":        rewards,
        "ep_lengths":     ep_lengths,
        "successes":      successes,
        "object_results": object_results,
    }


def compute_metrics(eval_data: dict, algo: str) -> dict:
    """
    Summarise raw episode data into the 5 report metrics.

    Metric 1 — Mean Episode Reward
    Metric 2 — Success Rate
    Metric 3 — Average Episode Length
    Metric 4 — Learning Speed (timestep of first consistent success)
                read from evaluations.npz training log
    Metric 5 — Stability (std of episode rewards across runs)
    """
    rewards    = np.array(eval_data["rewards"])
    ep_lengths = np.array(eval_data["ep_lengths"])
    successes  = np.array(eval_data["successes"], dtype=float)

    # Per-object success rates
    obj_success = {}
    for obj, results in eval_data["object_results"].items():
        if results:
            obj_success[obj] = float(np.mean(results))
        else:
            obj_success[obj] = float("nan")

    # Metric 4: learning speed from evaluations.npz
    learning_speed_ts = _find_learning_speed(algo)

    return {
        "algo":              algo,
        # Metric 1
        "mean_reward":       float(np.mean(rewards)),
        "std_reward":        float(np.std(rewards)),
        # Metric 2
        "success_rate":      float(np.mean(successes)),
        # Per-object breakdown
        "success_per_object": obj_success,
        # Metric 3
        "mean_ep_length":    float(np.mean(ep_lengths)),
        "std_ep_length":     float(np.std(ep_lengths)),
        # Metric 4
        "learning_speed_ts": learning_speed_ts,
        # Metric 5
        "stability_std":     float(np.std(rewards)),
    }


def _find_learning_speed(algo: str, success_threshold: float = 0.5) -> int:
    """
    Metric 4 — Learning Speed.

    Reads the evaluations.npz saved by EvalCallback and finds the first
    timestep where mean reward exceeded 50% of the maximum observed reward
    (proxy for 'consistently succeeding').

    Returns -1 if the log file is not found or never crossed the threshold.
    """
    exp_name = EXP_NAMES[algo]
    npz_path = os.path.join("logs", exp_name, "evaluations.npz")

    if not os.path.exists(npz_path):
        return -1

    data      = np.load(npz_path)
    timesteps = data["timesteps"]            # shape (n_evals,)
    results   = data["results"]              # shape (n_evals, n_eval_episodes)

    mean_rewards = results.mean(axis=1)
    max_reward   = mean_rewards.max()

    # Find first timestep where mean reward crosses the threshold
    threshold = success_threshold * max_reward
    above     = np.where(mean_rewards >= threshold)[0]

    if len(above) == 0:
        return -1   # never reached threshold

    return int(timesteps[above[0]])



# Training curve loading (evaluations.npz)


def load_training_curves(algo: str):
    """
    Load the reward training curve saved by EvalCallback.

    Returns
    -------
    timesteps    : np.ndarray  (n_evals,)
    mean_rewards : np.ndarray  (n_evals,)
    std_rewards  : np.ndarray  (n_evals,)
    """
    exp_name = EXP_NAMES[algo]
    npz_path = os.path.join("logs", exp_name, "evaluations.npz")

    if not os.path.exists(npz_path):
        print(f"  ⚠ No evaluations.npz for {algo} at {npz_path} — skipping curve")
        return None, None, None

    data      = np.load(npz_path)
    timesteps = data["timesteps"]
    results   = data["results"]   # shape (n_evals, n_eval_episodes)

    mean_rewards = results.mean(axis=1)
    std_rewards  = results.std(axis=1)

    return timesteps, mean_rewards, std_rewards


def load_success_rate_curves(algo: str):
    """
    Load the success rate curve saved by EvalCallback.
    SB3 logs this when info dicts contain 'is_success'.
    Stored in evaluations.npz under key 'successes' if available.

    Returns
    -------
    timesteps     : np.ndarray  (n_evals,)
    success_rates : np.ndarray  (n_evals,)   values in [0, 1]
    """
    exp_name = EXP_NAMES[algo]
    npz_path = os.path.join("logs", exp_name, "evaluations.npz")

    if not os.path.exists(npz_path):
        return None, None

    data = np.load(npz_path)
    timesteps = data["timesteps"]

    # SB3 stores success info under 'successes' key when is_success is present
    if "successes" in data:
        successes     = data["successes"]            # shape (n_evals, n_eval_eps)
        success_rates = successes.mean(axis=1)
    else:
        # Fallback: infer success from reward (success reward = +100)
        results       = data["results"]
        success_rates = (results.mean(axis=1) > 50.0).astype(float)

    return timesteps, success_rates



# Graph 1: Training Reward vs Timesteps
def plot_training_reward(all_metrics: dict, algos: list):
    fig, ax = plt.subplots(figsize=(10, 5))

    any_plotted = False
    for algo in algos:
        ts, mean_r, std_r = load_training_curves(algo)
        if ts is None:
            continue

        color = ALGO_COLORS[algo]
        ax.plot(ts, mean_r, label=algo, color=color, linewidth=2)
        ax.fill_between(
            ts,
            mean_r - std_r,
            mean_r + std_r,
            alpha=0.15,
            color=color,
        )
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No evaluations.npz files found.\nRun training first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Graph 1 — Training Reward vs Timesteps", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "graph1_training_reward.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")



# Graph 2: Success Rate Bar Chart
def plot_success_rate(all_metrics: dict, algos: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: overall success rate ────────────────────────────────────────
    ax = axes[0]
    rates  = [all_metrics[a]["success_rate"] * 100 for a in algos if a in all_metrics]
    colors = [ALGO_COLORS[a] for a in algos if a in all_metrics]
    valid_algos = [a for a in algos if a in all_metrics]

    bars = ax.bar(valid_algos, rates, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Overall Success Rate", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Right: per-object success rate grouped bar ────────────────────────
    ax2    = axes[1]
    objs   = PandaPickEnv.OBJECT_NAMES
    n_objs = len(objs)
    x      = np.arange(n_objs)
    width  = 0.25

    for i, algo in enumerate(valid_algos):
        per_obj = all_metrics[algo]["success_per_object"]
        rates_obj = [per_obj.get(obj, 0.0) * 100 for obj in objs]
        offset = (i - len(valid_algos) / 2 + 0.5) * width
        ax2.bar(x + offset, rates_obj, width, label=algo,
                color=ALGO_COLORS[algo], edgecolor="white", linewidth=1.0)

    ax2.set_xticks(x)
    ax2.set_xticklabels([o.capitalize() for o in objs], fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_title("Success Rate per Object", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Graph 2 — Success Rate Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "graph2_success_rate.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")



# Graph 3: Episode Length Bar Chart
def plot_episode_length(all_metrics: dict, algos: list):
    fig, ax = plt.subplots(figsize=(8, 5))

    valid_algos = [a for a in algos if a in all_metrics]
    means  = [all_metrics[a]["mean_ep_length"] for a in valid_algos]
    stds   = [all_metrics[a]["std_ep_length"]  for a in valid_algos]
    colors = [ALGO_COLORS[a] for a in valid_algos]

    bars = ax.bar(
        valid_algos, means,
        yerr=stds,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
        capsize=6,
        error_kw={"elinewidth": 2, "ecolor": "black", "alpha": 0.6},
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{mean:.0f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_ylabel("Average Episode Length (steps)", fontsize=12)
    ax.set_title("Graph 3 — Average Episode Length\n(shorter = faster task completion)", 
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 220)
    ax.axhline(200, color="gray", linestyle="--", alpha=0.5, label="Max steps (200)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "graph3_episode_length.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")



# Graph 4: Learning Curve (Success Rate vs Timesteps)
def plot_learning_curve(all_metrics: dict, algos: list):
    fig, ax = plt.subplots(figsize=(10, 5))

    any_plotted = False
    for algo in algos:
        ts, sr = load_success_rate_curves(algo)
        if ts is None:
            continue

        color = ALGO_COLORS[algo]
        ax.plot(ts, sr * 100, label=algo, color=color, linewidth=2)

        # Mark the learning speed point
        ls_ts = all_metrics.get(algo, {}).get("learning_speed_ts", -1)
        if ls_ts > 0:
            # Find the y value at that timestep
            idx = np.argmin(np.abs(ts - ls_ts))
            ax.axvline(ls_ts, color=color, linestyle=":", alpha=0.6, linewidth=1.5)
            ax.scatter([ls_ts], [sr[idx] * 100], color=color, s=80, zorder=5)

        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No evaluations.npz files found.\nRun training first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.axhline(50, color="gray", linestyle="--", alpha=0.4, label="50% success threshold")
    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Graph 4 — Learning Curve (Success Rate vs Timesteps)\n"
                 "Dotted vertical lines = timestep of first consistent success",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "graph4_learning_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")



# CSV export
def save_csv(all_metrics: dict, algos: list):
    path = os.path.join(RESULTS_DIR, "metrics_summary.csv")

    rows = []
    for algo in algos:
        if algo not in all_metrics:
            continue
        m = all_metrics[algo]

        row = {
            "Algorithm":              algo,
            "Mean Episode Reward":    f"{m['mean_reward']:.2f}",
            "Std Episode Reward":     f"{m['std_reward']:.2f}",
            "Success Rate (%)":       f"{m['success_rate']*100:.1f}",
            "Mean Episode Length":    f"{m['mean_ep_length']:.1f}",
            "Std Episode Length":     f"{m['std_ep_length']:.1f}",
            "Learning Speed (steps)": m['learning_speed_ts'] if m['learning_speed_ts'] > 0 else "N/A",
            "Stability (Reward Std)": f"{m['stability_std']:.2f}",
        }

        # Add per-object success rates
        for obj in PandaPickEnv.OBJECT_NAMES:
            rate = m["success_per_object"].get(obj, float("nan"))
            row[f"Success {obj.capitalize()} (%)"] = (
                f"{rate*100:.1f}" if not np.isnan(rate) else "N/A"
            )

        rows.append(row)

    if not rows:
        print("  No metrics to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved → {path}")


# Console summary table
def print_summary(all_metrics: dict, algos: list):
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    header = f"{'Metric':<35} " + "  ".join(f"{a:>10}" for a in algos if a in all_metrics)
    print(header)
    print("-" * 70)

    valid = [a for a in algos if a in all_metrics]

    def row(label, key, fmt=".2f", scale=1.0):
        vals = []
        for a in valid:
            v = all_metrics[a].get(key, float("nan"))
            if isinstance(v, float) and not np.isnan(v):
                vals.append(f"{v*scale:{fmt}}")
            else:
                vals.append("  N/A")
        print(f"  {label:<33} " + "  ".join(f"{v:>10}" for v in vals))

    row("Metric 1 — Mean Reward",       "mean_reward",       fmt="+.2f")
    row("         ± Std Dev",           "std_reward",         fmt=".2f")
    row("Metric 2 — Success Rate (%)",  "success_rate",       fmt=".1f", scale=100)
    row("Metric 3 — Mean Ep Length",    "mean_ep_length",     fmt=".1f")
    row("         ± Std Dev",           "std_ep_length",      fmt=".1f")
    row("Metric 5 — Stability Std",     "stability_std",      fmt=".2f")

    # Metric 4 separately (int or N/A)
    vals = []
    for a in valid:
        ls = all_metrics[a].get("learning_speed_ts", -1)
        vals.append(f"{ls:>10,}" if ls > 0 else "       N/A")
    print(f"  {'Metric 4 — Learning Speed (ts)':<33} " + "  ".join(vals))

    print("=" * 70)

    # Per-object breakdown
    print("\n  PER-OBJECT SUCCESS RATES (%)")
    print("-" * 70)
    for obj in PandaPickEnv.OBJECT_NAMES:
        vals = []
        for a in valid:
            r = all_metrics[a]["success_per_object"].get(obj, float("nan"))
            vals.append(f"{r*100:>10.1f}" if not np.isnan(r) else "       N/A")
        print(f"  {obj.capitalize():<33} " + "  ".join(vals))
    print("=" * 70)



# Main
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO / SAC / TD3 on PandaPickEnv")
    parser.add_argument("--algos",      nargs="+", default=["PPO", "SAC", "TD3"],
                        choices=["PPO", "SAC", "TD3"],
                        help="Algorithms to evaluate (default: all three)")
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="Evaluation episodes per algorithm (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  PANDA PICK-AND-LIFT — ALGORITHM COMPARISON")
    print(f"  Algorithms   : {', '.join(args.algos)}")
    print(f"  Eval episodes: {args.n_episodes}")
    print(f"  Results dir  : {RESULTS_DIR}/")
    print("=" * 60)

    all_metrics = {}

    # ── Evaluate each algorithm ───────────────────────────────────────────
    for algo in args.algos:
        print(f"\n{'─'*60}")
        print(f"  Loading {algo}...")
        try:
            model, vec_env, normalised = load_model_and_env(algo)
            print(f"  Model loaded. VecNormalize: {'yes' if normalised else 'no'}")
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {algo}: {e}")
            continue

        eval_data = evaluate_model(model, vec_env, n_episodes=args.n_episodes, algo=algo)
        metrics   = compute_metrics(eval_data, algo)
        all_metrics[algo] = metrics

        vec_env.close()

    if not all_metrics:
        print("\n  No models were successfully loaded. Exiting.")
        return

    # ── Console summary ───────────────────────────────────────────────────
    print_summary(all_metrics, args.algos)

    # ── Graphs ────────────────────────────────────────────────────────────
    print(f"\n  Generating graphs → {RESULTS_DIR}/")
    plot_training_reward(all_metrics, args.algos)
    plot_success_rate(all_metrics, args.algos)
    plot_episode_length(all_metrics, args.algos)
    plot_learning_curve(all_metrics, args.algos)

    # ── CSV ───────────────────────────────────────────────────────────────
    print(f"\n  Saving CSV → {RESULTS_DIR}/metrics_summary.csv")
    save_csv(all_metrics, args.algos)

    print(f"\n  All done! Check the {RESULTS_DIR}/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()