"""
envs/reward_utils.py — Modular, shaped reward components for pick-and-place.

Reward shaping is critical for sparse-reward robotics tasks. We decompose
the task into a curriculum of sub-rewards so the agent gets signal even
before completing the full task.

Shaping philosophy:
  R_total = reach → grasp → lift → place → success
Each stage only activates once the previous is achieved, preventing
"reward hacking" (e.g., getting lift reward without actually grasping).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from first.config import REWARD_CONFIG as RC


@dataclass
class RobotState:
    """Snapshot of the environment state passed to reward functions."""
    ee_pos: np.ndarray          # End-effector XYZ (3,)
    ee_vel: np.ndarray          # End-effector velocity (3,)
    obj_pos: np.ndarray         # Current target object XYZ (3,)
    obj_vel: np.ndarray         # Object velocity (3,)
    goal_pos: np.ndarray        # Goal placement XYZ (3,)
    gripper_width: float        # Current gripper opening (0=closed, 1=open)
    prev_ee_pos: np.ndarray     # EE position at previous step (for delta)
    is_grasped: bool            # True if object is being held
    object_height: float        # Current Z height of object
    table_height: float         # Z height of table surface
    in_collision: bool          # True if self/table collision detected


def compute_reach_reward(state: RobotState) -> Tuple[float, Dict]:
    """
    Reward proportional to improvement in EE-to-object distance.
    Uses delta reward (improvement) rather than raw distance to avoid
    the agent getting stuck at a fixed distance from the object.
    """
    dist_now = np.linalg.norm(state.ee_pos - state.obj_pos)
    dist_prev = np.linalg.norm(state.prev_ee_pos - state.obj_pos)
    
    # Improvement reward: positive if getting closer, negative if moving away
    delta = dist_prev - dist_now
    
    # Add small bonus for being very close (within grasp range)
    proximity_bonus = max(0.0, 0.05 - dist_now) * 10.0
    
    reward = RC["reach_coeff"] * delta + proximity_bonus
    info = {"reach_dist": dist_now, "reach_delta": delta}
    return reward, info


def compute_grasp_reward(state: RobotState, just_grasped: bool) -> Tuple[float, Dict]:
    """
    One-time reward for successfully grasping the object.
    'just_grasped' is True only on the step the grasp is detected
    to avoid giving continuous reward for holding.
    """
    reward = RC["grasp_coeff"] if just_grasped else 0.0
    info = {"grasped": state.is_grasped, "just_grasped": just_grasped}
    return reward, info


def compute_lift_reward(state: RobotState, lift_threshold: float) -> Tuple[float, Dict]:
    """
    Shaped reward for lifting the object off the table.
    Only active while grasping. Scales with height above table.
    """
    if not state.is_grasped:
        return 0.0, {"lift_height": 0.0}
    
    height_above_table = state.object_height - state.table_height
    # Clip to [0, lift_threshold] and normalize to [0, 1]
    norm_height = np.clip(height_above_table / lift_threshold, 0.0, 1.0)
    reward = RC["lift_coeff"] * norm_height
    info = {"lift_height": height_above_table, "lift_progress": norm_height}
    return reward, info


def compute_place_reward(state: RobotState, goal_threshold: float) -> Tuple[float, Dict]:
    """
    Reward for moving a held object toward the goal position.
    Uses the same delta approach as reach reward.
    Only active while grasping.
    """
    if not state.is_grasped:
        return 0.0, {"place_dist": float("inf")}
    
    dist_to_goal = np.linalg.norm(state.obj_pos - state.goal_pos)
    reward = RC["place_coeff"] * max(0.0, goal_threshold - dist_to_goal) / goal_threshold
    info = {"place_dist": dist_to_goal}
    return reward, info


def compute_penalties(
    state: RobotState,
    dropped: bool,
    action: np.ndarray
) -> Tuple[float, Dict]:
    """
    Penalty terms to discourage undesirable behaviors:
    - Time penalty: nudges agent to be efficient
    - Drop penalty: discourages letting go of object mid-air
    - Collision penalty: discourages hitting the table or self
    - Action smoothness: optional regularization on large actions
    """
    penalty = RC["time_penalty"]  # Always applied (per step)
    
    if dropped:
        penalty += RC["drop_penalty"]
    
    if state.in_collision:
        penalty += RC["collision_penalty"]
    
    # Optional: penalize large, jerky actions (helps with sim stability)
    action_reg = -0.001 * np.sum(np.square(action))
    penalty += action_reg
    
    info = {
        "dropped": dropped,
        "in_collision": state.in_collision,
        "action_reg": action_reg,
    }
    return penalty, info


def compute_total_reward(
    state: RobotState,
    action: np.ndarray,
    just_grasped: bool,
    dropped: bool,
    object_placed: bool,
    all_done: bool,
    lift_threshold: float,
    goal_threshold: float,
) -> Tuple[float, Dict]:
    """
    Aggregate all reward components.
    Returns total reward and a breakdown dict for logging/debugging.
    """
    info = {}
    total = 0.0

    r_reach, i_reach = compute_reach_reward(state)
    r_grasp, i_grasp = compute_grasp_reward(state, just_grasped)
    r_lift, i_lift = compute_lift_reward(state, lift_threshold)
    r_place, i_place = compute_place_reward(state, goal_threshold)
    r_pen, i_pen = compute_penalties(state, dropped, action)

    total = r_reach + r_grasp + r_lift + r_place + r_pen

    if object_placed:
        total += RC["success_bonus"]
    if all_done:
        total += RC["all_done_bonus"]

    info.update(i_reach)
    info.update(i_grasp)
    info.update(i_lift)
    info.update(i_place)
    info.update(i_pen)
    info["r_reach"] = r_reach
    info["r_grasp"] = r_grasp
    info["r_lift"] = r_lift
    info["r_place"] = r_place
    info["r_penalties"] = r_pen
    info["r_total"] = total

    return total, info
