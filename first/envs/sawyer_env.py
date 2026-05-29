"""
envs/sawyer_env.py — Gymnasium environment for Sawyer pick-and-place in MuJoCo.

Design decisions:
  - Uses MuJoCo's native gymnasium wrapper for rendering support
  - Observation is normalized to [-1, 1] for neural network stability
  - Objects are randomized each episode to force generalization
  - Grasp detection uses a contact-force heuristic (robust, no wrist sensor needed)
  - Action space is joint velocity control (smoother than position control for PPO)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import xml.etree.ElementTree as ET

from first.envs.reward_utils import RobotState, compute_total_reward
from first.config import ENV_CONFIG, REWARD_CONFIG


# ─── Object definitions ──────────────────────────────────────────────────────
# Each object has a MuJoCo geom type, size params, and mass.
OBJECT_SPECS = {
    "cube":     {"type": "box",      "size": [0.03, 0.03, 0.03], "mass": 0.1,  "rgba": "0.8 0.2 0.2 1"},
    "sphere":   {"type": "sphere",   "size": [0.035],             "mass": 0.08, "rgba": "0.2 0.7 0.2 1"},
    "cylinder": {"type": "cylinder", "size": [0.025, 0.04],       "mass": 0.12, "rgba": "0.2 0.3 0.9 1"},
    "box":      {"type": "box",      "size": [0.04, 0.025, 0.02], "mass": 0.15, "rgba": "0.9 0.6 0.1 1"},
    "cone":     {"type": "cylinder", "size": [0.02, 0.05],        "mass": 0.09, "rgba": "0.7 0.1 0.7 1"},
}


class SawyerPickPlaceEnv(gym.Env):
    """
    Sawyer 7-DOF arm: pick and place 5 objects in MuJoCo.

    Episode flow:
      1. Reset: spawn 5 objects randomly on table, sample goal positions
      2. Agent must pick each object (one at a time) and place at its goal
      3. Episode ends when all 5 are placed OR max_episode_steps reached
      4. Success = all 5 objects within goal_threshold of their goals
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Sawyer joint limits [lower, upper] in radians
    JOINT_LIMITS = np.array([
        [-3.0503, 3.0503],   # j0
        [-3.8095, 2.2736],   # j1
        [-3.0426, 3.0426],   # j2
        [-3.0439, 3.0439],   # j3
        [-2.9761, 2.9761],   # j4
        [-2.9761, 2.9761],   # j5
        [-4.7124, 4.7124],   # j6
    ])

    def __init__(self, render_mode: Optional[str] = None, n_objects = 5, max_episode_steps = 200):
        super().__init__()
        self.render_mode = render_mode
        self.n_objects = n_objects
        self.max_episode_steps = max_episode_steps
        self.cfg = ENV_CONFIG

        # Build the MuJoCo XML model dynamically
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/rethink_robotics_sawyer/sawyer.xml")
        self.data = mujoco.MjData(self.model)

        # Cache body/joint/sensor IDs for fast lookup
        self._cache_mujoco_ids()

        # ── Observation space ────────────────────────────────────────────────
        # [joint_pos(7), joint_vel(7), ee_pos(3), ee_quat(4), gripper(2),
        #  target_obj_pos(3), target_obj_quat(4), goal_pos(3)] = 33 dims
        obs_dim = 7 + 7 + 3 + 4 + 2 + 3 + 4 + 3  # = 33
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space ─────────────────────────────────────────────────────
        # 7 joint velocity deltas + 1 gripper command, all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Episode state
        self._step_count = 0
        self._current_obj_idx = 0          # Which object we're targeting
        self._objects_placed = [False] * n_objects
        self._is_grasped = False
        self._prev_ee_pos = np.zeros(3)
        self._prev_was_grasped = False
        self._goal_positions = []
        self._object_init_positions = []

        # Viewer (for human render mode)
        self._viewer = None

    # ────────────────────────────────────────────────────────────────────────
    # Core Gym API
    # ────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset episode tracking
        self._step_count = 0
        self._current_obj_idx = 0
        self._objects_placed = [False] * self.n_objects
        self._is_grasped = False
        self._prev_was_grasped = False

        # Randomize object positions on the table (within a safe region)
        self._spawn_objects()

        # Sample distinct goal positions (where to place each object)
        self._goal_positions = self._sample_goal_positions()

        # Set arm to a neutral starting pose
        self._set_arm_neutral()

        # Forward simulation to settle physics
        mujoco.mj_forward(self.model, self.data)
        self._prev_ee_pos = self._get_ee_pos().copy()

        obs = self._get_obs()
        info = {"objects_placed": 0, "current_target": 0}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Apply joint velocity commands
        self._apply_action(action)

        # Step physics (multiple substeps for stability)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # Detect grasp state change
        just_grasped = False
        just_dropped = False
        self._is_grasped = self._detect_grasp()
        if self._is_grasped and not self._prev_was_grasped:
            just_grasped = True
        if not self._is_grasped and self._prev_was_grasped:
            just_dropped = True
        self._prev_was_grasped = self._is_grasped

        # Check if current object is placed at goal
        obj_placed = False
        if self._is_grasped or just_dropped:
            obj_pos = self._get_object_pos(self._current_obj_idx)
            goal_pos = self._goal_positions[self._current_obj_idx]
            if np.linalg.norm(obj_pos - goal_pos) < self.cfg["goal_threshold"]:
                self._objects_placed[self._current_obj_idx] = True
                obj_placed = True
                # Advance to next unplaced object
                self._advance_target()

        all_done = all(self._objects_placed)

        # Build reward state struct
        state = RobotState(
            ee_pos=self._get_ee_pos(),
            ee_vel=self._get_ee_vel(),
            obj_pos=self._get_object_pos(self._current_obj_idx),
            obj_vel=self._get_object_vel(self._current_obj_idx),
            goal_pos=self._goal_positions[self._current_obj_idx],
            gripper_width=self._get_gripper_width(),
            prev_ee_pos=self._prev_ee_pos,
            is_grasped=self._is_grasped,
            object_height=self._get_object_pos(self._current_obj_idx)[2],
            table_height=0.0,    # Table surface at Z=0 in our XML
            in_collision=self._detect_collision(),
        )

        reward, reward_info = compute_total_reward(
            state=state,
            action=action,
            just_grasped=just_grasped,
            dropped=just_dropped,
            object_placed=obj_placed,
            all_done=all_done,
            lift_threshold=self.cfg["lift_height"],
            goal_threshold=self.cfg["goal_threshold"],
        )

        self._prev_ee_pos = state.ee_pos.copy()

        obs = self._get_obs()
        terminated = all_done
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "objects_placed": sum(self._objects_placed),
            "current_target": self._current_obj_idx,
            "is_grasped": self._is_grasped,
            "is_success": all_done,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers — observations
    # ────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Assemble and normalize the 33-dim observation vector."""
        joint_pos = self.data.qpos[self._arm_joint_ids].copy()
        joint_vel = self.data.qvel[self._arm_joint_ids].copy()

        # Normalize joint positions to [-1, 1] using joint limits
        joint_pos_norm = (
            2.0 * (joint_pos - self.JOINT_LIMITS[:, 0])
            / (self.JOINT_LIMITS[:, 1] - self.JOINT_LIMITS[:, 0])
            - 1.0
        )
        joint_vel_norm = np.clip(joint_vel / np.pi, -1.0, 1.0)

        ee_pos = self._get_ee_pos()
        ee_quat = self.data.xquat[self._ee_body_id].copy()
        ee_pos_norm = np.clip(ee_pos / 1.0, -1.0, 1.0)  # Assume workspace ≤ 1m

        gripper_width = np.array([self._get_gripper_width(), self._get_gripper_width()])

        obj_pos = self._get_object_pos(self._current_obj_idx)
        obj_quat = self._get_object_quat(self._current_obj_idx)
        obj_pos_norm = np.clip(obj_pos / 1.0, -1.0, 1.0)

        goal_pos = self._goal_positions[self._current_obj_idx]
        goal_pos_norm = np.clip(goal_pos / 1.0, -1.0, 1.0)

        obs = np.concatenate([
            joint_pos_norm,   # 7
            joint_vel_norm,   # 7
            ee_pos_norm,      # 3
            ee_quat,          # 4
            gripper_width,    # 2
            obj_pos_norm,     # 3
            obj_quat,         # 4
            goal_pos_norm,    # 3
        ]).astype(np.float32)

        return obs

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers — actions
    # ────────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray):
        """Map [-1,1] action to joint velocity commands + gripper."""
        MAX_VEL = 0.5  # rad/s max joint speed

        # Joint velocity commands (first 7 dims)
        joint_vel_cmds = action[:7] * MAX_VEL
        self.data.ctrl[self._arm_ctrl_ids] = joint_vel_cmds

        # Gripper (last dim): +1=open, -1=close
        gripper_cmd = action[7]
        MAX_GRIPPER = 0.04  # Max finger separation in meters
        self.data.ctrl[self._gripper_ctrl_ids] = gripper_cmd * MAX_GRIPPER

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers — grasp & collision detection
    # ────────────────────────────────────────────────────────────────────────

    def _detect_grasp(self) -> bool:
        """
        Heuristic grasp detection: object must be
          1. Close to the end-effector
          2. Above the table (lifted)
          3. Gripper is partially closed
        """
        obj_pos = self._get_object_pos(self._current_obj_idx)
        ee_pos = self._get_ee_pos()
        dist = np.linalg.norm(ee_pos - obj_pos)
        gripper_closed = self._get_gripper_width() < 0.06
        above_table = obj_pos[2] > 0.02
        return dist < self.cfg["grasp_threshold"] and gripper_closed and above_table

    def _detect_collision(self) -> bool:
        """Check for contacts that indicate a collision with the table or self."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if geom1 and geom2:
                if "table" in geom1 or "table" in geom2:
                    if "arm" in geom1 or "arm" in geom2:
                        return True
        return False

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers — state queries
    # ────────────────────────────────────────────────────────────────────────

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ee_body_id].copy()

    def _get_ee_vel(self) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                  self._ee_body_id, vel, 0)
        return vel[3:].copy()  # Linear velocity (last 3)

    def _get_object_pos(self, idx: int) -> np.ndarray:
        return self.data.xpos[self._obj_body_ids[idx]].copy()

    def _get_object_vel(self, idx: int) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                  self._obj_body_ids[idx], vel, 0)
        return vel[3:].copy()

    def _get_object_quat(self, idx: int) -> np.ndarray:
        return self.data.xquat[self._obj_body_ids[idx]].copy()

    def _get_gripper_width(self) -> float:
        return float(self.data.qpos[self._gripper_joint_ids[0]] +
                     self.data.qpos[self._gripper_joint_ids[1]])

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers — episode management
    # ────────────────────────────────────────────────────────────────────────

    def _spawn_objects(self):
        """Randomly place objects on the table surface."""
        table_x = [-0.2, 0.2]
        table_y = [0.3, 0.7]
        positions = []
        for i in range(self.n_objects):
            for _ in range(100):  # Rejection sampling to avoid overlap
                x = self.np_random.uniform(*table_x)
                y = self.np_random.uniform(*table_y)
                pos = np.array([x, y, 0.05])
                if all(np.linalg.norm(pos[:2] - p[:2]) > 0.08 for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array([
                    self.np_random.uniform(*table_x),
                    self.np_random.uniform(*table_y),
                    0.05
                ]))
            # Set object freejoint position
            start = self.model.jnt_qposadr[self._obj_free_joint_ids[i]]
            self.data.qpos[start:start+3] = positions[-1]
            self.data.qpos[start+3:start+7] = [1, 0, 0, 0]  # Identity quaternion

        self._object_init_positions = positions

    def _sample_goal_positions(self):
        """Sample placement goal positions (right side of table)."""
        goals = []
        goal_x = [0.25, 0.45]
        goal_y = [0.3, 0.7]
        for _ in range(self.n_objects):
            for _ in range(100):
                x = self.np_random.uniform(*goal_x)
                y = self.np_random.uniform(*goal_y)
                pos = np.array([x, y, 0.01])
                if all(np.linalg.norm(pos[:2] - g[:2]) > 0.07 for g in goals):
                    goals.append(pos)
                    break
            else:
                goals.append(np.array([
                    self.np_random.uniform(*goal_x),
                    self.np_random.uniform(*goal_y),
                    0.01
                ]))
        return goals

    def _set_arm_neutral(self):
        """Move arm to a safe home position."""
        home = np.array([0.0, -0.5, 0.0, 1.0, 0.0, 1.2, 0.0])
        for i, jid in enumerate(self._arm_joint_ids):
            start = self.model.jnt_qposadr[jid]
            self.data.qpos[start] = home[i]

    def _advance_target(self):
        """Move to the next unplaced object."""
        for i in range(self.n_objects):
            if not self._objects_placed[i]:
                self._current_obj_idx = i
                return
        self._current_obj_idx = self.n_objects - 1  # All done

    # ────────────────────────────────────────────────────────────────────────
    # MuJoCo ID caching
    # ────────────────────────────────────────────────────────────────────────

    def _cache_mujoco_ids(self):
        """Cache frequently-accessed MuJoCo IDs to avoid repeated lookups."""
        # Arm joints (j0..j6)
        self._arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_j{i}")
            for i in range(7)
        ]
        # Arm actuators
        self._arm_ctrl_ids = list(range(7))
        # Gripper
        self._gripper_ctrl_ids = [7, 8]
        self._gripper_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_l_finger"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_r_finger"),
        ]
        # End-effector body
        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        # Object bodies and free joints
        self._obj_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{i}")
            for i in range(self.n_objects)
        ]
        self._obj_free_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{i}_free")
            for i in range(self.n_objects)
        ]

    # ────────────────────────────────────────────────────────────────────────
    # XML model builder
    # ────────────────────────────────────────────────────────────────────────

    def _build_xml(self) -> str:
        """
        Dynamically generate MuJoCo XML for Sawyer + table + 5 objects.
        In production, replace the arm kinematics with the official Sawyer MJCF
        from Rethink Robotics or the robosuite library.
        """
        shapes = list(OBJECT_SPECS.keys())
        objects_xml = ""
        for i in range(self.n_objects):
            spec = OBJECT_SPECS[shapes[i % len(shapes)]]
            size_str = " ".join(str(s) for s in spec["size"])
            objects_xml += f"""
        <body name="object_{i}" pos="0 0.5 0.05">
            <freejoint name="object_{i}_free"/>
            <geom type="{spec['type']}" size="{size_str}" mass="{spec['mass']}"
                  rgba="{spec['rgba']}" condim="4" friction="1 0.005 0.0001"/>
            <site name="object_{i}_site" size="0.005"/>
        </body>"""

        xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="sawyer_pick_place">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="Newton"/>
    
    <default>
        <joint limited="true" damping="0.1" armature="0.01"/>
        <geom contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
    </default>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.4 0.4 0.4"
                 rgb2="0.6 0.6 0.6" width="300" height="300"/>
        <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <worldbody>
        <!-- Ground plane -->
        <geom name="floor" type="plane" size="5 5 0.1" material="grid_mat"/>
        
        <!-- Lighting -->
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
               pos="0 0 3" dir="0 0 -1"/>

        <!-- Table -->
        <body name="table" pos="0 0.5 0">
            <geom name="table_top" type="box" size="0.4 0.3 0.01" pos="0 0 0"
                  rgba="0.8 0.7 0.5 1" mass="10"/>
            <geom name="table_leg1" type="cylinder" size="0.02 0.35" pos="-0.35 -0.25 -0.35"
                  rgba="0.5 0.4 0.3 1" mass="1"/>
            <geom name="table_leg2" type="cylinder" size="0.02 0.35" pos="0.35 -0.25 -0.35"
                  rgba="0.5 0.4 0.3 1" mass="1"/>
            <geom name="table_leg3" type="cylinder" size="0.02 0.35" pos="-0.35 0.25 -0.35"
                  rgba="0.5 0.4 0.3 1" mass="1"/>
            <geom name="table_leg4" type="cylinder" size="0.02 0.35" pos="0.35 0.25 -0.35"
                  rgba="0.5 0.4 0.3 1" mass="1"/>
        </body>

        <!-- Sawyer Robot Arm (simplified kinematic chain) -->
        <!-- NOTE: Replace with official Sawyer MJCF for production use -->
        <!-- Get it from: github.com/vikashplus/robohive or robosuite -->
        <body name="base" pos="0 0 0.01">
            <geom name="base_geom" type="cylinder" size="0.1 0.01" rgba="0.3 0.3 0.3 1"/>
            
            <body name="arm_link0" pos="0 0 0.08">
                <joint name="arm_j0" type="hinge" axis="0 0 1"
                       range="{self.JOINT_LIMITS[0,0]} {self.JOINT_LIMITS[0,1]}"/>
                <geom name="arm_link0_geom" type="capsule" size="0.04 0.07"
                      rgba="0.9 0.9 0.9 1" mass="2"/>

                <body name="arm_link1" pos="0 0 0.14">
                    <joint name="arm_j1" type="hinge" axis="0 1 0"
                           range="{self.JOINT_LIMITS[1,0]} {self.JOINT_LIMITS[1,1]}"/>
                    <geom name="arm_link1_geom" type="capsule" size="0.035 0.15"
                          rgba="0.85 0.85 0.85 1" mass="1.8"/>

                    <body name="arm_link2" pos="0 0 0.3">
                        <joint name="arm_j2" type="hinge" axis="0 0 1"
                               range="{self.JOINT_LIMITS[2,0]} {self.JOINT_LIMITS[2,1]}"/>
                        <geom name="arm_link2_geom" type="capsule" size="0.03 0.13"
                              rgba="0.9 0.9 0.9 1" mass="1.5"/>

                        <body name="arm_link3" pos="0 0 0.26">
                            <joint name="arm_j3" type="hinge" axis="0 1 0"
                                   range="{self.JOINT_LIMITS[3,0]} {self.JOINT_LIMITS[3,1]}"/>
                            <geom name="arm_link3_geom" type="capsule" size="0.028 0.12"
                                  rgba="0.85 0.85 0.85 1" mass="1.2"/>

                            <body name="arm_link4" pos="0 0 0.24">
                                <joint name="arm_j4" type="hinge" axis="0 0 1"
                                       range="{self.JOINT_LIMITS[4,0]} {self.JOINT_LIMITS[4,1]}"/>
                                <geom name="arm_link4_geom" type="capsule" size="0.025 0.1"
                                      rgba="0.9 0.9 0.9 1" mass="0.9"/>

                                <body name="arm_link5" pos="0 0 0.2">
                                    <joint name="arm_j5" type="hinge" axis="0 1 0"
                                           range="{self.JOINT_LIMITS[5,0]} {self.JOINT_LIMITS[5,1]}"/>
                                    <geom name="arm_link5_geom" type="capsule" size="0.023 0.08"
                                          rgba="0.85 0.85 0.85 1" mass="0.6"/>

                                    <body name="arm_link6" pos="0 0 0.16">
                                        <joint name="arm_j6" type="hinge" axis="0 0 1"
                                               range="{self.JOINT_LIMITS[6,0]} {self.JOINT_LIMITS[6,1]}"/>
                                        <geom name="arm_link6_geom" type="capsule" size="0.02 0.06"
                                              rgba="0.9 0.9 0.9 1" mass="0.4"/>

                                        <!-- Wrist / End-effector -->
                                        <body name="end_effector" pos="0 0 0.12">
                                            <geom name="wrist_geom" type="box"
                                                  size="0.04 0.02 0.02" rgba="0.3 0.3 0.3 1" mass="0.2"/>
                                            
                                            <!-- Gripper fingers -->
                                            <body name="gripper_l" pos="-0.02 0 0.03">
                                                <joint name="gripper_l_finger" type="slide"
                                                       axis="-1 0 0" range="0 0.04"/>
                                                <geom name="gripper_l_geom" type="box"
                                                      size="0.008 0.012 0.025" rgba="0.2 0.2 0.2 1" mass="0.05"/>
                                            </body>
                                            <body name="gripper_r" pos="0.02 0 0.03">
                                                <joint name="gripper_r_finger" type="slide"
                                                       axis="1 0 0" range="0 0.04"/>
                                                <geom name="gripper_r_geom" type="box"
                                                      size="0.008 0.012 0.025" rgba="0.2 0.2 0.2 1" mass="0.05"/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <!-- Objects to pick and place -->
        {objects_xml}

    </worldbody>

    <!-- Actuators: velocity-controlled joints -->
    <actuator>
        <velocity name="act_j0" joint="arm_j0" kv="10"/>
        <velocity name="act_j1" joint="arm_j1" kv="10"/>
        <velocity name="act_j2" joint="arm_j2" kv="10"/>
        <velocity name="act_j3" joint="arm_j3" kv="10"/>
        <velocity name="act_j4" joint="arm_j4" kv="5"/>
        <velocity name="act_j5" joint="arm_j5" kv="5"/>
        <velocity name="act_j6" joint="arm_j6" kv="5"/>
        <position name="act_gripper_l" joint="gripper_l_finger" kp="200"/>
        <position name="act_gripper_r" joint="gripper_r_finger" kp="200"/>
    </actuator>

</mujoco>"""
        return xml

    def _render_rgb(self) -> np.ndarray:
        """Render an RGB frame for video logging."""
        width, height = 640, 480
        renderer = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data, camera="track_ee" if "track_ee" in
                              [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                               for i in range(self.model.ncam)] else -1)
        return renderer.render()
