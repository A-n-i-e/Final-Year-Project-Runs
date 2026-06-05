"""
panda_pick_env.py  —  Franka Panda Pick-and-Lift
=================================================

Observation (42-dim, PRIVILEGED)
---------------------------------
  0: 7   Arm joint positions (rad)
  7:14   Arm joint velocities (rad/s)
 14:15   Gripper jaw separation  — from qpos  (0=closed, 1=open)
 15:16   Gripper openness        — from qpos  (0=closed, 1=open)
 16:19   End-effector position   (x, y, z)
 19:22   Object position         (x, y, z)
 22:25   Relative vector         obj_pos − ee_pos
 25:29   Object quaternion       (w, qx, qy, qz)
 29:32   Target position         (x, y, z)
 32:37   Object one-hot          (5 object types)
 37:38   is_grasped flag         (0 or 1)
 38:39   Curriculum stage        (0–4)
 39:41   Finger contact flags    (left, right)
 41:42   Height progress         [0, 1]
          ──────
          42 total

Action (4-dim)
--------------
  action[0:3]  (dx, dy, dz) Cartesian EE displacement — ±5 cm/step
  action[3]    gripper: +1 = open, −1 = close
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import inverse_kinematics as ik



# XML snippets
OBJECT_XMLS = {
    "cube": """
        <body name="cube" pos="0.45 0 0.25">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
                rgba="0.8 0.2 0.2 1" mass="0.1" condim="4"
                friction="1.5 0.05 0.001"/>
          <site name="cube_site" pos="0 0 0" size="0.005"/>
        </body>""",

    "sphere": """
        <body name="sphere" pos="0.45 0 0.255">
          <freejoint name="sphere_joint"/>
          <geom name="sphere_geom" type="sphere" size="0.025"
                rgba="0.2 0.6 0.2 1" mass="0.1" condim="4"
                friction="1.5 0.05 0.001"/>
          <site name="sphere_site" pos="0 0 0" size="0.005"/>
        </body>""",

    "cylinder": """
        <body name="cylinder" pos="0.45 0 0.26">
          <freejoint name="cylinder_joint"/>
          <geom name="cylinder_geom" type="cylinder" size="0.02 0.03"
                rgba="0.2 0.2 0.8 1" mass="0.1" condim="4"
                friction="1.5 0.05 0.001"/>
          <site name="cylinder_site" pos="0 0 0" size="0.005"/>
        </body>""",

    "box": """
        <body name="box" pos="0.45 0 0.25">
          <freejoint name="box_joint"/>
          <geom name="box_geom" type="box" size="0.04 0.025 0.02"
                rgba="0.8 0.6 0.1 1" mass="0.15" condim="4"
                friction="1.5 0.05 0.001"/>
          <site name="box_site" pos="0 0 0" size="0.005"/>
        </body>""",

    "cone": """
        <body name="cone" pos="0.45 0 0.26">
          <freejoint name="cone_joint"/>
          <geom name="cone_geom" type="cylinder" size="0.02 0.03"
                rgba="0.7 0.1 0.7 1" mass="0.08" condim="4"
                friction="1.5 0.05 0.001"/>
          <site name="cone_site" pos="0 0 0" size="0.005"/>
        </body>""",
}

# Table: body centre z=0.1, geom half-height=0.1 → top surface at z=0.20
TABLE_XML = """
    <body name="table" pos="0.45 0 0.1">
      <geom name="table_geom" type="box" size="0.35 0.35 0.1"
            rgba="0.9 0.8 0.6 1" contype="1" conaffinity="1"
            friction="1 0.005 0.0001"/>
    </body>"""

TABLE_HEIGHT          = 0.20   # top surface of table (m)
LIFT_HEIGHT_THRESHOLD = 0.20   # object must rise this far above table top

# Curriculum
CURRICULUM_WINDOW    = 20
CURRICULUM_THRESHOLD = 0.70


def _build_xml(base_xml_path: str) -> str:
    """
    Inject table and all 5 objects into scene.xml.
    Finger pad geoms come from the updated panda.xml (already hardcoded).
    Written next to scene.xml so <include> and asset paths resolve.
    """
    with open(base_xml_path, "r") as f:
        xml_text = f.read()
    if "</worldbody>" not in xml_text:
        raise ValueError("Could not find </worldbody> in scene XML.")

    all_objects_xml = "\n".join(OBJECT_XMLS.values())
    xml_text = xml_text.replace(
        "</worldbody>",
        f"{TABLE_XML}\n{all_objects_xml}\n  </worldbody>"
    )
    return xml_text



# Environment
class PandaPickEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    OBJECT_NAMES       = ["cube", "sphere", "cylinder", "box", "cone"]
    JOINT_NAMES        = ["joint1","joint2","joint3",
                          "joint4","joint5","joint6","joint7"]
    FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]
    EE_SITE_NAME       = "pinch"
    # FIX-1: corrected path
    BASE_XML_PATH      = "franka_emika_panda/scene.xml"
    # Gripper pointing straight down — consistent grasp orientation
    EE_TARGET_QUAT     = np.array([0.0, 0.7071068, -0.7071068, 0.0])
    OBS_DIM            = 42

    def __init__(
        self,
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        initial_stage: int = 0,
        use_curriculum: bool = True,
        hold_steps: int = 10,        # SO-1: must hold lift this many steps
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.render_mode       = render_mode
        self.use_curriculum    = use_curriculum
        self.hold_steps        = hold_steps   # SO-1

        self.current_step     = 0
        self._hold_count      = 0             # SO-1
        self._viewer          = None

        # FIX-3: initialise here, not just in reset()
        self._prev_grasped    = False
        self._prev_action     = np.zeros(4, dtype=np.float32)   # SO-3

        self.curriculum_stage   = int(np.clip(initial_stage, 0, 4))
        self._episode_results: list[int] = []

        self.current_object_idx  = 0
        self.current_object_name = self.OBJECT_NAMES[0]

        self.target_pos = np.array(
            [0.45, 0.0, TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._load_model()


    # Model loading — once at init
    def _load_model(self):
        xml_str   = _build_xml(self.BASE_XML_PATH)
        scene_dir = os.path.dirname(os.path.abspath(self.BASE_XML_PATH))
        tmp_path  = os.path.join(scene_dir, "_panda_pick_all_objects.xml")

        with open(tmp_path, "w") as fh:
            fh.write(xml_str)

        self._tmp_xml_path = tmp_path
        self.model = mujoco.MjModel.from_xml_path(tmp_path)
        self.data  = mujoco.MjData(self.model)

        # Arm joint addresses
        self._qpos_indices = [
            self.model.joint(n).qposadr[0] for n in self.JOINT_NAMES
        ]
        self._qvel_indices = [
            self.model.joint(n).dofadr[0] for n in self.JOINT_NAMES
        ]

        # Finger joint qpos 
        self._finger_qpos_indices = []
        for name in self.FINGER_JOINT_NAMES:
            try:
                self._finger_qpos_indices.append(
                    self.model.joint(name).qposadr[0]
                )
            except Exception:
                pass

        # End-effector site
        self._ee_site_id = self.model.site(self.EE_SITE_NAME).id

        # All object bodies and free-joint addresses
        self._obj_qpos_starts = {}
        self._obj_body_ids    = {}
        self._obj_geom_ids    = {}
        for name in self.OBJECT_NAMES:
            self._obj_qpos_starts[name] = (
                self.model.joint(f"{name}_joint").qposadr[0]
            )
            self._obj_body_ids[name] = self.model.body(name).id
            self._obj_geom_ids[name] = self.model.geom(f"{name}_geom").id

        self._refresh_object_refs()

        # Finger body IDs 
        self._left_finger_body_id  = self.model.body("left_finger").id
        self._right_finger_body_id = self.model.body("right_finger").id
        self._n_actuators          = self.model.nu

        # Finger pad geom IDs 
        self._left_pad_geom_id  = None
        self._right_pad_geom_id = None
        self._use_geom_contact  = False
        try:
            self._left_pad_geom_id  = self.model.geom("left_finger_pad").id
            self._right_pad_geom_id = self.model.geom("right_finger_pad").id
            self._use_geom_contact  = True
        except Exception:
            pass   # fall back to body-level

    def _refresh_object_refs(self):
        self._obj_qpos_start = self._obj_qpos_starts[self.current_object_name]
        self._obj_body_id    = self._obj_body_ids[self.current_object_name]
        self._obj_geom_id    = self._obj_geom_ids[self.current_object_name]



    # Grasp detection
    def _get_finger_contacts(self) -> tuple[bool, bool]:
        """
        Returns (left_contact, right_contact).
        Uses named geom pads when available, body-level otherwise.
        Only counts contacts with meaningful penetration (dist < 2 mm).
        """
        left_contact  = False
        right_contact = False

        if self._use_geom_contact:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if abs(c.dist) > 0.002:
                    continue
                g1, g2 = c.geom1, c.geom2
                if g1 == self._obj_geom_id or g2 == self._obj_geom_id:
                    other = g2 if g1 == self._obj_geom_id else g1
                    if other == self._left_pad_geom_id:
                        left_contact = True
                    elif other == self._right_pad_geom_id:
                        right_contact = True
                if left_contact and right_contact:
                    break
        else:
            obj_body = self._obj_body_id
            for i in range(self.data.ncon):
                c  = self.data.contact[i]
                b1 = self.model.geom_bodyid[c.geom1]
                b2 = self.model.geom_bodyid[c.geom2]
                if b1 == obj_body or b2 == obj_body:
                    other = b2 if b1 == obj_body else b1
                    if other == self._left_finger_body_id:
                        left_contact = True
                    elif other == self._right_finger_body_id:
                        right_contact = True
                if left_contact and right_contact:
                    break

        return left_contact, right_contact

    def _object_is_grasped(self) -> bool:
        """Both fingers contact object AND gripper is at least 20% closed."""
        if self._n_actuators > 7:
            gripper_openness = float(self.data.ctrl[7]) / 255.0
            if gripper_openness > 0.80:
                return False
        left, right = self._get_finger_contacts()
        return left and right


    # Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step  = 0
        self._hold_count   = 0          # SO-1: reset hold counter
        self._prev_grasped = False
        self._prev_action  = np.zeros(4, dtype=np.float32)  # SO-3

        # Pick random object
        self.current_object_idx  = int(
            self.np_random.integers(0, len(self.OBJECT_NAMES))
        )
        self.current_object_name = self.OBJECT_NAMES[self.current_object_idx]
        self._refresh_object_refs()

        # Reset to home keyframe
        key_id = self.model.keyframe("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Bury all inactive objects underground
        for name in self.OBJECT_NAMES:
            s = self._obj_qpos_starts[name]
            self.data.qpos[s:s+3]   = [0.0, 0.0, -5.0]
            self.data.qpos[s+3]     = 1.0
            self.data.qpos[s+4:s+7] = 0.0

        # Place active object randomly on table surface
        obj_x = float(self.np_random.uniform(0.35, 0.55))
        obj_y = float(self.np_random.uniform(-0.15, 0.15))
        obj_z = TABLE_HEIGHT + 0.03
        s = self._obj_qpos_start
        self.data.qpos[s:s+3]   = [obj_x, obj_y, obj_z]
        self.data.qpos[s+3]     = 1.0
        self.data.qpos[s+4:s+7] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # FIX-4: curriculum pre-positioning via direct qpos (no physics loop)
        obj_pos = np.array([obj_x, obj_y, obj_z])
        self._apply_curriculum_reset(obj_pos)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._build_info()


    # Curriculum reset 
    def _apply_curriculum_reset(self, obj_pos: np.ndarray):
        """
        Pre-position EE using direct IK solve + qpos assignment.
        No physics simulation — resets are instantaneous.

        Stage 0: home pose, full task (hardest — trained last)
        Stage 1: EE above object, gripper open
        Stage 2: EE at grasp height, gripper open
        Stage 3: EE at grasp height, gripper closed
        Stage 4: pre-grasped, halfway to lift height (easiest — trained first)
        """
        stage = self.curriculum_stage
        if stage == 0:
            return

        def _set_ee_to(target: np.ndarray):
            """Single IK solve → directly write qpos (no simulation)."""
            try:
                result = ik.qpos_from_site_pose(
                    mjmodel    = self.model,
                    mjdata     = self.data,
                    site_name  = self.EE_SITE_NAME,
                    target_pos = target,
                    target_quat= self.EE_TARGET_QUAT,
                    joint_names= self.JOINT_NAMES,
                )
                for i, idx in enumerate(self._qpos_indices):
                    self.data.qpos[idx] = result[0][i]
                mujoco.mj_forward(self.model, self.data)
            except Exception:
                pass

        above_pos = obj_pos + np.array([0.0, 0.0, 0.05])
        grasp_pos = obj_pos + np.array([0.0, 0.0, 0.002])

        if stage >= 1:
            _set_ee_to(above_pos)

        if stage >= 2:
            _set_ee_to(grasp_pos)

        if stage >= 3:
            # Close gripper directly via ctrl
            if self._n_actuators > 7:
                self.data.ctrl[7] = 20.0
            mujoco.mj_forward(self.model, self.data)

        if stage >= 4:
            # Pre-lift object via direct qpos
            s = self._obj_qpos_start
            self.data.qpos[s + 2] = obj_pos[2] + LIFT_HEIGHT_THRESHOLD * 0.5
            mujoco.mj_forward(self.model, self.data)



    # Curriculum advance
    def _record_episode_result(self, success: bool):
        """
        Advance curriculum stage when rolling success rate
        exceeds threshold. Progresses: 4 → 3 → 2 → 1 → 0 (hardest).
        """
        if not self.use_curriculum:
            return

        self._episode_results.append(1 if success else 0)
        if len(self._episode_results) > CURRICULUM_WINDOW:
            self._episode_results.pop(0)
        if len(self._episode_results) < CURRICULUM_WINDOW:
            return

        rolling = float(np.mean(self._episode_results))
        if rolling >= CURRICULUM_THRESHOLD and self.curriculum_stage > 0:
            self.curriculum_stage -= 1
            self._episode_results  = []
            print(
                f"[Curriculum] → stage {self.curriculum_stage} "
                f"(rolling={rolling:.1%})"
            )


    # Step
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.current_step += 1

        # ── 1. Cartesian target ───────────────────────────────────────────
        EE_SCALE      = 0.05
        dx, dy, dz    = action[:3] * EE_SCALE
        gripper_cmd   = float(action[3])

        curr_ee_pos   = self.data.site_xpos[self._ee_site_id].copy()
        target_ee_pos = curr_ee_pos + np.array([dx, dy, dz])

        # Workspace safety bounds
        target_ee_pos[0] = np.clip(target_ee_pos[0], 0.20, 0.70)
        target_ee_pos[1] = np.clip(target_ee_pos[1], -0.40, 0.40)
        target_ee_pos[2] = np.clip(target_ee_pos[2], TABLE_HEIGHT + 0.01, 0.90)

        # ── 2. IK solve ───────────────────────────────────────────────────
        try:
            ik_result = ik.qpos_from_site_pose(
                mjmodel    = self.model,
                mjdata     = self.data,
                site_name  = self.EE_SITE_NAME,
                target_pos = target_ee_pos,
                target_quat= self.EE_TARGET_QUAT,
                joint_names= self.JOINT_NAMES,
            )
            joint_targets = ik_result[0][:7]
            ik_success    = True
        except Exception:
            joint_targets = np.array(
                [self.data.qpos[i] for i in self._qpos_indices]
            )
            ik_success = False

        # ── 3. Apply controls ─────────────────────────────────────────────
        ctrl = np.zeros(self._n_actuators, dtype=np.float64)
        ctrl[:7] = joint_targets
        if self._n_actuators > 7:
            ctrl[7] = 127.5 + 127.5 * gripper_cmd
        self.data.ctrl[:] = ctrl

        # ── 4. SO-2: 10 fixed physics substeps ───────────────────────────
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # ── 5. Reward + termination ───────────────────────────────────────
        info                = self._build_info()
        reward, terminated  = self._compute_reward(action)
        truncated           = self.current_step >= self.max_episode_steps

        # SO-1: hold counter — must maintain lift for hold_steps
        if info["is_lifted"] and info["object_grasped"]:
            self._hold_count += 1
        else:
            self._hold_count  = 0

        is_success          = self._hold_count >= self.hold_steps
        if is_success:
            terminated = True

        info["is_success"]   = is_success
        info["hold_count"]   = self._hold_count
        info["ik_success"]   = ik_success

        # Record for curriculum when episode ends
        if terminated or truncated:
            self._record_episode_result(is_success)

        # Update previous state for next step
        self._prev_grasped = info["object_grasped"]
        self._prev_action  = action.copy()   # SO-3

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info



    # Info dict
    def _build_info(self) -> dict:
        ee_pos     = self.data.site_xpos[self._ee_site_id]
        obj_pos    = self.data.xpos[self._obj_body_id]
        is_grasped = self._object_is_grasped()
        left_c, right_c = self._get_finger_contacts()

        rolling = (float(np.mean(self._episode_results))
                   if self._episode_results else 0.0)

        return {
            "object_name":        self.current_object_name,
            "object_height":      float(obj_pos[2]),
            "is_lifted":          bool(
                obj_pos[2] >= TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD
            ),
            "dist_ee_obj":        float(np.linalg.norm(ee_pos - obj_pos)),
            "dist_obj_goal":      float(
                np.linalg.norm(obj_pos - self.target_pos)
            ),
            "object_grasped":     bool(is_grasped),
            "left_contact":       bool(left_c),
            "right_contact":      bool(right_c),
            "curriculum_stage":   self.curriculum_stage,
            "rolling_success":    rolling,
            "using_geom_contact": self._use_geom_contact,
        }


    # Observation  (42-dim)
    def _get_obs(self) -> np.ndarray:
        joint_pos = np.array(
            [self.data.qpos[i] for i in self._qpos_indices], dtype=np.float32
        )
        joint_vel = np.array(
            [self.data.qvel[i] for i in self._qvel_indices], dtype=np.float32
        )

        # FIX-2: both gripper signals read from qpos (actual position)
        if self._finger_qpos_indices:
            total_opening = sum(
                self.data.qpos[i] for i in self._finger_qpos_indices
            )
            # jaw_sep: sum of both finger joints (max ~0.08 m when fully open)
            jaw_sep_norm  = np.array(
                [np.clip(total_opening / 0.08, 0.0, 1.0)], dtype=np.float32
            )
            # gripper openness: average of both finger joints, normalised
            gripper_state = np.array(
                [np.clip(total_opening / 0.08, 0.0, 1.0)], dtype=np.float32
            )
        else:
            # Fallback: ctrl-based (command, not actual position)
            raw = (float(self.data.ctrl[7]) / 255.0
                   if self._n_actuators > 7 else 0.5)
            jaw_sep_norm  = np.array([raw], dtype=np.float32)
            gripper_state = np.array([raw], dtype=np.float32)

        ee_pos   = self.data.site_xpos[self._ee_site_id].astype(np.float32)
        obj_pos  = self.data.xpos [self._obj_body_id].astype(np.float32)
        obj_quat = self.data.xquat[self._obj_body_id].astype(np.float32)
        rel_vec  = obj_pos - ee_pos

        is_grasped = np.array(
            [float(self._object_is_grasped())], dtype=np.float32
        )
        left_c, right_c = self._get_finger_contacts()
        contact_flags   = np.array(
            [float(left_c), float(right_c)], dtype=np.float32
        )

        h_above = max(0.0, float(obj_pos[2]) - TABLE_HEIGHT)
        h_prog  = np.array(
            [min(h_above / LIFT_HEIGHT_THRESHOLD, 1.0)], dtype=np.float32
        )

        one_hot = np.zeros(len(self.OBJECT_NAMES), dtype=np.float32)
        one_hot[self.current_object_idx] = 1.0

        stage_raw = np.array([float(self.curriculum_stage)], dtype=np.float32)

        obs = np.concatenate([
            joint_pos,        #  7  →  0: 7
            joint_vel,        #  7  →  7:14
            jaw_sep_norm,     #  1  → 14:15
            gripper_state,    #  1  → 15:16
            ee_pos,           #  3  → 16:19
            obj_pos,          #  3  → 19:22
            rel_vec,          #  3  → 22:25
            obj_quat,         #  4  → 25:29
            self.target_pos,  #  3  → 29:32
            one_hot,          #  5  → 32:37
            is_grasped,       #  1  → 37:38
            stage_raw,        #  1  → 38:39
            contact_flags,    #  2  → 39:41
            h_prog,           #  1  → 41:42
        ])                    # = 42

        assert obs.shape == (self.OBS_DIM,), f"Obs shape mismatch: {obs.shape}"
        return obs


    # Reward
    def _compute_reward(self, action: np.ndarray):
        """
        Dense shaped reward: reach → grasp → lift → hold.

        Component                Condition              Value
        ───────────────────────  ─────────────────────  ─────────────
        Reach (tanh)             always                 [0.0, 1.0]
        Push-down penalty        obj below table+5mm    -(depth*50)
        Gripper-close shaping    near object            [0.0, 0.15]
        Grasp bonus (per-step)   grasped                +0.25
        Drop penalty             grasp lost this step   -2.0
        Lift progress            grasped only           [0.0, 3.0]
        SO-4: Binary lift bonus  obj_z > table+2cm      +1.0
        SO-5: Target height bon. obj_z > lift_height    +1.0
        Target pull              grasped only           ≤0
        SO-3: Action rate pen.   obj_z > lift/2         -0.01*||Δa||²
        SO-1: Hold success       held for hold_steps    +10.0
        Failure                  fell / wandered        -50.0
        """
        ee_pos  = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.xpos[self._obj_body_id]

        dist_ee_obj   = float(np.linalg.norm(ee_pos  - obj_pos))
        dist_obj_goal = float(np.linalg.norm(obj_pos - self.target_pos))
        gripper_open  = (float(self.data.ctrl[7]) / 255.0
                         if self._n_actuators > 7 else 0.5)
        is_grasped    = self._object_is_grasped()
        obj_z         = float(obj_pos[2])

        # Reach — tanh-shaped, bounded [0, 1]
        reach_reward = 1.0 - float(np.tanh(10.0 * dist_ee_obj))

        # Push-down penalty
        push_threshold = TABLE_HEIGHT + 0.005
        push_penalty   = 0.0
        if obj_z < push_threshold:
            push_penalty = -(push_threshold - obj_z) * 50.0

        # Gripper-close shaping (wider decay → less exploitable)
        proximity           = float(np.exp(-dist_ee_obj / 0.08))
        gripper_close_bonus = 0.15 * proximity * (1.0 - gripper_open)

        # Per-step grasp bonus
        grasp_bonus = 0.25 if is_grasped else 0.0

        # Drop penalty — only fires on the exact step grasp is lost
        drop_penalty = -2.0 if (self._prev_grasped and not is_grasped) else 0.0

        # Continuous lift reward (gated on grasp)
        lift_reward = 0.0
        if is_grasped:
            h_above         = max(0.0, obj_z - TABLE_HEIGHT)
            height_progress = min(h_above / LIFT_HEIGHT_THRESHOLD, 1.0)
            lift_reward     = 3.0 * height_progress

        # SO-4: binary bonus when object clears the table by 2 cm
        binary_lift_bonus = 1.0 if obj_z > TABLE_HEIGHT + 0.02 else 0.0

        # SO-5: bonus when object reaches full target height
        target_height_bonus = 1.0 if obj_z > TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD else 0.0

        # Target pull (gated on grasp — moves object, not arm)
        target_pull = -0.5 * dist_obj_goal if is_grasped else 0.0

        # SO-3: action rate penalty — penalises jerk, not magnitude
        action_penalty = 0.0
        half_lift      = TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD * 0.5
        if obj_z > half_lift:
            action_delta   = action - self._prev_action
            action_penalty = -0.01 * float(np.sum(action_delta ** 2))

        reward = (
            reach_reward
            + push_penalty
            + gripper_close_bonus
            + grasp_bonus
            + drop_penalty
            + lift_reward
            + binary_lift_bonus      
            + target_height_bonus     
            + target_pull
            + action_penalty          
        )

        # Hold success bonus (fires when hold_count reaches threshold)
        if self._hold_count >= self.hold_steps:
            reward    += 10.0
            terminated = True
            return float(reward), True

        # Failure — tightened fall threshold (2 cm below table top)
        terminated   = False
        object_fell  = obj_z < TABLE_HEIGHT - 0.02
        arm_wandered = dist_ee_obj > 0.80

        if object_fell or arm_wandered:
            reward    -= 50.0
            terminated = True

        return float(reward), terminated


    # Rendering
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            else:
                self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            frame = renderer.render()
            renderer.close()
            return frame

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if hasattr(self, "_tmp_xml_path") and os.path.exists(self._tmp_xml_path):
            os.remove(self._tmp_xml_path)



# Sanity check
if __name__ == "__main__":
    print("Running sanity check...")
    print(f"  TABLE_HEIGHT          : {TABLE_HEIGHT} m")
    print(f"  LIFT_HEIGHT_THRESHOLD : {LIFT_HEIGHT_THRESHOLD} m")
    print(f"  Target pos            : z = {TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD} m")

    env = PandaPickEnv(
        max_episode_steps=20,
        render_mode=None,
        initial_stage=4,
        use_curriculum=True,
        hold_steps=10,
    )

    obs, info = env.reset(seed=42)
    print(f"\nEpisode reset")
    print(f"  Object          : {info['object_name']}")
    print(f"  Obs shape       : {obs.shape}")              # (42,)
    print(f"  Curriculum stage: {info['curriculum_stage']}")
    print(f"  Geom contact    : {info['using_geom_contact']}")

    total_reward = 0.0
    for t in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  step {t+1:2d}  rew={reward:+6.3f}  "
            f"h={info['object_height']:.3f}  "
            f"hold={info['hold_count']:2d}/{env.hold_steps}  "
            f"grasped={info['object_grasped']}  "
            f"success={info['is_success']}"
        )
        if terminated or truncated:
            break

    print(f"\nTotal reward : {total_reward:.3f}")
    env.close()
    print("Done.")