"""
panda_pick_env.py
=================
A Gymnasium environment for training a PPO agent to pick and lift 5 different
objects using the Franka Emika Panda arm + Hand (MuJoCo Menagerie).

Key design decisions
--------------------
1. TASK-SPACE CONTROL (IK - Inverse Kinematics)
   PPO outputs (dx, dy, dz, gripper) — 4 numbers.
   The IK solver converts the desired EE position into 7 joint angles.
   This is much easier for PPO than learning raw joint control.

2. SINGLE MODEL LOAD
   All 5 objects are injected into the XML once at startup.
   Inactive objects are buried underground (-5 m) each episode.
   The viewer never flickers or closes between episodes.

3. PRIVILEGED OBSERVATION (36-dim)
   No camera. PPO receives perfect state info: joint states, EE pos,
   object pos/orientation, target pos, and object type one-hot.

4. CONTACT-BASED GRASP DETECTION
   Uses MuJoCo body contact data to confirm both finger bodies
   are simultaneously touching the object — not just proximity.

Observation space : 36-dimensional (see _get_obs for full breakdown)
Action space      : 4-dimensional  [dx, dy, dz, gripper]

"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import inverse_kinematics as ik

# ---------------------------------------------------------------------------
# XML snippets
# ---------------------------------------------------------------------------

OBJECT_XMLS = {
    "cube": """
        <body name="cube" pos="0.45 0 0.25">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
                rgba="0.8 0.2 0.2 1" mass="0.1" condim="4" friction="1 0.005 0.0001"/>
          <site name="cube_site" pos="0 0 0" size="0.005"/>
        </body>""",
    "sphere": """
        <body name="sphere" pos="0.45 0 0.255">
          <freejoint name="sphere_joint"/>
          <geom name="sphere_geom" type="sphere" size="0.025"
                rgba="0.2 0.6 0.2 1" mass="0.1" condim="4" friction="1 0.005 0.0001"/>
          <site name="sphere_site" pos="0 0 0" size="0.005"/>
        </body>""",
    "cylinder": """
        <body name="cylinder" pos="0.45 0 0.26">
          <freejoint name="cylinder_joint"/>
          <geom name="cylinder_geom" type="cylinder" size="0.02 0.03"
                rgba="0.2 0.2 0.8 1" mass="0.1" condim="4" friction="1 0.005 0.0001"/>
          <site name="cylinder_site" pos="0 0 0" size="0.005"/>
        </body>""",
    "box": """
        <body name="box" pos="0.45 0 0.25">
          <freejoint name="box_joint"/>
          <geom name="box_geom" type="box" size="0.04 0.025 0.02"
                rgba="0.8 0.6 0.1 1" mass="0.15" condim="4" friction="1 0.005 0.0001"/>
          <site name="box_site" pos="0 0 0" size="0.005"/>
        </body>""",
    "cone": """
        <body name="cone" pos="0.45 0 0.26">
          <freejoint name="cone_joint"/>
          <geom name="cone_geom" type="cylinder" size="0.02 0.03"
                rgba="0.7 0.1 0.7 1" mass="0.08" condim="4" friction="1 0.005 0.0001"/>
          <site name="cone_site" pos="0 0 0" size="0.005"/>
        </body>""",
}


TABLE_XML = """
    <body name="table" pos="0.45 0 0.1">
      <geom name="table_geom" type="box" size="0.35 0.35 0.1"
            rgba="0.9 0.8 0.6 1" contype="1" conaffinity="1"
            friction="1 0.005 0.0001"/>
    </body>"""


TABLE_HEIGHT            = 0.20   # metres — top surface of the table in world frame
LIFT_HEIGHT_THRESHOLD   = 0.20   # agent must raise object this far above the table top


def _build_xml(base_xml_path: str) -> str:
    """
    Inject the table AND all 5 objects into the Panda scene XML.
    All objects are present from the start; reset() buries the inactive ones.
    The temp file is written next to scene.xml so <include> and asset paths
    resolve correctly.
    """
    with open(base_xml_path, "r") as f:
        xml_text = f.read()
    if "</worldbody>" not in xml_text:
        raise ValueError("Could not find </worldbody> in the scene XML.")
    all_objects_xml = "\n".join(OBJECT_XMLS.values())
    xml_text = xml_text.replace(
        "</worldbody>",
        f"{TABLE_XML}\n{all_objects_xml}\n  </worldbody>"
    )
    return xml_text


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PandaPickEnv(gym.Env):
    """
    Franka Panda pick-and-lift environment.

    ── Observation (36-dim, PRIVILEGED) ─────────────────────────────────────
    Index   Size  Description                          Privileged?
    ──────  ────  ───────────────────────────────────  ───────────
      0: 7    7   Arm joint positions (rad)             No  (joint encoders)
      7:14    7   Arm joint velocities (rad/s)          No  (joint encoders)
     14:15    1   Gripper opening  0=closed … 1=open    No  (gripper encoder)
     15:18    3   End-effector position (x, y, z)       No  (FK from joints)
     18:21    3   Object position (x, y, z)             YES (needs perception)
     21:24    3   Relative vector  obj_pos − ee_pos     YES (derived)
     24:28    4   Object quaternion (w, qx, qy, qz)     YES (needs pose est.)
     28:31    3   Target position (x, y, z)             YES (task spec.)
     31:36    5   Object one-hot (which of 5 types)     Partial
    ──────  ────
              36  total

    ── Action (4-dim) ───────────────────────────────────────────────────────
    action[0:3]  (dx, dy, dz) Cartesian EE displacement, ±5 cm per step
    action[3]    gripper:  +1 = open,  −1 = close
    """

    metadata      = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    OBJECT_NAMES  = ["cube", "sphere", "cylinder", "box", "cone"]
    JOINT_NAMES   = ["joint1", "joint2", "joint3",
                     "joint4", "joint5", "joint6", "joint7"]
    EE_SITE_NAME  = "pinch"
    BASE_XML_PATH = "franka_emika_panda/scene.xml"

    # Gripper pointing straight down (w=0, x=0.707, y=-0.707, z=0)
    EE_TARGET_QUAT = np.array([0.0, 0.7071068, -0.7071068, 0.0])

    def __init__(
        self,
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        control_timestep: float = 0.02,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.render_mode       = render_mode
        self.control_timestep  = control_timestep
        self.current_step      = 0
        self._viewer           = None

        self.current_object_idx  = 0
        self.current_object_name = self.OBJECT_NAMES[0]

        # Fixed lift target: directly above the table centre
        self.target_pos = np.array(
            [0.45, 0.0, TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD], dtype=np.float32
        )

        # ── Spaces ────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # ── Load model ONCE (all 5 objects embedded) ──────────────────────
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading — called ONCE at init, not every episode
    # ------------------------------------------------------------------
    def _load_model(self):
        """
        Write the patched XML (table + all 5 objects) next to scene.xml so
        MuJoCo can resolve <include> and asset paths, then load it.
        Called once at __init__; reset() repositions objects instead.
        """
        xml_str   = _build_xml(self.BASE_XML_PATH)
        scene_dir = os.path.dirname(os.path.abspath(self.BASE_XML_PATH))
        tmp_path  = os.path.join(scene_dir, "_panda_pick_all_objects.xml")
        with open(tmp_path, "w") as fh:
            fh.write(xml_str)

        self._tmp_xml_path = tmp_path
        self.model = mujoco.MjModel.from_xml_path(tmp_path)
        self.data  = mujoco.MjData(self.model)

        # ── Arm joint addresses ──────────────────────────────────────────
        self._qpos_indices = [
            self.model.joint(n).qposadr[0] for n in self.JOINT_NAMES
        ]
        self._qvel_indices = [
            self.model.joint(n).dofadr[0] for n in self.JOINT_NAMES
        ]

        # ── End-effector site ────────────────────────────────────────────
        self._ee_site_id = self.model.site(self.EE_SITE_NAME).id

        # ── Cache all object body + free-joint addresses ─────────────────
        self._obj_qpos_starts = {}
        self._obj_body_ids    = {}
        for name in self.OBJECT_NAMES:
            self._obj_qpos_starts[name] = (
                self.model.joint(f"{name}_joint").qposadr[0]
            )
            self._obj_body_ids[name] = self.model.body(name).id

        # Set convenience references for the current active object
        self._obj_qpos_start = self._obj_qpos_starts[self.current_object_name]
        self._obj_body_id    = self._obj_body_ids[self.current_object_name]

        # ── Finger body IDs for contact-based grasp detection ────────────
        self._left_finger_body_id  = self.model.body("left_finger").id
        self._right_finger_body_id = self.model.body("right_finger").id
        self._n_actuators = self.model.nu

    # ------------------------------------------------------------------
    # Grasp detection — body-contact based
    # ------------------------------------------------------------------
    def c(self) -> bool:
        """
        Return True when BOTH finger bodies are simultaneously contacting
        the active object body.

        Why body-level and not geom-level?
        The Panda's finger collision geoms in Menagerie's panda.xml are
        unnamed, so we cannot look them up by name.  Instead we check which
        BODY each geom in a contact pair belongs to using model.geom_bodyid.

        A contact pair (g1, g2) involves our object if one geom belongs to
        the object body.  We then check whether the other geom belongs to
        the left or right finger body.  Both fingers must be in contact
        simultaneously for this to count as a real grasp.
        """
        obj_body = self._obj_body_id
        fingers_in_contact = set()
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == obj_body or b2 == obj_body:
                other = b2 if b1 == obj_body else b1
                if other == self._left_finger_body_id:
                    fingers_in_contact.add("left")
                elif other == self._right_finger_body_id:
                    fingers_in_contact.add("right")
        return len(fingers_in_contact) == 2

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # FIX (Bug 2): reset the one-time grasp bonus flag each episode
        self._grasp_rewarded = False

        # Pick a random object for this episode
        self.current_object_idx  = int(self.np_random.integers(0, len(self.OBJECT_NAMES)))
        self.current_object_name = self.OBJECT_NAMES[self.current_object_idx]

        # Update convenience references
        self._obj_qpos_start = self._obj_qpos_starts[self.current_object_name]
        self._obj_body_id    = self._obj_body_ids[self.current_object_name]

        # Reset arm to home keyframe (upright pose, gripper open)
        key_id = self.model.keyframe("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Bury ALL inactive objects 5 m underground so they don't interfere
        for name in self.OBJECT_NAMES:
            s = self._obj_qpos_starts[name]
            self.data.qpos[s + 0] = 0.0
            self.data.qpos[s + 1] = 0.0
            self.data.qpos[s + 2] = -5.0   # underground
            self.data.qpos[s + 3] = 1.0    # unit quaternion
            self.data.qpos[s + 4] = 0.0
            self.data.qpos[s + 5] = 0.0
            self.data.qpos[s + 6] = 0.0

        # Place the ACTIVE object randomly on the table surface
        obj_x = float(self.np_random.uniform(0.35, 0.55))
        obj_y = float(self.np_random.uniform(-0.15, 0.15))
        obj_z = TABLE_HEIGHT + 0.03   # slightly above table top
        s = self._obj_qpos_start
        self.data.qpos[s + 0] = obj_x
        self.data.qpos[s + 1] = obj_y
        self.data.qpos[s + 2] = obj_z
        self.data.qpos[s + 3] = 1.0
        self.data.qpos[s + 4] = 0.0
        self.data.qpos[s + 5] = 0.0
        self.data.qpos[s + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        obs  = self._get_obs()
        info = {"object_name": self.current_object_name}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.current_step += 1

        # ── 1. Compute target EE position in Cartesian space ─────────────
        EE_SCALE = 0.05   # ±1 action unit → ±5 cm
        dx, dy, dz  = action[:3] * EE_SCALE
        gripper_cmd = float(action[3])

        curr_ee_pos   = self.data.site_xpos[self._ee_site_id].copy()
        target_ee_pos = curr_ee_pos + np.array([dx, dy, dz])

        # Workspace safety bounds — keep EE above table, within reach
        target_ee_pos[0] = np.clip(target_ee_pos[0], 0.20, 0.70)
        target_ee_pos[1] = np.clip(target_ee_pos[1], -0.40, 0.40)
        target_ee_pos[2] = np.clip(target_ee_pos[2], TABLE_HEIGHT + 0.01, 0.90)

        # ── 2. Inverse Kinematics ─────────────────────────────────────────
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
            # Menagerie actuator8: 0 = closed, 255 = open
            ctrl[7] = 127.5 + 127.5 * gripper_cmd
        self.data.ctrl[:] = ctrl

        # ── 4. Simulate ───────────────────────────────────────────────────
        n_substeps = max(1, int(self.control_timestep / self.model.opt.timestep))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        # ── 5. Reward + termination ───────────────────────────────────────
        reward, terminated = self._compute_reward(action)
        truncated = self.current_step >= self.max_episode_steps

        ee_pos  = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.xpos[self._obj_body_id]

        # FIX (Bug 5): is_success uses a geometric check, not a reward threshold.
        # "reward > 50" was reachable via accumulated shaping bugs alone.
        dist_obj_goal = float(np.linalg.norm(obj_pos - self.target_pos))
        info = {
            "is_success": (
                terminated
                and float(obj_pos[2]) >= TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD
                and dist_obj_goal < 0.10
                and self._object_is_grasped()
            ),
            "object_name":    self.current_object_name,
            "object_height":  float(obj_pos[2]),
            "dist_ee_obj":    float(np.linalg.norm(ee_pos - obj_pos)),
            "object_grasped": self._object_is_grasped(),
            "ik_success":     ik_success,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation (36-dim)
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """
        36-dimensional privileged state observation.

        Index   Size  Description
        ──────  ────  ─────────────────────────────────────
          0: 7    7   Arm joint positions (rad)
          7:14    7   Arm joint velocities (rad/s)
         14:15    1   Gripper opening  (0=closed, 1=open)
         15:18    3   End-effector position (x, y, z)
         18:21    3   Object position (x, y, z)
         21:24    3   Relative vector  obj_pos - ee_pos
         24:28    4   Object quaternion (w, qx, qy, qz)
         28:31    3   Target position (x, y, z)
         31:36    5   Object one-hot  (5 object types)
        ──────  ────
               36    total
        """
        joint_pos = np.array(
            [self.data.qpos[i] for i in self._qpos_indices], dtype=np.float32
        )
        joint_vel = np.array(
            [self.data.qvel[i] for i in self._qvel_indices], dtype=np.float32
        )

        # Gripper ctrl in [0, 255] → normalise to [0, 1]
        if self._n_actuators > 7:
            gripper_state = np.array(
                [np.clip(self.data.ctrl[7] / 255.0, 0.0, 1.0)], dtype=np.float32
            )
        else:
            gripper_state = np.array([0.5], dtype=np.float32)

        ee_pos   = self.data.site_xpos[self._ee_site_id].astype(np.float32)
        obj_pos  = self.data.xpos [self._obj_body_id].astype(np.float32)
        obj_quat = self.data.xquat[self._obj_body_id].astype(np.float32)
        rel_vec  = obj_pos - ee_pos

        one_hot = np.zeros(len(self.OBJECT_NAMES), dtype=np.float32)
        one_hot[self.current_object_idx] = 1.0

        obs = np.concatenate([
            joint_pos,        #  7  →  0: 7
            joint_vel,        #  7  →  7:14
            gripper_state,    #  1  → 14:15
            ee_pos,           #  3  → 15:18
            obj_pos,          #  3  → 18:21
            rel_vec,          #  3  → 21:24
            obj_quat,         #  4  → 24:28
            self.target_pos,  #  3  → 28:31
            one_hot,          #  5  → 31:36
        ])                    # = 36

        assert obs.shape == (36,), f"Obs shape mismatch: {obs.shape}"
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray):
        """
        Stage-based dense reward:  reach → grasp → lift → success.

        Component                  Condition            Value / step
        ─────────────────────────  ───────────────────  ──────────────────────
        Reach shaping              always               -dist(ee, object)
        Gripper shaping            always               ≤ +0.30  (exp-decayed)
        Grasp bonus (one-time)     first contact        +5.0     (flag-gated)
        Lift shaping               object contact only  ≤ +3.0   (height frac.)
        Goal pull                  object contact only  -0.5 * dist(obj, goal)
        Action regularisation      always               -0.01 * ||a||²
        ─────────────────────────  ───────────────────  ──────────────────────
        Success                    lifted + near + grip +100,  terminates
        Failure (fell off table)   obj_z < table - 5cm  -50,  terminates
        Failure (arm wandered)     dist(ee,obj) > 80 cm -50,  terminates

        """
        ee_pos  = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.xpos[self._obj_body_id]

        dist_ee_obj   = float(np.linalg.norm(ee_pos  - obj_pos))
        dist_obj_goal = float(np.linalg.norm(obj_pos - self.target_pos))
        gripper_open  = float(self.data.ctrl[7]) / 255.0 if self._n_actuators > 7 else 0.5

        # Compute once — used by multiple terms below
        is_grasped = self._object_is_grasped()

        # ── Reach ────────────────────────────────────────────────────────
        # Continuous pull of the EE toward the object.
        reward = -dist_ee_obj

        # ── Gripper shaping (FIX Bug 3) ──────────────────────────────────
        # Exponential decay: reward ≈ 0 when dist > ~15 cm, peaks at +0.30
        # when the EE is touching the object.  No exploitable binary threshold.
        reach_factor = float(np.exp(-dist_ee_obj / 0.05))
        reward += 0.30 * reach_factor * (1.0 - gripper_open)

        # ── Contact-based grasp bonus (FIX Bug 2) ────────────────────────
        # One-time +5 the first step both fingers touch the object.
        # self._grasp_rewarded is reset to False in reset().
        if is_grasped and not self._grasp_rewarded:
            reward += 5.0
            self._grasp_rewarded = True

        # ── Continuous lift reward (FIX Bug 1) ───────────────────────────
        # Height-progress shaping only fires when the object is physically
        # in the grasp (body contact).  Old code used gripper_open < 0.3,
        # which gave free reward even when the gripper was just closed in air.
        height_above_table = max(0.0, obj_pos[2] - TABLE_HEIGHT)
        height_progress    = min(height_above_table / LIFT_HEIGHT_THRESHOLD, 1.0)
        if is_grasped:
            reward += 3.0 * height_progress

        # ── Object-to-target pull (FIX Bug 4) ────────────────────────────
        # Only applied once the object is confirmed grasped so the agent is
        # not mis-incentivised to move its arm (not the object) to the goal.
        if is_grasped:
            reward -= 0.5 * dist_obj_goal

        # ── Action regularisation ────────────────────────────────────────
        reward -= 0.01 * float(np.sum(action ** 2))

        # ── Success ───────────────────────────────────────────────────────
        terminated = False
        lifted     = obj_pos[2] >= TABLE_HEIGHT + LIFT_HEIGHT_THRESHOLD
        near_goal  = dist_obj_goal < 0.10

        # Also require is_grasped so a thrown object can't trigger success
        # during its ballistic arc.
        if lifted and near_goal and is_grasped:
            reward    += 100.0
            terminated = True

        # ── Failure ───────────────────────────────────────────────────────
        object_fell  = obj_pos[2] < TABLE_HEIGHT - 0.05
        arm_wandered = dist_ee_obj > 0.80
        if object_fell or arm_wandered:
            reward    -= 50.0
            terminated = True

        return float(reward), terminated

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
        # Clean up the patched XML file
        scene_dir = os.path.dirname(os.path.abspath(self.BASE_XML_PATH))
        tmp_path  = os.path.join(scene_dir, "_panda_pick_all_objects.xml")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Sanity check  (python panda_pick_env.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running environment sanity check...")
    env = PandaPickEnv(max_episode_steps=10, render_mode=None)
    obs, info = env.reset(seed=42)
    print(f"Object this episode    : {info['object_name']}")
    print(f"Observation shape      : {obs.shape}")          # (36,)
    print(f"Joint pos    (obs 0:7) : {obs[0:7].round(3)}")
    print(f"EE pos     (obs 15:18) : {obs[15:18].round(3)}")
    print(f"Object pos (obs 18:21) : {obs[18:21].round(3)}")
    print(f"Rel vec    (obs 21:24) : {obs[21:24].round(3)}")
    print(f"Target pos (obs 28:31) : {obs[28:31].round(3)}")
    print(f"One-hot    (obs 31:36) : {obs[31:36]}")

    total_reward = 0.0
    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  step {t+1:2d}  reward={reward:+.3f}  "
            f"height={info['object_height']:.3f}  "
            f"grasped={info['object_grasped']}  "
            f"ik_ok={info['ik_success']}  "
            f"success={info['is_success']}"
        )
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    env.close()
    print("Done.")