"""Closed-loop industrial work orders with measured MuJoCo contact conditions."""

from dataclasses import dataclass
from functools import lru_cache
import math

from gait import Footsteps, Frame, LegIK, pelvis_quat, yaw_quat
from gait_policy import FootFeedback
from navigation import Aisles, ContactWatch
from plant import BASKET_FLOOR, BASKET_X, CATALOG, GRIP_HALF, PAN_MASS, PAN_STIFFNESS, PUMP_FLIGHT, ROOT, build, named_id, mj, np
from factory_layout import PUMP_EXTENT
from pump_survey import (
    CART_X, COVER_CLEAR, COVER_VIEW, DOG_WAIT, FLIGHTS, GATE_VIEW, LANES, LANE_EAST, LANE_NAMES,
    LANE_WEST, LOWER_VIEW, PUMPS, UNDERPIPE_ENTRY, UNDERPIPE_VIEW, PumpSurvey, PumpVariant,
)


DT = .02
ARM_HOME = np.array([.10, -.25, .03, .87, 0., 0., 0.])
FEATURES = 23
CORRECTION = .026


class TaskFault(RuntimeError):
    pass


@dataclass
class Guide:
    xyz: np.ndarray
    yaw: float = 0.
    crouch: float = 0.
    crouch_applied: float = 0.
    frame: Frame | None = None


def squash(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_matrix(quat):
    result = np.empty(9)
    mj.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


def spread(target, value, amount):
    return value + np.clip(target - value, -amount, amount)


@lru_cache(maxsize=1)
def load_policy():
    import onnxruntime as ort

    path = ROOT / "results/g1-contact-policy.onnx"
    if not path.exists():
        raise FileNotFoundError("缺少服务器导出的 G1 接触策略")
    cfg = ort.SessionOptions()
    cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), cfg, providers=["CPUExecutionProvider"])
    return lambda obs: session.run(None, {"observation": obs[None].astype(np.float32)})[0][0]


class Simulation:
    def __init__(self, scene, seed=17, policy=None, capture=False, gait_policy=None, capture_gait=False, gait_teacher=False,
                 factory=None):
        if scene not in CATALOG or not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
            raise ValueError("场景或种子无效")
        self.scene, self.seed = scene, seed
        self.factory = factory
        self.random = np.random.default_rng(seed)
        variant = PumpVariant.from_seed(seed) if scene == "pump" else None
        self.alternate = variant is not None and variant.closed_a
        xml, self.layout = build(scene, self.alternate, variant, factory)
        self.aisles = Aisles(dict(self.layout, extent=PUMP_EXTENT) if factory else self.layout)
        self.model = mj.MjModel.from_xml_string(xml)
        self.data = mj.MjData(self.model)
        self.kin = mj.MjData(self.model)
        self.policy = policy
        self.capture = capture
        self.transitions = []
        self.robot = {"g1": self._robot("pelvis", "floating_base_joint", 29, 0),
                      "go2": self._robot("go_base", "go_float", 12, 29)}
        self.arm = self.robot["g1"]["q"] + 22 + np.arange(7)
        self.arm_dof = self.robot["g1"]["v"] + 22 + np.arange(7)
        self.limits = self.model.jnt_range[1 + 22:1 + 29].copy()
        self.grip = self.id("site", "grip")
        self.clamp = self.id("body", "clamp")
        self.jaws = [self.id("joint", f"jaw_{side}_slide") for side in ("left", "right")]
        self.jaw_ctrl = [self.id("actuator", f"close_{side}") for side in ("left", "right")]
        self.pads = [self.id("geom", f"pad_{side}") for side in ("left", "right")]
        self.probe = self.id("geom", "probe")
        self.handle = self.id("geom", "piece_handle") if self.scene in GRIP_HALF else None
        self.item_geom = self.id("geom", "piece_body") if self.scene in GRIP_HALF else None
        self.part = self.id("body", "workpiece") if self.scene in GRIP_HALF else None
        part_joint = self.id("joint", "piece_float") if self.part is not None else None
        self.part_q = self.model.jnt_qposadr[part_joint] if part_joint is not None else None
        self.part_v = self.model.jnt_dofadr[part_joint] if part_joint is not None else None
        self.button = self.model.jnt_qposadr[self.id("joint", "button_slide")] if scene == "pack" else None
        self.door = self.model.jnt_qposadr[self.id("joint", "door_hinge")] if scene in ("pump", "air") else None
        self.cell = self.model.jnt_qposadr[self.id("joint", "cell_slide")] if scene == "store" else None
        self.basket_clip = self.id("equality", "basket_clip") if scene == "pump" else None
        self.grip_constraint = "hand_grasp"
        self.cargo_name = "记录器"
        self.clearance = ContactWatch(self.model, self.robot, self.part, self.pads, self.probe)
        self.log = []
        self.metrics = {"grasps": 0, "placements": 0, "samples": 0, "buttonMm": 0., "doorDegrees": 0.,
                        "maxGripN": 0., "maxProbeN": 0., "policyCalls": 0, "gaitPolicyCalls": 0, "reroutes": 0,
                        "aisleDetours": 0, "yieldFrames": 0, "basketPickups": 0, "unexpectedContactFrames": 0}
        self.guides = {"g1": Guide(np.array((1.31, 2.55, 2.8) if scene == "pump" else (-4.25, -.52, 0.))),
                       "go2": Guide(np.array((-4.59, -2.31, 0.)))}
        self.mocap = {key: self.model.body_mocapid[self.id("body", f"{key}_guide")] for key in self.guides}
        self.hand_goal = None
        self.access_entry = None
        self.basket_access = False
        self.closed = False
        self.holding = False
        self.retracting = False
        self.held_inverse = np.array([1., 0., 0., 0.])
        self.attachment = None
        self.twists = np.zeros(6)
        self.last_grip = None
        self.hand_speed = np.zeros(3)
        self.q_arm = ARM_HOME.copy()
        self.left_force = self.right_force = self.probe_force = 0.
        self.hand_mode = 0.
        self.status = "ready"
        self.step = -1
        self.ticks = 0
        self.focus = None
        self.reading = None
        self.target_station = None if scene == "pump" else "target"
        self.survey = None
        self.inbox = {}
        if variant is not None:
            self.cart = self.model.body_mocapid[self.id("body", "service_cart")]
            self.move_cart = variant.move_cart
            self.survey = PumpSurvey(self.model, self.data, self.robot, self.part, self.layout["extent"], compact=bool(factory))
            self.metrics.update(mapUpdates=0, messages=0, cartMoves=0, transferLane=None, previousLane=None,
                                selectedFlight=None, coverProof=0, blindProof=0, verifiedPump=None)
        self._initialize()
        self.leg_ik = LegIK(self.model, self.kin, self.robot)
        self.footsteps = {}
        self.previous_leg_ref = {}
        for key, robot in self.robot.items():
            offsets = self.leg_ik.positions(key, self.data) - self.guides[key].xyz
            offsets[:, 2] -= .010 if key == "g1" else .013
            self.footsteps[key] = Footsteps(key, offsets, DT)
            self.previous_leg_ref[key] = self.data.qpos[robot["q"]:robot["q"]+12].copy()
        self.leg_actuators = np.concatenate([np.arange(robot["act"], robot["act"]+12) for robot in self.robot.values()])
        self.foot_feedback = (FootFeedback(self.model, self.data, self.robot, DT, gait_policy, capture_gait)
                              if gait_policy is not None or capture_gait or gait_teacher else None)
        self.program = self.work_factory() if factory else getattr(self, f"work_{scene}")()

    def id(self, category, name):
        return named_id(self.model, getattr(mj.mjtObj, f"mjOBJ_{category.upper()}"), name)

    def _robot(self, body, joint, n, act):
        j = self.id("joint", joint)
        return {"body": self.id("body", body), "root": self.model.jnt_qposadr[j],
                "q": self.model.jnt_qposadr[j] + 7, "v": self.model.jnt_dofadr[j] + 6,
                "rv": self.model.jnt_dofadr[j], "n": n, "act": act}

    @property
    def time(self):
        return self.ticks * DT

    def _posture(self, key, legs=None, frame=None):
        if key == "go2":
            return np.tile([0., .78, -1.56], 4) if legs is None else legs
        if legs is None:
            legs = np.tile([-.125, 0., 0., .25, -.125, 0.], 2)
        swing = 0. if frame is None else frame.arm_swing * (.52 if self.holding else 1.)
        relaxed = 0. if frame is None else frame.walking
        elbow = .86 + .43*relaxed
        waist = (0., 0., 0.) if frame is None else (-.85*frame.pelvis_twist, -.75*frame.pelvis_roll, 0.)
        result = np.r_[legs, waist, [.10+swing, .24, -.03, elbow-.18*swing, 0., 0., 0.], self.q_arm]
        if self.hand_goal is None and not self.holding and not self.retracting:
            result[22] -= swing
            result[25] += .43*relaxed + .18*swing
        return result

    def _move_legs(self, key, robot, correction=None):
        g = self.guides[key]
        d, m = self.data, self.model
        frame = g.frame
        g.crouch_applied = float(spread(g.crouch, g.crouch_applied, .045))
        lowering = (frame.lowering if frame is not None else 0.) + .137*g.crouch_applied
        root = g.xyz + [0., 0., self.standing[key] - lowering]
        if frame is not None:
            feet, headings = frame.feet, frame.sole_yaw
        else:
            feet = self.footsteps[key].planted(g.xyz, g.yaw)
            headings = np.full(len(feet), g.yaw)
        if correction is not None:
            feet = feet + correction
        attitude = yaw_quat(g.yaw)
        if key == "g1":
            pitches = np.zeros(2)
            if frame is not None:
                attitude = pelvis_quat(g.yaw+frame.pelvis_twist, frame.pelvis_pitch, frame.pelvis_roll)
                pitches = frame.sole_pitch
            legs = self.leg_ik.g1(root, attitude, feet, headings, pitches, d.qpos)
        else:
            legs = self.leg_ik.go2(root, g.yaw, feet)

        idx = slice(robot["act"], robot["act"] + 12)
        speed = np.clip((legs - self.previous_leg_ref[key]) / DT, -8., 8.)
        self.previous_leg_ref[key] = legs
        kp = m.actuator_gainprm[idx, 0]
        kv = -m.actuator_biasprm[idx, 2]
        feedback = -m.actuator_forcerange[idx, 0] * .16
        v = robot["v"]
        gravity = np.clip(d.qfrc_bias[v:v+12], -feedback, feedback)
        requested = legs + np.clip((kv*speed + gravity) / kp, -.32, .32)
        d.mocap_pos[self.mocap[key]] = root
        d.mocap_quat[self.mocap[key]] = attitude
        actuator, n = robot["act"], robot["n"]
        d.ctrl[actuator:actuator+n] = self._posture(key, requested, frame)

    def _initialize(self):
        d = self.data
        self.standing = {"g1": .78, "go2": .326}
        for key, info in self.robot.items():
            r, q, n = info["root"], info["q"], info["n"]
            g = self.guides[key]
            d.qpos[r:r+3] = g.xyz + [0., 0., self.standing[key]]
            d.qpos[r+3:r+7] = yaw_quat(g.yaw)
            d.qpos[q:q+n] = self._posture(key)
        mj.mj_forward(self.model, d)
        for key, pads in (("g1", [f"{s}_pad_{i}" for s in ("left", "right") for i in range(4)]),
                          ("go2", [f"go_{leg}" for leg in ("FL", "FR", "RL", "RR")])):
            ids = [self.id("geom", name) for name in pads]
            lowest = min(d.geom_xpos[i, 2] - self.model.geom_size[i, 0] for i in ids)
            self.standing[key] += self.guides[key].xyz[2] - lowest + .004
            d.qpos[self.robot[key]["root"] + 2] = self.guides[key].xyz[2] + self.standing[key]
        if self.part is not None:
            p = self.layout["points"].get("pick")
            if p is None:
                p = self.guides["go2"].xyz + [BASKET_X, 0., self.standing["go2"] + BASKET_FLOOR + GRIP_HALF["pump"][2] + .031]
            else:
                p = np.array(p) + np.r_[self.random.uniform(-.008, .008, size=2), 0.]
            self.part_mass = float(self.random.uniform(.30, .43))
            self.model.body_mass[self.part] = self.part_mass
            d.qpos[self.part_q:self.part_q+3] = p
            d.qpos[self.part_q+3:self.part_q+7] = [1., 0., 0., 0.]
        mj.mj_forward(self.model, d)
        if self.scene == "pump":
            self.bind("basket_clip", self.id("body", "go_basket"))
        elif self.part is not None:
            self.bind("source_grasp", self.id("body", "source_fixture"))
        for key, item in self.robot.items():
            r = item["root"]
            d.mocap_pos[self.mocap[key]] = d.qpos[r:r+3]
            d.mocap_quat[self.mocap[key]] = d.qpos[r+3:r+7]
            d.ctrl[item["act"]:item["act"]+item["n"]] = self._posture(key)
        mj.mj_forward(self.model, d)
        self.last_grip = d.site_xpos[self.grip].copy()

    def bind(self, name, first):
        # Store the relative transform measured at contact. Coupling never changes qpos.
        eid = self.id("equality", name)
        data = self.data
        q1, q2 = data.xquat[first], data.xquat[self.part]
        inv = q1 * [1., -1., -1., -1.]
        q = np.empty(4)
        mj.mju_mulQuat(q, inv, q2)
        transform = self.model.eq_data[eid]
        transform[3:6] = data.xmat[first].reshape(3, 3).T @ (data.xpos[self.part] - data.xpos[first])
        transform[6:10] = q
        if self.attachment is not None:
            data.eq_active[self.attachment] = False
        self.attachment = eid
        data.eq_active[eid] = True

    def unbind(self):
        if self.attachment is not None:
            self.data.eq_active[self.attachment] = False
            self.attachment = None

    def _read_contacts(self):
        left = right = probe = 0.
        support = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            a, b = contact.geom1, contact.geom2
            pair = {a, b}
            if not pair.intersection((self.handle, self.item_geom, self.probe)):
                continue
            mj.mj_contactForce(self.model, self.data, index, self.twists)
            load = float(self.twists[0])
            if pair.intersection((self.handle, self.item_geom)):
                left += load if self.handle in pair and self.pads[0] in pair else 0.
                right += load if self.handle in pair and self.pads[1] in pair else 0.
                surface = b if a in (self.handle, self.item_geom) else a
                support[surface] = support.get(surface, 0.) + load
            if self.probe in pair:
                probe += load
        self.support = support
        self.left_force, self.right_force, self.probe_force = left, right, probe
        self.metrics["maxGripN"] = max(self.metrics["maxGripN"], left + right)
        self.metrics["maxProbeN"] = max(self.metrics["maxProbeN"], probe)

    def observation(self, target):
        d, s = self.data, self.robot["g1"]
        relative = d.site_xpos[self.grip] - d.xpos[s["body"]]
        goal = target - d.xpos[s["body"]]
        contacts = np.array([self.left_force + self.right_force, self.probe_force]) / 20.
        return np.r_[relative, self.hand_speed * .2, goal, d.qpos[self.arm] * .35,
                     contacts, float(self.closed), float(self.holding), self.hand_mode,
                     float(self.scene == "air" or self.scene == "pump" and self.hand_mode == 1.),
                     float(self.scene == "pack" or self.factory and self.hand_mode == .85)].astype(np.float32)

    def teacher(self, observation):
        error = observation[6:9] - observation[:3]
        return np.clip(error * 4.4 - observation[3:6] * .58, -.94, .94)

    def _arm_ik(self, target, yaw):
        d = self.kin
        d.qpos[:] = self.data.qpos
        q = self.q_arm.copy()
        desired = yaw_quat(yaw)
        if self.holding or self.retracting:
            held_pose = np.empty(4)
            mj.mju_mulQuat(held_pose, desired, self.held_inverse)
            desired = held_pose
        jp, jr = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        rotation_error, current = np.zeros(3), np.empty(4)
        for _ in range(6):
            d.qpos[self.arm] = q
            mj.mj_kinematics(self.model, d)
            mj.mj_comPos(self.model, d)
            matrix = d.site_xmat[self.grip]
            mj.mju_mat2Quat(current, matrix)
            mj.mju_subQuat(rotation_error, desired, current)
            positional = target - d.site_xpos[self.grip]
            task = np.r_[positional, .068 * (matrix.reshape(3, 3) @ rotation_error)]
            if np.linalg.norm(task) < .002:
                break
            mj.mj_jacSite(self.model, d, jp, jr, self.grip)
            jac = np.vstack((jp[:, self.arm_dof], .068 * jr[:, self.arm_dof]))
            dq = np.linalg.solve(jac.T @ jac + np.eye(7)*.0018, jac.T @ task + .00022 * (ARM_HOME-q))
            q = np.clip(q + np.clip(dq, -.31, .31), self.limits[:, 0]+.004, self.limits[:, 1]-.004)
        self.q_arm = spread(q, self.q_arm, .075)

    def _physics(self):
        d, m = self.data, self.model
        walking = any(guide.frame is not None for guide in self.guides.values())
        if walking:
            before_pos, before_quat = d.mocap_pos.copy(), d.mocap_quat.copy()
            before_legs = d.ctrl[self.leg_actuators]
        goal = self.hand_goal
        if goal is None and self.holding:
            g = self.guides["g1"]
            attitude = quat_matrix(yaw_quat(g.yaw))
            goal = g.xyz + attitude @ [.301, -.175, .88]
        sample = None
        if goal is not None:
            obs = self.observation(goal)
            if len(obs) != FEATURES:
                raise ValueError("Unexpected policy input size")
            expected = self.teacher(obs)
            correction = expected + self.random.uniform(-.047, .047, 3) if self.capture else np.zeros(3)
            if self.policy is not None:
                correction = np.asarray(self.policy(obs), dtype=np.float64).reshape(3)
                if not np.all(np.isfinite(correction)):
                    raise TaskFault("策略产生无效动作")
                self.metrics["policyCalls"] += 1
            self._arm_ik(goal + CORRECTION*np.clip(correction, -1, 1), self.guides["g1"].yaw)
            if self.capture:
                sample = (obs, np.asarray(correction, dtype=np.float32), expected, goal.copy())
        else:
            self.q_arm = spread(ARM_HOME, self.q_arm, .058)

        offsets = self.foot_feedback.command(d, self.guides, self.random) if self.foot_feedback is not None else {}
        for key, robot in self.robot.items():
            self._move_legs(key, robot, offsets.get(key))
        arm_act = np.arange(22, 29)
        d.ctrl[arm_act] += np.clip(d.qfrc_bias[self.arm_dof], -10., 10.) / m.actuator_gainprm[arm_act, 0]
        d.ctrl[self.jaw_ctrl] = .040 if self.closed else 0.
        if walking:
            target_pos, target_quat = d.mocap_pos.copy(), d.mocap_quat.copy()
            target_legs = d.ctrl[self.leg_actuators]
            for u in (.2, .4, .6, .8, 1.):
                d.mocap_pos[:] = before_pos + u*(target_pos-before_pos)
                quat = before_quat + u*(target_quat-before_quat)
                d.mocap_quat[:] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
                d.ctrl[self.leg_actuators] = before_legs + u*(target_legs-before_legs)
                mj.mj_step(m, d)
        else:
            mj.mj_step(m, d, nstep=5)
        mj.mj_forward(m, d)
        self._read_contacts()
        collision = self.clearance.check(d)
        self.metrics["unexpectedContactFrames"] = self.clearance.frames
        if collision is not None:
            raise TaskFault(f"碰撞保护 · {collision}")
        if self.foot_feedback is not None:
            self.foot_feedback.sample(d)
            self.metrics["gaitPolicyCalls"] = self.foot_feedback.calls
        grip = d.site_xpos[self.grip].copy()
        self.hand_speed = .55 * self.hand_speed + .45 * (grip - self.last_grip) / DT
        self.last_grip = grip
        if sample is not None:
            obs, action, expected, goal = sample
            self.transitions.append((obs, action, expected, self.observation(goal)))
        if self.button is not None:
            self.metrics["buttonMm"] = max(self.metrics["buttonMm"], float(d.qpos[self.button]) * 1000)
        if self.door is not None:
            self.metrics["doorDegrees"] = max(self.metrics["doorDegrees"], math.degrees(d.qpos[self.door]))
        if self.survey is not None and self.ticks % 16 == 0:
            commissioning = self.factory and self.step < 0
            self.survey.scan("go2", show_sweep=bool(commissioning and self.ticks % 48 == 0))
            if not commissioning and self.guides["g1"].xyz[2] > 2.79:
                self.survey.scan("g1")
            self.metrics["mapUpdates"] = self.survey.revision

    def event(self, note, kind="info"):
        self.log.append({"time": round(self.time, 3), "text": note, "kind": kind, "step": self.step})

    def send(self, sender, receiver, signal, text, **payload):
        self.inbox[receiver, signal] = payload
        self.metrics["messages"] += 1
        self.event(text, "message")
        self.log[-1].update(sender=sender, receiver=receiver, signal=signal, payload=payload)

    def phase(self, number, note=None):
        self.step = number
        self.event(note or CATALOG[self.scene]["steps"][number])

    def pause(self, seconds):
        for _ in range(round(seconds/DT)):
            yield

    def wait(self, predicate, seconds, fault, stable=1):
        streak = 0
        for _ in range(round(seconds/DT)):
            yield
            streak = streak + 1 if predicate() else 0
            if streak >= stable:
                return
        raise TaskFault(fault)

    def follow_steps(self, key, frames):
        guide = self.guides[key]
        other = "go2" if key == "g1" else "g1"
        for frame in frames:
            waiting = 0
            while self.conflict(key, other, frame.ground):
                waiting += 1
                self.metrics["yieldFrames"] += 1
                if waiting == 1:
                    self.event(f"{'G1' if key == 'g1' else 'Go2'} 暂停让行 · 共用通道", "wait")
                if waiting > round(5./DT):
                    raise TaskFault("共用通道仍被占用，两台机器人已停止")
                yield
            guide.frame = frame
            guide.xyz[:] = frame.ground
            guide.yaw = frame.yaw
            guide.crouch = max(0., guide.crouch - .04)
            yield
        guide.frame = None

    def conflict(self, key, other, next_position):
        neighbor = self.guides[other]
        if abs(next_position[2]-neighbor.xyz[2]) >= .58:
            return False
        # The basket has a separate, slower rear approach with Go2 parked.
        gap = .535 if self.basket_access and key == "g1" and neighbor.frame is None else .72
        axis = next_position[:2]-neighbor.xyz[:2]
        if axis @ axis >= gap*gap:
            return False
        current = self.guides[key].xyz[:2]-neighbor.xyz[:2]
        return axis @ axis < current @ current - 1.e-5

    def drive(self, key, points, speed=None, yaw=0., direct=False):
        if key == "g1":
            yield from self.leave_access()
        guide = self.guides[key]
        planner = self.footsteps[key]
        cruise = speed or (.75 if key == "g1" else .90)
        for raw in points:
            point = np.array(raw, dtype=float)
            start = guide.xyz.copy()
            flight = PUMP_FLIGHT if abs(point[2] - start[2]) > .05 else None
            route = [point] if flight is not None else self.aisles.route(key, start, point)
            if route is None or direct and len(route) != 1:
                raise TaskFault(f"{key.upper()} 无可用通道 · 目标 ({point[0]:.2f}, {point[1]:.2f})")
            if len(route) > 1:
                self.metrics["aisleDetours"] += 1
            for waypoint in route:
                start = guide.xyz.copy()
                vector = waypoint[:2] - start[:2]
                if np.linalg.norm(vector) < .018:
                    continue
                heading = math.atan2(vector[1], vector[0])
                yield from self.turn(key, heading)
                yield from self.follow_steps(key, planner.walk(start, waypoint, guide.yaw, cruise, stairs=flight))
        if yaw is not None:
            yield from self.turn(key, yaw)
            yield from self.pause(.22)

    def turn(self, key, yaw):
        guide = self.guides[key]
        angle = squash(yaw - guide.yaw)
        if abs(angle) > .055:
            yield from self.follow_steps(key, self.footsteps[key].turn(guide.xyz.copy(), guide.yaw, angle))
        guide.yaw += squash(yaw - guide.yaw)

    def leave_access(self):
        if self.access_entry is None:
            return
        guide = self.guides["g1"]
        self.hand_goal = None
        start = guide.xyz.copy()
        yield from self.follow_steps("g1", self.footsteps["g1"].walk(start, self.access_entry, guide.yaw, .56))
        self.access_entry = None
        self.basket_access = False

    def approach(self, point, clearance=.385, basket=False):
        self.hand_goal = None
        guide = self.guides["g1"]
        lateral = .199 if basket else .166
        goal = np.array((point[0]-clearance, point[1]+lateral, 0.))
        if self.access_entry is None or np.linalg.norm(goal-guide.xyz) >= .11:
            entry = goal - [.38, 0., 0.]
            departure = self.access_entry if self.access_entry is not None else guide.xyz
            path = [entry]
            if (self.scene == "pump" and (not self.factory or self.order_index == 0)
                    and point[0] > LANE_EAST and abs(entry[0]-departure[0]) > .68):
                path = [(departure[0], -2.10, 0.), (entry[0], -2.10, 0.), entry]
            yield from self.drive("g1", path, speed=.76 if self.holding else .88)
            local_path = self.aisles.route("g1", entry, goal)
            if local_path is None or len(local_path) != 1:
                raise TaskFault("工作位前方净空不足，无法伸手")
            self.basket_access = basket
            yield from self.follow_steps("g1", self.footsteps["g1"].walk(entry, goal, guide.yaw, .38 if basket else .51))
            self.access_entry = entry
        guide.crouch = float(np.clip((.715-point[2])*3.60, 0., .95))
        yield from self.pause(.44)

    def raise_hand(self, wanted, seconds=.50):
        origin = self.data.site_xpos[self.grip].copy()
        target = np.array(wanted)
        for fraction in np.linspace(0., 1., max(2, round(seconds/DT))):
            fraction = fraction*fraction*(3-2*fraction)
            self.hand_goal = origin+(target-origin)*fraction
            yield

    def handle_point(self):
        return self.data.geom_xpos[self.handle].copy()

    def grasp(self, navigate=True):
        self.holding = self.closed = False
        self.hand_mode = .25
        target = self.handle_point()
        in_basket = self.basket_clip is not None and self.attachment == self.basket_clip
        if navigate:
            yield from self.approach(target, .403 if in_basket else .385, basket=in_basket)
        for attempt in range(3):
            self.closed = False
            target = self.handle_point()
            self.guides["g1"].crouch = .65 if in_basket else float(np.clip((.715-target[2])*3.60, 0., .97))
            if in_basket:
                yield from self.raise_hand(target + [-.213, 0., .119], .55)
                yield from self.raise_hand(target + [0., 0., .100], .63)
                yield from self.raise_hand(target, .59)
            else:
                yield from self.raise_hand(target + [-.082, 0., .022], .48)
            for _ in range(85):
                self.hand_goal = self.handle_point()
                yield
                if np.linalg.norm(self.data.site_xpos[self.grip] - self.handle_point()) < .021:
                    break
            self.closed = True
            trim = 0.
            for _ in range(75):
                self.hand_goal = self.handle_point() + [0., trim, 0.]
                yield
                if min(self.left_force, self.right_force) >= .10:
                    self.bind(self.grip_constraint, self.clamp)
                    self.holding = True
                    grip_quat = np.empty(4)
                    mj.mju_mat2Quat(grip_quat, self.data.site_xmat[self.grip])
                    mj.mju_mulQuat(self.held_inverse, self.data.xquat[self.part]*[1., -1., -1., -1.], grip_quat)
                    self.metrics["grasps"] += 1
                    self.event(f"夹爪承重 {self.left_force + self.right_force:.1f} N · 取件确认", "pass")
                    lift = self.data.site_xpos[self.grip].copy() + ([0., 0., .129] if in_basket else [-.025, 0., .096])
                    yield from self.raise_hand(lift, .49)
                    if in_basket:
                        self.metrics["basketPickups"] += 1
                        self.event(f"{self.cargo_name}已从筐口上方取出 · Go2 原地驻留", "pass")
                    self.guides["g1"].crouch = 0.
                    self.hand_goal = None
                    yield from self.pause(.25)
                    return
                if max(self.left_force, self.right_force) > .15:
                    trim = float(np.clip(trim + .0015*np.sign(self.left_force-self.right_force), -.016, .016))
            self.event(f"夹持反馈不充分 · 重新对位 {attempt+1}", "wait")
            self.closed = False
            yield from self.pause(.18)
        distance = float(np.linalg.norm(self.data.site_xpos[self.grip] - self.handle_point()))
        raise TaskFault(f"夹持接触未建立，手腕距离 {distance*1000:.0f} mm")

    def place(self, station, magnetic=False):
        point = np.array(self.layout["points"][station])
        if not self.holding:
            raise TaskFault("G1 当前夹爪未承载记录物")
        yield from self.approach(point)
        self.hand_mode = .62
        grip = self.data.site_xpos[self.grip]
        offset = grip-self.data.xpos[self.part]
        yield from self.raise_hand(point + offset + [0., 0., .090], .55)
        for fraction in np.linspace(1., 0., 43):
            offset = self.data.site_xpos[self.grip] - self.data.xpos[self.part]
            self.hand_goal = point + offset + [0., 0., .018+.081*fraction]
            yield
        for _ in range(120):
            self.hand_goal = point + self.data.site_xpos[self.grip] - self.data.xpos[self.part] + [0., 0., .014]
            yield
            error = self.data.xpos[self.part]-point
            if np.linalg.norm(error[:2]) < .048 and abs(error[2]) < .048:
                break
        else:
            raise TaskFault("物件未对准支座")
        if magnetic or self.scene == "store":
            for _ in range(60):
                self.hand_goal = point + self.data.site_xpos[self.grip] - self.data.xpos[self.part] + [0., 0., -.004]
                yield
                if self.seated(station):
                    if magnetic:
                        holder = f"magnet_{station}"
                        self.bind(holder, self.id("body", holder))
                    elif station == "target":
                        self.bind("locating_target", self.id("body", "locating_target"))
                    else:
                        self.bind("locating_reference", self.id("body", "scale_pan"))
                    break
            else:
                self.unbind()
        else:
            self.unbind()
        self.closed = self.holding = False
        self.retracting = True
        self.hand_goal = self.data.site_xpos[self.grip].copy()
        yield from self.pause(.19)
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy() + [-.015, 0., .048], .29)
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy() + [-.16, 0., .013], .48)
        support = self.id("geom", f"rest_{station}")
        yield from self.wait(lambda: self.seated(station),
                             2.8, "支座未取得稳定载荷", 10)
        if magnetic and self.attachment is None:
            self.bind(f"magnet_{station}", self.id("body", f"magnet_{station}"))
        elif station == "reference" and self.scene == "cnc":
            self.bind("locating_reference", self.id("body", "locating_reference"))
        elif station == "reference" and self.scene == "store" and self.attachment is None:
            self.bind("locating_reference", self.id("body", "scale_pan"))
        self.metrics["placements"] += 1
        self.event(f"支座承重 {self.support.get(support, 0.):.1f} N · 松手复核通过", "pass")
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy() + [-.13, 0., .083], .48)
        self.retracting = False
        if self.scene not in ("pack", "store") or station == "target" and self.scene == "store":
            self.hand_goal = None

    def seated(self, station):
        point = np.array(self.layout["points"][station])
        error = self.data.xpos[self.part]-point
        support = self.id("geom", f"rest_{station}")
        speed = float(np.linalg.norm(self.data.qvel[self.part_v:self.part_v+3]))
        upright = self.data.xmat[self.part, 8] > .84
        secured = self.attachment is not None and not self.holding
        return ((self.support.get(support, 0.) > .2 or secured) and upright and speed < .065
                and abs(error[0]) < .125 and abs(error[1]) < .105 and abs(error[2]) < .045)

    def measure(self, station, text, value, unit):
        surface = self.id("geom", f"rest_{station}")
        yield from self.wait(lambda: self.support.get(surface, 0.) > .18 or self.scene == "store" and self.seated(station), 2.8,
                             "测量位载荷中断", 32)
        if unit == "g":
            g = -self.model.opt.gravity[2]
            samples = []
            for _ in range(110):
                load = -PAN_STIFFNESS*self.data.qpos[self.cell] - PAN_MASS*g if self.cell is not None else self.support.get(surface, 0.)
                if self.seated(station) and abs(load - self.part_mass*g) < .12*self.part_mass*g:
                    samples.append(load)
                else:
                    samples.clear()
                if len(samples) >= 20:
                    break
                yield
            else:
                raise TaskFault("秤台读数未稳定")
            value = float(np.mean(samples))*1000/g
        if self.scene == "store":
            self.hand_goal = None
        if self.scene == "pack":
            self.bind("locating_target", self.id("body", "locating_target"))
            self.hand_goal = None
        self.metrics["samples"] += 1
        self.reading = {"name": text, "value": round(value, 2), "unit": unit}
        self.event(f"{text} {value:.2f} {unit}", "pass")

    def arrived(self, key, radius=.24):
        r = self.robot[key]["root"]
        return np.linalg.norm(self.data.qpos[r:r+2]-self.guides[key].xyz[:2]) < radius

    def parallel(self, *steps):
        active = list(steps)
        while active:
            remaining = []
            for work in active:
                if next(work, False) is not False:
                    remaining.append(work)
            active = remaining
            if active:
                yield

    def route_scan(self, locations, note):
        yield from self.drive("go2", locations)
        yield from self.wait(lambda: self.arrived("go2"), 1.1, "Go2 未到观察位", 10)
        self.event(note, "pass")

    def report_lanes(self, signal):
        for lane, y in LANES.items():
            yield from self.drive("go2", [(LANE_WEST, y, 0.)])
            self.survey.scan("go2", show_sweep=True)
            self.survey.observe_lane(lane)
        notation = " · ".join(f"{LANE_NAMES[key]}{'畅通' if state == 'clear' else '有障碍'}"
                              for key, state in self.survey.lanes.items())
        self.send("go2", "g1", signal, notation, revision=self.survey.revision,
                  lanes=dict(self.survey.lanes), landings=dict(self.survey.landings))

    def scout_pump(self):
        yield from self.drive("go2", [LOWER_VIEW], yaw=np.pi/2)
        self.survey.scan("go2", show_sweep=True)
        self.survey.observe_landings()
        yield from self.report_lanes("initial_map")
        yield from self.drive("go2", [UNDERPIPE_ENTRY, UNDERPIPE_VIEW], yaw=np.pi)
        markers = [name for name in PUMPS if self.survey.visible("go2", f"drip_{name}")]
        hot = [name for name in markers if self.model.geom(f"drip_{name}").rgba[0] > .85]
        if len(markers) != len(PUMPS) or len(hot) != 1:
            raise TaskFault(f"接液指示未完整取证 · 已确认 {len(markers)} 路")
        name = hot[0]
        geom = self.id("geom", f"drip_{name}")
        self.survey.finding = {"pump": name, "point": self.data.geom_xpos[geom].round(3).tolist(),
                               "source": self.survey.position("go2").round(3).tolist()}
        self.metrics["blindProof"] = len(markers)
        self.send("go2", "g1", "finding", f"{name} 的接液指示变色 · 已从管架下核验三路", pump=name)
        yield from self.drive("go2", [UNDERPIPE_ENTRY, COVER_CLEAR])

    def inspect_pump_cover(self):
        site = self.id("site", "door_touch")
        target = self.data.site_xpos[site].copy()
        yield from self.approach(target, .251)
        point = self.data.geom_xpos[self.id("geom", "cover_witness")]
        heading = math.atan2(point[1]-COVER_VIEW[1], point[0]-COVER_VIEW[0])
        yield from self.drive("go2", [COVER_VIEW], yaw=heading)
        occluder, _, _ = self.survey.cast("go2", point)
        if occluder != "door_panel":
            raise TaskFault("Go2 未确认指示片受观察盖遮挡")
        self.send("go2", "g1", "request_cover", "指示片被观察盖挡住 · 请求开盖", occluder=occluder)
        yield from self.wait(lambda: ("g1", "request_cover") in self.inbox, .7, "G1 未收到开盖请求")
        self.hand_mode = 1.
        yield from self.raise_hand(target + [-.066, 0., 0.], .48)
        for i in range(165):
            advance = min(i/16, 1.)
            advance = advance*advance*(3.-2.*advance)
            self.hand_goal = self.data.site_xpos[site] + [-.062+.154*advance, 0., 0.]
            yield
            if self.data.qpos[self.door] > .36 and self.metrics["maxProbeN"] > .15:
                break
        else:
            raise TaskFault(f"观察盖开启反馈不足 · {self.metrics['doorDegrees']:.1f}°")
        self.send("g1", "go2", "cover_open", "观察盖已打开 · 请从低位观察点取证",
                  angle=round(math.degrees(self.data.qpos[self.door]), 1))
        seen = 0
        for _ in range(125):
            self.hand_goal = self.data.site_xpos[site] + [.082, 0., 0.]
            yield
            if (("go2", "cover_open") in self.inbox and self.data.qpos[self.door] > .32
                    and self.survey.visible("go2", "cover_witness")):
                seen += 1
                if seen >= 8:
                    break
            else:
                seen = 0
        else:
            raise TaskFault("Go2 未在开盖窗口看到指示片")
        self.metrics["coverProof"] = 1
        self.metrics["samples"] += 1
        self.survey.evidence = {"point": point.round(3).tolist(), "source": self.survey.position("go2").round(3).tolist(),
                                "angle": round(math.degrees(self.data.qpos[self.door]), 1)}
        self.send("go2", "g1", "evidence", "已看到观察盖内的绿色指示片 · 正退出观察位", **self.survey.evidence)
        for _ in self.drive("go2", [COVER_CLEAR]):
            self.hand_goal = self.data.site_xpos[site] + [.082, 0., 0.]
            yield
        if not self.arrived("go2"):
            raise TaskFault("Go2 尚未退出观察区")
        self.send("go2", "g1", "clear_of_cover", "已离开观察区 · 可以复位")
        for _ in self.wait(lambda: ("g1", "evidence") in self.inbox and ("g1", "clear_of_cover") in self.inbox,
                           1., "G1 未收到取证和退场回执，观察盖保持打开"):
            self.hand_goal = self.data.site_xpos[site] + [.082, 0., 0.]
            yield
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy() + [-.21, 0., .11], .65)
        self.hand_goal = None
        yield from self.wait(lambda: self.data.qpos[self.door] < .09, 3., "观察盖未回位", 12)
        self.send("g1", "go2", "cover_closed", "观察盖已复位 · 前往交接位")
        yield from self.leave_access()

    def move_service_cart(self):
        start = self.data.mocap_pos[self.cart].copy()
        other = max(LANES.values(), key=lambda lane: abs(lane-start[1]))
        self.event("检修车换位 · 旧通道图已过期", "wait")
        self.survey.lanes.clear()
        for i in range(1, 64):
            fraction = i/63
            eased = fraction*fraction*(3.-2.*fraction)
            self.data.mocap_pos[self.cart] = [CART_X, start[1]+(other-start[1])*eased, 0.]
            yield
        self.metrics["cartMoves"] += 1

    def work_pump(self):
        self.phase(0)
        yield from self.parallel(self.scout_pump(), self.drive("g1", [(1.89, 2.69, 2.8)], speed=.62))
        alert = self.inbox.get(("g1", "finding"))
        lower = self.inbox.get(("g1", "initial_map"))
        if alert is None or lower is None:
            raise TaskFault("G1 在上层等待 Go2 的地图和接液指示")
        self.phase(1)
        pump = alert["pump"]
        plate = f"vfd_{pump}"
        x = self.data.geom_xpos[self.id("geom", plate)][0]
        yield from self.drive("g1", [(x, 2.49, 2.8)], yaw=np.pi/2)
        if not self.survey.visible("g1", plate):
            raise TaskFault("G1 未能从上层核验控制柜位号")
        self.metrics["verifiedPump"] = pump
        self.send("g1", "go2", "verified_pump", f"上层控制柜位号与低位证据一致 · 确认 {pump}", pump=pump)
        yield from self.drive("g1", [GATE_VIEW], yaw=np.pi)
        for flight in FLIGHTS:
            self.survey.observe_gate(flight)
        selected = next((flight for flight in FLIGHTS if self.survey.gates[flight] == "clear"
                         and lower["landings"].get(flight) == "clear"), None)
        if selected is None:
            raise TaskFault("没有上下口均已确认的检修梯")
        self.metrics["selectedFlight"] = selected
        self.metrics["reroutes"] += int(selected != "A")
        self.send("g1", "go2", "descent", f"上口与 Go2 下口观测一致 · 选用 {selected} 梯", flight=selected)
        y = FLIGHTS[selected]
        yield from self.drive("g1", [(.15, y, 2.8), (-5.37, y, 0.), (-5.60, y, 0.), (-5.60, -1.16, 0.)], speed=.74)
        self.phase(2)
        self.focus = "door"
        yield from self.inspect_pump_cover()
        self.phase(3)
        self.focus = "dock"
        yield from self.drive("go2", [self.layout["points"]["dock"]])
        yield from self.wait(lambda: self.arrived("go2"), 1.1, "Go2 未到交接位", 10)
        self.send("go2", "g1", "handoff_ready", "驻车完成 · 可从箩筐提取记录器")
        yield from self.wait(lambda: ("g1", "handoff_ready") in self.inbox, .7, "G1 未收到驻车回执")
        yield from self.grasp()
        yield from self.leave_access()
        self.send("g1", "go2", "handoff_done", "记录器已接收 · 请复核横向通道")
        self.phase(4)
        previous = next((key for key in LANES if lower["lanes"].get(key) == "clear"), None)
        self.metrics["previousLane"] = previous
        if self.move_cart:
            yield from self.move_service_cart()
        yield from self.drive("go2", [(-2.55, -2.23, 0.)])
        yield from self.report_lanes("transfer_map")
        yield from self.drive("go2", [DOG_WAIT])
        confirmed = self.inbox.get(("g1", "transfer_map"))
        lane = next((key for key in LANES if confirmed and confirmed["lanes"].get(key) == "clear"), None)
        if lane is None:
            raise TaskFault("两条横向通道的净空均未得到 Go2 确认")
        self.metrics["transferLane"] = lane
        if lane != previous:
            self.metrics["reroutes"] += 1
        left, right = (LANE_WEST, LANES[lane], 0.), (LANE_EAST, LANES[lane], 0.)
        self.survey.route = {"lane": lane, "flight": selected, "revision": confirmed["revision"],
                             "points": [left, right]}
        self.send("g1", "go2", "route_ack", f"接受地图 V{confirmed['revision']:02d} · 经{LANE_NAMES[lane]}携件",
                  lane=lane, revision=confirmed["revision"])
        yield from self.drive("g1", [left], speed=.71)
        yield from self.drive("g1", [right], speed=.71, direct=True)
        self.phase(5)
        self.target_station = self.focus = pump
        yield from self.place(pump, magnetic=True)
        yield from self.measure(pump, f"{pump} 模拟振动 RMS", 6.05+self.random.uniform(-.19, .19), "mm/s")
        yield from self.drive("g1", [(PUMPS[pump]-.84, -2.16, 0.)], speed=.68)
        self.send("g1", "go2", "complete", f"{pump} 记录器已固定 · 工单签收", pump=pump)

    def work_cnc(self):
        self.phase(0)
        yield from self.route_scan([(-3.20, -2.43, 0.)], "备料标签 T-114 与换型工单一致")
        self.phase(1)
        self.focus = "pick"
        yield from self.grasp()
        self.phase(2)
        self.focus = "reference"
        yield from self.place("reference")
        yield from self.measure("reference", "刀柄预调高度", 74.0+self.random.uniform(-.13, .13), "mm")
        yield from self.grasp()
        self.phase(3)
        self.focus = "target"
        yield from self.place("target")
        self.phase(4)
        yield from self.parallel(self.drive("g1", [(1.4, -.67, 0.)]),
                                 self.route_scan([(2.62, -2.43, 0.)], "T-114 已装入换型托盘"))

    def work_pack(self):
        self.phase(0)
        yield from self.route_scan([(-.44, -2.41, 0.)], "抽检小盒停在滚筒剔除支路")
        self.phase(1)
        point = np.array(self.layout["points"]["button"])
        self.focus = "button"
        yield from self.approach(point, .340)
        self.hand_mode = .85
        yield from self.raise_hand(point + [-.063, 0., 0.], .53)
        self.hand_goal = point + [.015, 0., 0.]
        yield from self.wait(lambda: self.data.qpos[self.button] > .015 and self.probe_force > .16,
                             2.45, "抽样确认按钮未达到有效行程", 5)
        self.event(f"按钮有效行程 {self.data.qpos[self.button]*1000:.1f} mm", "pass")
        yield from self.raise_hand(point + [-.12, 0., .07], .36)
        self.hand_goal = None
        self.phase(2)
        self.focus = "pick"
        yield from self.grasp()
        self.phase(3)
        self.focus = "target"
        yield from self.place("target")
        yield from self.measure("target", "复核称重", self.part_mass*1000, "g")
        self.phase(4)
        yield from self.parallel(self.drive("g1", [(1.45, -.52, 0.)]),
                                 self.route_scan([(1.93, -2.41, 0.)], "抽检支路清空 · 样品留在秤台"))

    def work_store(self):
        self.phase(0)
        yield from self.route_scan([(-3.17, -2.32, 0.), (2.85, -2.95, 0.)],
                                   "A-12 上的料盒标签指向 B-07")
        self.phase(1)
        self.focus = "pick"
        yield from self.grasp()
        self.phase(2)
        self.focus = "reference"
        yield from self.place("reference")
        yield from self.measure("reference", "料盒复核质量", self.part_mass*1000, "g")
        yield from self.grasp()
        self.phase(3)
        self.focus = "target"
        yield from self.place("target")
        self.phase(4)
        yield from self.parallel(self.drive("g1", [(1.44, -.73, 0.)]),
                                 self.route_scan([(2.63, -2.95, 0.), (-2.80, -2.95, 0.)],
                                                 "B-07 确认有料盒 · A-12 空位复扫"))

    def work_air(self):
        self.phase(0)
        yield from self.route_scan([(-2.67, -2.43, 0.)], "保养状态有效 · 检查隔离的观察罩")
        self.phase(1)
        self.focus = "door"
        site = self.id("site", "door_touch")
        target = self.data.site_xpos[site].copy()
        yield from self.approach(target, .251)
        self.hand_mode = 1.
        yield from self.raise_hand(target + [-.066, 0., 0.], .48)
        for i in range(165):
            advance = min(i/16, 1.)
            advance = advance*advance*(3.-2.*advance)
            self.hand_goal = self.data.site_xpos[site] + [-.062+.154*advance, 0., 0.]
            yield
            if self.data.qpos[self.door] > .36 and self.metrics["maxProbeN"] > .15:
                break
        else:
            raise TaskFault(f"观察罩开启反馈不足，最大角 {self.metrics['doorDegrees']:.1f}°")
        self.event(f"观察罩开启 {math.degrees(self.data.qpos[self.door]):.0f}°", "pass")
        self.phase(2)
        scan = self.drive("go2", [(-.46, -1.13, 0.), (.70, -.70, 0.)], yaw=np.pi/2)
        samples = 0
        for _ in range(650):
            self.hand_goal = self.data.site_xpos[site] + [.082, 0., 0.]
            next(scan, False)
            yield
            gap = np.linalg.norm(self.data.xpos[self.robot["go2"]["body"], :2] - [.70, -.70])
            settled = self.guides["go2"].frame is None
            samples = samples+1 if settled and gap < .23 and self.data.qpos[self.door] > .32 else 0
            if samples > 25:
                self.metrics["samples"] += 1
                self.reading = {"name": "过滤压差", "value": round(12.8+self.random.uniform(-.35, .35), 2), "unit": "kPa"}
                self.event("Go2 已在观察位复核压差指示", "pass")
                break
        else:
            raise TaskFault("Go2 未获得开罩条件下的有效观察")
        self.phase(3)
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy() + [-.21, 0., .11], .65)
        self.hand_goal = None
        yield from self.wait(lambda: self.data.qpos[self.door] < .09, 3., "观察罩未回位", 12)
        self.event("回弹铰链闭合反馈有效", "pass")
        self.phase(4)
        yield from self.parallel(self.drive("g1", [(-1.23, -.66, 0.)]),
                                 self.drive("go2", [(-1.31, -1.47, 0.)]))
        self.event("观察窗闭合 · 巡检记录签收", "pass")

    def tick(self):
        if self.status in ("done", "fault"):
            return False
        self.status = "running"
        try:
            next(self.program)
            self._physics()
        except StopIteration:
            if self.part is not None and (self.target_station is None or not self.seated(self.target_station)):
                self.status = "fault"
                self.event("工单结束时，物件未稳定留在目标支座", "fault")
            elif self.door is not None and self.data.qpos[self.door] >= .09:
                self.status = "fault"
                self.event("工单结束时，观察罩闭合反馈无效", "fault")
            else:
                self.status = "done"
            return False
        except TaskFault as exc:
            self.status = "fault"
            self.event(str(exc), "fault")
            return False
        self.ticks += 1
        limit = 1300 if self.factory else 230 if self.survey is not None else 130
        if self.ticks > round(limit/DT):
            self.status = "fault"
            self.event(f"工单超过 {limit} 秒仿真时长", "fault")
            return False
        return True

    def packet(self, from_log=0, map_revision=None):
        d = self.data
        actors = {}
        for key, robot in self.robot.items():
            root = robot["root"]
            q = robot["q"]
            actors[key] = {"pose": d.qpos[root:root+7].round(6).tolist(),
                           "joints": d.qpos[q:q+robot["n"]].round(6).tolist(),
                           "speed": round(float(np.linalg.norm(d.qvel[robot["rv"]:robot["rv"]+2])), 3),
                           "moving": self.guides[key].frame is not None}
        part = None if self.part is None else d.qpos[self.part_q:self.part_q+7].round(6).tolist()
        packet = {"ok": True, "scene": self.scene, "seed": self.seed, "status": self.status, "step": self.step,
                "t": round(self.time, 3), "actors": actors, "part": part, "focus": self.focus,
                "grip": [round(float(d.qpos[self.model.jnt_qposadr[joint]]), 5) for joint in self.jaws],
                "gripN": round(self.left_force+self.right_force, 2), "holding": self.holding,
                "button": 0. if self.button is None else float(d.qpos[self.button]),
                "door": 0. if self.door is None else float(d.qpos[self.door]),
                "reading": self.reading, "metrics": self.metrics,
                "events": self.log[from_log:], "eventCount": len(self.log)}
        if self.survey is not None:
            packet["survey"] = self.survey.snapshot(map_revision)
            if self.status == "running":
                detection = self.survey.go2_detection()
                if detection:
                    packet["survey"]["go2Detection"] = detection
            packet["cart"] = d.mocap_pos[self.cart].round(4).tolist()
        return packet
