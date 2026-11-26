"""An isolated MuJoCo experiment for G1 climbing a fixed ladder."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco as mj
import numpy as np


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
CONTROL_DT = .02
SUBSTEPS = 10
LADDER_X = .34
RUNG_BOTTOM = .26
RUNG_HALF_DEPTH = .038
TREAD_HALF = .018
RUNG_FACE = LADDER_X - RUNG_HALF_DEPTH
PALM_RADIUS = .021
PENETRATION_TOLERANCE = .0025
START_HAND_RUNG = 4
RESIDUAL_LIMIT = np.array([.018, .015, .019])
LIMBS = ("left_grip", "right_grip", "left_toe", "right_toe")
SUPPORT_GEOMS = ("left_palm", "right_palm", "left_sole", "right_sole")
RUNG_COUNT = 10
FLOOR_SUPPORT = -1
PLATFORM_SUPPORT = -2


@dataclass(frozen=True)
class Setup:
    spacing: float = .20
    friction: float = 1.05
    reach: float = -.086
    rounds: int = 3
    gait: str = "stepwise"
    floor: float = 0.
    rungs: int = RUNG_COUNT
    platform: tuple[float, float] | None = None


def ladder_boxes(setup):
    top = RUNG_BOTTOM + (setup.rungs-1)*setup.spacing + .1
    for side, y in (("left", .39), ("right", -.39)):
        yield f"{side}_rail", (LADDER_X, y, (top+setup.floor)/2), (.036, .026, (top-setup.floor)/2)
    for number in range(setup.rungs):
        yield f"rung_{number}", (LADDER_X, 0., RUNG_BOTTOM+number*setup.spacing), (.038, .385, TREAD_HALF)


def build_model(setup):
    xml = ET.parse(ROOT / "model/g1.xml").getroot()
    xml.set("model", "g1-ladder")
    xml.find("option").attrib.update(timestep=str(CONTROL_DT / SUBSTEPS), iterations="48")
    xml.remove(xml.find("keyframe"))
    world = xml.find("worldbody")
    sole_outline = ((-.088, -.029), (-.078, -.037), (.076, -.037), (.088, -.025),
                    (.088, .025), (.076, .037), (-.078, .037), (-.088, .029))
    sole_vertices = " ".join(f"{x} {y} {z}" for z in (-.014, .014) for x, y in sole_outline)
    ET.SubElement(xml.find("asset"), "mesh", name="ladder_sole", vertex=sole_vertices)
    # The original meshes remain visible and take part in ladder collisions.
    # The new sole plates replace the original four point contacts per foot.
    for geom in world.findall(".//geom"):
        if geom.get("class") == "foot":
            geom.attrib.update(contype="0", conaffinity="0", group="4")
        else:
            geom.attrib.update(contype="1", conaffinity="2", rgba=".68 .73 .77 1")
    surface = {"contype": "2", "conaffinity": "1", "priority": "2", "solref": ".004 1",
               "solimp": ".97 .998 .0005", "margin": ".0015",
               "friction": f"{setup.friction} .005 .0001"}
    ET.SubElement(world, "geom", name="floor", type="plane", pos=f"0 0 {setup.floor}", size="3 3 .1",
                  rgba=".27 .32 .36 1", **surface)
    for name, center, half in ladder_boxes(setup):
        color = ".80 .58 .22 1" if name.startswith("rung") else ".66 .39 .09 1"
        ET.SubElement(world, "geom", name=name, type="box", pos=" ".join(map(str, center)),
                      size=" ".join(map(str, half)), rgba=color, **surface)
    if setup.platform is not None:
        height, edge = setup.platform
        ET.SubElement(world, "geom", name="landing", type="box", pos=f"0 {edge+.85} {height-.08}",
                      size="1.25 .85 .08", rgba=".42 .48 .50 1", **surface)
    constraints = ET.SubElement(xml, "equality")
    cover = {"group": "3", "contype": "1", "conaffinity": "2", "rgba": ".38 .46 .50 1", "density": "0"}
    waist = world.find(".//body[@name='waist_roll_link']")
    ET.SubElement(waist, "geom", name="waist_housing", type="capsule",
                  fromto="0 0 -.05 0 0 .007", size=".039", **cover)
    for arm, side in enumerate(("left", "right")):
        wrist = world.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(wrist, "geom", name=f"{side}_palm_mount", type="capsule",
                      fromto=".022 0 0 .061 0 0", size=".012", **cover)
        ET.SubElement(wrist, "geom", name=f"{side}_palm", type="sphere", pos=".061 0 0",
                      size=str(PALM_RADIUS), **cover)
        ET.SubElement(wrist, "site", name=LIMBS[arm], pos=".061 0 0", size=".006", rgba=".29 .63 .83 1", group="5")
        ankle = world.find(f".//body[@name='{side}_ankle_roll_link']")
        ET.SubElement(ankle, "geom", name=f"{side}_sole", type="mesh", mesh="ladder_sole",
                      pos=".034 0 -.021", **cover)
        ET.SubElement(ankle, "site", name=f"{side}_toe", pos=".108 0 -.035", size=".006",
                      rgba=".23 .44 .60 1", group="5")
    for name in LIMBS:
        ET.SubElement(world, "site", name=f"{name}_target", size=".004", rgba="0 0 0 0",
                      pos=f"{LADDER_X} 0 {RUNG_BOTTOM}")
        ET.SubElement(constraints, "weld", name=f"{name}_hold",
                      site1=name, site2=f"{name}_target",
                      active="false", solref=".010 1" if name.endswith("_toe") else ".016 1",
                      torquescale=".36" if name.endswith("_toe") else ".17")
    ET.SubElement(world, "light", pos="-1.3 -2 3.0", dir="0.35 0.4 -1",
                  diffuse=".75 .75 .75", castshadow="true")
    ET.SubElement(world, "light", pos="-1 2 2.2", dir=".4 -.5 -.7", diffuse=".45 .48 .52")
    return mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


class ReferenceIK:
    def __init__(self, model, sites):
        self.model = model
        self.data = mj.MjData(model)
        self.feedback = mj.MjData(model)
        self.sites = sites
        self.q = model.qpos0.copy()
        self.dofs = (np.arange(21, 28), np.arange(28, 35), np.arange(6, 12), np.arange(12, 18))
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros_like(self.jac_pos)
        self.limits = model.jnt_range[1:]
        waist_pitch = model.joint("waist_pitch_joint").qposadr[0]
        self.q[waist_pitch] = .24
        for start in (6, 12):
            self.q[[start+1, start+4, start+5]] = [-.55, .9, -.35]
        for start in (21, 28):
            self.q[[start+1, start+4]] = [-1.25, .82]

    def refresh(self, data):
        mj.mj_kinematics(self.model, data)
        mj.mj_comPos(self.model, data)

    def adjust(self, data, limb, position, orientation_weight):
        site, dofs = self.sites[limb], self.dofs[limb]
        position_error = position-data.site_xpos[site]
        mj.mj_jacSite(self.model, data, self.jac_pos, self.jac_rot, site)
        rotation = data.site_xmat[site].reshape(3, 3)
        twist = .5*sum(np.cross(rotation[:, axis], np.eye(3)[axis]) for axis in range(3))
        jac = np.vstack((self.jac_pos[:, dofs], orientation_weight*self.jac_rot[:, dofs]))
        error = np.r_[position_error, orientation_weight*twist]
        update = jac.T @ np.linalg.solve(jac @ jac.T + .0003*np.eye(len(error)), error)
        lower, upper = self.limits[dofs-6].T
        data.qpos[dofs+1] = np.clip(data.qpos[dofs+1]+np.clip(update, -.19, .19), lower+.008, upper-.008)
        return np.linalg.norm(error)

    def solve(self, pelvis, sites):
        self.q[:3] = pelvis
        self.data.qpos[:] = self.q
        for _ in range(24):
            self.refresh(self.data)
            biggest = max(self.adjust(self.data, i, target, .18 if i >= 2 else .13)
                          for i, target in enumerate(sites))
            if biggest < .0016:
                break
        self.q[:] = self.data.qpos
        return self.q.copy()

    def moving_limb(self, reference, actual, limb, target, orientation_weight=None):
        indices = self.dofs[limb]+1
        self.feedback.qpos[:] = actual
        if orientation_weight is None:
            orientation_weight = .18 if limb >= 2 else .045
        for _ in range(17):
            self.refresh(self.feedback)
            if self.adjust(self.feedback, limb, target, orientation_weight) < .002:
                break
        self.refresh(self.feedback)
        reference[indices] = self.feedback.qpos[indices]
        return reference


def eased(t):
    return t*t*(3.-2.*t)


def transfer_target(start, end, phase, foot, platform=False):
    # Leave the front face before lifting past the next rung.
    if platform:
        clear_height = max(start[2], end[2]) + .085
        retract = np.array([end[0], start[1], start[2]+.022])
        above_edge = np.array([end[0], start[1], clear_height])
        above = np.array([end[0], end[1], clear_height])
        contact_goal = end - np.array([0., 0., .003])
        path = ((0., start), (.23, retract), (.48, above_edge), (.80, above), (1., contact_goal))
    elif foot:
        retracted = np.array([RUNG_FACE-.106, start[1], start[2]+.015])
        lifted = np.array([retracted[0], end[1], end[2]+.060])
        above = end + np.array([0., 0., .060])
        contact_goal = end - np.array([0., 0., .002])
        path = ((0., start), (.22, retracted), (.62, lifted), (.80, above), (1., contact_goal))
    else:
        offset = np.array([-.075, 0., 0.])
        contact_goal = end + np.array([.009, 0., 0.])
        path = ((0., start), (.20, start+offset), (.65, end+offset), (1., contact_goal))
    for (t0, a), (t1, b) in zip(path, path[1:]):
        if phase <= t1:
            return a + (b-a)*eased((phase-t0)/(t1-t0))
    return contact_goal


class Climb:
    def __init__(self, setup=Setup(), seed=17):
        self.setup = setup
        self.random = np.random.default_rng(seed)
        self.model = build_model(setup)
        self.data = mj.MjData(self.model)
        self.sites = np.array([self.model.site(name).id for name in LIMBS])
        self.anchors = [self.model.site(f"{name}_target").id for name in LIMBS]
        self.eq = [self.model.equality(f"{name}_hold").id for name in LIMBS]
        self.grip_geoms = [self.model.geom(name).id for name in SUPPORT_GEOMS]
        self.pads = {geom: leg for leg, geom in enumerate(self.grip_geoms[2:], 2)}
        self.rungs = {self.model.geom(f"rung_{i}").id: i for i in range(setup.rungs)}
        self.surfaces = set(self.rungs) | {self.model.geom(name).id for name in ("left_rail", "right_rail", "floor")}
        self.treads = {**self.rungs}
        if setup.platform is not None:
            landing = self.model.geom("landing").id
            self.surfaces.add(landing)
            self.treads[landing] = PLATFORM_SUPPORT
        self.ik = ReferenceIK(self.model, self.sites)
        self.bias_x = self.random.uniform(-.005, .005)
        self.hand_offsets = [.148, -.148]
        self.targets = np.array([self.rung_point(i, START_HAND_RUNG if i < 2 else 0) for i in range(4)])
        self.pelvis = np.array([setup.reach+self.bias_x, 0, RUNG_BOTTOM+TREAD_HALF+.586])
        self.ctrl_limits = self.model.actuator_ctrlrange[:, 1].copy()
        self.stiffness = np.r_[np.tile([250., 200., 120., 270., 175., 105.], 2),
                              [170., 210., 195.], np.tile([90., 90., 75., 95., 38., 32., 25.], 2)]
        self.damping = np.r_[np.tile([10., 9., 5., 10., 6., 5.], 2), [7., 8., 8.],
                            np.tile([4., 4., 3., 4., 2., 1.4, 1.1], 2)]
        self.jac = np.zeros((3, self.model.nv))
        self.rot = np.zeros_like(self.jac)
        self.load = np.zeros(4)
        self.support = [None, None]
        self.attached = [None]*4
        self.events = []
        self.frames = []
        self.traces = []
        self.calls = 0
        self.stats = {"max_hand_force_n": 0., "max_feet_force_n": 0.,
                      "max_pad_normal_force_n": 0., "unsupported_frames": 0,
                      "foot_swing_support_loss_frames": 0,
                      "max_ladder_penetration_mm": 0., "penetrating_substeps": 0}
        self.worst_collision = None

    def rung_point(self, limb, rung):
        if rung == FLOOR_SUPPORT:
            return self.floor_targets[limb]
        if rung == PLATFORM_SUPPORT:
            return self.platform_targets[limb-2]
        hand = limb < 2
        y = self.hand_offsets[limb] if hand else (.110 if limb == 2 else -.110)
        x = RUNG_FACE - PALM_RADIUS if hand else RUNG_FACE + .017
        return np.array([x, y, RUNG_BOTTOM + rung*self.setup.spacing + (0. if hand else TREAD_HALF)])

    def attach(self, limb, rung):
        goal = self.rung_point(limb, rung)
        mj.mj_fwdPosition(self.model, self.data)
        site = self.sites[limb]
        position = self.data.site_xpos[site]
        gap = float(np.linalg.norm(position - goal))
        fixture = {FLOOR_SUPPORT: "floor", PLATFORM_SUPPORT: "landing"}.get(rung, f"rung_{rung}")
        pair = {self.grip_geoms[limb], self.model.geom(fixture).id}
        touching = any({c.geom1, c.geom2} == pair and c.dist >= -PENETRATION_TOLERANCE
                       and (limb < 2 or abs(c.frame[2]) > .8) for c in self.data.contact)
        orientation = self.data.site_xmat[site]
        toe_tolerance = .018 if rung == PLATFORM_SUPPORT else .010
        on_tread = (limb < 2 or ((rung < 0 or position[0] >= RUNG_FACE+.006) and orientation[8] > .95
                                and abs(position[2]-goal[2]) < toe_tolerance))
        if rung == PLATFORM_SUPPORT:
            on_tread &= position[1] > self.setup.platform[1]+.046
        if gap > (.05 if rung < 0 else .025) or not touching or not on_tread:
            return False, gap
        self.lock(limb, rung, position, orientation, goal[2])
        return True, gap

    def lock(self, limb, rung, position, orientation, height):
        anchor = self.anchors[limb]
        self.model.site_pos[anchor] = position
        if limb < 2:
            mj.mju_mat2Quat(self.model.site_quat[anchor], orientation)
        else:
            self.model.site_pos[anchor, 2] = height + .001
            heading = np.arctan2(orientation[3], orientation[0])
            self.model.site_quat[anchor] = [np.cos(heading/2), 0., 0., np.sin(heading/2)]
        self.data.eq_active[self.eq[limb]] = True
        self.attached[limb] = rung

    def audit_contacts(self):
        deepest = 0.
        for contact in self.data.contact:
            a, b = contact.geom1, contact.geom2
            if a in self.surfaces:
                a, b = b, a
            if b not in self.surfaces or self.model.geom_bodyid[a] == 0:
                continue
            depth = -float(contact.dist)
            if depth > deepest:
                deepest = depth
            if depth*1000 > self.stats["max_ladder_penetration_mm"]:
                self.stats["max_ladder_penetration_mm"] = depth*1000
                self.worst_collision = {"time_s": round(float(self.data.time), 3),
                                        "robot": self.model.geom(a).name or self.model.body(self.model.geom_bodyid[a]).name,
                                        "surface": self.model.geom(b).name}
        if deepest > PENETRATION_TOLERANCE:
            self.stats["penetrating_substeps"] += 1

    def measure(self):
        self.load.fill(0.)
        self.support[:] = self.attached[2:]
        constraint_type = int(mj.mjtConstraint.mjCNSTR_EQUALITY)
        for limb, eq in enumerate(self.eq):
            rows = (self.data.efc_type[:self.data.nefc] == constraint_type) & (self.data.efc_id[:self.data.nefc] == eq)
            self.load[limb] = np.linalg.norm(self.data.efc_force[:self.data.nefc][rows][:3])
        wrench = np.empty(6)
        pad_force = 0.
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = contact.geom1, contact.geom2
            if g1 in self.treads:
                g1, g2 = g2, g1
            if g1 in self.pads and g2 in self.treads:
                mj.mj_contactForce(self.model, self.data, index, wrench)
                leg = self.pads[g1]
                normal = max(wrench[0], 0.)
                self.load[leg] += normal
                pad_force += normal
                if normal > 1. and self.support[leg-2] is None:
                    self.support[leg-2] = self.treads[g2]
        self.stats["max_hand_force_n"] = max(self.stats["max_hand_force_n"], float(max(self.load[:2])))
        self.stats["max_feet_force_n"] = max(self.stats["max_feet_force_n"], float(sum(self.load[2:])))
        self.stats["max_pad_normal_force_n"] = max(self.stats["max_pad_normal_force_n"], float(pad_force))
        if np.count_nonzero(self.load > 2.) < 2:
            self.stats["unsupported_frames"] += 1
        if self.setup.gait == "skip" and None in self.attached[2:]:
            supported = [index for index, rung in enumerate(self.attached) if rung is not None]
            if len(supported) < 3 or np.any(self.load[supported] <= 2.):
                self.stats["foot_swing_support_loss_frames"] += 1

    def observation(self, limb, goal, velocity, previous, phase):
        center = self.pelvis
        local_pos = self.data.site_xpos[self.sites[limb]] - center
        site_velocity = (self.data.site_xpos[self.sites[limb]] - previous) / CONTROL_DT
        dofs = self.ik.dofs[limb]
        arm = float(limb < 2)
        mass = float(self.model.body_subtreemass[1])
        return np.r_[local_pos, .20 * site_velocity, goal-center, .20*velocity,
                     self.data.qpos[dofs[:4]+1]*.35, self.load[limb]/(9.81*mass),
                     self.data.qvel[2]*.40, arm, float(limb%2), phase,
                     self.setup.spacing, self.data.qpos[0]-center[0]].astype(np.float32)

    def drive(self, q, limb=-1, residual=None):
        joint_reference = q[7:].copy()
        if residual is not None:
            dofs = self.ik.dofs[limb]
            mj.mj_jacSite(self.model, self.ik.feedback, self.jac, self.rot, self.sites[limb])
            jac = self.jac[:, dofs]
            correction = jac.T @ np.linalg.solve(jac @ jac.T + .003*np.eye(3), RESIDUAL_LIMIT*residual)
            joint_reference[dofs-6] += np.clip(correction, -.19, .19)
        for _ in range(SUBSTEPS):
            torque = (self.stiffness*(joint_reference-self.data.qpos[7:])
                      - self.damping*self.data.qvel[6:] + self.data.qfrc_bias[6:])
            self.data.ctrl[:] = np.clip(torque, -self.ctrl_limits, self.ctrl_limits)
            mj.mj_step(self.model, self.data)
            self.audit_contacts()
        self.measure()

    def record(self, label):
        self.frames.append((self.data.qpos.copy(), self.load.copy(),
                            list(self.support), self.data.eq_active.copy(), label))

    def hold(self, ticks, label, capture):
        target = self.ik.solve(self.pelvis, self.targets)
        for _ in range(ticks):
            self.drive(target)
            if capture:
                self.record(label)
            yield

    def shift_limb(self, limb, rung, rise, policy, collect, capture, *, stance=None):
        start = self.targets[limb].copy()
        end = self.rung_point(limb, rung)
        from_rung = self.attached[limb]
        advance = None if from_rung is None else rung-from_rung
        support = {name: self.attached[index]+1 for index, name in enumerate(LIMBS)
                   if index != limb and self.attached[index] is not None}
        self.data.eq_active[self.eq[limb]] = False
        self.attached[limb] = None
        trace = []
        previous = self.data.site_xpos[self.sites[limb]].copy()
        last_goal = start.copy()
        foot = limb >= 2
        fast = self.setup.gait == "skip"
        platform = rung == PLATFORM_SUPPORT
        ticks = 64 if limb >= 2 else 49
        settle, hold = 20, 11
        if fast:
            ticks = (78 if abs(advance) == 2 else 60) if foot else 42
            settle, hold = 24, 7
        if platform:
            ticks, settle, hold = 104, 34, 14
        locked = False
        gap = 0.
        label = f"{LIMBS[limb]} -> rung {rung+1}"
        initial_pelvis = self.pelvis.copy()
        for step in range(ticks+settle):
            phase = min((step+1)/ticks, 1.)
            if stance is not None:
                self.pelvis[:] = initial_pelvis + (stance-initial_pelvis)*eased(phase)
            elif rise:
                self.pelvis[2] = initial_pelvis[2] + rise*eased(phase)
            goal = transfer_target(start, end, phase, foot, platform)
            self.targets[limb] = goal
            velocity = (goal-last_goal)/CONTROL_DT
            q = self.ik.moving_limb(self.ik.solve(self.pelvis, self.targets), self.data.qpos, limb, goal,
                                    .52 if platform else None)
            current = self.data.site_xpos[self.sites[limb]].copy()
            obs = self.observation(limb, goal, velocity, previous, phase)
            teacher = np.clip((.66*(goal-current)+.032*(velocity-(current-previous)/CONTROL_DT))/RESIDUAL_LIMIT, -.95, .95)
            action = None
            if policy is not None:
                action = np.clip(policy(obs[None])[0], -1., 1.)
                self.calls += 1
            elif collect:
                action = np.clip(teacher*.80 + self.random.normal(0., .15, size=3), -1., 1.)
            self.drive(q, limb, action)
            if collect:
                following = self.observation(limb, goal, velocity, current, phase)
                trace.append((obs, np.asarray(action, dtype=np.float32), teacher.astype(np.float32), following))
            previous = current
            last_goal = goal
            if capture:
                self.record(label)
            yield
            if fast and phase == 1.:
                locked, gap = self.attach(limb, rung)
                if locked:
                    break
        self.targets[limb] = end
        if collect:
            self.traces.append(trace)
        if not fast:
            locked, gap = self.attach(limb, rung)
        retried = not locked
        if retried:
            locked, gap = yield from self.reseat(limb, rung, end, capture)
        self.events.append({"limb": LIMBS[limb], "from_rung": None if from_rung is None else from_rung+1,
                            "rung": rung+1, "rungs_advanced": advance, "support": support,
                            "gap_m": round(gap, 4), "retried": retried, "latched": locked,
                            "landed_at_s": round(float(self.data.time), 2) if locked else None})
        if not locked:
            return False
        yield from self.hold(hold, label, capture)
        return True

    def reseat(self, limb, rung, end, capture):
        label = f"{LIMBS[limb]} adjusting on rung {rung+1}"
        foot = limb >= 2
        for step in range(30):
            blend = eased(min((step+1)/19, 1.))
            stance = self.pelvis.copy()
            self.targets[limb] = end.copy()
            if rung < 0:
                self.targets[limb, 2] -= .010*blend
            else:
                stance[2 if foot else 0] += (.050 if foot else .040)*blend
                if foot:
                    self.targets[limb, 2] += .055*np.sin(np.pi*blend) + .021*blend
            q = self.ik.solve(stance, self.targets)
            self.drive(self.ik.moving_limb(q, self.data.qpos, limb, self.targets[limb],
                                          .52 if rung == PLATFORM_SUPPORT else None))
            if capture:
                self.record(label)
            yield
            if step >= 14:
                locked, _ = self.attach(limb, rung)
                if locked:
                    self.targets[limb] = end
                    return True, float(np.linalg.norm(self.data.site_xpos[self.sites[limb]]-end))
        self.targets[limb] = end
        return False, float(np.linalg.norm(self.data.site_xpos[self.sites[limb]]-end))

    def transfers(self, level):
        yield 0, START_HAND_RUNG+level, None
        yield 1, START_HAND_RUNG+level, None
        if self.setup.gait == "stepwise":
            yield 2, level, None
            yield 3, level, level
        else:
            leading = 2+(level-1)%2
            yield leading, level, level-.48
            if level == self.setup.rounds:
                yield 5-leading, level, level

    def ready(self):
        self.data.qpos[:] = self.ik.solve(self.pelvis, self.targets)
        mj.mj_forward(self.model, self.data)
        self.audit_contacts()
        for limb in range(4):
            if not self.attach(limb, START_HAND_RUNG if limb < 2 else 0)[0]:
                raise RuntimeError(f"Initial ladder support is out of reach: {LIMBS[limb]}")

    def move_stance(self, pelvis, targets, ticks):
        start_pelvis, start_targets = self.pelvis.copy(), self.targets.copy()
        for i in range(ticks):
            blend = eased((i+1)/ticks)
            self.pelvis[:] = start_pelvis + blend*(pelvis-start_pelvis)
            self.targets[:] = start_targets + blend*(targets-start_targets)
            self.drive(self.ik.solve(self.pelvis, self.targets))
            yield

    def mount(self, pose, prepared=None):
        # Align the standing guides before handing off to the free-base ladder model.
        if prepared is not None:
            if np.linalg.norm(pose[:3]-prepared[:3]) > .03 or abs(pose[3:7] @ prepared[3:7]) < .997:
                return False
            soles = []
            for geom in self.grip_geoms[2:]:
                mesh = self.model.geom_dataid[geom]
                base = self.model.mesh_vertadr[mesh]
                soles.append((geom, self.model.mesh_vert[base:base+self.model.mesh_vertnum[mesh]]))
            for tick in range(1, 22):
                self.data.qpos[:] = pose + eased(tick/21)*(prepared-pose)
                self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
                mj.mj_kinematics(self.model, self.data)
                lowest = min(self.data.geom_xpos[geom, 2]+np.min(vertices @ self.data.geom_xmat[geom, 6:9])
                             for geom, vertices in soles)
                self.data.qpos[2] += max(0., self.setup.floor-lowest)
                mj.mj_fwdPosition(self.model, self.data)
                self.audit_contacts()
                yield
            pose = prepared
        self.data.qpos[:] = pose
        mj.mj_forward(self.model, self.data)
        self.ground_pose = pose.copy()
        self.floor_targets = self.data.site_xpos[self.sites].copy()
        if np.max(np.abs(self.floor_targets[2:, 2]-self.setup.floor)) > .016:
            return False
        self.floor_targets[2:, 2] = self.setup.floor
        self.targets[:] = self.floor_targets
        destination = self.pelvis.copy()
        self.pelvis[:] = pose[:3]
        for limb in (2, 3):
            site = self.sites[limb]
            self.lock(limb, FLOOR_SUPPORT, self.floor_targets[limb], self.data.site_xmat[site], self.setup.floor)
        yield from self.move_stance(destination, self.floor_targets, 55)
        for limb in range(4):
            rung = START_HAND_RUNG if limb < 2 else 0
            if not (yield from self.shift_limb(limb, rung, 0., None, False, False)):
                return False
        return not self.stats["penetrating_substeps"]

    def dismount(self):
        for limb in (3, 2):
            if not (yield from self.shift_limb(limb, FLOOR_SUPPORT, 0., None, False, False)):
                return False
        for limb in (0, 1):
            self.data.eq_active[self.eq[limb]] = False
            self.attached[limb] = None
        yield from self.move_stance(self.ground_pose[:3], self.floor_targets, 60)
        return not self.stats["penetrating_substeps"]

    def step_onto_platform(self):
        height, edge = self.setup.platform
        self.platform_targets = np.array(((.15, edge+.19, height), (.173, edge+.082, height)))
        stance = np.array((-.070, .218, height+.505))
        if not (yield from self.shift_limb(2, PLATFORM_SUPPORT, 0., None, False, False, stance=stance)):
            return False
        for limb, lateral in ((0, .254), (1, .044)):
            self.hand_offsets[limb] = lateral
            if not (yield from self.shift_limb(limb, self.attached[limb], 0., None, False, False)):
                return False
        stance = np.array((-.068, .363, height+.565))
        if not (yield from self.shift_limb(3, PLATFORM_SUPPORT, 0., None, False, False, stance=stance)):
            return False
        for limb in (0, 1):
            self.data.eq_active[self.eq[limb]] = False
            self.attached[limb] = None
        standing = np.array((.023, edge+.155, height+.748))
        targets = self.targets.copy()
        targets[:2] = standing + np.array(((.095, .16, -.105), (.095, -.16, -.105)))
        yield from self.move_stance(standing, targets, 76)
        return (self.support == [PLATFORM_SUPPORT]*2 and min(self.load[2:]) > 2.
                and not self.stats["penetrating_substeps"])

    def ascent(self, policy=None, collect=False, capture=False):
        start_time, start_event = float(self.data.time), len(self.events)
        yield from self.hold(38, "Starting stance", capture)
        base_start = float(self.data.qpos[2])
        reference_start = self.pelvis[2]
        self.ascent_moves = []
        success = True
        count = 0
        for level in range(1, self.setup.rounds+1):
            for limb, rung, height in self.transfers(level):
                rise = 0. if height is None else reference_start + height*self.setup.spacing - self.pelvis[2]
                start = self.attached[limb]
                if not (yield from self.shift_limb(limb, rung, rise, policy, collect, capture)):
                    success = False
                    break
                self.ascent_moves.append((limb, start, rise))
            if not success:
                break
            count += 1
        yield from self.hold(24, "Final stance", capture)
        mj.mj_fwdPosition(self.model, self.data)
        self.audit_contacts()
        rise = float(self.data.qpos[2])-base_start
        stable = (success and self.attached[2:] == [self.setup.rounds]*2
                  and not self.stats["unsupported_frames"] and not self.stats["foot_swing_support_loss_frames"]
                  and not self.stats["penetrating_substeps"])
        passed = stable and rise > self.setup.spacing*self.setup.rounds-.060
        return {"passed": bool(passed), "gait": self.setup.gait, "rounds_completed": count, "pelvis_rise_m": round(rise,4),
                "rung_spacing_m": self.setup.spacing, "seconds": round(float(self.data.time)-start_time,2),
                "policy_calls": self.calls, "feet_final_rungs": [None if n is None else n+1 for n in self.support],
                "final_feet_force_n": np.round(self.load[2:],2).tolist(),
                **{key: round(value,2) for key,value in self.stats.items()},
                "worst_collision": self.worst_collision, "transfers": self.events[start_event:]}

    def descent(self):
        for limb, rung, rise in reversed(self.ascent_moves):
            if not (yield from self.shift_limb(limb, rung, -rise, None, False, False)):
                return False
        yield from self.hold(20, "Bottom stance", False)
        return self.attached == [START_HAND_RUNG]*2 + [0, 0] and not self.stats["penetrating_substeps"]

    def run(self, policy=None, collect=False, capture=False):
        self.ready()
        generator = self.ascent(policy, collect, capture)
        while True:
            try:
                next(generator)
            except StopIteration as result:
                return result.value


def inference(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return lambda obs: session.run(None, {"observation": obs})[0]


def save_trace(climb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    poses, loads, feet, grips, labels = zip(*climb.frames)
    np.savez_compressed(path, poses=poses, loads=loads,
                        feet=[[0 if n is None else n+1 for n in pair] for pair in feet],
                        grips=grips, labels=labels, spacing=climb.setup.spacing, dt=CONTROL_DT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--spacing", type=float, default=.20)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--gait", choices=("stepwise", "skip"), default="stepwise")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    climb = Climb(Setup(spacing=args.spacing, rounds=args.rounds, gait=args.gait), args.seed)
    result = climb.run(inference(args.policy) if args.policy else None, capture=args.trace is not None)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.trace:
        save_trace(climb, args.trace)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
