"""Measured foot tracking corrections, evaluated as one ONNX batch per step."""

from functools import lru_cache
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

from gait import GO2_FEET, G1_SOLES


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
ACTION_LIMIT = np.array([.026, .023, .017])
PAD_NAMES = {
    "g1": tuple(tuple(f"{side}_pad_{i}" for i in range(4)) for side in ("left", "right")),
    "go2": tuple(("go_" + side,) for side in GO2_FEET),
}


@lru_cache(maxsize=1)
def load_gait_policy():
    import onnxruntime as ort

    path = ROOT / "results/foot-tracking-policy.onnx"
    cfg = ort.SessionOptions()
    cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), cfg, providers=["CPUExecutionProvider"])
    return lambda obs: session.run(None, {"observation": obs})[0]


class FootFeedback:
    def __init__(self, model, data, robots, dt, policy=None, record=False):
        self.model = model
        self.robots = robots
        self.dt = dt
        self.policy = policy
        self.record = record
        self.soles = [model.site(name).id for name in G1_SOLES]
        self.paws = [model.geom("go_" + name).id for name in GO2_FEET]
        self.contacts = {model.geom(name).id: (key, i) for key, pads in PAD_NAMES.items()
                         for i, foot in enumerate(pads) for name in foot}
        self.floor = {i for i in range(model.ngeom)
                      if (model.geom(i).name in ("slab", "walkway") or model.geom(i).name.startswith("tread_"))}
        self.loads = {key: np.zeros(len(feet)) for key, feet in PAD_NAMES.items()}
        self.actual = self.measure_positions(data)
        self.velocity = {key: np.zeros_like(points) for key, points in self.actual.items()}
        self.reference = {}
        self.trim = {key: np.zeros_like(points) for key, points in self.actual.items()}
        self.traces = {key: [[] for _ in points] for key, points in self.actual.items()}
        self.pending = {}
        self.force = np.zeros(6)
        self.calls = 0

    def measure_positions(self, data):
        return {"g1": data.site_xpos[self.soles].copy(), "go2": data.geom_xpos[self.paws].copy()}

    def observe(self, data, key, frame, planned_velocity):
        robot = self.robots[key]
        basis = data.xmat[robot["body"]].reshape(3, 3)
        root = data.xpos[robot["body"]]
        points = self.actual[key]
        out = np.zeros((len(points), 23), dtype=np.float32)
        out[:, :3] = (points - root) @ basis
        out[:, 3:6] = self.velocity[key] @ basis * .20
        out[:, 6:9] = (frame.feet - root) @ basis
        out[:, 9:12] = planned_velocity @ basis * .20
        width = 6 if key == "g1" else 3
        q = robot["q"]
        out[:, 12:12+min(width, 4)] = data.qpos[q:q+12].reshape(len(points), width)[:, :4] * .35
        out[:, 16] = self.loads[key] / (120. if key == "g1" else 35.)
        vel = robot["rv"]
        out[:, 17] = np.linalg.norm(data.qvel[vel:vel+2]) * .40
        out[:, 18] = float(key == "g1")
        out[:, 19] = frame.support
        out[:, 20] = float(frame.ground[2] > .06)
        out[:, 21] = np.linspace(-1., 1., len(points))
        out[:, 22] = data.qvel[vel+5] * .25
        return out, basis

    @staticmethod
    def teacher(obs):
        support = obs[:, 19:20]
        error = obs[:, 6:9] - obs[:, :3]
        velocity_error = (obs[:, 9:12] - obs[:, 3:6]) / .20
        gain = .25 + .38*support
        adjustment = gain*error + (.005 + .012*support)*velocity_error
        # Do not drive through the floor to erase a nominal preload.
        adjustment[:, 2] = np.maximum(adjustment[:, 2], -.009)
        return np.clip(adjustment / ACTION_LIMIT, -.95, .95).astype(np.float32)

    def command(self, data, guides, random):
        commands = {}
        for key, guide in guides.items():
            frame = guide.frame
            if frame is None:
                self.reference.pop(key, None)
                self.trim[key].fill(0.)
                continue
            last = self.reference.get(key, frame.feet)
            velocity = np.clip((frame.feet - last) / self.dt, -4., 4.)
            self.reference[key] = frame.feet
            obs, basis = self.observe(data, key, frame, velocity)
            commands[key] = (frame, velocity, obs, basis)
        if not commands:
            return {}
        observations = np.concatenate([item[2] for item in commands.values()])
        teacher = self.teacher(observations)
        if self.policy is not None:
            action = self.policy(observations)
            if action.shape != teacher.shape or not np.isfinite(action).all():
                raise ValueError("落脚修正策略输出无效")
            self.calls += 1
        else:
            action = teacher
            if self.record:
                action = action + random.uniform(-.10, .10, teacher.shape).astype(np.float32)
        offset = 0
        corrections = {}
        for key, (frame, velocity, obs, basis) in commands.items():
            end = offset+len(obs)
            act = np.clip(action[offset:end], -1., 1.)
            shift = (act * ACTION_LIMIT) @ basis.T
            self.trim[key][:] = .48*self.trim[key] + .52*shift
            corrections[key] = self.trim[key]
            if self.record:
                self.pending[key] = (frame, velocity, obs, act, teacher[offset:end])
            offset = end
        return corrections

    def sample(self, data):
        positions = self.measure_positions(data)
        for key, feet in positions.items():
            self.velocity[key][:] = (feet - self.actual[key]) / self.dt
            self.loads[key].fill(0.)
        self.actual = positions
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = contact.geom1, contact.geom2
            if a in self.floor:
                a, b = b, a
            if b in self.floor and a in self.contacts:
                key, leg = self.contacts[a]
                mj.mj_contactForce(self.model, data, i, self.force)
                self.loads[key][leg] += max(0., self.force[0])
        for key, (frame, velocity, obs, action, teacher) in self.pending.items():
            following, _ = self.observe(data, key, frame, velocity)
            for i, trace in enumerate(self.traces[key]):
                trace.append((obs[i], action[i], teacher[i], following[i]))
        self.pending.clear()
