"""Measure physical foot contact, ground-relative slip and heading on work orders."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MUJOCO_GL"] = "disable"

import mujoco as mj
import numpy as np

from gait_policy import PAD_NAMES, load_gait_policy
from sim import DT, Simulation, load_policy, quat_matrix


class ContactAudit:
    def __init__(self, sim):
        self.sim = sim
        self.ids = {key: [[sim.id("geom", name) for name in foot] for foot in feet]
                    for key, feet in PAD_NAMES.items()}
        self.floor = {}
        self.tread_names = {}
        for geom in range(sim.model.ngeom):
            if sim.model.geom_bodyid[geom] != 0:
                continue
            name = mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_GEOM, geom) or ""
            location = sim.model.geom_pos[geom]
            size = sim.model.geom_size[geom]
            stair = (sim.scene == "pump" and sim.model.geom_type[geom] == mj.mjtGeom.mjGEOM_BOX
                     and abs(size[0] - .151) < .002 and abs(size[1] - .57) < .002
                     and abs(location[2] - size[2]) < .001)
            landing = sim.scene == "pump" and abs(size[0]-3.05) < .001 and abs(location[2]-2.72) < .001
            if name in ("slab", "walkway") or name.startswith("tread_") or stair or landing:
                self.floor[geom] = round(location[2] + size[2], 3)
            if name.startswith("tread_"):
                _, flight, level = name.split("_")
                self.tread_names[geom] = (flight, int(level))
        self.foot_ids = set(g for feet in self.ids.values() for foot in feet for g in foot)
        self.g1_pads = {pad for foot in self.ids["g1"] for pad in foot}
        self.stair_steps = {"a": set(), "b": set()}
        self.previous = sim.data.geom_xpos.copy()
        self.motion = {key: [] for key in self.ids}
        self.flat_slip = {key: [] for key in self.ids}
        self.stair_slip = {key: [] for key in self.ids}
        self.headings = {key: [] for key in self.ids}
        self.treads = {key: set() for key in self.ids}
        self.force = np.zeros(6)

    def update(self):
        data = self.sim.data
        loads = {}
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = contact.geom1, contact.geom2
            if b in self.foot_ids and a in self.floor:
                a, b = b, a
            if a in self.foot_ids and b in self.floor:
                mj.mj_contactForce(self.sim.model, data, i, self.force)
                value, height = loads.get(a, (0., 0.))
                loads[a] = (value + self.force[0], max(height, self.floor[b]))
                if b in self.tread_names and a in self.g1_pads and self.force[0] > 1.:
                    flight, step = self.tread_names[b]
                    self.stair_steps[flight].add(step)

        for key, feet in self.ids.items():
            info = self.sim.robot[key]
            horizontal = data.qvel[info["rv"]:info["rv"]+2]
            speed = float(np.linalg.norm(horizontal))
            if speed > .10:
                q = data.qpos[info["root"]+3:info["root"]+7]
                direction = quat_matrix(q)[:2, 0]
                agreement = np.clip(direction @ horizontal / speed, -1, 1)
                self.headings[key].append(float(np.degrees(np.arccos(agreement))))
            turning = abs(data.qvel[info["rv"]+5]) > .17
            if speed <= .10 and not turning:
                continue

            supported = 0
            for foot in feet:
                load = sum(loads.get(pad, (0., 0.))[0] for pad in foot)
                if load < 2.0:
                    continue
                supported += 1
                drift = 0.
                for pad in foot:
                    weight, height = loads.get(pad, (0., 0.))
                    if height > 0. and weight > 1.:
                        self.treads[key].add(height)
                    drift += weight * np.linalg.norm(data.geom_xpos[pad, :2] - self.previous[pad, :2]) / DT
                if any(0. < loads.get(pad, (0., 0.))[1] < 2.8 for pad in foot):
                    if speed > .10:
                        self.stair_slip[key].append(float(drift/load))
                elif speed > .10:
                    self.flat_slip[key].append(float(drift/load))
            self.motion[key].append(supported)
        self.previous[:] = data.geom_xpos

    def report(self):
        def quantile(values, q, factor=1.):
            return round(float(np.percentile(values, q)*factor), 3) if values else None

        result = {}
        for key in self.ids:
            motion = np.asarray(self.motion[key])
            flat = self.flat_slip[key]
            stairs = self.stair_slip[key]
            result[key] = {
                "motionFrames": len(motion),
                "supportedPct": (round(100*float(np.mean(motion >= (1 if key == "g1" else 2))), 2)
                                 if len(motion) else None),
                "flatContactSamples": len(flat),
                "flatSlipMedianCmS": quantile(flat, 50, 100),
                "flatSlipP90CmS": quantile(flat, 90, 100),
                "stairSlipP90CmS": quantile(stairs, 90, 100),
                "headingErrorP90Deg": quantile(self.headings[key], 90),
                "treadLevels": sorted(self.treads[key]),
            }
        return result


def evaluate(case):
    scene, seed, controller = case
    gait_policy = load_gait_policy() if controller == "onnx" else None
    sim = Simulation(scene, seed, policy=load_policy(), gait_policy=gait_policy, gait_teacher=controller == "teacher")
    audit = ContactAudit(sim)
    while sim.tick():
        audit.update()
    report = {"scene": scene, "seed": seed, "state": sim.status, "seconds": round(sim.time, 2),
            "gaitCalls": 0 if sim.foot_feedback is None else sim.foot_feedback.calls,
            "last": sim.log[-1]["text"], "contact": audit.report()}
    if scene == "pump":
        report["stairSteps"] = {flight: sorted(steps) for flight, steps in audit.stair_steps.items()}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=("pump", "cnc", "pack", "store", "air"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 28))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--controller", choices=("planned", "teacher", "onnx"), default="planned")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [(scene, seed, args.controller) for scene in args.scenes for seed in args.seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        reports = list(pool.map(evaluate, cases))
    result = {"sampleSeconds": DT, "controller": args.controller, "trials": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        compact = {key: [info["flatSlipMedianCmS"], info["flatSlipP90CmS"], info["headingErrorP90Deg"],
                         info["supportedPct"], len(info["treadLevels"])]
                   for key, info in report["contact"].items()}
        print(report["scene"], report["seed"], report["state"], report["seconds"], compact, flush=True)


if __name__ == "__main__":
    main()
