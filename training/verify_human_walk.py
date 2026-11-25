"""Measure G1 joint posture and sole contact during a straight, unloaded walk."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MUJOCO_GL"] = "disable"


def walk_report(source):
    sys.path.insert(0, str(source.resolve()))
    from gait_policy import PAD_NAMES, load_gait_policy
    from sim import Simulation, load_policy, mj, np

    sim = Simulation("air", 17, policy=load_policy(), gait_policy=load_gait_policy())
    model, state = sim.model, sim.data
    info = sim.robot["g1"]
    root, q, v = (info[key] for key in ("root", "q", "rv"))
    start = sim.guides["g1"].xyz.copy()
    end = start + [5., 0., 0.]
    sim.program = sim.drive("g1", [end], speed=.78)
    slab = sim.id("geom", "slab")
    soles = [sim.id("site", f"{side}_sole") for side in ("left", "right")]
    arms = [[sim.id("body", f"{side}_{link}_link") for link in ("shoulder_pitch", "elbow", "wrist_yaw")]
            for side in ("left", "right")]
    pads = {sim.id("geom", name): (foot, pad // 2) for foot, names in enumerate(PAD_NAMES["g1"])
            for pad, name in enumerate(names)}
    air = [0, 0]
    last_toe = [False, False]
    force = np.zeros(6)
    knees, swing_knees, elbows, hips, arm_difference, speeds = [], [], [], [], [], []
    supported = moving = landings = heel_strikes = toe_offs = 0
    while sim.tick():
        load = np.zeros((2, 2))
        for i, contact in enumerate(state.contact):
            a, b = contact.geom1, contact.geom2
            if a == slab:
                a, b = b, a
            if b == slab and a in pads:
                mj.mj_contactForce(model, state, i, force)
                load[pads[a]] += force[0]
        speed = float(state.qvel[v])
        progress = float(state.qpos[root] - start[0])
        cruising = .38 < progress < 4.62 and speed > .38
        if cruising:
            moving += 1
            supported += int(np.max(load.sum(axis=1)) >= 2.)
            for arm in arms:
                shoulder, elbow, wrist = state.xpos[arm]
                upper, lower = elbow-shoulder, wrist-elbow
                bend = upper @ lower / (np.linalg.norm(upper)*np.linalg.norm(lower))
                elbows.append(math.degrees(math.acos(float(np.clip(bend, -1., 1.)))))
            hips.append(float(state.qpos[root+2]))
            speeds.append(speed)
            arm_difference.append(float(np.degrees(state.qpos[q+15] - state.qpos[q+22])))
        for side, sole in enumerate(soles):
            back, front = load[side]
            pitch = -math.degrees(math.asin(float(np.clip(state.site_xmat[sole, 6], -1., 1.))))
            knee = float(np.degrees(state.qpos[q+3+6*side]))
            if back + front >= 2.:
                if air[side] >= 3 and cruising:
                    landings += 1
                    heel_strikes += int(back >= 2. and front < 1. and pitch < -2.)
                air[side] = 0
                last_toe[side] = front >= 2. and back < 1. and pitch > 3.
                if cruising and abs(state.site_xpos[sole, 0]-state.qpos[root]) < .09:
                    knees.append(knee)
            else:
                air[side] += 1
                if cruising:
                    if air[side] == 3 and last_toe[side]:
                        toe_offs += 1
                    swing_knees.append(knee)

    def percentile(values, level):
        return round(float(np.percentile(values, level)), 3) if len(values) else None

    return {
        "state": sim.status, "timeSeconds": sim.time, "distanceM": round(float(state.qpos[root]-start[0]), 3),
        "cruiseFrames": moving, "supportedPct": round(100*supported/max(moving, 1), 2),
        "medianStanceKneeDeg": percentile(knees, 50), "swingKneeP90Deg": percentile(swing_knees, 90),
        "medianElbowFlexionDeg": percentile(elbows, 50), "medianPelvisM": percentile(hips, 50),
        "opposedArmDifferenceP90Deg": percentile(np.abs(arm_difference), 90),
        "medianForwardMps": percentile(speeds, 50), "landings": landings, "heelFirstLandings": heel_strikes,
        "toeOffs": toe_offs, "unexpectedContactFrames": sim.metrics["unexpectedContactFrames"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = walk_report(args.source)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
