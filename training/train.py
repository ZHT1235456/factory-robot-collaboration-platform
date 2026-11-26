"""Train and verify an object-aware, reference-conditioned SuperMimic residual."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MUJOCO_GL"] = "disable"

import numpy as np


SCENES = ("pump", "cnc", "pack", "store", "air")
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SEEDS = (117, 128, 239, 340, 451, 562)
TEST_SEEDS = (17, 28, 103, 206, 407, 510)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect_one(request):
    from sim import Simulation, load_policy

    scene, seed, skill = request
    if skill == "gait":
        sim = Simulation(scene, seed, policy=load_policy(), capture_gait=True)
    else:
        sim = Simulation(scene, seed, capture=True)
    while sim.tick():
        pass
    traces = ([trace for legs in sim.foot_feedback.traces.values() for trace in legs]
              if skill == "gait" else [sim.transitions])
    arrays = [tuple(np.stack(points) for points in zip(*trace)) for trace in traces if trace]
    result = {"scene": scene, "seed": seed, "status": sim.status, "seconds": round(sim.time, 2),
              "samples": sum(len(trace) for trace in traces), "note": sim.log[-1]["text"]}
    return result, arrays


def collect(directory, workers, skill):
    directory.mkdir(parents=True, exist_ok=True)
    records, fragments, starts = [], [], []
    offset = 0
    with ProcessPoolExecutor(workers) as pool:
        for result, traces in pool.map(collect_one, [(s, n, skill) for s in SCENES for n in TRAIN_SEEDS]):
            records.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            for values in traces:
                obs, _, _, following = values
                neighbors = np.max(np.abs(obs[1:, :6] - following[:-1, :6]), axis=1) < 1.e-5
                if len(neighbors) >= 3:
                    starts.extend(np.flatnonzero(neighbors[:-2] & neighbors[1:-1] & neighbors[2:]) + offset)
                offset += len(obs)
                fragments.append(values)
    obs, action, teacher, following = (np.concatenate(parts) for parts in zip(*fragments))
    np.savez_compressed(directory / "transitions.npz", obs=obs, action=action, teacher=teacher,
                        next=following, starts=np.array(starts, dtype=np.int64))
    write_json(directory / "collection.json", records)
    print(f"{len(records)} trials, {len(obs)} measured transitions, {len(starts)} four-step windows", flush=True)


def architectures(torch, n):
    nn = torch.nn
    policy = nn.Sequential(nn.Linear(n, 80), nn.SiLU(), nn.Linear(80, 80), nn.SiLU(), nn.Linear(80, 3), nn.Tanh())
    world = nn.Sequential(nn.Linear(n+3, 128), nn.SiLU(), nn.Linear(128, 96), nn.SiLU(), nn.Linear(96, 8))
    return policy, world


def fit(directory, device, rounds, *, seeds=TRAIN_SEEDS):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(73614)
    with np.load(directory / "transitions.npz") as arrays:
        obs, action, teacher, following, starts = [
            torch.as_tensor(arrays[key], device=device) for key in ("obs", "action", "teacher", "next", "starts")]
    policy, world = [network.to(device) for network in architectures(torch, obs.shape[1])]
    wp = torch.optim.AdamW(world.parameters(), lr=.002)
    ap = torch.optim.AdamW(policy.parameters(), lr=.0018)
    dynamic = torch.tensor([0, 1, 2, 3, 4, 5, 16, 17], device=device)
    scale = torch.tensor([30., 30., 30., 8., 8., 8., 1., 1.], device=device)
    expected = (following[:, dynamic] - obs[:, dynamic]) * scale
    reports = []
    for cycle in range(3):
        world.requires_grad_(True)
        for _ in range(rounds):
            ii = torch.randint(len(obs), (512,), device=device)
            prediction = world(torch.cat((obs[ii], action[ii]), dim=1))
            world_loss = F.smooth_l1_loss(prediction, expected[ii])
            wp.zero_grad(set_to_none=True)
            world_loss.backward()
            wp.step()
        world.requires_grad_(False)
        for _ in range(rounds):
            ii = torch.randint(len(obs), (512,), device=device)
            imitation = F.mse_loss(policy(obs[ii]), teacher[ii])
            j = starts[torch.randint(len(starts), (256,), device=device)]
            state = obs[j]
            tracking = 0.
            for k in range(4):
                act = policy(state)
                delta = world(torch.cat((state, act), dim=1)) / scale
                prediction = state[:, dynamic] + delta
                desired = following[j+k, 6:9]
                tracking += (.86**k) * F.smooth_l1_loss((prediction[:, :3]-desired)*4., torch.zeros_like(desired))
                state = following[j+k].clone()
                state[:, dynamic] = prediction
            loss = imitation + .14 * tracking / 4
            ap.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.5)
            ap.step()
        record = {"round": cycle+1, "worldLoss": float(world_loss.detach().cpu()),
                  "teacherLoss": float(imitation.detach().cpu()), "rolloutLoss": float(tracking.detach().cpu())}
        reports.append(record)
        print(record, flush=True)
    checkpoint = {f"{name}.{key}": param.detach().cpu().numpy()
                  for name, model in (("actor", policy), ("world", world)) for key, param in model.state_dict().items()}
    np.savez_compressed(directory / "checkpoint.npz", **checkpoint)
    hardware = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
    write_json(directory / "training.json", {"method": "reference-conditioned actor, learned differentiable dynamics, "
                                                        "four-step imagined policy optimization",
                                            "hardware": hardware, "torch": torch.__version__, "seed": 73614,
                                            "input": list(obs.shape[1:]), "rolloutSteps": 4, "samples": len(obs),
                                            "windows": len(starts), "rounds": reports, "trainSeeds": list(seeds)})


def export(directory, output, skill):
    import onnx
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    arrays = np.load(directory / "transitions.npz")
    weights = np.load(directory / "checkpoint.npz")
    report = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    policy, world = architectures(torch, arrays["obs"].shape[1])
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    prefix = f"g1-{skill}" if skill in ("ladder", "ladder-skip") else (
        "foot-tracking" if skill == "gait" else "g1-contact")
    for key, network, value, filename, input_name, output_name in [
        ("actor", policy, arrays["obs"][:96], f"{prefix}-policy.onnx", "observation", "correction"),
        ("world", world, np.concatenate((arrays["obs"][:96], arrays["action"][:96]), axis=1),
         f"{prefix}-world.onnx", "observation_and_action", "scaled_dynamic_delta"),
    ]:
        network.load_state_dict({name: torch.from_numpy(weights[f"{key}.{name}"]) for name in network.state_dict()})
        network.eval()
        path = output / filename
        data = torch.from_numpy(value)
        torch.onnx.export(network, data, str(path), opset_version=17, input_names=[input_name],
                          output_names=[output_name], dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}})
        onnx.checker.check_model(onnx.load(str(path)))
        config = ort.SessionOptions()
        config.intra_op_num_threads = config.inter_op_num_threads = 1
        restored = ort.InferenceSession(str(path), config, providers=["CPUExecutionProvider"])
        max_error = float(np.max(np.abs(restored.run(None, {input_name: value})[0] - network(data).detach().numpy())))
        if max_error > 1.e-5:
            raise RuntimeError(f"ONNX parity failed for {filename}: {max_error}")
        result[key] = {"file": filename, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parityMaxError": max_error}
        print(result[key], flush=True)
    report["export"] = result
    report["collection"] = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    report["skill"] = skill
    report_name = "training.json" if skill == "contact" else f"{skill}-training.json"
    write_json(output / report_name, report)


def verify_one(request):
    from gait_policy import load_gait_policy
    from pump_survey import LANES, PumpVariant
    from sim import Simulation, load_policy
    from verify_gait import ContactAudit

    scene, seed, skill = request
    gait_policy = load_gait_policy() if skill == "gait" else None
    sim = Simulation(scene, seed, policy=load_policy(), gait_policy=gait_policy)
    audit = ContactAudit(sim) if scene == "pump" else None
    while sim.tick():
        if audit is not None:
            audit.update()
    result = {"scene": scene, "seed": seed, "status": sim.status,
              "seconds": round(sim.time, 2), "metrics": sim.metrics,
              "result": sim.reading, "last": sim.log[-1]["text"]}
    if sim.foot_feedback is not None:
        result["gaitCalls"] = sim.foot_feedback.calls
    pump_passed = True
    if audit is not None:
        variant = PumpVariant.from_seed(seed)
        flight = "b" if variant.closed_a else "a"
        other = "a" if variant.closed_a else "b"
        initial_lane, = (key for key in LANES if key != variant.cart_lane)
        selected_lane = variant.cart_lane if variant.move_cart else initial_lane
        result["stairs"] = {"openFlight": flight.upper(), "loadedSteps": sorted(audit.stair_steps[flight]),
                            "otherFlight": sorted(audit.stair_steps[other])}
        messages = {entry["signal"]: entry for entry in sim.log if "signal" in entry}
        signals = ("initial_map", "finding", "verified_pump", "descent", "request_cover", "cover_open",
                   "evidence", "clear_of_cover", "cover_closed", "handoff_ready", "handoff_done",
                   "transfer_map", "route_ack", "complete")
        floors = sim.survey.maps()
        result["survey"] = {"observedCells": {key: len(grid)-grid.count("0") for key, grid in floors.items()},
                            "initialMap": messages.get("initial_map", {}).get("payload"),
                            "transferMap": messages.get("transfer_map", {}).get("payload"),
                            "coverEvidence": sim.survey.evidence,
                            "timeline": {signal: entry["time"] for signal, entry in messages.items()}}
        if tuple(messages) == signals:
            first, last = (messages[signal]["payload"] for signal in ("initial_map", "transfer_map"))
            route = messages["route_ack"]["payload"]
            pump_passed = (len(audit.stair_steps[flight]) == 16 and not audit.stair_steps[other]
                           and sim.metrics["selectedFlight"] == flight.upper()
                           and sim.metrics["basketPickups"] == 1 and sim.metrics["blindProof"] == 3
                           and sim.metrics["verifiedPump"] == variant.alarm and sim.target_station == variant.alarm
                           and sim.metrics["coverProof"] == 1 and sim.metrics["doorDegrees"] > 20.
                           and sim.metrics["cartMoves"] == int(variant.move_cart)
                           and sim.metrics["previousLane"] == initial_lane and sim.metrics["transferLane"] == selected_lane
                           and first["lanes"][initial_lane] == last["lanes"][selected_lane] == "clear"
                           and first["lanes"][variant.cart_lane] == "blocked"
                           and last["revision"] >= first["revision"] and route["revision"] == last["revision"]
                           and route["lane"] == selected_lane and all(result["survey"]["observedCells"].values())
                           and "2" in floors["lower"] and messages["cover_open"]["time"] < messages["evidence"]["time"]
                           < messages["clear_of_cover"]["time"] < messages["cover_closed"]["time"])
        else:
            pump_passed = False
    counts = {"pump": (1, 1, 2), "cnc": (2, 2, 1), "pack": (1, 1, 1), "store": (2, 2, 1), "air": (0, 0, 1)}
    result["passed"] = (sim.status == "done" and sim.metrics["policyCalls"] > 0 and pump_passed and
                        sim.metrics["unexpectedContactFrames"] == 0 and
                        all(sim.metrics[name] >= count for name, count in zip(("grasps", "placements", "samples"), counts[scene])) and
                        (gait_policy is None or result["gaitCalls"] > 0) and
                        (scene != "pack" or sim.metrics["buttonMm"] >= 15.) and
                        (scene != "air" or sim.metrics["doorDegrees"] >= 25.))
    return result


def verify_missing_pump_receipt(request):
    from gait_policy import load_gait_policy
    from sim import Simulation, load_policy

    dropped, skill = request

    class LostReceipt(Simulation):
        def send(self, sender, receiver, signal, text, **payload):
            if signal != dropped:
                super().send(sender, receiver, signal, text, **payload)

    sim = LostReceipt("pump", 17, policy=load_policy(), gait_policy=load_gait_policy() if skill == "gait" else None)
    while sim.tick():
        pass
    signals = {entry.get("signal") for entry in sim.log}
    guarded = (sim.data.qpos[sim.door] > .32 and "cover_closed" not in signals if dropped == "clear_of_cover"
               else sim.survey.route is None and "route_ack" not in signals and sim.metrics["placements"] == 0)
    return {"dropped": dropped, "status": sim.status, "last": sim.log[-1]["text"],
            "passed": bool(sim.status == "fault" and guarded and sim.metrics["unexpectedContactFrames"] == 0)}


def verify(output, workers, seeds, skill, scenes):
    with ProcessPoolExecutor(workers) as pool:
        records = []
        for record in pool.map(verify_one, [(scene, seed, skill) for scene in scenes for seed in seeds]):
            records.append(record)
            print(f"{record['scene']} seed={record['seed']} {record['passed']} {record['seconds']}s {record['last']}", flush=True)
        faults = list(pool.map(verify_missing_pump_receipt, [(signal, skill) for signal in
                      ("clear_of_cover", "transfer_map")])) if "pump" in scenes else []
    report = {"passed": sum(r["passed"] for r in records), "total": len(records), "skill": skill,
              "trials": records, "faultTrials": faults}
    write_json(output / "validation.json", report)
    print(f"{report['passed']}/{report['total']} complete work orders", flush=True)
    for fault in faults:
        print(json.dumps(fault, ensure_ascii=False), flush=True)
    if report["passed"] != report["total"] or not all(fault["passed"] for fault in faults):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("collect", "fit", "export", "verify"))
    parser.add_argument("--run", type=Path, default=ROOT / ".training")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=360)
    parser.add_argument("--seeds", nargs="+", type=int, default=TEST_SEEDS)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--skill", choices=("contact", "gait"), default="contact")
    args = parser.parse_args()
    if args.phase == "collect":
        collect(args.run, args.workers, args.skill)
    elif args.phase == "fit":
        fit(args.run, args.device, args.rounds)
    elif args.phase == "export":
        export(args.run, args.output, args.skill)
    else:
        verify(args.output, args.workers, args.seeds, args.skill, args.scenes)
