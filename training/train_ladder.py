"""Collect simulated G1 ladder transitions and train a SuperMimic-style residual."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from train import export, fit, write_json

import numpy as np

from ladder import CONTROL_DT, PENETRATION_TOLERANCE, SUBSTEPS, Climb, Setup, inference, save_trace


TRAIN_SEEDS = tuple(range(341, 373))
TEST_SEEDS = (81, 236, 498, 575, 602, 651, 729, 804, 933, 1120, 1224, 1449)


def setup_for(seed, gait):
    random = np.random.default_rng(seed)
    return Setup(spacing=float(random.uniform(.191, .207)),
                 friction=float(random.uniform(.81, 1.23)),
                 reach=float(random.uniform(-.095, -.078)), gait=gait)


def collect_trial(request):
    seed, gait = request
    climb = Climb(setup_for(seed, gait), seed)
    trial = climb.run(collect=True)
    traces = [tuple(np.stack(component) for component in zip(*trace)) for trace in climb.traces]
    trial["seed"] = seed
    return trial, traces


def collect(directory, workers, gait):
    directory.mkdir(parents=True, exist_ok=True)
    trials, fragments, starts = [], [], []
    offset = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for trial, traces in pool.map(collect_trial, ((seed, gait) for seed in TRAIN_SEEDS)):
            trials.append(trial)
            print(f"seed {trial['seed']}: {trial['rounds_completed']} levels, {len(traces)} transfers", flush=True)
            for trace in traces:
                count = len(trace[0])
                starts.extend(range(offset, offset+count-3))
                offset += count
                fragments.append(trace)
    if not starts or not any(trial["passed"] for trial in trials):
        raise RuntimeError("Ladder collection produced no complete reference trial")
    obs, action, teacher, following = (np.concatenate(columns) for columns in zip(*fragments))
    np.savez_compressed(directory / "transitions.npz", obs=obs, action=action, teacher=teacher,
                        next=following, starts=np.array(starts, dtype=np.int64))
    write_json(directory / "collection.json", trials)
    print(f"{len(trials)} trials, {len(obs)} measured transitions, {len(starts)} four-step windows", flush=True)


def check_seed(request):
    policy_path, gait, seed = request
    policy = inference(policy_path)
    setup = setup_for(seed, gait)
    return {"seed": seed, "setup": vars(setup), "baseline": Climb(setup, seed).run(),
            "trained": Climb(setup, seed).run(policy)}


def benchmark(output, trace_path, workers, gait, skill, stepwise_policy):
    policy_path = output / f"g1-{skill}-policy.onnx"
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for pair in pool.map(check_seed, ((policy_path, gait, seed) for seed in TEST_SEEDS)):
            results.append(pair)
            print(f"seed {pair['seed']}: baseline {pair['baseline']['rounds_completed']}, "
                  f"policy {pair['trained']['rounds_completed']}", flush=True)
    demo = Climb(Setup(gait=gait), 17)
    result = demo.run(inference(policy_path), capture=True)
    save_trace(demo, trace_path)
    summary = {key: sum(trial[key]["passed"] for trial in results) for key in ("baseline", "trained")}
    report = {"description": "Held-out simulations with varied rung spacing, grip stance and contact friction",
              "collision_audit": {"rate_hz": SUBSTEPS/CONTROL_DT,
                                  "allowed_penetration_mm": PENETRATION_TOLERANCE*1000},
              "seeds": TEST_SEEDS, "complete": summary, "demo": result, "trials": results}
    if stepwise_policy is not None:
        standard = Climb(Setup(), 17).run(inference(stepwise_policy))
        if not standard["passed"]:
            raise RuntimeError("The stepwise policy did not complete the comparison climb")
        report["stepwise_comparison"] = standard
    write_json(output / f"{skill}-validation.json", report)
    print(f"Held-out ladder trials {summary}, demo rise {result['pelvis_rise_m']:.3f} m", flush=True)
    if not result["passed"]:
        raise RuntimeError("The trained policy did not complete the standard ladder demo")
    if gait == "skip" and not any(move["rungs_advanced"] == 2 and move["latched"]
                                  for move in result["transfers"]):
        raise RuntimeError("The trained policy did not complete a two-rung transfer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "collect", "fit", "verify"), default="all", nargs="?")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--updates-per-round", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gait", choices=("stepwise", "skip"), default="stepwise")
    parser.add_argument("--stepwise-policy", type=Path, help="Compare the demo to this stepwise policy.")
    args = parser.parse_args()
    skill = "ladder-skip" if args.gait == "skip" else "ladder"
    directory = args.run or Path(".training") / skill
    if args.stage in ("all", "collect"):
        collect(directory, args.workers, args.gait)
    if args.stage in ("all", "fit"):
        fit(directory, args.device, args.updates_per_round, seeds=TRAIN_SEEDS)
        export(directory, args.output, skill)
    if args.stage in ("all", "verify"):
        benchmark(args.output, directory / "demo.npz", args.workers, args.gait, skill, args.stepwise_policy)


if __name__ == "__main__":
    main()
