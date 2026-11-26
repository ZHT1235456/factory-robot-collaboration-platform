"""Verify the expanded factory paths, handoffs and work-order dependencies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MUJOCO_GL"] = "disable"

from factory_layout import FACTORY, FILTER_FOR_PUMP, HEAT_LADDER, PORTALS, PUMP_EXTENT, UPPER_Z
from factory_mission import FactoryRun
from gait_policy import load_gait_policy
from pump_survey import PumpVariant
from sim import load_policy, np
from verify_gait import ContactAudit
from yard_layout import yard_fault
from yard_mission import YardRun


REQUIRED_SEQUENCE = (
    "mapping_start", "floor_map", "floor_ack", "complete",
    "filter_start", "filter_alarm", "filter_verified", "filter_permit", "filter_switch",
    "filter_isolated", "heat_scout", "heat_map", "heat_route", "gate_clear",
    "heat_ladder_clear", "heat_ladder_start", "heat_ladder_landed", "heat_ladder_read", "heat_lift_board",
    "heat_high_ok", "heat_low_ok",
    "spare_order", "stock_verified", "spare_weighed", "spare_dock", "spare_loaded", "return_dock",
    "return_seated", "return_permit", "return_switch", "return_visual", "factory_complete",
)


def run_trial(request):
    seed, dropped = request

    class MissingReceipt(FactoryRun):
        def send(self, sender, receiver, signal, text, **payload):
            if signal != dropped:
                super().send(sender, receiver, signal, text, **payload)

    sim = MissingReceipt(seed, policy=load_policy(), gait_policy=load_gait_policy())
    audit = ContactAudit(sim)
    waited_for_map = True
    while sim.tick():
        if sim.g1_floor is None:
            waited_for_map &= sim.guides["g1"].frame is None
        if sim.order_index == 0 and sim.step == 1:
            audit.update()
    result = {"seed": seed, "status": sim.status, "seconds": round(sim.time, 2), "last": sim.log[-1]["text"],
              "completedOrders": list(sim.completed_orders)}
    metrics = sim.metrics
    if dropped:
        if dropped == "floor_map":
            guard = sim.g1_floor is None and sim.step == -1 and waited_for_map and not sim.completed_orders
        elif dropped == "heat_map":
            guard = sim.order_index == 2 and sim.step == 2 and metrics["selectedPortal"] is None
        elif dropped == "heat_ladder_clear":
            guard = sim.order_index == 2 and sim.step == 3 and metrics["heatLadder"] is None
        else:
            guard = (sim.order_index == 4 and sim.step == 4
                     and "return_check" not in metrics["validatedSwitches"])
        result.update(dropped=dropped, passed=bool(sim.status == "fault" and guard
                                                  and metrics["unexpectedContactFrames"] == 0))
        return result

    orders = FACTORY["orders"]
    expected = [(i, j, step) for i, order in enumerate(orders) for j, step in enumerate(order["steps"])]
    observed = [(entry["order"], entry["step"], entry["text"]) for entry in sim.log if entry["kind"] == "info"]
    pump = PumpVariant.from_seed(seed)
    flight = "b" if pump.closed_a else "a"
    alternate = "a" if pump.closed_a else "b"
    portal, = (gate for gate in PORTALS if gate != sim.factory.closed_portal)
    indices = {entry["signal"]: i for i, entry in enumerate(sim.log) if "signal" in entry}
    protocol = [indices.get(signal, -1) for signal in REQUIRED_SEQUENCE]
    mapped = metrics["modelledFloor"]
    deliveries = {signal: sim.log[indices[signal]]["payload"]["map"]
                  for signal in ("floor_map", "heat_map") if signal in indices}
    routes = metrics["mappedRoutes"]
    map_passed = (
        waited_for_map and mapped is not None and mapped["go2Scans"] > 0 and mapped["g1Scans"] == 0
        and mapped["coverage"] >= .88 and len(deliveries) == 2
        and deliveries["floor_map"]["revision"] == mapped["revision"]
        and deliveries["floor_map"]["cells"] != deliveries["heat_map"]["cells"]
        and [(route["area"], route["revision"]) for route in routes] == [
            ("filter", deliveries["floor_map"]["revision"]), ("heat", deliveries["heat_map"]["revision"])]
        and not sim.g1_floor.cells.flags.writeable
        and not np.shares_memory(sim.g1_floor.cells, sim.survey.grids["lower"])
    )
    grid = sim.survey.grids["lower"]
    seen = {}
    for order in orders:
        x1, y1, x2, y2 = order["bounds"]
        a, b = np.floor((np.array([x1, y1])-sim.survey.origin)/sim.survey.resolution).astype(int)
        c, d = np.ceil((np.array([x2, y2])-sim.survey.origin)/sim.survey.resolution).astype(int)
        seen[order["id"]] = int(np.count_nonzero(grid[b:d, a:c]))
    result.update(metrics=metrics, phases=len(observed), stairs=sorted(audit.stair_steps[flight]), portals=sim.portals,
                  lowerObservedCells=seen, upperObservedCells=int(np.count_nonzero(sim.survey.grids["upper"])))
    climb = metrics["heatLadder"]
    cross_steps = ["right_toe" if level % 2 == 0 else "left_toe" for level in range(2, HEAT_LADDER.rounds+1)]
    ladder_passed = (climb is not None and climb["finished"] and climb["passed"] and climb["policy_calls"] > 0
                     and climb["rounds_completed"] == HEAT_LADDER.rounds and climb["pelvis_rise_m"] > UPPER_Z-.06
                     and climb["start_floor_m"] == 0. and climb["landed_floor_m"] == UPPER_Z and climb["landing_feet"] == 2
                     and not climb["penetrating_substeps"] and not climb["foot_swing_support_loss_frames"]
                     and metrics["heatLiftStops"] == [0.]
                     and [move["limb"] for move in climb["transfers"] if move["rungs_advanced"] == 2] == cross_steps)
    result["passed"] = bool(
        sim.status == "done" and observed == expected and sim.completed_orders == [order["id"] for order in orders]
        and len(audit.stair_steps[flight]) == 16 and not audit.stair_steps[alternate]
        and metrics["verifiedPump"] == pump.alarm and metrics["verifiedFilter"] == FILTER_FOR_PUMP[pump.alarm]
        and metrics["selectedPortal"] == portal and sim.portals == {gate: "clear" if gate == portal else "blocked"
                                                                  for gate in PORTALS}
        and metrics["requestedSpare"] == sim.factory.spare and metrics["basketPickups"] == 2
        and metrics["cargoLoads"] == 1 and metrics["coverProof"] == 1 and metrics["spareDelivered"]
        and 380. < metrics["spareMassG"] < 470. and metrics["acceptedReturn"]
        and set(metrics["validatedSwitches"]) == {f"filter_{metrics['verifiedFilter']}", "return_check"}
        and all(stroke > 15 for stroke in metrics["validatedSwitches"].values())
        and np.linalg.norm(sim.data.qpos[sim.recorder_q:sim.recorder_q+3]-sim.layout["points"][pump.alarm]) < .08
        and sim.seated("return_rest") and metrics["unexpectedContactFrames"] == 0
        and metrics["policyCalls"] > 0 and metrics["gaitPolicyCalls"] > 0
        and all(seen.values()) and result["upperObservedCells"] > 0 and map_passed and ladder_passed
        and protocol[0] >= 0 and all(a < b for a, b in zip(protocol, protocol[1:]))
    )
    return result


def run_yard(seed):
    sim = YardRun(seed, gait_policy=load_gait_policy())
    while sim.tick():
        pass
    metrics = sim.metrics
    messages = [event["signal"] for event in sim.log if "signal" in event]
    passed = (sim.status == "done" and sim.completed_orders == ["tower"] and metrics["tower"] == yard_fault(seed)
              and metrics["mappedRoute"] == sim.floor_delivery["revision"] and metrics["yardCoverage"] >= .65
              and metrics["unexpectedContactFrames"] == 0 and metrics["gaitPolicyCalls"] > 0
              and messages == ["mapping_start", "floor_map", "floor_ack", "yard_alarm", "yard_gauge", "yard_complete"])
    return {"seed": seed, "passed": passed, "status": sim.status, "seconds": round(sim.time, 2),
            "metrics": metrics, "last": sim.log[-1]["text"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", default=[0, 11, 14, 17], type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--output", type=Path, default=Path("results/factory-validation.json"))
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--yard-seeds", nargs="*", type=int, default=[14, 17])
    args = parser.parse_args()
    requests = [(seed, None) for seed in args.seeds]
    if not args.skip_faults:
        requests.extend((17, signal) for signal in ("floor_map", "heat_map", "heat_ladder_clear", "return_permit"))
    with ProcessPoolExecutor(args.workers) as workers:
        outdoor = [workers.submit(run_yard, seed) for seed in args.yard_seeds]
        results = []
        for result in workers.map(run_trial, requests):
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        yard = [trial.result() for trial in outdoor]
        for trial in yard:
            print(json.dumps({"yard": trial}, ensure_ascii=False), flush=True)
    trials = [result for result in results if "dropped" not in result]
    faults = [result for result in results if "dropped" in result]
    report = {"extent": FACTORY["extent"], "floorAreaM2": round(float(np.prod(FACTORY["extent"])), 2),
              "areaMultiple": round(float(np.prod(FACTORY["extent"])/np.prod(PUMP_EXTENT)), 3),
              "orders": len(FACTORY["orders"]), "checkpoints": sum(len(order["steps"]) for order in FACTORY["orders"]),
              "passed": sum(result["passed"] for result in trials), "total": len(trials),
              "trials": trials, "faultTrials": faults, "yardTrials": yard}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if not all(result["passed"] for result in results + yard):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
