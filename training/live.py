"""Local request/response worker for the Three.js console."""

import json
import os
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MUJOCO_GL"] = "disable"

from gait_policy import load_gait_policy
from sim import Simulation, load_policy
from factory_mission import FactoryRun


class Worker:
    def __init__(self):
        self.sessions = {}

    def request(self, message):
        op = message.get("op")
        client = message.get("client")
        if not isinstance(client, str) or not 1 <= len(client) <= 64:
            raise ValueError("无效的会话")
        if op == "reset":
            scene, seed = message.get("scene"), message.get("seed", 17)
            policies = {"policy": load_policy(), "gait_policy": load_gait_policy()}
            runner = FactoryRun(seed, **policies) if scene == "factory" else Simulation(scene, seed, **policies)
            if len(self.sessions) >= 4 and client not in self.sessions:
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[client] = runner
            return dict(runner.packet(), layout=runner.layout)
        if op == "close":
            self.sessions.pop(client, None)
            return {"ok": True}
        if op != "advance" or client not in self.sessions:
            raise ValueError("工单会话已过期，请重置")
        runner = self.sessions[client]
        count = message.get("count", 1)
        previous = message.get("events", 0)
        revision = message.get("mapRevision")
        maximum = 32 if runner.factory else 8
        if type(count) is not int or not 1 <= count <= maximum or type(previous) is not int or previous < 0:
            raise ValueError("无效的步进请求")
        for _ in range(count):
            if not runner.tick():
                break
        return runner.packet(previous, revision)


def main():
    supervisor = Worker()
    for line in sys.stdin:
        try:
            result = supervisor.request(json.loads(line))
        except Exception as exc:
            result = {"ok": False, "error": str(exc) or type(exc).__name__}
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
