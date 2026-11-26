"""Go2 maps and reads the low return indicators before G1 checks the matching gauge."""

import math

from occupancy import ReceivedFloor
from sim import Simulation, TaskFault
from yard_layout import TOWER_X, YARD_SCANS, YARD_WAIT


class YardRun(Simulation):
    def __init__(self, seed, **options):
        self.mapping_stage = "ready"
        self.survey_stops = 0
        self.floor_delivery = None
        self.completed_orders = []
        super().__init__("yard", seed, **options)
        self.metrics.update(tower=None, towerTemperature=None, yardCoverage=None, mappedRoute=None)

    def require(self, robot, signal):
        report = self.inbox.get((robot, signal))
        if report is None:
            raise TaskFault("冷却水站缺少双机确认回执")
        return report

    def work_yard(self):
        self.phase(0)
        self.mapping_stage = "surveying"
        self.send("go2", "g1", "mapping_start", "开始测绘冷却塔周边 · G1 等待接图")
        for x, y in YARD_SCANS:
            yield from self.drive("go2", [(x, y, 0.)], yaw=None)
            self.survey.scan("go2", show_sweep=True)
            self.survey_stops += 1
        coverage = self.survey.lower_coverage()
        if coverage < .65:
            raise TaskFault(f"室外巡检区已测覆盖不足 · {coverage:.0%}")
        self.metrics["yardCoverage"] = round(coverage, 4)
        floor = self.survey.floor_message()
        self.mapping_stage = "sending"
        self.send("go2", "g1", "floor_map", f"室外地图 V{floor['revision']} 已交付 · 覆盖 {coverage:.0%}", map=floor)
        delivery_event = len(self.log)-1
        yield from self.pause(.9)
        received = ReceivedFloor(self.require("g1", "floor_map")["map"], self.survey.origin, self.survey.resolution,
                                 (self.survey.width, self.survey.height))
        self.mapping_stage = "received"
        self.floor_delivery = {"event": delivery_event, "revision": received.revision}
        self.send("g1", "go2", "floor_ack", f"已接收室外地图 V{received.revision} · 等待异常塔位")

        self.phase(1)
        alarms = []
        for tower, x in TOWER_X.items():
            yield from self.drive("go2", [(x, -.38, 0.)], yaw=math.pi/2)
            name = f"yard_low_{tower}"
            if not self.survey.visible("go2", name):
                raise TaskFault(f"Go2 未获得 {tower} 低位回水指示的无遮挡观测")
            if self.model.geom(name).rgba[0] > .75:
                alarms.append(tower)
        if len(alarms) != 1:
            raise TaskFault("冷却塔低位回水异常需要人工复核")
        tower, = alarms
        self.send("go2", "g1", "yard_alarm", f"{tower} 低位回水指示异常 · 请核对对应温度表", tower=tower)
        yield from self.drive("go2", [YARD_WAIT], yaw=math.pi/2)

        self.phase(2)
        tower = self.require("g1", "yard_alarm")["tower"]
        self.focus = f"yard_{tower}"
        start = self.guides["g1"].xyz.copy()
        goal = (TOWER_X[tower], .16, 0.)
        route = received.route(start, goal, self.aisles)
        if route is None:
            raise TaskFault("收到的地图中未找到前往温度表的已观测通道")
        self.metrics["mappedRoute"] = received.revision
        self.survey.route = {"area": "tower", "description": f"{tower} 温度表",
                             "revision": received.revision, "points": [start.tolist(), *(p.tolist() for p in route)]}
        yield from self.drive("g1", route, yaw=math.pi/2, direct=True)

        self.phase(3)
        if not self.survey.visible("g1", f"yard_gauge_{tower}"):
            raise TaskFault(f"G1 未看到 {tower} 温度表")
        temperature = round(34.5 + (self.seed % 12)*.13, 1)
        self.metrics.update(tower=tower, towerTemperature=temperature)
        self.reading = {"name": f"{tower} 模拟回水温度", "value": temperature, "unit": "°C"}
        self.send("g1", "go2", "yard_gauge", f"{tower} 塔位一致 · 模拟回水温度 {temperature:.1f} °C", tower=tower)
        yield from self.drive("g1", [(TOWER_X[tower]+1.35, -2.12, 0.)])

        self.phase(4)
        confirmed = self.require("go2", "yard_gauge")["tower"]
        yield from self.drive("go2", [(TOWER_X[confirmed], -.38, 0.)], yaw=math.pi/2)
        if not self.survey.visible("go2", f"yard_low_{confirmed}"):
            raise TaskFault("Go2 未能复核低位回水指示")
        self.send("go2", "g1", "yard_complete", f"{confirmed} 回水指示已复测 · 冷却塔巡检联签")
        self.require("g1", "yard_complete")
        self.completed_orders.append("tower")
        yield from self.pause(.45)

    def packet(self, from_log=0, map_revision=None):
        packet = super().packet(from_log, map_revision)
        packet.update(order=0, completed=list(self.completed_orders))
        packet["mapping"] = {"stage": self.mapping_stage, "stops": self.survey_stops,
                             "total": len(YARD_SCANS), "received": self.floor_delivery}
        return packet
