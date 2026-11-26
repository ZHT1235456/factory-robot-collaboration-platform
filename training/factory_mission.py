"""Five linked work orders executed in one expanded cooling-water hall."""

import math
from functools import lru_cache

from factory_layout import (
    FACTORY, FILTER_X, GATE_NORTH, GATE_SOUTH, GREEN, LIFT_WAIT, LIFT_XY, UPPER_Z,
    HEAT_DOG_WAIT, HEAT_LADDER, HEAT_LADDER_APPROACH, HEAT_LADDER_ENTRY, HEAT_LADDER_ORIGIN,
    HEAT_VIEW_STAND, HEAT_VIEWS, PORTALS, PORTAL_NAMES,
    RETURN_BUTTON, RETURN_DOCK, RETURN_REST, RETURN_VIEW, SPARE_DOCK, SPARE_NAMES, SPARE_POINTS, SURVEY_POSES,
    FactoryVariant,
)
from navigation import Aisles
from occupancy import ReceivedFloor
from plant import BASKET_FLOOR, GRIP_HALF, ROOT, mj
from gait import yaw_quat
from ladder import Climb, PLATFORM_SUPPORT, eased, inference
from sim import Simulation, TaskFault, np


RECEIPTS = {
    "floor_map": "Go2 下层占用地图",
    "complete": "泵组测振签收", "filter_alarm": "低位压差记录", "filter_verified": "过滤控制位号",
    "filter_permit": "过滤区清场许可", "filter_switch": "过滤按钮行程", "filter_isolated": "过滤隔离确认",
    "heat_map": "换热入口地图", "heat_route": "换热入口采纳回执", "gate_clear": "跨区入口清场",
    "heat_ladder_clear": "换热检修梯梯口净空",
    "heat_high_ok": "换热位号核对", "heat_low_ok": "换热低位复检", "spare_order": "备件领料单",
    "stock_verified": "备件货位核对", "spare_weighed": "备件称重", "spare_dock": "备件区驻车",
    "spare_loaded": "箩筐载荷确认", "return_dock": "回水区驻车", "return_seated": "托架就位",
    "return_permit": "回水验收许可", "return_switch": "回水按钮行程", "return_visual": "回水指示回读",
}


@lru_cache(maxsize=1)
def ladder_policy():
    return inference(ROOT / "results/g1-ladder-skip-policy.onnx")


class FactoryRun(Simulation):
    def __init__(self, seed, **options):
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("工单种子无效")
        self.order_index = 0
        self.completed_orders = []
        self.portals = {}
        self.cargo = "recorder"
        self.mapping_stage = "ready"
        self.survey_stops = 0
        self.g1_floor = None
        self.floor_delivery = None
        self.ladder = None
        self.lift_moving = False
        super().__init__("pump", seed, factory=FactoryVariant.from_seed(seed), **options)
        self.recorder_q = self.part_q
        self.spare_part = self.id("body", "spare_piece")
        joint = self.id("joint", "spare_float")
        self.spare_q, self.spare_v = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        self.basket = self.id("body", "go_basket")
        self.basket_liner = self.id("geom", "basket_liner")
        self.buttons = {key: self.model.jnt_qposadr[self.id("joint", f"slide_{key}")]
                        for key in self.layout["factory"]["buttons"]}
        self.indicators = {key: self.id("geom", key) for key in self.layout["factory"]["indicators"]}
        self.lift = self.model.body_mocapid[self.id("body", "heat_lift")]
        g1 = self.robot["g1"]
        self.ladder_stance = self.data.qpos[g1["root"]:g1["q"]+g1["n"]].copy()
        self.ladder_stance[:3] = [HEAT_LADDER_ORIGIN[0]-HEAT_LADDER_ENTRY[0], 0.,
                                  self.standing["g1"]+HEAT_LADDER.floor]
        spare_geoms = np.flatnonzero(self.model.body_rootid[self.model.geom_bodyid] == self.spare_part)
        self.clearance.payload.update(spare_geoms)
        self.model.geom_group[spare_geoms] = 5
        self.metrics.update(verifiedFilter=None, selectedPortal=None, requestedSpare=None, cargoLoads=0,
                            spareMassG=None, spareDelivered=False, validatedSwitches={}, acceptedReturn=False,
                            modelledFloor=None, mappedRoutes=[], heatLadder=None, heatLiftStops=[])

    def event(self, note, kind="info"):
        super().event(note, kind)
        self.log[-1]["order"] = self.order_index

    def phase(self, number, note=None):
        super().phase(number, note or FACTORY["orders"][self.order_index]["steps"][number])

    def receipt(self, robot, signal):
        message = self.inbox.get((robot, signal))
        if message is None:
            name = "G1" if robot == "g1" else "Go2"
            raise TaskFault(f"{name} 未收到回执 · {RECEIPTS[signal]}")
        return message

    def find_color(self, robot, geometry):
        if not self.survey.visible(robot, geometry):
            raise TaskFault(f"{robot.upper()} 未取得指示牌的无遮挡观测 · {geometry}")
        return self.model.geom(geometry).rgba[:3]

    def confirm_control(self, key):
        self.focus = key
        point = np.array(self.layout["points"][key])
        self.button = self.buttons[key]
        yield from self.approach(point, .340)
        self.hand_mode = .85
        yield from self.raise_hand(point + [-.063, 0., 0.], .53)
        self.hand_goal = point + [.015, 0., 0.]
        yield from self.wait(lambda: self.data.qpos[self.button] > .015 and self.probe_force > .16,
                             2.45, "确认按钮未达到有效行程", 5)
        mm = round(float(self.data.qpos[self.button])*1000, 2)
        self.metrics["validatedSwitches"][key] = mm
        self.reading = {"name": "确认行程", "value": mm, "unit": "mm"}
        self.event(f"按钮行程 {mm:.1f} mm · 接触确认", "pass")
        yield from self.raise_hand(point + [-.12, 0., .07], .36)
        self.hand_goal = None
        yield from self.wait(lambda: self.data.qpos[self.button] < .004, 1.6, "按钮未正常回弹", 5)
        self.button = None
        yield from self.leave_access()

    def receive_floor(self, signal):
        report = self.receipt("g1", signal)
        try:
            received = ReceivedFloor(report["map"], self.survey.origin, self.survey.resolution,
                                     (self.survey.width, self.survey.height))
        except (KeyError, TypeError, ValueError) as error:
            raise TaskFault("G1 收到的下层栅格不完整或坐标不匹配") from error
        if self.g1_floor is not None and received.revision < self.g1_floor.revision:
            raise TaskFault("G1 收到过期的下层地图")
        index = next(i for i in range(len(self.log)-1, -1, -1)
                     if self.log[i].get("signal") == signal and self.log[i].get("receiver") == "g1")
        self.g1_floor = received
        self.floor_delivery = {"event": index, "revision": received.revision}
        return report

    def build_floor_map(self):
        pump_aisles = self.aisles
        self.aisles = Aisles(self.layout)
        self.mapping_stage = "surveying"
        self.send("go2", "g1", "mapping_start", "开始采集厂房下层地图 · G1 请等待接图")
        for x, y in SURVEY_POSES:
            yield from self.drive("go2", [(x, y, 0.)], yaw=None)
            self.survey.scan("go2", show_sweep=True)
            self.survey_stops += 1
        yield from self.turn("go2", 0.)
        yield from self.pause(.45)
        self.aisles = pump_aisles
        coverage = self.survey.lower_coverage()
        zones = {order["id"]: self.survey.lower_coverage(order["bounds"]) for order in FACTORY["orders"]}
        if coverage < .88 or min(zones.values()) < .65:
            raise TaskFault(f"下层地图观测不足 · 全厂 {coverage:.0%}，最低工区 {min(zones.values()):.0%}")
        floor = self.survey.floor_message()
        self.metrics["modelledFloor"] = {"revision": floor["revision"], "coverage": round(coverage, 4),
                                         "zones": {key: round(value, 4) for key, value in zones.items()},
                                         "go2Scans": self.survey.scans["go2"], "g1Scans": self.survey.scans["g1"],
                                         "seconds": round(self.time, 2)}
        self.mapping_stage = "sending"
        self.send("go2", "g1", "floor_map", f"下层建图完成 · 交付 V{floor['revision']} · 已观测 {coverage:.0%}",
                  map=floor)
        yield from self.pause(1.)
        self.receive_floor("floor_map")
        self.mapping_stage = "received"
        self.send("g1", "go2", "floor_ack", f"收到下层地图 V{self.g1_floor.revision} · 开始执行联动工单",
                  revision=self.g1_floor.revision)
        yield from self.pause(.70)

    def drive_on_floor(self, goal, area, description):
        if self.g1_floor is None:
            raise TaskFault("G1 等待 Go2 的下层地图")
        start = self.guides["g1"].xyz.copy()
        route = self.g1_floor.route(start, goal, self.aisles)
        if route is None:
            raise TaskFault(f"地图 V{self.g1_floor.revision} 未找到已观测安全路线 · {description}")
        self.metrics["mappedRoutes"].append({"area": area, "revision": self.g1_floor.revision, "legs": len(route)})
        self.survey.route = {"area": area, "description": description, "revision": self.g1_floor.revision,
                             "points": [start.tolist(), *(point.tolist() for point in route)]}
        yield from self.drive("g1", route, direct=True)

    def work_factory(self):
        work_orders = (self.work_pump, self.work_filter, self.work_heat, self.work_spares, self.work_return)
        yield from self.build_floor_map()
        for index, work in enumerate(work_orders):
            self.order_index = index
            self.reading = self.focus = None
            if index == 1:
                self.aisles = Aisles(self.layout)
            yield from work()
            order = FACTORY["orders"][index]
            self.completed_orders.append(order["id"])
            self.event(f"{order['name']}已验收", "pass")

    @staticmethod
    def filter_view(key, offset=.24):
        return (FILTER_X[key]+offset, -5.42, 0.)

    def work_filter(self):
        self.phase(0)
        pump = self.receipt("go2", "complete")["pump"]
        self.send("go2", "g1", "filter_start", f"收到 {pump} 测振签收 · 前往过滤支路复核压差", pump=pump)
        yield from self.drive("go2", [(-5.61, -6.27, 0.), (8.76, -6.27, 0.)])
        g1_x = float(self.guides["g1"].xyz[0])
        yield from self.drive("g1", [(g1_x, -7.43, 0.)], yaw=None)
        yield from self.drive_on_floor((8.47, -7.43, 0.), "filter", "过滤支路转场")
        self.phase(1)
        detected = []
        for key in FILTER_X:
            yield from self.drive("go2", [self.filter_view(key)], yaw=math.pi/2)
            if self.find_color("go2", f"filter_low_{key}")[0] > .8:
                detected.append(key)
        if len(detected) != 1:
            raise TaskFault("过滤组压差信号需要人工复核")
        key = detected[0]
        self.send("go2", "g1", "filter_alarm", f"低位差压牌确认 {key} 异常 · 请核对控制位号", filter=key)
        yield from self.drive("go2", [(FILTER_X[key]+1.28, -6.56, 0.)])
        self.phase(2)
        key = self.receipt("g1", "filter_alarm")["filter"]
        yield from self.drive("g1", [self.filter_view(key)], yaw=math.pi/2)
        self.find_color("g1", f"filter_id_{key}")
        self.metrics["verifiedFilter"] = key
        self.send("g1", "go2", "filter_verified", f"{key} 控制位号一致 · 等待观察位清场", filter=key)
        self.phase(3)
        matched = self.receipt("go2", "filter_verified")
        if not self.arrived("go2") or matched["filter"] != key:
            raise TaskFault("过滤隔离位未得到清场确认")
        self.send("go2", "g1", "filter_permit", f"观察位清场 · 允许确认 {key} 支路的检修隔离", filter=key)
        self.phase(4)
        yield from self.confirm_control(f"filter_{self.receipt('g1', 'filter_permit')['filter']}")
        self.model.geom_rgba[self.indicators[f"isolated_{key}"], :3] = GREEN
        self.send("g1", "go2", "filter_switch", f"{key} 隔离确认行程有效 · 请读回状态", filter=key)
        self.phase(5)
        confirmed = self.receipt("go2", "filter_switch")["filter"]
        yield from self.drive("go2", [self.filter_view(confirmed, .55)], yaw=math.pi/2)
        if self.find_color("go2", f"isolated_{confirmed}")[1] < .60:
            raise TaskFault("过滤设备未返回隔离状态")
        self.send("go2", "g1", "filter_isolated", f"{confirmed} 隔离指示已读回 · 换热工单可开始", filter=confirmed)

    def work_heat(self):
        self.phase(0)
        key = self.receipt("g1", "filter_isolated")["filter"]
        self.send("g1", "go2", "heat_scout", f"已接收 {key} 隔离结果 · 先勘测两处换热区入口")
        yield from self.drive("go2", [(8.61, -5.48, 0.), (8.61, -.74, 0.)])
        self.phase(1)
        for gate, x in PORTALS.items():
            yield from self.drive("go2", [(x, GATE_SOUTH, 0.)], yaw=math.pi/2)
            ends = [(x+offset, GATE_NORTH, z) for offset, z in ((-.31, .32), (.31, .32), (0., .82))]
            self.portals[gate] = "clear" if all(self.survey.clear_to("go2", end) for end in ends) else "blocked"
            self.survey.scan("go2", show_sweep=True)
        note = " · ".join(f"{PORTAL_NAMES[gate]}{'畅通' if state == 'clear' else '有障碍'}"
                          for gate, state in self.portals.items())
        floor = self.survey.floor_message()
        self.send("go2", "g1", "heat_map", note, portals=dict(self.portals), revision=floor["revision"], map=floor)
        self.phase(2)
        report = self.receive_floor("heat_map")
        chosen = next((gate for gate, state in report["portals"].items() if state == "clear"), None)
        if chosen is None:
            raise TaskFault("两处入口净空均未确认")
        self.metrics["selectedPortal"] = chosen
        x = PORTALS[chosen]
        approach, crossing = (x, GATE_SOUTH, 0.), (x, GATE_NORTH, 0.)
        self.survey.route = {"area": "heat", "revision": report["revision"], "points": [approach, crossing],
                             "description": PORTAL_NAMES[chosen]}
        self.send("g1", "go2", "heat_route", f"采纳地图 V{report['revision']} · 经{PORTAL_NAMES[chosen]}进入换热区",
                  portal=chosen)
        self.receipt("go2", "heat_route")
        yield from self.drive("go2", [approach])
        yield from self.drive("go2", [crossing], direct=True)
        yield from self.drive("go2", [(x, LIFT_WAIT[1], 0.), HEAT_VIEWS["low"]], yaw=math.pi/2)
        self.send("go2", "g1", "gate_clear", "Go2 已通过并离开入口 · G1 可以通过")
        self.receipt("g1", "gate_clear")
        g1_x = float(self.guides["g1"].xyz[0])
        yield from self.drive("g1", [(g1_x, -5.48, 0.), (x, -5.48, 0.), approach])
        yield from self.drive_on_floor(crossing, "heat", "换热"+PORTAL_NAMES[chosen])
        self.phase(3)
        yield from self.drive("go2", [HEAT_DOG_WAIT], yaw=-.8)
        if not all(self.survey.clear_to("go2", (HEAT_LADDER_ENTRY[0], HEAT_LADDER_ENTRY[1]+offset, .24))
                   for offset in (-.26, 0., .26)):
            raise TaskFault("Go2 未确认换热检修梯梯口净空")
        self.send("go2", "g1", "heat_ladder_clear", "一层梯口净空 · G1 可以登梯上楼")
        self.receipt("g1", "heat_ladder_clear")
        yield from self.inspect_heat_ladder()
        self.send("g1", "go2", "heat_high_ok", "二层 H2 已复检 · G1 已回到一层", unit="H2")
        self.phase(4)
        self.receipt("go2", "heat_high_ok")
        yield from self.drive("go2", [HEAT_VIEWS["low"]], yaw=math.pi/2)
        low = self.find_color("go2", "heat_low")
        request = "seal" if low[0] > .8 else "screen"
        self.send("go2", "g1", "heat_low_ok",
                  f"{'接液指示变色' if request == 'seal' else '回流筛网需复核'} · 建议领取 {SPARE_NAMES[request]}",
                  spare=request)
        self.phase(5)
        request = self.receipt("g1", "heat_low_ok")["spare"]
        self.metrics["requestedSpare"] = request
        self.send("g1", "go2", "spare_order", f"上位和低位证据已联签 · 领用 {SPARE_NAMES[request]}", spare=request)

    def ride_heat_lift(self, level):
        guide = self.guides["g1"]
        start = float(self.data.mocap_pos[self.lift, 2])
        if abs(guide.xyz[2]-start) > .025 or np.linalg.norm(guide.xyz[:2]-LIFT_XY) > .12:
            raise TaskFault("G1 未在升降台内站稳")
        self.lift_moving = True
        for step in range(1, 261):
            height = start + (level-start)*eased(step/260)
            self.data.mocap_pos[self.lift, 2] = height
            guide.xyz[2] = height
            yield
        self.lift_moving = False
        self.metrics["heatLiftStops"].append(level)
        yield from self.pause(.24)

    def inspect_heat_ladder(self):
        self.focus = "heat_ladder"
        x = float(self.guides["g1"].xyz[0])
        yield from self.drive("g1", [(x, LIFT_WAIT[1], 0.), (HEAT_LADDER_APPROACH[0], LIFT_WAIT[1], 0.),
                                     HEAT_LADDER_APPROACH, HEAT_LADDER_ENTRY], yaw=math.pi)

        climb = Climb(HEAT_LADDER, 17)
        robot = self.robot["g1"]
        coordinates = slice(robot["root"], robot["q"]+robot["n"])
        speed = slice(robot["rv"], robot["v"]+robot["n"])
        origin = np.array(HEAT_LADDER_ORIGIN)
        turn = yaw_quat(math.pi)
        inverse = yaw_quat(-math.pi)
        state = self.data.qpos[coordinates].copy()
        state[:3] -= origin
        state[:2] *= -1
        mj.mju_mulQuat(state[3:7], inverse, self.data.qpos[robot["root"]+3:robot["root"]+7])

        def synchronize():
            pose = climb.data.qpos
            self.data.qpos[coordinates] = pose
            self.data.qpos[robot["root"]:robot["root"]+3] = origin + [-pose[0], -pose[1], pose[2]]
            mj.mju_mulQuat(self.data.qpos[robot["root"]+3:robot["root"]+7], turn, pose[3:7])
            self.data.qvel[speed] = climb.data.qvel
            self.data.qvel[robot["rv"]:robot["rv"]+2] *= -1

        def move(stage, motion):
            self.ladder = {"stage": stage, "step": None, "feet": [None, None]}
            seen = len(climb.events)
            while True:
                try:
                    next(motion)
                except StopIteration as stop:
                    return stop.value
                self.ladder = {"stage": stage, "step": self.ladder["step"],
                               "feet": [None if rung is None or rung < 0 else rung+1 for rung in climb.attached[2:]]}
                for event in climb.events[seen:]:
                    if stage == "ascending" and event["rungs_advanced"] == 2 and event["latched"]:
                        self.ladder["step"] = {"limb": event["limb"], "from": event["from_rung"], "to": event["rung"]}
                seen = len(climb.events)
                if climb.stats["penetrating_substeps"]:
                    hit = climb.worst_collision
                    raise TaskFault(f"登梯碰撞保护 · {hit['robot']} / {hit['surface']}")
                yield

        # The ladder has its own contact solver at the same plant coordinates.
        # The plant's G1 body is its sensing and rendering proxy while on the ladder.
        robot_geoms = self.clearance.owners == 1
        masks = self.model.geom_contype[robot_geoms].copy(), self.model.geom_conaffinity[robot_geoms].copy()
        guide_weld = self.id("equality", "g1_guidance")
        self.model.geom_contype[robot_geoms] = self.model.geom_conaffinity[robot_geoms] = 0
        self.data.eq_active[guide_weld] = False
        self.external_g1 = synchronize
        self.send("g1", "go2", "heat_ladder_start", "一层梯口已就位 · 开始跨档上楼")
        landed = False
        try:
            if not (yield from move("mounting", climb.mount(state, self.ladder_stance))):
                raise TaskFault("G1 未取得梯下支撑")
            result = yield from move("ascending", climb.ascent(ladder_policy()))
            if not result["passed"] or result["policy_calls"] <= 0:
                raise TaskFault("G1 跨档上梯未完成")
            landed = yield from move("landing", climb.step_onto_platform())
            if not landed:
                raise TaskFault("G1 未取得二层平台双脚支撑")
            self.metrics["heatLadder"] = dict(result, finished=True, start_floor_m=HEAT_LADDER_ENTRY[2],
                                               landed_floor_m=UPPER_Z, landing_feet=climb.support.count(PLATFORM_SUPPORT),
                                               max_ladder_penetration_mm=climb.stats["max_ladder_penetration_mm"])
        finally:
            self.external_g1 = None
            self.ladder = None
            self.model.geom_contype[robot_geoms], self.model.geom_conaffinity[robot_geoms] = masks
            self.data.eq_active[guide_weld] = True
            root = robot["root"]
            self.data.mocap_pos[self.mocap["g1"]] = self.data.qpos[root:root+3]
            self.data.mocap_quat[self.mocap["g1"]] = self.data.qpos[root+3:root+7]
            self.guides["g1"].xyz[:] = (*self.data.qpos[root:root+2], UPPER_Z if landed else HEAT_LADDER_ENTRY[2])
            self.previous_leg_ref["g1"] = self.data.qpos[robot["q"]:robot["q"]+12].copy()

        self.send("g1", "go2", "heat_ladder_landed", "双脚已踏上二层格栅 · 前往 H2 巡检台", level=UPPER_Z)
        guide = self.guides["g1"]
        # Step sideways away from the edge before turning into the walkway.
        clear = np.array((guide.xyz[0], HEAT_VIEW_STAND[1]+.48, UPPER_Z))
        yield from self.follow_steps("g1", self.footsteps["g1"].walk(guide.xyz.copy(), clear, guide.yaw, .33))
        self.focus = "heat_station"
        yield from self.drive("g1", [HEAT_VIEW_STAND], yaw=-math.pi/2)
        self.find_color("g1", "heat_high")
        self.reading = {"name": "二层巡检位号", "value": "H2", "unit": ""}
        self.send("g1", "go2", "heat_ladder_read", "已在二层巡检台核对 H2 位号 · 准备返回一层")
        yield from self.pause(.72)
        yield from self.drive("g1", [(LIFT_XY[0], HEAT_VIEW_STAND[1], UPPER_Z), (*LIFT_XY, UPPER_Z)], yaw=math.pi/2)
        self.send("g1", "go2", "heat_lift_board", "已进入下行升降台 · 返回一层")
        yield from self.ride_heat_lift(0.)
        yield from self.drive("g1", [LIFT_WAIT], yaw=math.pi/2)

    def activate_spare(self, request):
        self.part = self.spare_part
        self.part_q, self.part_v = self.spare_q, self.spare_v
        self.handle = self.id("geom", "spare_handle")
        self.item_geom = self.id("geom", "spare_body")
        self.part_mass = float(self.model.body_mass[self.part])
        self.attachment = self.id("equality", "spare_stock")
        self.grip_constraint = "spare_hold"
        self.basket_clip = self.id("equality", "spare_basket")
        self.cargo, self.cargo_name = "spare", SPARE_NAMES[request]

    def load_basket(self):
        offset = np.array([0., 0., BASKET_FLOOR+GRIP_HALF["pump"][2]+.031])
        point = self.data.xpos[self.basket].copy()+offset
        yield from self.approach(point+[0., 0., .076], .403, basket=True)
        self.guides["g1"].crouch = .78
        self.hand_mode = .62
        grip_offset = self.data.site_xpos[self.grip]-self.data.xpos[self.part]
        yield from self.raise_hand(point+grip_offset+[-.17, 0., .134], .62)
        yield from self.raise_hand(point+grip_offset+[0., 0., .123], .62)
        for dz in np.linspace(.123, -.010, 65):
            error = point+[0., 0., dz]-self.data.xpos[self.part]
            self.hand_goal = self.data.site_xpos[self.grip]+np.clip(error*1.55, -.075, .075)
            yield
        def over_liner():
            dx, dy, dz = self.data.xmat[self.basket].reshape(3, 3).T @ (self.data.xpos[self.part]-point)
            return abs(dx) < .067 and abs(dy) < .053 and -.018 < dz < .043 and self.data.xmat[self.part, 8] > .89

        yield from self.wait(over_liner, 1.7, "备件未进入筐口安全释放范围", 8)
        self.unbind()
        self.holding = self.closed = False
        self.retracting = True
        self.hand_goal = self.data.site_xpos[self.grip].copy()
        yield from self.wait(lambda: over_liner() and self.support.get(self.basket_liner, 0.) > .14
                             and np.linalg.norm(self.data.qvel[self.part_v:self.part_v+3]) < .060,
                             2.2, "箩筐未取得稳定载荷", 10)
        self.bind("spare_basket", self.basket)
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy()+[0., 0., .132], .56)
        yield from self.raise_hand(self.data.site_xpos[self.grip].copy()+[-.20, 0., .025], .48)
        self.retracting = False
        yield from self.leave_access()
        self.metrics["cargoLoads"] += 1

    def work_spares(self):
        self.phase(0)
        request = self.receipt("go2", "spare_order")["spare"]
        yield from self.drive("go2", [(13.62, 5.15, 0.), (-9.0, 5.15, 0.), (-12.54, 6.43, 0.)])
        yield from self.drive("g1", [(-8.30, LIFT_WAIT[1], 0.), (-10.88, 5.28, 0.)])
        self.phase(1)
        available = []
        for key, (x, y, _) in SPARE_POINTS.items():
            yield from self.drive("go2", [(x+.16, y-1.65, 0.)], yaw=math.pi/2)
            if self.find_color("go2", f"spare_{key}")[1] > .60:
                available.append(key)
        if available != [request]:
            raise TaskFault("需要的备件与 Go2 读到的货位不一致")
        self.send("go2", "g1", "stock_verified", f"已核对两处货位 · {SPARE_NAMES[request]} 可领用", spare=request)
        yield from self.drive("go2", [(-10.29, 9.15, 0.)])
        self.phase(2)
        request = self.receipt("g1", "stock_verified")["spare"]
        self.activate_spare(request)
        self.focus = "spare_stock"
        yield from self.grasp()
        self.phase(3)
        self.focus = "spare_scale"
        yield from self.place("spare_scale")
        yield from self.measure("spare_scale", "备件实测质量", 0., "g")
        self.metrics["spareMassG"] = self.reading["value"]
        self.send("g1", "go2", "spare_weighed", f"备件质量 {self.reading['value']:.0f} g · 请求停车装载", spare=request)
        self.phase(4)
        self.receipt("go2", "spare_weighed")
        yield from self.drive("go2", [SPARE_DOCK])
        yield from self.wait(lambda: self.arrived("go2"), 1.0, "Go2 未在备件间驻车", 10)
        self.send("go2", "g1", "spare_dock", "在备件间交接位驻车 · 允许装筐")
        self.phase(5)
        self.receipt("g1", "spare_dock")
        yield from self.grasp()
        yield from self.load_basket()
        self.send("g1", "go2", "spare_loaded", f"{SPARE_NAMES[request]} 已装筐 · 卡扣确认 · 可运往回水区", spare=request)

    def work_return(self):
        self.phase(0)
        order = self.receipt("go2", "spare_loaded")
        if self.attachment != self.basket_clip:
            raise TaskFault("备件未锁定在箩筐中")
        self.focus = "return_dock"
        yield from self.drive("go2", [(-9.11, 5.32, 0.), (-9.11, RETURN_DOCK[1], 0.), RETURN_DOCK])
        yield from self.wait(lambda: self.arrived("go2"), 1.0, "Go2 未在回水区驻车", 10)
        self.send("go2", "g1", "return_dock", "已运抵回水交接位并驻车 · 等待 G1 卸件", spare=order["spare"])
        self.phase(1)
        approach_y = RETURN_DOCK[1]-1.83
        yield from self.drive("g1", [(-7.75, 5.32, 0.), (-7.75, approach_y, 0.), (-16.66, approach_y, 0.)])
        self.receipt("g1", "return_dock")
        yield from self.grasp()
        yield from self.drive("g1", [(-16.23, RETURN_DOCK[1]+1., 0.), (-13.64, RETURN_REST[1], 0.)])
        self.phase(2)
        self.focus = "return_rest"
        yield from self.place("return_rest", magnetic=True)
        self.target_station = "return_rest"
        self.metrics["spareDelivered"] = bool(self.seated(self.target_station))
        self.send("g1", "go2", "return_seated", "备件在验收托架稳定就位 · 请复核隔离和验收位")
        self.phase(3)
        self.receipt("go2", "return_seated")
        isolated = self.receipt("g1", "filter_isolated")["filter"]
        if self.model.geom_rgba[self.indicators[f"isolated_{isolated}"], 1] < .6 or not self.seated("return_rest"):
            raise TaskFault("过滤隔离或备件就位条件未满足")
        yield from self.drive("go2", [RETURN_VIEW], yaw=math.pi/2)
        if self.find_color("go2", "return_status")[1] > .60:
            raise TaskFault("回水确认前的指示状态异常")
        self.send("go2", "g1", "return_permit", "隔离回执与现场状态相符 · 允许执行回水验收确认")
        self.phase(4)
        self.receipt("g1", "return_permit")
        aisle_y = RETURN_REST[1]+.09
        yield from self.drive("g1", [(-13.62, aisle_y, 0.), (RETURN_BUTTON[0]-.95, aisle_y, 0.)], speed=.71)
        yield from self.confirm_control("return_check")
        self.model.geom_rgba[self.indicators["return_status"], :3] = GREEN
        self.send("g1", "go2", "return_switch", "回水验收按钮行程有效 · 等待 Go2 读回")
        self.phase(5)
        self.receipt("go2", "return_switch")
        if self.find_color("go2", "return_status")[1] < .60 or not self.seated("return_rest"):
            raise TaskFault("回水验收状态或托架承重不稳定")
        self.metrics["acceptedReturn"] = True
        self.send("go2", "g1", "return_visual", "回水验收绿灯已读回 · 备件就位")
        self.receipt("g1", "return_visual")
        self.send("g1", "go2", "factory_complete", "泵组、过滤、换热、备件、回水五张工单双机签收")
        yield from self.pause(.36)

    def packet(self, from_log=0, map_revision=None):
        packet = super().packet(from_log, map_revision)
        packet.update(scene="factory", order=self.order_index, completed=list(self.completed_orders),
                      portals=dict(self.portals), cargo=self.cargo, ladder=self.ladder,
                      liftHeight=round(float(self.data.mocap_pos[self.lift, 2]), 5), liftMoving=self.lift_moving)
        if self.ladder is not None:
            g1 = packet["actors"]["g1"]
            g1["moving"] = True
            root = self.robot["g1"]["rv"]
            g1["speed"] = round(float(np.linalg.norm(self.data.qvel[root:root+3])), 3)
        packet["mapping"] = {"stage": self.mapping_stage, "stops": self.survey_stops, "total": len(SURVEY_POSES),
                             "received": self.floor_delivery}
        packet["part"] = self.data.qpos[self.recorder_q:self.recorder_q+7].round(6).tolist()
        packet["spare"] = self.data.qpos[self.spare_q:self.spare_q+7].round(6).tolist()
        packet["devices"] = {
            "buttons": {key: round(float(self.data.qpos[address]), 5) for key, address in self.buttons.items()},
            "indicators": {key: self.model.geom_rgba[geom, :3].round(3).tolist() for key, geom in self.indicators.items()},
        }
        return packet
