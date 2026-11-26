"""One MuJoCo world for the two robots, the fixtures and their contact geometry."""

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from gait import StairFlight
from pump_survey import CART_X, COVER, FLIGHTS, LANES, PUMPS, UNDERPIPE_Y, PumpVariant
from factory_layout import FACTORY, HEAT_LADDER_ENTRY, HEAT_VIEW_PORT, LIFT_XY, PUMP_EXTENT, UPPER_DECKS, add_factory
from yard_layout import YARD, TOWER_X, build_yard


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
CATALOG = {item["id"]: item for item in json.loads((ROOT / "scenes/catalog.json").read_text(encoding="utf-8"))}
CATALOG["yard"] = {**YARD, "steps": YARD["orders"][0]["steps"]}
ROBOT_XML = {name: ET.parse(ROOT / f"model/{name}.xml").getroot() for name in ("g1", "go2")}

STEEL = (.39, .46, .51)
BLUE = (.20, .38, .54)
INK = (.17, .23, .27)
WHITE = (.86, .89, .90)
YELLOW = (.88, .65, .19)
FLOOR = (.81, .83, .83)
BROWN = (.67, .50, .33)
PAN_MASS = .22
PAN_STIFFNESS = 2200.
PUMP_FLIGHT = StairFlight()
BASKET_X = -.16
BASKET_FLOOR = .130

GALV = (.55, .60, .62)
RUBBER = (.17, .19, .20)
CONCRETE = (.79, .80, .79)

GRIP_HALF = {
    "pump": (.047, .032, .028),
    "cnc": (.063, .030, .022),
    "pack": (.046, .028, .034),
    "store": (.066, .038, .035),
}
HANDLE_Z = {"pump": .076, "cnc": .017, "pack": .022, "store": .038}

POSITIONS = {
    "pump": {"dock": (-3.40, -1.75, 0.), "door": COVER, **{name: (x, -1.45, .72) for name, x in PUMPS.items()}},
    "cnc": {"pick": (-2.85, .62, .70), "reference": (-.22, .62, .78), "target": (2.75, .62, .70)},
    "pack": {"button": (-2.30, -.38, .76), "pick": (-.15, .60, .70), "target": (2.42, .60, .71)},
    "store": {"pick": (-2.64, .55, .66), "reference": (.05, -.95, .74), "target": (2.75, .55, .66)},
    "air": {"door": (.42, .60, .73)},
}


def vec(values):
    return " ".join(f"{float(x):.8g}" for x in values)


def named_id(model, kind, name):
    value = mj.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Missing MuJoCo object {name}")
    return value


class Plant:
    def __init__(self, scene, alternate, variant, factory=None):
        self.scene = scene
        self.points = POSITIONS.get(scene, {}).copy()
        self.alternate = alternate
        self.variant = variant
        self.factory = factory
        self.shapes = []
        self.basket = []
        self.cart = []
        self.obstacles = []
        self.labels = []
        self.width, self.depth = (FACTORY["extent"] if factory else
                                  YARD["extent"] if scene == "yard" else PUMP_EXTENT if scene == "pump" else (11.8, 7.8))
        self.xml = deepcopy(ROBOT_XML["g1"])
        self.xml.set("model", f"collaboration-{scene}")
        self.xml.remove(self.xml.find("keyframe"))
        self.world = self.xml.find("worldbody")
        self.xml.find("option").attrib.update(timestep=".004", iterations="20", cone="elliptic")
        self.equality = ET.SubElement(self.xml, "equality")
        self._attach_go2()
        self._position_servos()
        self._gripper()
        for name, body in (("g1", "pelvis"), ("go2", "go_base")):
            ET.SubElement(self.world, "body", name=f"{name}_guide", mocap="true")
            ET.SubElement(self.equality, "weld", name=f"{name}_guidance", body1=f"{name}_guide", body2=body,
                          relpose="0 0 0 1 0 0 0", solref=".010 1", torquescale=".7")
        self._structure()
        getattr(self, f"_{scene}")()
        if factory:
            add_factory(self)
        if scene in GRIP_HALF:
            self._workpiece()

    def _attach_go2(self):
        source = deepcopy(ROBOT_XML["go2"])
        for node in source.iter():
            for attr in ("name", "class", "childclass", "joint", "body1", "body2", "geom1", "geom2"):
                if attr in node.attrib:
                    node.set(attr, f"go_{node.get(attr)}")
        for tag in ("default", "worldbody", "actuator"):
            self.xml.find(tag).extend(source.find(tag))
        base = self.world.find('body[@name="go_base"]')
        base.find("freejoint").set("name", "go_float")
        self.go2 = base
        self._attach_basket(base)

    def _attach_basket(self, base):
        basket = ET.SubElement(base, "body", name="go_basket", pos=vec((BASKET_X, 0, 0)))
        ET.SubElement(basket, "inertial", pos=vec((0, 0, BASKET_FLOOR+.0225)),
                      mass=".27", diaginertia=".0022 .0031 .0039")
        blue = (.21, .36, .43)

        def board(x, y, z, w, d, h, color, name=None, visible=True, surface="plastic"):
            center, half = (x, y, BASKET_FLOOR+z), (w/2, d/2, h/2)
            if visible:
                self.basket.append(self.shape("box", center, half, color, None, surface))
            if name:
                self.geom("box", center, half, color, parent=basket, name=name, conaffinity="7")

        board(0, 0, -.012, .34, .254, .024, RUBBER, "basket_liner", surface="rubber")
        board(.006, 0, -.0285, .268, .221, .010, INK, "basket_carrier", surface="metal")
        for side in (-1, 1):
            side_id = "neg" if side < 0 else "pos"
            board(0, side*.120, .035, .34, .015, .073, blue, f"basket_side_{side_id}", False)
            board(0, side*.120, .0075, .34, .014, .019, blue)
            board(0, side*.120, .0615, .347, .022, .019, blue)
            for x in np.linspace(-.15, .15, 8):
                board(float(x), side*.120, .0335, .016, .013, .049, blue)
            top = .071 if side > 0 else .053
            board(side*.164, 0, (top-.0015)/2, .015, .234, top+.0015, blue, f"basket_end_{side_id}", False)
            board(side*.164, 0, .0075, .013, .236, .019, blue)
            board(side*.164, 0, top-.0095, .022, .245, .019, blue)
            for y in np.linspace(-.101, .101, 6):
                board(side*.164, float(y), (top+.0065)/2, .014, .016, top-.0065, blue)

            # The pads straddle the body shell. Short risers connect them to the carrier.
            for end, x in (("rear", -.020), ("front", .106)):
                suffix = f"{end}_{side_id}"
                board(x, side*.049, .066-BASKET_FLOOR, .050, .037, .022, RUBBER,
                      f"basket_mount_pad_{suffix}", surface="rubber")
                board(x, side*.049, .0885-BASKET_FLOOR, .041, .031, .024, STEEL,
                      f"basket_mount_riser_{suffix}", surface="metal")

    def _position_servos(self):
        for index, motor in enumerate(self.xml.find("actuator")):
            if index < 29:
                torque = float(motor.get("ctrlrange").split()[1])
                arm = index >= 15
                gain = (180. if index not in (19, 20, 21, 26, 27, 28) else 52.) if arm else 285.
                damping = 9. if arm else 14.
            else:
                torque = 45.43 if (index - 29) % 3 == 2 else 23.7
                gain, damping = 75., 3.5
            motor.tag = "position"
            motor.attrib.pop("ctrlrange", None)
            motor.attrib.pop("class", None)
            motor.attrib.update(kp=str(gain), kv=str(damping), forcelimited="true", forcerange=vec((-torque, torque)))

    def _gripper(self):
        wrist = self.world.find('.//body[@name="right_wrist_yaw_link"]')
        palm = ET.SubElement(wrist, "body", name="clamp", pos=".045 0 0")
        ET.SubElement(palm, "geom", name="gripper_housing", type="box", size=".030 .040 .020", pos=".026 0 0",
                      mass=".165", contype="32", conaffinity="1")
        ET.SubElement(palm, "site", name="grip", pos=".107 0 0", size=".008")
        ET.SubElement(palm, "geom", name="probe", type="sphere", pos=".137 0 0", size=".013",
                      mass=".015", contype="16", conaffinity="8")
        for side, sign in (("left", 1), ("right", -1)):
            finger = ET.SubElement(palm, "body", name=f"jaw_{side}", pos=vec((.106, sign * .048, 0)))
            joint = f"jaw_{side}_slide"
            ET.SubElement(finger, "joint", name=joint, type="slide", axis=vec((0, -sign, 0)),
                          limited="true", range="0 .040", damping="1.0", armature=".001")
            ET.SubElement(finger, "geom", name=f"pad_{side}", type="box", size=".035 .009 .023",
                          mass=".035", friction="1.8 .03 .006", contype="4", conaffinity="3")
            ET.SubElement(self.xml.find("actuator"), "position", name=f"close_{side}", joint=joint,
                          kp="270", kv="3", ctrllimited="true", ctrlrange="0 .040",
                          forcelimited="true", forcerange="-18 18")

    @staticmethod
    def shape(kind, center, half, color, rotation, surface):
        return {"type": kind, "p": center, "s": half, "color": color, "q": rotation, "surface": surface}

    def geom(self, kind, center, half, color, parent=None, name=None, collide=True, rotation=None,
             surface="paint", nav=True, render=True, **extra):
        root = self.world if parent is None else parent
        if parent is None:
            if render:
                self.shapes.append(self.shape(kind, center, half, color, rotation, surface))
            if collide and nav:
                axes = np.eye(3).ravel()
                if rotation is not None:
                    mj.mju_quat2Mat(axes, np.asarray(rotation))
                axes = axes.reshape(3, 3)
                if kind == "cylinder":
                    axis = axes[:, 2]
                    radius, height = half
                    bounds = np.abs(axis)*height + np.sqrt(np.maximum(0., 1.-axis*axis))*radius
                else:
                    bounds = np.abs(axes) @ half
                self.obstacles.append(tuple(np.r_[center-bounds, center+bounds]))
        if collide:
            attrs = dict(type=kind, pos=vec(center), size=vec(half), rgba=vec((*color, 1.)),
                         contype="1", conaffinity="3", friction="1.05 .012 .002")
            if name:
                attrs["name"] = name
            if rotation:
                attrs["quat"] = vec(rotation)
            attrs.update(extra)
            return ET.SubElement(root, "geom", attrs)
        return None

    def box(self, x, y, z, w, d, h, color, **kwargs):
        return self.geom("box", (x, y, z), (w / 2, d / 2, h / 2), color, **kwargs)

    def tube(self, a, b, radius, color, collide=False, **kwargs):
        delta = np.array(b) - a
        length = float(np.linalg.norm(delta))
        unit = delta / length
        cross = np.cross((0., 0., 1.), unit)
        q = np.r_[1 + unit[2], cross]
        q = q / np.linalg.norm(q) if np.linalg.norm(q) > 1.e-8 else np.array([0., 1., 0., 0.])
        self.geom("cylinder", tuple((np.array(a) + b) / 2), (radius, length / 2), color,
                  collide=collide, rotation=tuple(float(n) for n in q), **kwargs)

    def stair_stringer(self, y, name):
        lower = np.array((-5.065, y, .074))
        upper = np.array((-.255, y, 2.682))
        delta = upper - lower
        span = float(np.linalg.norm(delta))
        angle = math.atan2(delta[0], delta[2]) / 2
        rotation = (math.cos(angle), 0., math.sin(angle), 0.)
        self.geom("box", tuple((lower+upper)/2), (.090, .045, span/2), STEEL, name=name, rotation=rotation,
                  nav=False, surface="metal")

    def label(self, title, x, y, z):
        self.labels.append({"name": title, "p": (x, y, z)})

    def flange(self, x, y, z, radius, vertical=False):
        start, end = ((x, y, z-.023), (x, y, z+.023)) if vertical else ((x-.023, y, z), (x+.023, y, z))
        self.tube(start, end, radius, GALV, surface="metal")
        for i in range(8):
            angle = math.tau*i/8
            a, b = .77*radius*math.cos(angle), .77*radius*math.sin(angle)
            if vertical:
                lo, hi = (x+a, y+b, z+.023), (x+a, y+b, z+.037)
            else:
                lo, hi = (x-.037, y+a, z+b), (x-.023, y+a, z+b)
            self.tube(lo, hi, .008, INK, surface="metal")

    def gauge(self, x, y, z, radius):
        self.tube((x, y+.019, z), (x, y-.022, z), radius, GALV, surface="metal")
        self.tube((x, y-.023, z), (x, y-.026, z), radius*.84, WHITE, surface="porcelain")
        for i in range(8):
            angle = math.radians(-38 + 37*i)
            dx, dz = math.cos(angle)*radius, math.sin(angle)*radius
            self.tube((x+.60*dx, y-.027, z+.60*dz), (x+.73*dx, y-.027, z+.73*dz), .0017, INK)
        self.tube((x, y-.030, z), (x+.37*radius, y-.030, z+.47*radius), .0027, YELLOW)

    def lamp(self, x, y, z):
        self.box(x, y, z, .093, .089, .059, INK, collide=False)
        for i, color in enumerate(((.21, .52, .32), (.92, .63, .21), (.57, .24, .20))):
            level = z + .073 + i*.048
            self.tube((x, y, level-.022), (x, y, level+.022), .033, color,
                      surface="display" if i == 0 else "plastic")

    def bench(self, point, name, color, kind="table"):
        x, y, z = point
        top = z - GRIP_HALF[self.scene][2] - .031
        sprung_pan = self.scene == "store" and name == "reference"
        center, width, depth = x + .165, .51, .50
        self.box(center, y, top - .024, width, depth, .048, color, name=None if sprung_pan else f"rest_{name}",
                 collide=not sprung_pan, surface="brushed")
        if sprung_pan:
            pan = ET.SubElement(self.world, "body", name="scale_pan", pos=vec((center, y, top - .024)))
            ET.SubElement(pan, "joint", name="cell_slide", type="slide", axis="0 0 1", range="-.015 .004",
                          limited="true", stiffness=str(PAN_STIFFNESS), damping="36")
            self.box(0, 0, 0, width, depth, .048, color, parent=pan, name=f"rest_{name}",
                     mass=str(PAN_MASS))
            self.box(center, y, top - .095, width-.014, depth-.010, .035, STEEL)
        for dx in (.012, .34):
            for dy in (-.205, .205):
                leg_height = top - (.113 if sprung_pan else .048)
                self.box(x + dx, y + dy, leg_height / 2, .048, .048, leg_height, STEEL)
                self.box(x + dx, y + dy, .014, .073, .073, .028, RUBBER, collide=False, surface="rubber")
        for dy in (-.204, .204):
            self.box(x+.175, y+dy, .16, .335, .037, .041, STEEL, collide=False)
        self.box(x+.34, y, .16, .040, .43, .041, STEEL, collide=False)
        if kind == "scale":
            self.box(x + .24, y + .194, top + .059, .185, .083, .100, INK)
            self.box(x + .24, y + .151, top + .070, .128, .005, .038, (.36, .72, .69),
                     collide=False, surface="display")
            for dx in (-.040, 0., .040):
                self.box(x + .24+dx, y + .149, top + .034, .019, .008, .011, RUBBER, collide=False, surface="rubber")
        if kind == "nest":
            for side in (-1, 1):
                self.box(x + .04, y + side * .087, top + .017, .18, .022, .034, INK, surface="metal")
        return top

    def _structure(self):
        self.box(0, 0, -.10, self.width, self.depth, .20, FLOOR, name="slab", nav=False,
                 surface="factory-floor" if self.factory else "concrete")
        if self.scene == "yard":
            return
        width, depth = PUMP_EXTENT if self.factory else (self.width, self.depth)
        if not self.factory:
            for x in (-3.94, .04, 3.95):
                self.box(x, 0, .0008, .009, depth-.04, .0016, (.65, .66, .65), collide=False)
            self.box(0, .02, .0009, width-.04, .009, .0018, (.65, .66, .65), collide=False)
        wall = 4.97 if self.scene == "pump" else 3.42
        self.box(0, 3.84, .26, width, .13, .52, CONCRETE, name="back_wall_base", surface="concrete")
        self.box(0, 3.86, (wall-.80+.48)/2, width, .090, wall-.80-.48, WHITE, surface="panel")
        self.box(0, 3.86, wall+.16, width, .105, .32, WHITE, surface="panel")
        self.box(0, 3.839, wall-.40, width, .024, .80, (.46, .64, .70), collide=False, surface="window")
        for x in np.linspace(-5.84, 5.84, 13):
            self.box(float(x), 3.823, wall-.40, .033, .053, .80, STEEL, collide=False, surface="metal")
        for x in (-5.76, -2.93, .0, 2.93, 5.76):
            self.box(x, 3.78, (wall+.32)/2, .138, .19, wall+.32, GALV, surface="panel")
            self.box(x, 3.74, .071, .27, .25, .142, CONCRETE, surface="concrete")
        for z in (.53, wall-.82, wall+.01):
            self.box(0, 3.807, z, width, .093, .039, STEEL, collide=False, surface="metal")
        self.tube((-5.58, 3.713, wall-.96), (5.61, 3.713, wall-.96), .018, GALV, surface="metal")
        for x in (-4.63, -.90, 2.71):
            self.box(x, 3.702, wall-.96, .112, .061, .099, STEEL, collide=False, surface="metal")
            self.box(x+.004, 3.52, wall+.02, 1.19, .145, .059, STEEL, collide=False, surface="metal")
            self.box(x+.004, 3.52, wall-.014, 1.10, .107, .010, WHITE, collide=False, surface="light")
        self.box(5.84, 0, .32, .12, depth, .64, WHITE, name="side_wall_base", surface="concrete")
        if not self.factory:
            lane = -2.95 if self.scene in ("store", "pump") else -2.43
            for y in (lane-.36, lane+.36):
                self.box(0, y, .002, 11.10, .036, .004, YELLOW, collide=False)
            for x in (-5.54, 5.54):
                self.box(x, lane, .002, .038, .75, .004, YELLOW, collide=False)

    def _yard(self):
        build_yard(self, self.variant)

    def _pump(self):
        tread = "grating" if self.factory else "tread"
        self.box(2.75, 1.94, PUMP_FLIGHT.upper - .08, 6.1, 3.61, .16, GALV, name="walkway",
                 nav=False, surface=tread)
        for x in (.18, 3.0, 5.43):
            self.box(x, .22, 1.38, .125, .125, 2.74, STEEL, surface="metal")
            self.box(x, .22, .033, .28, .25, .066, GALV, surface="metal")
            if self.factory:
                for dx in (-.095, .095):
                    self.tube((x+dx, .22, .066), (x+dx, .22, .087), .021, STEEL, surface="brushed")
        self.box(2.76, .21, 2.64, 6.05, .16, .19, STEEL, collide=False, surface="metal")
        if self.factory:
            self.box(2.74, .151, 2.815, 5.94, .052, .084, STEEL, collide=False, surface="metal")
            for start, end in ((.145, 1.29), (2.65, 3.729)):
                self.box(5.72, (start+end)/2, 2.815, .055, end-start, .084, STEEL,
                         collide=False, surface="metal")
        for letter, y in FLIGHTS.items():
            flight = letter.lower()
            for i, x in enumerate(PUMP_FLIGHT.centers):
                height = (i + 1) * PUMP_FLIGHT.riser
                self.box(x, y, height-.023, PUMP_FLIGHT.tread + .002, 1.14, .046, GALV,
                         name=f"tread_{flight}_{i + 1}", surface=tread)
                self.box(x - .14, y, height + .002, .025, 1.10, .004, YELLOW, collide=False)
            for side in (-.59, .59):
                side_name = "south" if side < 0 else "north"
                self.stair_stringer(y+side*.935, f"stringer_{flight}_{side_name}")
                for level, suffix in ((.95, "top"), (.46, "mid")):
                    self.tube((-4.95, y + side, .175+level), (-.45, y + side, 2.80+level), .024,
                              YELLOW, True, nav=False, name=f"rail_{flight}_{side_name}_{suffix}")
                for i in (0, 5, 10, 15):
                    x = -4.95 + .30 * i
                    z = (i + 1) * .175
                    self.tube((x, y + side, z), (x, y + side, z+.95), .022, YELLOW, True, nav=False,
                              name=f"rail_post_{flight}_{side_name}_{i}")
                    self.box(x, y+side, z+.013, .104, .046, .026, GALV, collide=False, surface="metal")
            if not self.factory:
                self.box(-5.49, y-.52, .002, .73, .031, .004, YELLOW, collide=False)
                self.box(-5.49, y+.52, .002, .73, .031, .004, YELLOW, collide=False)
                self.box(-5.85, y, .002, .036, 1.07, .004, YELLOW, collide=False)
        for i, x in enumerate((2.68, 3.30, 3.92), 1):
            self.box(x, 3.39, 3.64, .54, .46, 1.68, INK)
            self.box(x, 3.145, 3.66, .485, .032, 1.54, (.59, .66, .69), collide=False, surface="panel")
            self.box(x, 3.124, 3.96, .281, .022, .205, INK, collide=False, surface="rubber")
            self.box(x, 3.110, 3.97, .239, .006, .159, (.27, .61, .65), name=f"vfd_P{i}",
                     nav=False, surface="display")
            self.box(x+.180, 3.120, 3.56, .032, .026, .148, INK, collide=False, surface="metal")
            for z in np.linspace(3.08, 3.29, 8):
                self.box(x, 3.122, float(z), .31, .012, .013, STEEL, collide=False, surface="rubber")
            self.lamp(x, 3.34, 4.52)
            self.label(f"P{i} 柜", x, 3.09, 4.24)
        for y, edge in ((.145, "south"), (3.729, "north")):
            for i, x in enumerate((.0, .9, 1.8, 2.7, 3.6, 4.5, 5.60)):
                self.tube((x, y, 2.80), (x, y, 3.75), .024, YELLOW, True, name=f"landing_post_{edge}_{i}")
            for z in (3.27, 3.75):
                self.tube((-.24, y, z), (5.72, y, z), .025, YELLOW, True)
        for z in (3.27, 3.75):
            self.tube((-.245, 1.48, z), (-.245, 2.22, z), .025, YELLOW, True)
            segments = ((.145, 1.29), (2.65, 3.729)) if self.factory else ((.145, 3.729),)
            for start, end in segments:
                self.tube((5.72, start, z), (5.72, end, z), .025, YELLOW, True)
        self.tube((-.13, -.47, 1.57), (5.19, -.47, 1.57), .094, BLUE, True, name="pump_header")
        self.tube((-.18, -.83, .38), (5.26, -.83, .38), .079, STEEL, True, name="pump_return")
        for index, (name, x) in enumerate(PUMPS.items(), 1):
            self.box(x, -.46, .084, 1.49, .66, .168, CONCRETE, surface="concrete")
            self.box(x, -.46, .197, 1.29, .49, .061, STEEL, surface="metal")
            self.tube((x-.52, -.46, .46), (x-.21, -.46, .46), .247, BLUE, True, name=f"pump_volute_{index}")
            self.tube((x+.015, -.46, .46), (x+.48, -.46, .46), .183, BLUE, True, name=f"pump_motor_{index}")
            self.tube((x+.47, -.46, .46), (x+.55, -.46, .46), .177, INK, surface="metal")
            self.tube((x-.27, -.46, .46), (x+.035, -.46, .46), .098, YELLOW, True, name=f"shaft_guard_{index}")
            self.tube((x-.37, -.46, .55), (x-.37, -.46, 1.57), .076, BLUE, True)
            self.tube((x-.35, -.47, .38), (x-.35, -.84, .38), .064, STEEL, True)
            self.flange(x-.48, -.46, .46, .211)
            for z in (.73, 1.24, 1.43):
                self.flange(x-.37, -.46, z, .113, vertical=True)
            self.flange(x+.04, -.47, 1.57, .123)
            for rib in range(11):
                xi = x + .068 + rib*.036
                self.tube((xi, -.46, .46), (xi+.012, -.46, .46), .193, BLUE, surface="metal")
            self.box(x+.24, -.46, .661, .173, .138, .083, BLUE, collide=False, surface="panel")
            self.tube((x-.37, -.49, 1.19), (x-.37, -.685, 1.19), .025, GALV, surface="metal")
            self.gauge(x-.37, -.70, 1.19, .064)
            self.tube((x-.37, -.46, .96), (x-.37, -.46, 1.05), .109, STEEL, surface="metal")
            self.tube((x-.37, -.49, 1.035), (x-.37, -.66, 1.035), .018, STEEL, surface="metal")
            self.tube((x-.37, -.667, 1.035), (x-.37, -.696, 1.035), .085, (.57, .22, .16))
            self.label(f"P{index}", x+.23, -.60, .78)
            self.bench(self.points[name], name, STEEL)
        dock_x, dock_y, _ = self.points["dock"]
        if not self.factory:
            for offset in (-.365, .365):
                self.box(dock_x, dock_y+offset, .002, 1.02, .026, .004, YELLOW, collide=False)
            for offset in (-.495, .495):
                self.box(dock_x+offset, dock_y, .002, .028, .74, .004, YELLOW, collide=False)
        self.label("交接位", dock_x-.14, dock_y+.24, .075)
        self.label("B 检修梯", -.59, 2.82, 3.64)
        if self.variant.closed_a:
            self.box(-.27, .89, 3.30, .067, 1.09, .96, YELLOW, name="stair_a_gate")
            for y in np.linspace(.46, 1.30, 5):
                self.box(-.230, float(y), 3.64, .010, .079, .094, INK, collide=False)
            self.label("A 检修梯 · 封闭", -.21, .89, 3.92)
        else:
            self.label("A 检修梯", -.59, .89, 3.64)
        self._inspection_bay()

    def _inspection_bay(self):
        ET.SubElement(self.go2, "site", name="survey_go2", pos=".295 0 .048", size=".006")
        torso = self.world.find('.//body[@name="torso_link"]')
        ET.SubElement(torso, "site", name="survey_g1", pos=".10 0 .45", size=".006")
        y = UNDERPIPE_Y
        self.box(-4.22, y, .8475, 1.80, .90, .135, STEEL, name="low_pipe_rack", surface="metal")
        for side in (-1, 1):
            self.box(-4.22, y+side*.485, .395, 1.80, .070, .79, GALV, name=f"low_rack_side_{side}",
                     surface="metal")
            self.box(-4.22, y+side*.485, .018, 1.88, .162, .036, INK, collide=False, surface="metal")
        for offset in (-.257, 0., .257):
            self.tube((-5.11, y+offset, .947), (-3.32, y+offset, .947), .037, BLUE, surface="metal")
            for x in (-4.97, -4.09, -3.44):
                self.tube((x-.024, y+offset, .947), (x+.024, y+offset, .947), .063, GALV, surface="metal")
        if self.factory:
            # The clamping collars sit on galvanized saddles, leaving the inspection opening clear.
            for x in (-4.97, -4.09, -3.44):
                self.box(x, y, .900, .087, .84, .032, GALV, collide=False, surface="brushed")
            for side in (-1, 1):
                base_y = y+side*.485
                for x in (-5.06, -3.39):
                    self.box(x, base_y, .414, .068, .088, .756, STEEL, collide=False, surface="metal")
                    self.tube((x, base_y, .036), (x, base_y, .056), .023, GALV, surface="brushed")
        self.box(-5.145, y, .41, .050, 1.04, .82, GALV, name="sample_bank_wall", surface="panel")
        for index, name in enumerate(PUMPS):
            axis = y + (index-1)*.23
            self.box(-5.055, axis, .296, .112, .144, .135, INK, surface="panel")
            color = (.91, .43, .14) if name == self.variant.alarm else (.67, .78, .72)
            self.box(-4.996, axis, .318, .012, .102, .065, color, name=f"drip_{name}", nav=False,
                     surface="display")
            self.tube((-5.045, axis, .376), (-5.045, axis, .775), .017, BLUE, surface="metal")
        self.label("低位管架", -4.01, y, 1.06)
        if not self.factory:
            for lane_y in LANES.values():
                for edge in (-.39, .39):
                    for x in np.arange(-3.40, -.30, .42):
                        self.box(float(x), lane_y+edge, .002, .25, .026, .004, YELLOW, collide=False)
        for x, lane_y, text in ((-2.71, LANES["near"], "服务通道"), (-2.71, LANES["far"], "南侧回程通道")):
            self.label(text, x, lane_y+.45, .07)

        self._service_cart()
        x, y, z = COVER
        self.box(x+.55, y-.024, .40, .23, .60, .80, STEEL, name="inspection_box", surface="panel")
        for side in (-.362, .204):
            self.box(x+.184, y+side, .733, .37, .040, .547, INK, surface="panel")
        self.box(x+.184, y-.079, .448, .37, .61, .046, STEEL, surface="panel")
        self.box(x+.184, y-.079, 1.007, .37, .61, .042, STEEL, surface="panel")
        self.box(x+.102, y-.155, .505, .085, .067, .068, INK, surface="panel")
        self.box(x+.068, y-.155, .576, .027, .087, .064, (.20,.71,.44), name="cover_witness", nav=False,
                 surface="display")
        self._inspection_door(COVER)
        self.label("检修观察盖", x+.20, y, 1.05)

    def _service_cart(self):
        cart = ET.SubElement(self.world, "body", name="service_cart", mocap="true",
                             pos=vec((CART_X, LANES[self.variant.cart_lane], 0.)))
        self.geom("box", (0., 0., .39), (.40, .36, .285), STEEL, parent=cart, name="cart_frame")
        self.geom("box", (.371, 0., .80), (.027, .29, .095), YELLOW, parent=cart, name="cart_handle")

        def panel(center, size, color, surface="metal"):
            self.cart.append(self.shape("box", center, tuple(value/2 for value in size), color, None, surface))

        axle = (math.sqrt(.5), math.sqrt(.5), 0., 0.)
        panel((0., 0., .39), (.80, .72, .57), STEEL, "panel")
        panel((0., 0., .692), (.846, .76, .05), INK, "rubber")
        for y in (-.369, .369):
            panel((0., y, .723), (.84, .023, .029), GALV)
        for x in (-.410, .410):
            panel((x, 0., .723), (.024, .72, .029), GALV)
        for z in (.235, .372, .509):
            panel((0., -.362, z), (.718, .014, .123), (.32, .38, .42), "panel")
            panel((0., -.386, z+.032), (.47, .027, .020), GALV, "brushed")
            for x in (-.208, .208):
                panel((x, -.375, z+.032), (.022, .027, .028), GALV)
        for y in (-.252, .252):
            self.cart.append(self.shape("cylinder", (.371, y, .794), (.021, .086), YELLOW, None, "metal"))
        self.cart.append(self.shape("cylinder", (.371, 0., .874), (.021, .262), YELLOW, axle, "metal"))
        for x in (-.30, .30):
            for y in (-.285, .285):
                panel((x, y, .115), (.056, .104, .106), GALV)
                self.cart.append(self.shape("cylinder", (x, y, .075), (.074, .042), RUBBER, axle, "rubber"))
                self.cart.append(self.shape("cylinder", (x, y, .075), (.030, .046), GALV, axle, "brushed"))

    def _cnc(self):
        for index, x in enumerate((-1.40, 2.73), 1):
            self.box(x, 2.57, .143, 2.18, 1.25, .267, INK, name=f"cnc_base_{index}")
            self.box(x, 2.59, 1.11, 2.18, 1.23, 1.67, WHITE, name=f"cnc_enclosure_{index}", surface="panel")
            self.box(x, 2.60, 1.975, 2.10, 1.17, .059, STEEL, collide=False, surface="panel")
            self.box(x-.28, 1.963, 1.18, 1.40, .042, 1.45, INK, collide=False, surface="rubber")
            self.box(x-.29, 1.938, .474, 1.41, .030, .058, GALV, collide=False, surface="metal")
            for dx in (-.614, .035):
                self.box(x+dx, 1.928, 1.172, .63, .047, 1.33, BLUE, collide=False, surface="panel")
                self.box(x+dx, 1.898, 1.28, .504, .015, .751, INK, collide=False, surface="rubber")
                self.box(x+dx, 1.886, 1.285, .455, .010, .70, (.23, .40, .47), collide=False, surface="window")
                self.box(x+dx+.24, 1.872, .894, .033, .056, .212, GALV, collide=False, surface="metal")
            self.box(x+.80, 1.89, 1.30, .424, .12, .534, STEEL, name=f"cnc_panel_{index}")
            self.box(x+.80, 1.826, 1.41, .355, .014, .232, INK, collide=False, surface="rubber")
            self.box(x+.80, 1.815, 1.421, .311, .009, .183, (.26, .51, .59), collide=False, surface="display")
            for row in range(4):
                for col in range(7):
                    self.box(x+.665+col*.044, 1.821, 1.257-row*.035, .028, .016, .022,
                             WHITE if col in (0, 2, 4) else INK, collide=False, surface="plastic")
            self.tube((x+.934, 1.822, 1.094), (x+.934, 1.790, 1.094), .034, (.66, .19, .12))
            self.lamp(x+.89, 2.01, 2.02)
            self.label(f"VMC {index:02}", x-.25, 1.88, 1.81)
            for offset in np.linspace(-.83, .83, 12):
                self.box(x+float(offset), 1.938, .32, .053, .012, .046, GALV, collide=False, surface="metal")
        for name, point in self.points.items():
            self.bench(point, name, BLUE if name == "target" else STEEL,
                       "nest" if name in ("reference", "target") else "table")
        self.label("刀具架", -2.78, .60, .93)
        self.label("预调支座", -.18, .60, 1.04)
        self.label("换型托盘", 2.79, .60, .94)
        self.box(-4.48, 2.46, .46, .81, .51, .66, BLUE, surface="panel")
        self.box(-4.48, 2.46, .805, .88, .57, .032, GALV, collide=False, surface="metal")
        for z in (.35, .49, .63):
            self.box(-4.48, 2.197, z, .73, .010, .113, STEEL, collide=False, surface="panel")
            self.box(-4.48, 2.184, z+.033, .32, .029, .019, GALV, collide=False, surface="metal")
        for yy in (2.23, 2.69):
            for xx in (-4.81, -4.15):
                self.tube((xx, yy - .035, .10), (xx, yy + .035, .10), .078, INK)

    def _pack(self):
        self.box(.63, 1.81, .54, 7.24, .82, .19, STEEL, name="line_frame", surface="metal")
        self.box(.63, 1.81, .628, 7.17, .65, .038, INK, surface="rubber")
        for y in (1.397, 2.224):
            self.box(.63, y, .662, 7.28, .041, .103, GALV, surface="brushed")
            self.box(.63, y, .583, 7.23, .039, .027, INK, collide=False, surface="rubber")
        for x in np.linspace(-2.89, 4.13, 49):
            self.tube((float(x), 1.434, .666), (float(x), 2.18, .666), .031, GALV, surface="brushed")
            self.tube((float(x), 1.376, .667), (float(x), 1.393, .667), .009, INK, surface="metal")
        for x in (-2.83, -.55, 1.78, 4.10):
            for y in (1.47, 2.15):
                self.box(x, y, .24, .073, .073, .48, STEEL)
                self.box(x, y, .018, .164, .117, .036, GALV, collide=False, surface="metal")
                self.tube((x, y, .037), (x, y, .095), .015, INK, surface="metal")
            self.box(x, 1.81, .237, .065, .68, .045, STEEL, collide=False, surface="metal")
        for i, x in enumerate((-2.05, -1.48, 2.73, 3.55)):
            self.box(x, 1.84, .817, .39, .35, .24, BROWN, surface="cardboard")
            self.box(x, 1.84, .939, .066, .349, .004, (.71, .58, .36), collide=False, surface="tape")
            self.box(x+.098, 1.663, .851, .111, .005, .068, WHITE, collide=False, surface="paper")
            for j in (0, 1, 3, 4, 6, 7):
                self.box(x+.060+j*.010, 1.659, .849, .004+(j%3)*.002, .003, .027,
                         INK, collide=False, surface="paper")
        self.tube((-2.32, 2.21, .65), (-2.32, 2.21, 1.02), .025, STEEL, surface="metal")
        self.lamp(-2.32, 2.21, 1.03)
        for y, sign in ((1.37, 1), (2.25, -1)):
            self.box(-.79, y, .769, .081, .047, .067, INK, surface="plastic")
            self.box(-.79, y+sign*.027, .771, .046, .009, .026, (.55, .20, .13),
                     collide=False, surface="display")
        self.bench(self.points["pick"], "pick", STEEL)
        for x in (-.355, .227):
            self.box(x, 1.24, .602, .044, .43, .082, STEEL, surface="metal")
        for y in np.linspace(.947, 1.44, 11):
            self.tube((-.34, float(y), .632), (.21, float(y), .632), .021, GALV, surface="brushed")
        self.bench(self.points["target"], "target", WHITE, "scale")
        x, y, z = self.points["button"]
        self.box(x + .16, y, .35, .120, .14, .70, STEEL)
        self.box(x + .087, y, z, .108, .23, .18, INK)
        self.tube((x+.032, y, z), (x+.059, y, z), .055, YELLOW, surface="plastic")
        self.box(x+.037, y-.075, z-.056, .015, .073, .026, WHITE, collide=False, surface="paper")
        press = ET.SubElement(self.world, "body", name="button", pos=vec((x, y, z)))
        ET.SubElement(press, "joint", name="button_slide", type="slide", axis="1 0 0",
                      range="0 .030", limited="true", stiffness="310", damping="2.5",
                      solreflimit=".005 1", solimplimit=".98 .999 .001")
        self.geom("cylinder", (0, 0, 0), (.042, .014), (.15, .59, .40), parent=press, name="button_head",
                  rotation=(.707107, 0, .707107, 0), mass=".035", contype="8", conaffinity="16")
        self.box(-.20, 1.425, .81, .056, .13, .20, YELLOW)
        self.box(-.20, 1.430, .94, .142, .16, .072, BLUE, collide=False, surface="panel")
        self.label("抽样确认", x + .30, y, 1.02)
        self.label("剔除支路", .04, .56, .91)
        self.label("复核秤", 2.57, .61, .98)

    def _store(self):
        for center in (-2.54, 2.86):
            for x in (center - .69, center + .69):
                for y in (1.119, 1.905):
                    self.box(x, y, 1.10, .068, .060, 2.20, BLUE, surface="panel")
                    self.box(x, y, .021, .151, .114, .042, GALV, collide=False, surface="metal")
                    for z in np.linspace(.16, 2.10, 26):
                        self.box(x, y-.032, float(z), .019, .004, .026, INK, collide=False, surface="rubber")
                self.tube((x, 1.119, .28), (x, 1.905, 1.21), .013, STEEL, surface="metal")
                self.tube((x, 1.905, 1.21), (x, 1.119, 2.12), .013, STEEL, surface="metal")
            for z in (.23, 1.10, 1.84):
                self.box(center, 1.515, z, 1.42, .81, .043, STEEL, surface="tread")
                self.box(center, 1.107, z-.012, 1.42, .058, .071, (.76, .48, .19), surface="panel")
                for dx in (-.43, -.02, .39):
                    bin_color = BLUE if dx < .39 else (.31, .42, .38)
                    self.box(center+dx, 1.505, z+.123, .31, .44, .20, bin_color, surface="plastic")
                    self.box(center+dx, 1.282, z+.207, .32, .028, .037, bin_color, collide=False, surface="plastic")
                    self.box(center+dx, 1.276, z+.135, .142, .009, .053, WHITE, collide=False, surface="paper")
                    self.box(center+dx, 1.278, z+.073, .151, .012, .030, INK, collide=False, surface="rubber")
                    for i in (0, 1, 3, 4, 6, 7):
                        self.box(center+dx-.049+i*.012, 1.270, z+.135, .005, .003, .026,
                                 INK, collide=False, surface="paper")
        self.bench(self.points["pick"], "pick", STEEL)
        self.bench(self.points["reference"], "reference", WHITE, "scale")
        self.bench(self.points["target"], "target", STEEL)
        self.label("库位 A-12", -2.63, .32, .99)
        self.label("复核台", .08, -1.03, 1.00)
        self.label("库位 B-07", 2.84, .32, .98)

    def _air(self):
        for i, x in enumerate((1.54, 3.69), 1):
            self.box(x, 1.51, .081, 1.85, 1.44, .162, STEEL, name=f"compressor_skid_{i}", surface="metal")
            self.box(x, 1.57, .857, 1.75, 1.28, 1.39, BLUE, name=f"compressor_body_{i}", surface="panel")
            self.box(x, 1.57, 1.567, 1.77, 1.30, .066, WHITE, collide=False, surface="panel")
            for offset in (-.401, .401):
                self.box(x+offset, .909, .968, .775, .047, 1.074, WHITE, surface="panel")
                self.box(x+offset-.282, .871, .935, .036, .049, .186, INK, collide=False, surface="metal")
            self.box(x+.40, .877, .942, .589, .022, .771, INK, collide=False, surface="rubber")
            for z in np.linspace(.605, 1.281, 16):
                self.box(x+.40, .859, float(z), .561, .032, .013, GALV, collide=False, surface="metal")
            self.box(x-.366, .879, 1.194, .339, .032, .231, INK, collide=False, surface="plastic")
            self.box(x-.366, .857, 1.248, .266, .009, .109, (.28, .55, .60), collide=False, surface="display")
            for button in range(5):
                self.box(x-.492+button*.064, .853, 1.157, .036, .012, .023,
                         WHITE, collide=False, surface="plastic")
            self.label(f"AC {i:02}", x-.32, .86, 1.46)
        for x in (-2.99, -2.31):
            for y in (2.57, 3.13):
                self.box(x, y, .169, .098, .108, .335, STEEL, surface="metal")
        self.tube((-2.65, 2.85, .462), (-2.65, 2.85, 2.35), .51, STEEL, True, name="receiver_shell", surface="metal")
        for z in (.468, 2.340):
            self.geom("ellipsoid", (-2.65, 2.85, z), (.51, .51, .239), STEEL, surface="metal")
            self.tube((-2.65, 2.85, z-.020), (-2.65, 2.85, z+.020), .518, GALV, surface="brushed")
        self.tube((-2.65, 2.85, 2.56), (-2.65, 2.85, 2.77), .060, STEEL, True, surface="metal")
        self.tube((-2.65, 2.85, 2.77), (4.35, 2.85, 2.77), .055, STEEL, True, surface="metal")
        for x in (1.54, 3.69):
            self.tube((x, 2.85, 2.77), (x, 2.15, 2.77), .050, STEEL, True, surface="metal")
            self.tube((x, 2.15, 2.77), (x, 2.15, 1.57), .050, STEEL, True, surface="metal")
            self.flange(x, 2.15, 2.12, .086, vertical=True)
            self.flange(x, 2.15, 1.64, .087, vertical=True)
        self.box(-.67, 2.79, .53, .79, .69, 1.07, BLUE)
        x, y, z = self.points["door"]
        self.box(x + .31, y - .16, .40, .27, .49, .80, STEEL)
        self.tube((x + .20, y - .19, .67), (x + .20, y - .19, 1.01), .054, BLUE)
        self.tube((x + .095, y - .16, .89), (x + .17, y - .16, .89), .060, WHITE)
        self.tube((x+.23, y-.17, .98), (x+.23, y-.17, 1.113), .027, GALV, surface="metal")
        self.gauge(x+.23, y-.208, 1.107, .061)
        self._inspection_door((x, y, z))
        self.label("过滤器观察罩", x + .13, y - .13, 1.19)
        self.label("储气罐", -2.65, 2.85, 2.94)

    def _inspection_door(self, point):
        x, y, z = point
        door = ET.SubElement(self.world, "body", name="inspection_door", pos=vec((x, y + .18, z)))
        ET.SubElement(door, "joint", name="door_hinge", type="hinge", axis="0 0 1",
                      range="0 .93", limited="true", stiffness="1.25", damping=".40")
        self.box(0, -.205, 0, .033, .41, .48, BLUE, parent=door, name="door_panel",
                 mass="1.3", contype="8", conaffinity="16")
        self.box(-.028, -.314, 0, .043, .075, .055, YELLOW, parent=door, name="door_handle",
                 mass=".07", contype="8", conaffinity="16")
        ET.SubElement(door, "site", name="door_touch", pos="-.027 -.316 0", size=".010")

    def _workpiece(self):
        scene = self.scene
        p = self.points.get("pick", (-2.71, -1.4, .522))
        part = ET.SubElement(self.world, "body", name="workpiece", pos=vec(p))
        ET.SubElement(part, "freejoint", name="piece_float")
        half = GRIP_HALF[scene]
        color = {"pump": YELLOW, "cnc": STEEL, "pack": BROWN, "store": BLUE}[scene]
        self.geom("box", (0, 0, -.031), half, color, parent=part, name="piece_body", mass=".33",
                  contype="2", conaffinity="7")
        self.geom("box", (0, 0, HANDLE_Z[scene]), (.031, .023, .015), INK, parent=part, name="piece_handle",
                  mass=".05", contype="2", conaffinity="7", friction="1.8 .03 .005")
        if scene == "pump":
            for side in (-1, 1):
                self.box(side*.025, 0, .0335, .008, .013, .074, INK, parent=part,
                         name=f"handle_mount_{side+1}", mass=".006", contype="2", conaffinity="3")
        for name, attach in [("hand_grasp", "clamp"), ("basket_clip", "go_basket")]:
            ET.SubElement(self.equality, "weld", name=name, body1=attach, body2="workpiece",
                          active="false", solref=".012 1")
        if scene != "pump":
            ET.SubElement(self.world, "body", name="source_fixture", pos=vec(p))
            ET.SubElement(self.equality, "weld", name="source_grasp", body1="source_fixture", body2="workpiece",
                          active="false", solref=".012 1")
        if scene == "pump":
            for name in PUMPS:
                x, y, z = self.points[name]
                ET.SubElement(self.world, "body", name=f"magnet_{name}", pos=vec((x, y, z)))
                ET.SubElement(self.equality, "weld", name=f"magnet_{name}", body1=f"magnet_{name}",
                              body2="workpiece", active="false", solref=".012 1")
        elif scene in ("cnc", "store"):
            for station in ("reference", "target") if scene == "store" else ("reference",):
                holder = f"locating_{station}"
                if scene == "store" and station == "reference":
                    parent = "scale_pan"
                else:
                    ET.SubElement(self.world, "body", name=holder, pos=vec(self.points[station]))
                    parent = holder
                ET.SubElement(self.equality, "weld", name=holder, body1=parent,
                              body2="workpiece", active="false", solref=".015 1")
        else:
            ET.SubElement(self.world, "body", name="locating_target", pos=vec(self.points["target"]))
            ET.SubElement(self.equality, "weld", name="locating_target", body1="locating_target",
                          body2="workpiece", active="false", solref=".015 1")

    def describe(self):
        layout = {"shapes": self.shapes, "labels": self.labels, "points": self.points, "handleZ": HANDLE_Z.get(self.scene),
                  "extent": [self.width, self.depth], "blocked": self.alternate, "obstacles": self.obstacles,
                  "basket": self.basket, "basketX": BASKET_X, "cart": self.cart}
        if self.scene == "pump":
            layout["surveyLandmarks"] = {
                "lower": [[f"{flight}梯", -5.30, y] for flight, y in FLIGHTS.items()]
                         + [[name, x, -.52] for name, x in PUMPS.items()]
                         + [["交接", *self.points["dock"][:2]], ["低位管架", -4.2, UNDERPIPE_Y]],
                "upper": [[f"{flight}梯", -.36, y] for flight, y in FLIGHTS.items()]
                         + [[f"{name} 柜", x, 3.34] for name, x in zip(PUMPS, (2.68, 3.30, 3.92))],
            }
        if self.factory:
            layout["factory"] = self.factory_devices
            layout["orders"] = FACTORY["orders"]
            layout["zoneLandmarks"] = self.factory_landmarks
            layout["floorMarkings"] = self.floor_markings
            layout["upperDecks"] = UPPER_DECKS
            layout["surveyLandmarks"]["lower"].append(["上楼梯口", *HEAT_LADDER_ENTRY[:2]])
            layout["surveyLandmarks"]["upper"].extend((
                ["检修梯出口", HEAT_LADDER_ENTRY[0], UPPER_DECKS[1][3]], ["H2 巡检台", *HEAT_VIEW_PORT[:2]],
                ["下行升降台", *LIFT_XY],
            ))
        if self.scene == "yard":
            layout.update(orders=YARD["orders"],
                          surveyLandmarks={"lower": [[name, x, 3.55] for name, x in TOWER_X.items()], "upper": []},
                          zoneLandmarks={"tower": [[name, x, 3.55] for name, x in TOWER_X.items()]})
        return layout


@lru_cache(maxsize=6)
def build(scene, alternate=False, variant=None, factory=None):
    if scene == "pump" and variant is None:
        variant = PumpVariant(alternate, "near", "P2", False)
    plant = Plant(scene, alternate, variant, factory)
    return ET.tostring(plant.xml, encoding="unicode"), plant.describe()
