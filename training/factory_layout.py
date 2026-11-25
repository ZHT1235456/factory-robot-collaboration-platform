"""Physical extensions to the existing pump room and their surveyed work positions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from factory_markings import floor_markings


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
FACTORY = json.loads((ROOT / "scenes/factory.json").read_text(encoding="utf-8"))
PUMP_EXTENT = (11.8, 10.4)

FILTER_X = {"F1": 11.75, "F2": 17.10}
SOUTH_PROCESS_Y = -2.37
FILTER_FOR_PUMP = {"P1": "F1", "P2": "F2", "P3": "F1"}
PORTALS = {"west": 9.55, "east": 20.35}
PORTAL_NAMES = {"west": "西门", "east": "东门"}
GATE_SOUTH, GATE_NORTH = 1.03, 3.78
HEAT_VIEWS = {"high": (16.20, 6.09, 0.), "low": (13.62, 6.21, 0.)}
SPARE_POINTS = {"seal": (-18.45, 8.68, .72), "screen": (-14.42, 8.68, .72)}
SPARE_NAMES = {"seal": "S1 密封件包", "screen": "S2 滤网组件"}
SPARE_SCALE = (-11.60, 7.85, .72)
SPARE_DOCK = (-10.24, 6.12, 0.)
RETURN_DOCK = (-15.35, SOUTH_PROCESS_Y-2.50, 0.)
RETURN_REST = (-12.80, SOUTH_PROCESS_Y-1.30, .72)
RETURN_BUTTON = (-18.65, SOUTH_PROCESS_Y-1.72, .76)
RETURN_VIEW = (-16.93, SOUTH_PROCESS_Y-2.74, 0.)
RETURN_STATUS = (-16.93, SOUTH_PROCESS_Y-.55, .53)

# Observation poses follow the ground-floor aisles, including behind the stock racks.
SURVEY_POSES = (
    (-6.75, -2.90), (-21.00, .10), (-22.45, 6.70), (-22.45, 11.79), (-11.20, 11.79),
    (0., 10.62), (0., 5.20), (5.10, 5.20), (10.90, 10.62), (21.10, 10.62),
    (21.15, 5.55), (13.80, 5.55), (7.48, 5.55), (6.72, -.36), (15.00, -.36),
    (22.10, -.36), (22.00, -5.96), (21.02, -9.52), (10.42, -9.52), (4.60, -5.95),
    (4.60, -3.25), (1.03, -3.05), (1.03, -9.52), (-13.50, -9.52), (-21.10, -9.52),
    (-20.50, -5.50), (-9.85, -5.50), (-6.90, -3.45), (-5.98, 2.30), (-5.72, -.62),
    (-4.59, -2.31),
)

BLUE = (.20, .38, .54)
GALV = (.57, .63, .65)
STEEL = (.36, .46, .51)
INK = (.17, .23, .27)
WHITE = (.85, .88, .87)
RUBBER = (.16, .20, .21)
YELLOW = (.88, .65, .19)
OCHRE = (.88, .40, .13)
GREEN = (.22, .67, .48)
WOOD = (.48, .35, .22)
KRAFT = (.67, .51, .34)


@dataclass(frozen=True)
class FactoryVariant:
    closed_portal: str
    spare: str

    @classmethod
    def from_seed(cls, seed):
        return cls(tuple(PORTALS)[(seed//24+seed//2) % len(PORTALS)],
                   tuple(SPARE_POINTS)[(seed//48+seed//3) % len(SPARE_POINTS)])


def point_string(point):
    return " ".join(f"{value:.8g}" for value in point)


def switch(plant, key, point, title):
    x, y, z = point
    plant.box(x+.16, y, .35, .12, .14, .70, STEEL, name=f"stand_{key}", surface="metal")
    plant.box(x+.087, y, z, .108, .23, .18, INK, surface="panel")
    plant.tube((x+.032, y, z), (x+.059, y, z), .055, YELLOW, surface="plastic")
    platform = ET.SubElement(plant.world, "body", name=f"button_{key}", pos=point_string(point))
    ET.SubElement(platform, "joint", name=f"slide_{key}", type="slide", axis="1 0 0", range="0 .030",
                  limited="true", stiffness="310", damping="2.5", solreflimit=".005 1", solimplimit=".98 .999 .001")
    plant.geom("cylinder", (0., 0., 0.), (.042, .014), GREEN, parent=platform,
               name=f"button_head_{key}", rotation=(.707107, 0., .707107, 0.), mass=".035", contype="8", conaffinity="16")
    plant.label(title, x+.17, y, 1.03)
    plant.factory_devices["buttons"][key] = point
    plant.points[key] = point


def indicator(plant, name, point, size, rgb, mutable=False):
    x, y, z = point
    w, d, h = size
    plant.box(x, y+d/2+.025, z, w+.07, .058, h+.05, INK, surface="panel")
    options = {"parent": plant.world} if mutable else {}
    plant.box(x, y, z, w, d, h, rgb, name=name, nav=False, surface="display", **options)
    if mutable:
        plant.factory_devices["indicators"][name] = (point, size, rgb)


def anchor_plate(plant, x, y, floor, width=.27, depth=.24):
    plant.box(x, y, floor+.010, width, depth, .020, STEEL, collide=False, surface="brushed")
    for dx in (-width*.34, width*.34):
        for dy in (-depth*.34, depth*.34):
            plant.tube((x+dx, y+dy, floor+.020), (x+dx, y+dy, floor+.039), .013, GALV, surface="metal")


def hall(plant):
    # Keep the central pump house dimensions and build the extension around its outside doors.
    for x in (-23.39, 23.39):
        plant.box(x, 0., .27, .20, 25.6, .54, WHITE, surface="concrete")
    plant.box(0., 12.79, .27, 46.8, .20, .54, WHITE, surface="concrete")
    plant.box(0., 12.81, 1.90, 46.8, .14, 2.72, WHITE, surface="panel")
    for x in (-23.20, -17.2, -11.4, -5.7, 0., 5.7, 11.4, 17.2, 23.20):
        plant.box(x, 12.60, 1.94, .18, .27, 3.88, STEEL, surface="metal")
        plant.box(x, 12.67, 3.63, .12, .12, .84, STEEL, collide=False, surface="metal")
        plant.box(x, 12.455, 1.94, .25, .043, 3.88, GALV, collide=False, surface="brushed")
        anchor_plate(plant, x, 12.55, 0., .37, .42)
    for x in (-20.1, -14.3, -8.6, -2.85, 2.85, 8.55, 14.3, 20.2):
        plant.box(x, 12.73, 3.55, 5.55, .045, .76, (.53, .69, .75), collide=False, surface="window")
        for offset in (-2.75, -1.38, 0., 1.38, 2.75):
            plant.box(x+offset, 12.695, 3.55, .039, .055, .78, STEEL, collide=False, surface="metal")
        for z in (3.18, 3.91):
            plant.box(x, 12.704, z, 5.55, .072, .045, GALV, collide=False, surface="metal")
        for pane in (-1.39, 1.39):
            plant.box(x+pane, 12.748, 1.88, .019, .012, 2.63, (.48, .53, .55), collide=False, surface="panel")
    plant.box(0., 12.710, .55, 46.75, .056, .058, GALV, collide=False, surface="metal")
    plant.box(0., 12.72, 4.0, 46.8, .22, .16, STEEL, collide=False, surface="metal")
    plant.label("中央检修通道", -1.0, -6.72, .05)


def pipe_run(plant, points, tube_radius, bend_radius, color=BLUE):
    """Sweep visual quarter elbows between right-angle pipe runs."""
    points = [np.array(point, dtype=float) for point in points]
    end = points[0]
    for i, corner in enumerate(points[1:-1], 1):
        first = corner-points[i-1]
        second = points[i+1]-corner
        first_length, second_length = np.linalg.norm(first), np.linalg.norm(second)
        radius = float(min(bend_radius, .44*first_length, .44*second_length))
        before, after = first/first_length, second/second_length
        inlet, outlet = corner-radius*before, corner+radius*after
        plant.tube(end, inlet, tube_radius, color, surface="metal")
        center = inlet+radius*after
        x_axis, y_axis = -after, before
        rotation = np.empty(4)
        mj.mju_mat2Quat(rotation, np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis))).ravel())
        plant.shapes.append(plant.shape("elbow", center.tolist(), (radius, tube_radius),
                                        color, rotation.tolist(), "metal"))
        end = outlet
    plant.tube(end, points[-1], tube_radius, color, surface="metal")


def overhead_lines(plant):
    row = SOUTH_PROCESS_Y
    pipe_run(plant, [(-20.31, row, 2.29), (-21.66, row, 2.29), (-21.66, row, 2.88),
                     (-21.66, 11.50, 2.88), (21.39, 11.50, 2.88), (21.39, -1.69, 2.88),
                     (21.39, -1.69, 2.18), (20.12, -1.69, 2.18)], .107, .34)
    pipe_run(plant, [(19.82, 11.50, 2.88), (19.82, 9.12, 2.88), (19.82, 9.12, 2.15)], .093, .24)
    plant.tube((-21.66, 11.92, 2.89), (21.39, 11.92, 2.89), .105, STEEL, surface="metal")
    for y, z in ((11.50, 2.88), (11.92, 2.89)):
        for x in (-20.65, -14.70, -8.68, -2.70, 3.32, 9.31, 15.30, 21.27):
            plant.tube((x-.025, y, z), (x+.025, y, z), .140, GALV, surface="brushed")
            distance = 12.76-y
            plant.box(x, y+distance/2, z-.14, .105, distance, .081, STEEL, collide=False, surface="metal")
    for x in (-19.7, -10.25, -.66, 10.11, 19.1):
        plant.flange(x, 11.50, 2.88, .151)
    for x, y, z in ((-20.58, row, 2.29), (20.57, -1.69, 2.18)):
        plant.flange(x, y, z, .152)
    for wall, pipe, stations in ((-23.39, -21.66, (-1.46, 3.55, 9.20)),
                                 (23.39, 21.39, (.25, 5.85, 9.52))):
        for y in stations:
            plant.box(wall, y, 1.64, .15, .17, 2.20, STEEL, collide=False, surface="metal")
            plant.tube((wall, y, 2.725), (pipe, y, 2.725), .041, STEEL, surface="metal")
            brace_tip = wall+(pipe-wall)*.86
            plant.tube((wall, y, 1.66), (brace_tip, y, 2.725), .032, GALV, surface="metal")
            plant.tube((pipe, y-.027, 2.88), (pipe, y+.027, 2.88), .131, GALV, surface="brushed")


def handwheel(plant, x, y, z):
    plant.tube((x, y, z-.155), (x, y, z+.011), .024, GALV, surface="metal")
    rim = [np.array((x+.101*math.cos(i*math.tau/16), y+.101*math.sin(i*math.tau/16), z))
           for i in range(16)]
    for a, b in zip(rim, rim[1:]+rim[:1]):
        plant.tube(a, b, .012, INK, surface="metal")
    for i in (0, 4, 8, 12):
        plant.tube((x, y, z), rim[i], .011, INK, surface="metal")


def filtration(plant):
    alarm = FILTER_FOR_PUMP[plant.variant.alarm]
    row = SOUTH_PROCESS_Y
    for key, x in FILTER_X.items():
        plant.box(x+.54, row-.01, .10, 2.36, 1.42, .20, GALV, surface="concrete")
        for offset in (0., 1.17):
            cx = x+offset
            plant.tube((cx, row, .22), (cx, row, 1.64), .47, BLUE, True, surface="metal")
            plant.tube((cx, row, 1.51), (cx, row, 1.62), .54, GALV, surface="metal")
            plant.tube((cx, row, 1.64), (cx, row, 1.97), .11, BLUE, surface="metal")
            for side in (-1, 1):
                plant.box(cx+side*.31, row, .20, .094, .60, .10, INK, collide=False, surface="metal")
                anchor_plate(plant, cx+side*.32, row, .198, .16, .60)
            for i in range(10):
                angle = i*math.tau/10
                px, py = cx+.47*math.cos(angle), row+.47*math.sin(angle)
                plant.tube((px, py, 1.62), (px, py, 1.661), .024, INK, surface="metal")
            plant.flange(cx, row, 1.86, .15, vertical=True)
        plant.tube((x-.60, row, 1.96), (x+1.83, row, 1.96), .10, BLUE, surface="metal")
        plant.tube((x-.60, row, .52), (x+1.83, row, .52), .09, STEEL, True, surface="metal")
        for dx in (-.48, .58, 1.69):
            plant.flange(x+dx, row, .52, .131)
        handwheel(plant, x+1.70, row, 2.20)
        plant.tube((x+1.70, row, 1.99), (x+1.70, row, 2.10), .049, BLUE, surface="metal")
        pipe_run(plant, [(x+.60, row, 1.96), (x+.60, -1.69, 1.96), (x+.60, -1.69, 2.18)],
                 .072, .098)
        plant.tube((x+.24, -2.76, 1.40), (x+.24, -2.99, 1.40), .019, GALV, surface="metal")
        plant.gauge(x+.24, -2.99, 1.40, .09)
        indicator(plant, f"filter_low_{key}", (x+.24, -3.02, .40), (.18, .018, .095),
                  OCHRE if key == alarm else GREEN)
        indicator(plant, f"filter_id_{key}", (x+.24, -3.03, 1.17), (.28, .018, .11), (.32, .61, .66))
        indicator(plant, f"isolated_{key}", (x+.55, -3.03, .68), (.14, .018, .103), OCHRE, mutable=True)
        for z in (.39, .68, 1.15):
            plant.box(x+.37, -2.96, z, .72, .065, .047, STEEL, collide=False, surface="metal")
        switch(plant, f"filter_{key}", (x-.45, -4.02, .76), f"{key} 隔离")
        plant.label(f"{key} 过滤组", x+.59, -2.55, 2.04)
    plant.tube((10.96, -1.69, 2.18), (20.12, -1.69, 2.18), .12, BLUE, surface="metal")
    for x in (10.98, 14.80, 19.59):
        plant.box(x, -1.69, 1.10, .11, .11, 2.20, STEEL, surface="metal")
        anchor_plate(plant, x, -1.69, 0.)


def mesh_guard(plant, x, width):
    y = 2.42
    plant.box(x, y, .28, width, .11, .56, STEEL, surface="panel", render=False)
    plant.box(x, y, .095, width, .075, .19, INK, collide=False, surface="metal")
    plant.tube((x-width/2, y, 1.03), (x+width/2, y, 1.03), .025, YELLOW, surface="metal")
    panels = math.ceil(width/1.65)
    for post in range(panels+1):
        px = x + (post/panels-.5)*(width-.08)
        plant.box(px, y, .52, .084, .083, 1.04, YELLOW, collide=False, surface="metal")
        anchor_plate(plant, px, y, 0., .165, .19)
    for px in np.arange(x-width/2+.095, x+width/2-.045, .122):
        plant.box(float(px), y, .600, .015, .022, .822, STEEL, collide=False, surface="metal")
    for z in np.arange(.24, .994, .119):
        plant.box(x, y, float(z), width-.085, .022, .015, STEEL, collide=False, surface="metal")


def loaded_pallet(plant, x):
    y = 2.46
    plant.box(x, y, .10, 2.40, .85, .20, STEEL, name="cross_bay_pallet", surface="metal", render=False)
    plant.box(x, y, .50, 2.43, .77, .65, OCHRE, name="cross_bay_closed", surface="panel", render=False)
    for off in (-.337, 0., .337):
        plant.box(x, y+off, .028, 2.40, .153, .056, WOOD, collide=False, surface="cardboard")
        for dx in (-1.06, -.35, .36, 1.06):
            plant.box(x+dx, y+off, .102, .21, .15, .10, WOOD, collide=False, surface="cardboard")
    for dx in np.linspace(-1.10, 1.10, 10):
        plant.box(x+float(dx), y, .177, .198, .85, .046, (.58, .42, .26), collide=False, surface="cardboard")
    for col, dx in enumerate((-.81, 0., .81)):
        for row in (0, 1):
            z = .352+row*.310
            clr = KRAFT if (col+row)%2 == 0 else (.59, .43, .29)
            plant.box(x+dx, y, z, .786, .743, .299, clr, collide=False, surface="cardboard")
            plant.box(x+dx, y-.372, z+.002, .047, .004, .296, (.75, .60, .42), collide=False, surface="tape")
            plant.box(x+dx+.235, y-.374, z+.041, .179, .004, .090, WHITE, collide=False, surface="paper")
            for i in (0, 1, 3, 4, 5):
                plant.box(x+dx+.191+i*.011, y-.377, z+.032, .003, .002, .046, INK,
                          collide=False, surface="paper")
        plant.box(x+dx, y, .813, .045, .736, .003, (.77, .61, .44), collide=False, surface="tape")
    for dx in (-.68, .68):
        plant.box(x+dx, y, .818, .043, .753, .007, INK, collide=False, surface="plastic")
        plant.box(x+dx, y-.378, .508, .043, .008, .621, INK, collide=False, surface="plastic")


def exchange(plant):
    for x, w in ((7.36, 1.32), (14.95, 7.94), (22.73, 1.62)):
        mesh_guard(plant, x, w)
        for dx in (-w/2+.04, 0., w/2-.04):
            plant.box(x+dx, 2.42, .52, .084, .083, 1.04, YELLOW, surface="metal", render=False)
    for name, x in PORTALS.items():
        plant.label(PORTAL_NAMES[name], x, 2.43, 1.28)
    loaded_pallet(plant, PORTALS[plant.factory.closed_portal])
    for number, x in enumerate((13.62, 16.20, 19.23), 1):
        plant.box(x, 8.88, .13, 2.01, 1.26, .26, GALV, surface="concrete")
        plant.box(x, 8.91, 1.01, 1.31, .79, 1.33, INK, name=f"exchanger_{number}", surface="metal")
        for plate in range(13):
            plant.box(x-.64+plate*.106, 8.91, 1.00, .038, .86, 1.34, GALV, collide=False, surface="brushed")
        for y in (8.48, 9.34):
            for z in (.59, 1.45):
                plant.tube((x-.76, y, z), (x+.77, y, z), .027, BLUE, surface="metal")
        for dx in (-.74, .74):
            plant.box(x+dx, 8.91, .98, .12, .95, 1.46, BLUE, collide=False, surface="metal")
            for y in (8.57, 9.27):
                anchor_plate(plant, x+dx, y, .26, .26, .23)
        for height in (.65, 1.38):
            plant.tube((x-.56, 8.31, height), (x-.56, 8.48, height), .064, BLUE, surface="metal")
            plant.tube((x-.56, 8.308, height), (x-.56, 8.355, height), .091, GALV, surface="brushed")
        plant.gauge(x+.46, 8.34, 1.47, .074)
        plant.tube((x+.46, 8.37, 1.47), (x+.46, 8.51, 1.47), .019, GALV, surface="metal")
        plant.tube((x+.56, 9.12, 1.28), (x+.56, 9.12, 2.14), .084, BLUE, surface="metal")
        plant.flange(x+.56, 9.12, 1.99, .118, vertical=True)
        plant.label(f"H{number} 板换", x, 8.85, 1.90)
    plant.tube((11.72, 9.12, 2.15), (21.05, 9.12, 2.15), .105, BLUE, surface="metal")
    for name, x, z, color in (("heat_low", 13.62, .42, OCHRE if plant.factory.spare == "seal" else GREEN),
                               ("heat_high", 16.20, 1.32, (.32, .61, .66))):
        plant.box(x, 7.73, .34, .14, .13, .68, GALV, surface="metal")
        plant.tube((x, 7.77, .39), (x, 7.77, z+.05), .035, STEEL, surface="metal")
        indicator(plant, name, (x, 7.64, z), (.28, .020, .10), color)


def spares(plant):
    for key, point in SPARE_POINTS.items():
        x, y, z = point
        plant.bench(point, key, GALV)
        indicator(plant, f"spare_{key}", (x+.16, y-.286, .46), (.24, .018, .095),
                  GREEN if key == plant.factory.spare else (.42, .54, .57))
        plant.label(SPARE_NAMES[key], x+.17, y, 1.02)
        if key != plant.factory.spare:
            plant.box(x+.04, y, z-.031, .126, .081, .056, STEEL, surface="metal")
    plant.bench(SPARE_SCALE, "spare_scale", GALV, "scale")
    plant.label("备件秤台", SPARE_SCALE[0]+.17, SPARE_SCALE[1], 1.02)
    plant.label("交接位", SPARE_DOCK[0], SPARE_DOCK[1], .06)
    for x in (-20.5, -17.0, -13.4):
        plant.box(x, 10.62, 1.26, 2.58, .69, 2.49, BLUE, surface="metal")
        for z in (.27, .74, 1.21, 1.69, 2.16):
            plant.box(x, 10.26, z, 2.53, .056, .072, GALV, collide=False, surface="metal")
            if z < 2.:
                for index, dx in enumerate((-.81, -.27, .27, .81)):
                    carton = z > 1. and index % 3 != 1
                    color = KRAFT if carton else (.34, .50, .62)
                    plant.box(x+dx, 10.20, z+.17, .43, .21, .26, color, collide=False,
                              surface="cardboard" if carton else "plastic")
                    if carton:
                        plant.box(x+dx, 10.093, z+.184, .069, .005, .233, (.75, .62, .45), collide=False, surface="tape")
                    else:
                        plant.box(x+dx, 10.092, z+.212, .198, .007, .038, INK, collide=False, surface="rubber")
                    plant.box(x+dx+.119, 10.090, z+.147, .094, .005, .044, WHITE, collide=False, surface="paper")
        for dx in (-1.27, 1.27):
            plant.box(x+dx, 10.20, 1.26, .068, .068, 2.49, INK, collide=False, surface="metal")
    part_position = SPARE_POINTS[plant.factory.spare]
    ET.SubElement(plant.world, "body", name="spare_source", pos=point_string(part_position))
    part = ET.SubElement(plant.world, "body", name="spare_piece", pos=point_string(part_position))
    ET.SubElement(part, "freejoint", name="spare_float")
    color = OCHRE if plant.factory.spare == "seal" else BLUE
    plant.geom("box", (0., 0., -.031), (.047, .032, .028), color, parent=part, name="spare_body",
               mass=".365", contype="2", conaffinity="7")
    plant.geom("box", (0., 0., .076), (.031, .023, .015), INK, parent=part, name="spare_handle",
               mass=".045", contype="2", conaffinity="7", friction="1.8 .03 .005")
    for side in (-1, 1):
        plant.box(side*.025, 0., .0335, .008, .013, .074, INK, parent=part, mass=".006",
                  contype="2", conaffinity="3")
    for joint, body, active in (("spare_hold", "clamp", False), ("spare_basket", "go_basket", False),
                                ("spare_stock", "spare_source", True), ("magnet_return_rest", "magnet_return_rest", False)):
        ET.SubElement(plant.equality, "weld", name=joint, body1=body, body2="spare_piece", active=str(active).lower(),
                      solref=".012 1")


def return_monitor(plant):
    row = SOUTH_PROCESS_Y
    for index, x in enumerate((-20.28, -17.97, -15.84), 1):
        plant.box(x, row, .09, 1.62, 1.32, .18, GALV, surface="concrete")
        plant.tube((x, row, .23), (x, row, 1.99), .51, BLUE, True, surface="metal")
        plant.tube((x, row, 1.90), (x, row, 2.01), .55, GALV, surface="metal")
        plant.tube((x, row, 2.01), (x, row, 2.29), .066, BLUE, surface="metal")
        for dx in (-.38, .38):
            for dy in (-.33, .33):
                anchor_plate(plant, x+dx, row+dy, .179, .245, .19)
        plant.geom("ellipsoid", (x, row, 2.00), (.511, .511, .155), BLUE, collide=False, surface="panel")
        plant.flange(x, row, 2.19, .115, vertical=True)
        plant.tube((x, row-.27, 1.40), (x, row-.50, 1.40), .021, GALV, surface="metal")
        plant.gauge(x, row-.50, 1.40, .092)
        plant.label(f"R{index} 缓冲罐", x, row-.01, 2.34)
    plant.tube((-20.32, row, 2.29), (-14.85, row, 2.29), .10, BLUE, surface="metal")
    x, y, _ = RETURN_STATUS
    plant.box(x, y+.13, .37, .50, .15, .74, STEEL, name="return_panel_pedestal", surface="panel")
    indicator(plant, "return_status", RETURN_STATUS, (.19, .018, .10), OCHRE, mutable=True)
    switch(plant, "return_check", RETURN_BUTTON, "回水确认")
    plant.bench(RETURN_REST, "return_rest", GALV, "nest")
    ET.SubElement(plant.world, "body", name="magnet_return_rest", pos=point_string(RETURN_REST))
    plant.label("验收托架", RETURN_REST[0]+.17, RETURN_REST[1], 1.03)
    plant.label("交接位", RETURN_DOCK[0], RETURN_DOCK[1], .07)


def add_factory(plant):
    plant.factory_devices = {"buttons": {}, "indicators": {}, "spareType": plant.factory.spare}
    plant.factory_landmarks = {
        "filter": [[key, x+.24, SOUTH_PROCESS_Y-.04] for key, x in FILTER_X.items()],
        "heat": [[PORTAL_NAMES[key], x, 2.42] for key, x in PORTALS.items()]
                + [["H1", 13.62, 8.93], ["H2", 16.20, 8.93], ["H3", 19.23, 8.93]],
        "spare": [[name[:2], *SPARE_POINTS[key][:2]] for key, name in SPARE_NAMES.items()]
                 + [["称重", *SPARE_SCALE[:2]], ["交接位", *SPARE_DOCK[:2]]],
        "return": [["交接位", *RETURN_DOCK[:2]], ["托架", *RETURN_REST[:2]], ["状态牌", *RETURN_STATUS[:2]]],
    }
    plant.points.update(spare_scale=SPARE_SCALE, spare_dock=SPARE_DOCK, return_dock=RETURN_DOCK,
                        spare_stock=SPARE_POINTS[plant.factory.spare], return_rest=RETURN_REST,
                        heat_station=(15.05, 7.88, .75), west_gate=(PORTALS["west"], 2.42, .4),
                        east_gate=(PORTALS["east"], 2.42, .4))
    plant.floor_markings = floor_markings(PORTALS, plant.points, FACTORY["orders"])
    hall(plant)
    overhead_lines(plant)
    filtration(plant)
    exchange(plant)
    return_monitor(plant)
    spares(plant)
