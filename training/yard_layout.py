"""A paved outdoor cooling-water yard with two cooling-tower cells."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from factory_layout import BLUE, GALV, GREEN, INK, OCHRE, STEEL, WHITE, YELLOW, anchor_plate, indicator, pipe_run


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
YARD = json.loads((ROOT / "scenes/yard.json").read_text(encoding="utf-8"))
TOWER_X = {"T1": -3.15, "T2": 3.15}
YARD_SCANS = ((-8.35, -4.8), (-8.72, 2.8), (-7.63, 5.72), (-.18, 5.82),
              (7.80, 5.79), (8.63, 2.35), (8.56, -4.66), (3.55, -4.82),
              (-.10, -4.82), (-4.10, -4.62), (-.10, -1.10), (-4.92, -1.33))
YARD_WAIT = (-6.60, -3.85, 0.)


def yard_fault(seed):
    return tuple(TOWER_X)[seed % len(TOWER_X)]


def build_yard(plant, variant):
    turf = (.30, .39, .28)
    for side in (-1, 1):
        plant.box(side*11.2, .56, .007, 1.45, 14.8, .014, turf, collide=False, surface="concrete")
    for y in (-5.86, -.37):
        plant.box(0., y, .003, 20.10, .062, .006, YELLOW, collide=False)

    for index, x in enumerate(np.linspace(-11.54, 11.54, 12)):
        plant.box(float(x), 7.47, .81, .062, .065, 1.62, STEEL, surface="metal")
        anchor_plate(plant, float(x), 7.47, 0., .145, .145)
        if index < 11:
            for offset in np.linspace(.16, 1.96, 11):
                plant.box(float(x+offset), 7.477, .87, .018, .018, 1.34, GALV, collide=False, surface="metal")
    for z in (.22, .78, 1.56):
        plant.box(0., 7.47, z, 23.14, .035, .045, GALV, surface="metal")

    for tower, x in TOWER_X.items():
        plant.box(x, 3.52, .14, 3.36, 2.79, .28, GALV, name=f"yard_pad_{tower}", surface="concrete")
        plant.box(x, 3.60, 1.44, 3.12, 2.53, 2.42, INK, name=f"yard_tower_{tower}", surface="metal")
        plant.box(x, 3.59, 2.80, 3.20, 2.61, .31, WHITE, surface="panel")
        plant.geom("cylinder", (x, 3.59, 3.01), (.96, .11), STEEL, surface="brushed")
        plant.geom("cylinder", (x, 3.59, 3.128), (.86, .009), INK, collide=False, surface="metal")
        for wing in range(8):
            angle = wing*np.pi/4
            plant.geom("box", (x+.43*np.cos(angle), 3.59+.43*np.sin(angle), 3.147),
                       (.37, .035, .012), GALV, collide=False,
                       rotation=(float(np.cos(angle/2)), 0., 0., float(np.sin(angle/2))), surface="metal")
        for side in (-1, 1):
            plant.box(x+side*1.51, 2.29, 1.51, .10, .08, 2.43, GALV, collide=False, surface="metal")
            for z in np.linspace(.54, 2.43, 15):
                plant.box(x, 2.29, float(z), 2.83, .088, .046, GALV, collide=False, surface="brushed")
        pipe_run(plant, [(x+1.04, 2.34, .53), (x+1.04, 1.32, .53), (x+.58, 1.32, .53)],
                 .089, .17, BLUE)
        plant.flange(x+1.04, 2.07, .53, .136)
        plant.flange(x+1.04, 1.57, .53, .136)
        plant.box(x, 1.53, .18, .13, .17, .36, STEEL, surface="metal")
        indicator(plant, f"yard_low_{tower}", (x, 1.41, .39), (.222, .016, .086),
                  OCHRE if variant == tower else GREEN)
        plant.box(x, 1.62, .78, .09, .09, 1.56, STEEL, surface="metal")
        indicator(plant, f"yard_gauge_{tower}", (x, 1.57, 1.39), (.188, .018, .112), (.25, .60, .65))
        plant.label(f"{tower} 冷却塔", x, 2.24, 2.81)
        plant.label(f"{tower} 回水读表位", x, 1.55, 1.69)
        plant.points[f"yard_{tower}"] = (x, 1.53, .87)

    for x in (-6.15, 0., 6.15):
        plant.box(x, .36, .008, .80, .25, .016, STEEL, nav=False, collide=False, surface="grating")
    for body, label, position in ((plant.world.find('.//body[@name="torso_link"]'), "g1", ".10 0 .45"),
                                  (plant.go2, "go2", ".305 0 .048")):
        ET.SubElement(body, "site", name=f"survey_{label}", pos=position, size=".006")
