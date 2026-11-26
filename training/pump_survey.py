"""Sensor observations and field coordinates for the circulating-water pump room."""

from dataclasses import dataclass
import math

import mujoco as mj
import numpy as np

from occupancy import FLOOR_FRAME, fuse_rays


PUMPS = {"P1": .65, "P2": 2.60, "P3": 4.56}
FLIGHTS = {"A": .89, "B": 2.81}
LANES = {"near": -2.10, "far": -3.58}
LANE_NAMES = {"near": "服务通道", "far": "南侧回程通道"}
LANE_WEST, LANE_EAST = -3.42, -.38
CART_X = -1.55
LOWER_VIEW = (-5.49, -.58, 0.)
UNDERPIPE_Y = -4.55
UNDERPIPE_ENTRY = (-2.78, UNDERPIPE_Y, 0.)
UNDERPIPE_VIEW = (-3.94, UNDERPIPE_Y, 0.)
COVER = (-2.83, -.71, .73)
COVER_VIEW = (-3.85, -1.36, 0.)
COVER_CLEAR = (-3.86, -2.45, 0.)
DOG_WAIT = (-4.70, -3.30, 0.)
GATE_VIEW = (.63, 1.87, 2.8)


@dataclass(frozen=True)
class PumpVariant:
    closed_a: bool
    cart_lane: str
    alarm: str
    move_cart: bool

    @classmethod
    def from_seed(cls, seed):
        return cls(bool(seed % 2), tuple(LANES)[seed//2 % 2],
                   tuple(PUMPS)[seed//4 % 3], bool(seed//12 % 2))


class PumpSurvey:
    """Fuse range returns into separate occupancy grids in the shared plant frame."""

    resolution = .25
    max_range = 5.0
    sweep_seconds = 1.15
    probe_seconds = .95

    def __init__(self, model, state, robots, item, extent, compact=False, upper_decks=None):
        self.model, self.state = model, state
        self.sites = {key: model.site(f"survey_{key}").id for key in robots}
        ignored = [robot["body"] for robot in robots.values()] + [item]
        roots = model.body_rootid[model.geom_bodyid]
        model.geom_group[np.isin(roots, ignored)] = 5
        dog = robots["go2"]["body"]
        # The nose colliders surround the optical origin. The trunk, legs, and basket can occlude it.
        nose = (model.geom_bodyid == dog) & (model.geom_type != mj.mjtGeom.mjGEOM_BOX)
        self.go2_occluders = (roots == dog) & ~nose
        model.geom_group[self.go2_occluders] = 4
        self.mask = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
        self.go2_scan_mask = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)
        self.origin = -np.asarray(extent)*.5
        self.width, self.height = np.ceil(np.asarray(extent)/self.resolution).astype(int)
        self.compact = compact
        self.upper_decks = upper_decks or ((-.55, .135, 5.80, 3.745),)
        cells = int(self.width*self.height)
        self.empty = f"a{cells:X}" if compact else "0"*cells
        self.grids = {level: np.zeros((self.height, self.width), np.uint8) for level in ("lower", "upper")}
        self.log_odds = {level: np.zeros_like(grid, dtype=np.int8) for level, grid in self.grids.items()}
        self.layer_versions = {"lower": 0, "upper": 0}
        angles = np.linspace(-np.pi, np.pi, 112, endpoint=False)
        self.rays = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        self.distance = np.empty(len(angles))
        self.hits = np.empty(len(angles), np.int32)
        self.steps = np.arange(.06, self.max_range, .14)
        self.revision = 0
        self.scans = {key: 0 for key in robots}
        self.lanes = {}
        self.landings = {}
        self.gates = {}
        self.finding = None
        self.evidence = None
        self.route = None
        self._go2_sweep = None
        self._go2_probe = None
        self._encoded = {}

    def position(self, key):
        return self.state.site_xpos[self.sites[key]]

    def cast(self, key, target):
        source = self.position(key)
        offset = np.asarray(target)-source
        length = float(np.linalg.norm(offset))
        if length < 1.e-6:
            return None, 0., length
        geom = np.empty(1, np.int32)
        direction = offset/length
        distance = mj.mj_ray(self.model, self.state, source, direction, self.mask, 1, -1, geom)
        if key == "go2":
            returned = bool(0. <= distance <= length)
            end = source + direction*(distance if returned else length)
            if self._go2_probe is None or self._go2_probe[0] != self.state.time:
                self._go2_probe = (self.state.time, {"origin": source.round(4).tolist(), "rays": []})
            self._go2_probe[1]["rays"].append([*end.round(4).tolist(), returned])
        name = self.model.geom(int(geom[0])).name if distance >= 0. else None
        return name, float(distance), length

    def visible(self, key, name, fov=53., distance=3.2):
        geom = self.model.geom(name).id
        target = self.state.geom_xpos[geom]
        offset = target-self.position(key)
        length = float(np.linalg.norm(offset))
        if not 1.e-6 < length <= distance:
            return False
        axis = self.state.site_xmat[self.sites[key]].reshape(3, 3)[:, 0]
        if float(axis @ offset)/length < math.cos(math.radians(fov)):
            return False
        return self.cast(key, target)[0] == name

    def clear_to(self, key, point, tolerance=.035):
        name, distance, length = self.cast(key, point)
        return name is None or distance >= length-tolerance

    def _cells(self, points):
        cells = np.floor((points[:, :2]-self.origin)/self.resolution).astype(np.int32)
        valid = (cells[:, 0] >= 0) & (cells[:, 0] < self.width) & (cells[:, 1] >= 0) & (cells[:, 1] < self.height)
        return cells[valid, 1], cells[valid, 0]

    def scan(self, key, *, show_sweep=False):
        source = self.position(key)
        level = "upper" if source[2] > 2.80 else "lower"
        mask = self.go2_scan_mask if key == "go2" else self.mask
        mj.mj_multiRay(self.model, self.state, source, self.rays.ravel(), mask, 1, -1,
                       self.hits, self.distance, len(self.rays), self.max_range)
        hits = (self.distance >= 0.) & (self.distance <= self.max_range)
        occluded = hits & self.go2_occluders[self.hits] if key == "go2" else np.zeros_like(hits)
        hits &= ~occluded
        if show_sweep and key == "go2":
            axis = self.state.site_xmat[self.sites[key]].reshape(3, 3)[:, 0]
            # Signed distances distinguish surface echoes from rays reaching the range limit.
            ranges = np.where(hits, self.distance, -self.max_range).round(3).tolist()
            for index in np.flatnonzero(occluded):
                ranges[index] = None
            self._go2_sweep = (self.state.time, {
                "origin": source.round(4).tolist(),
                "heading": round(math.atan2(axis[1], axis[0]), 4),
                "ranges": ranges,
            })
        limit = np.where(hits, np.maximum(0., self.distance-.12), self.max_range)
        limit[occluded] = 0.
        interior = source + self.rays[:, None, :]*self.steps[None, :, None]
        walls = source + self.rays[hits]*self.distance[hits, None]
        free_y, free_x = self._cells(interior[self.steps[None, :] < limit[:, None]])
        hit_y, hit_x = self._cells(walls)
        if self.compact and level == "upper":
            # Do not label air outside the raised deck as a surveyed walkway.
            def on_deck(x, y):
                center_x = self.origin[0]+(x+.5)*self.resolution
                center_y = self.origin[1]+(y+.5)*self.resolution
                return np.logical_or.reduce([(left <= center_x) & (center_x <= right) &
                                             (south <= center_y) & (center_y <= north)
                                             for left, south, right, north in self.upper_decks])
            clear = on_deck(free_x, free_y)
            obstacle = on_deck(hit_x, hit_y)
            free_y, free_x, hit_y, hit_x = free_y[clear], free_x[clear], hit_y[obstacle], hit_x[obstacle]
        previous = self.grids[level]
        grid = fuse_rays(self.log_odds[level], (free_y, free_x), (hit_y, hit_x))
        if not np.array_equal(grid, previous):
            self.grids[level] = grid
            self._encoded[level] = self.encode(grid.ravel())
            self.revision += 1
            self.layer_versions[level] += 1
        self.scans[key] += 1

    def go2_detection(self):
        effects = {}
        for kind, frame, duration in (("sweep", self._go2_sweep, self.sweep_seconds),
                                      ("probe", self._go2_probe, self.probe_seconds)):
            if frame is not None:
                measured, sample = frame
                age = self.state.time-measured
                if age < duration:
                    effects[kind] = dict(sample, progress=round(age/duration, 3))
        return effects

    def observe_lane(self, lane):
        # From the marked west survey pose the three low rays see the cart body
        # along the usable width. The upper ray catches a protruding cart handle.
        y = LANES[lane]
        endpoints = [(LANE_EAST, y+offset, z) for offset, z in
                     ((0., .26), (-.26, .36), (.26, .36), (0., .97))]
        self.lanes[lane] = "clear" if all(self.clear_to("go2", p) for p in endpoints) else "blocked"

    def observe_landings(self):
        for flight, y in FLIGHTS.items():
            point = (-5.52, y, self.position("go2")[2])
            self.landings[flight] = "clear" if self.clear_to("go2", point) else "blocked"

    def observe_gate(self, flight):
        target = (-.27, FLIGHTS[flight], 3.44)
        self.gates[flight] = "clear" if self.clear_to("g1", target, .016) else "blocked"

    def encode(self, cells):
        if not self.compact:
            return (cells+ord("0")).tobytes().decode("ascii")
        bounds = np.r_[0, np.flatnonzero(cells[1:] != cells[:-1])+1, len(cells)]
        return "".join(f"{chr(97+cells[start])}{end-start:X}" for start, end in zip(bounds[:-1], bounds[1:]))

    def maps(self):
        return {level: self._encoded.get(level, self.empty) for level in self.grids}

    def floor_message(self):
        return {"frame": FLOOR_FRAME, "revision": self.layer_versions["lower"],
                "encoding": "runs", "resolution": self.resolution,
                "origin": self.origin.tolist(), "size": [int(self.width), int(self.height)],
                "cells": self._encoded.get("lower", self.empty)}

    def lower_coverage(self, bounds=None):
        grid = self.grids["lower"]
        if bounds is not None:
            west, south = np.floor((np.asarray(bounds[:2])-self.origin)/self.resolution).astype(int)
            east, north = np.ceil((np.asarray(bounds[2:])-self.origin)/self.resolution).astype(int)
            grid = grid[max(0, south):min(self.height, north), max(0, west):min(self.width, east)]
        return float(np.count_nonzero(grid))/grid.size

    def snapshot(self, known_revision=None):
        result = {"revision": self.revision, "size": [int(self.width), int(self.height)],
                "resolution": self.resolution, "origin": self.origin.tolist(),
                "encoding": "runs" if self.compact else "digits",
                "floorVersions": dict(self.layer_versions),
                "lanes": dict(self.lanes), "landings": dict(self.landings), "gates": dict(self.gates),
                "observations": dict(self.scans), "finding": self.finding, "evidence": self.evidence, "route": self.route}
        if known_revision != self.revision:
            result["floors"] = self.maps()
        return result
