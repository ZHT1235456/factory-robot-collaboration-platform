"""Keep travel paths clear of the physical fixtures at each floor elevation."""

from functools import lru_cache
import heapq

import mujoco as mj
import numpy as np


FOOTPRINT = {"g1": (.24, 1.40), "go2": (.36, .62)}
CONTACT_DEPTH = .003
CONTACT_FORCE = 2.


class Aisles:
    def __init__(self, layout):
        self.obstacles = np.asarray(layout["obstacles"], dtype=float).reshape(-1, 6)
        self.extent = np.asarray(layout["extent"]) / 2

    @lru_cache(maxsize=8)
    def lanes(self, key, level):
        radius, height = FOOTPRINT[key]
        objects = self.obstacles
        bodies = objects[(objects[:, 5] > level+.035) & (objects[:, 2] < level+height)]
        rectangles = np.c_[bodies[:, :2]-radius, bodies[:, 3:5]+radius]
        if level > 1.:
            low = np.array((-.30, .135))
            high = np.array((5.80, 3.745))
        else:
            low, high = -self.extent, self.extent
        return rectangles, low+radius, high-radius

    @staticmethod
    def clear(start, end, boxes):
        # A slab intersection gives exact clearance for the swept planar centre.
        near = np.zeros(len(boxes))
        far = np.ones(len(boxes))
        for i in range(2):
            delta = end[i]-start[i]
            if abs(delta) < 1.e-8:
                far[(start[i] >= boxes[:, i+2]) | (start[i] <= boxes[:, i])] = -1.
                continue
            limits = (boxes[:, [i, i+2]]-start[i])/delta
            near = np.maximum(near, limits.min(axis=1))
            far = np.minimum(far, limits.max(axis=1))
        return not np.any(near <= far)

    def route(self, key, start, target):
        level = round(float(start[2]), 3)
        boxes, low, high = self.lanes(key, level)
        start, end = start[:2], target[:2]
        if np.any(end < low) or np.any(end > high) or not self.clear(end, end, boxes):
            return None
        if self.clear(start, end, boxes):
            return [target]

        # Walk around inflated corners when a direct path crosses a fixture.
        skirt = .027
        corners = []
        for left, front, right, back in boxes:
            for x in (left-skirt, right+skirt):
                for y in (front-skirt, back+skirt):
                    point = np.array((x, y))
                    if np.all((point >= low) & (point <= high)) and self.clear(point, point, boxes):
                        corners.append(point)
        nodes = np.array([start, end, *corners])
        cost = np.full(len(nodes), np.inf)
        cost[0] = 0.
        parent = {}
        frontier = [(float(np.linalg.norm(end-start)), 0., 0)]
        remaining = set(range(1, len(nodes)))
        while frontier:
            _, distance, here = heapq.heappop(frontier)
            if distance > cost[here]:
                continue
            if here == 1:
                waypoints = [target]
                while parent[here] != 0:
                    here = parent[here]
                    waypoints.append(np.r_[nodes[here], level])
                return waypoints[::-1]
            remaining.discard(here)
            for there in sorted(remaining):
                proposal = distance + float(np.linalg.norm(nodes[there]-nodes[here]))
                if proposal < cost[there] and self.clear(nodes[here], nodes[there], boxes):
                    cost[there] = proposal
                    parent[there] = here
                    heapq.heappush(frontier, (proposal+float(np.linalg.norm(end-nodes[there])), proposal, there))
        return None


class ContactWatch:
    """Watch loaded contacts outside the feet, the hand's tools and secured cargo."""

    def __init__(self, model, robot, part, jaws, probe):
        self.model = model
        roots = model.body_rootid[model.geom_bodyid]
        self.owners = np.zeros(model.ngeom, dtype=np.int8)
        for number, key in enumerate(("g1", "go2"), 1):
            self.owners[roots == robot[key]["body"]] = number
        self.names = tuple(model.geom(i).name for i in range(model.ngeom))
        self.labels = tuple(name or model.body(int(model.geom_bodyid[i])).name
                            for i, name in enumerate(self.names))
        self.floor = {i for i, name in enumerate(self.names)
                      if name in ("slab", "walkway") or name.startswith("tread_")}
        self.feet = {i for i, (name, label) in enumerate(zip(self.names, self.labels))
                     if name in ("go_FL", "go_FR", "go_RL", "go_RR") or
                     name.startswith(("left_pad_", "right_pad_")) or
                     label.endswith(("ankle_roll_link", "ankle_pitch_link"))}
        self.probe = probe
        self.controls = {i for i, name in enumerate(self.names)
                         if name.startswith("button_head") or name in ("door_handle", "door_panel")}
        self.payload = set(np.flatnonzero(roots == part)) if part is not None else set()
        self.carriers = set(jaws) | {i for i, name in enumerate(self.names) if name.startswith("basket_")}
        self.force = np.zeros(6)
        self.frames = 0
        self.streak = {}
        self.max_depth = 0.

    def check(self, data):
        observed = {}
        for i, contact in enumerate(data.contact):
            depth = -float(contact.dist)
            if depth < CONTACT_DEPTH:
                continue
            a, b = contact.geom1, contact.geom2
            a_owner, b_owner = self.owners[a], self.owners[b]
            if a_owner == b_owner:
                continue
            if not a_owner:
                a, b = b, a
            if ((a in self.feet and b in self.floor) or (a == self.probe and b in self.controls)
                    or (a in self.carriers and b in self.payload)):
                continue
            mj.mj_contactForce(self.model, data, i, self.force)
            if self.force[0] < CONTACT_FORCE:
                continue
            pair = (a, b)
            observed[pair] = max(observed.get(pair, 0.), depth)
        previous = self.streak
        self.streak = {pair: previous.get(pair, 0)+1 for pair in observed}
        if observed:
            self.frames += 1
            self.max_depth = max(self.max_depth, max(observed.values()))
        for (a, b), depth in observed.items():
            if depth >= .009 or self.streak[a, b] >= 4:
                return f"{self.labels[a]} / {self.labels[b]}"
        return None
