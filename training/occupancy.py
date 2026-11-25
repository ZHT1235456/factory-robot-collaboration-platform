"""Known-pose occupancy mapping and navigation on a received lower-floor grid."""

import heapq
import math
import re

import numpy as np

from navigation import Aisles


FLOOR_FRAME = "factory/lower"
FREE, OCCUPIED = 1, 2
RUN = re.compile(r"([abc])([0-9A-F]+)")


def fuse_rays(log_odds, free_cells, hit_cells):
    """One inverse sensor update per cell, with a stronger update for a return."""
    updates = np.zeros_like(log_odds)
    updates[free_cells] = -2
    updates[hit_cells] = 5
    np.clip(log_odds + updates, -10, 15, out=log_odds)
    return np.where(log_odds < 0, FREE, np.where(log_odds > 0, OCCUPIED, 0)).astype(np.uint8)


class ReceivedFloor:
    def __init__(self, message, origin, resolution, size):
        if (message["frame"] != FLOOR_FRAME or message["encoding"] != "runs" or
                message["origin"] != list(origin) or message["resolution"] != resolution or
                message["size"] != list(size) or type(message["revision"]) is not int or message["revision"] < 1):
            raise ValueError("Invalid floor map metadata")
        self.revision = message["revision"]
        self.origin = np.asarray(origin, dtype=float)
        self.resolution = resolution
        width, height = size
        cells = np.empty(width*height, dtype=np.uint8)
        offset = consumed = 0
        for match in RUN.finditer(message["cells"]):
            code, count = match.groups()
            end = offset + int(count, 16)
            if match.start() != consumed or not offset < end <= cells.size:
                raise ValueError("Invalid occupancy runs")
            cells[offset:end] = ord(code)-ord("a")
            offset, consumed = end, match.end()
        if offset != cells.size or consumed != len(message["cells"]):
            raise ValueError("Truncated occupancy grid")
        self.cells = cells.reshape(height, width)
        self.cells.flags.writeable = False

    def route(self, start, goal, aisles):
        boxes, low, high = aisles.lanes("g1", 0.)
        occupied = np.pad(self.cells == OCCUPIED, 1, constant_values=True)
        height, width = self.cells.shape
        free = self.cells == FREE
        # Inflate returned obstacles by one cell on each side for G1's footprint.
        for row in range(3):
            for col in range(3):
                free &= ~occupied[row:row+height, col:col+width]

        x = self.origin[0] + (np.arange(width)+.5)*self.resolution
        y = self.origin[1] + (np.arange(height)+.5)*self.resolution
        free &= (x[None, :] >= low[0]) & (x[None, :] <= high[0])
        free &= (y[:, None] >= low[1]) & (y[:, None] <= high[1])
        for west, south, east, north in boxes:
            free &= ~((x[None, :] >= west) & (x[None, :] <= east) &
                      (y[:, None] >= south) & (y[:, None] <= north))

        def clear(a, b):
            if not Aisles.clear(a, b, boxes):
                return False
            count = max(1, math.ceil(float(np.linalg.norm(b-a))/(self.resolution*.25)))
            samples = a + np.linspace(0., 1., count+1)[:, None]*(b-a)
            cols, rows = np.floor((samples-self.origin)/self.resolution).astype(int).T
            return (np.all((cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)) and
                    bool(np.all(free[rows, cols])))

        start, goal = np.asarray(start[:2]), np.asarray(goal[:2])
        if np.any(start < low) or np.any(start > high) or np.any(goal < low) or np.any(goal > high):
            return None
        if not clear(start, start) or not clear(goal, goal):
            return None
        if clear(start, goal):
            return [np.r_[goal, 0.]]
        a = tuple(np.floor((start-self.origin)/self.resolution).astype(int))
        b = tuple(np.floor((goal-self.origin)/self.resolution).astype(int))
        cells = self._search(free, a, b)
        if cells is None:
            return None

        path = [start, *(self.origin + (np.asarray(cell)+.5)*self.resolution for cell in cells), goal]
        result = []
        index = 0
        while index < len(path)-1:
            after = len(path)-1
            while after > index and not clear(path[index], path[after]):
                after -= 1
            if after == index:
                return None
            result.append(np.r_[path[after], 0.])
            index = after
        return result

    @staticmethod
    def _search(free, start, goal):
        def heuristic(cell):
            return math.hypot(cell[0]-goal[0], cell[1]-goal[1])

        height, width = free.shape
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        queue = [(heuristic(start), 0., start)]
        cost = {start: 0.}
        parent = {}
        while queue:
            _, distance, here = heapq.heappop(queue)
            if distance > cost[here]:
                continue
            if here == goal:
                path = [here]
                while here != start:
                    here = parent[here]
                    path.append(here)
                return path[::-1]
            x, y = here
            for dx, dy in directions:
                nx, ny = there = x+dx, y+dy
                if not (0 <= nx < width and 0 <= ny < height and free[ny, nx]):
                    continue
                if dx and dy and not (free[y, nx] and free[ny, x]):
                    continue
                proposal = distance + math.hypot(dx, dy)
                if proposal < cost.get(there, math.inf):
                    cost[there], parent[there] = proposal, here
                    heapq.heappush(queue, (proposal+heuristic(there), proposal, there))
        return None
