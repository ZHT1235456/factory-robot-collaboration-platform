"""World-space paint and slab joints for the connected factory floor."""

from pump_survey import FLIGHTS, LANES


def floor_markings(portals, stations, orders):
    marks = []

    def stroke(points, kind, width, radius=0., closed=False, dash=None):
        marks.append({"points": points, "kind": kind, "width": width, "radius": radius,
                      "closed": closed, "dash": dash})

    def rectangle(west, south, east, north, kind, width, radius, dash=None):
        stroke([(west, south), (east, south), (east, north), (west, north)],
               kind, width, radius, closed=True, dash=dash)

    for x in range(-20, 24, 4):
        stroke([(x, -12.92), (x, 12.79)], "joint", .019)
    for y in range(-12, 13, 4):
        stroke([(-23.44, y), (23.44, y)], "joint", .019)

    # Two open ends form the west aisle and wrap continuously around the pump house.
    stroke([(-22.88, 1.06), (-6.45, 1.06), (-6.45, 5.64), (6.49, 5.64),
            (6.49, -5.69), (-6.45, -5.69), (-6.45, -1.06), (-22.88, -1.06)],
           "aisle", .153, .84)

    south_zones = [order["bounds"] for order in orders if order["id"] in ("filter", "return")]
    for bounds in (
        (-22.05, 4.84, -8.87, 11.39),
        *south_zones,
        (7.63, 3.96, 22.52, 11.35),
    ):
        rectangle(*bounds, "zone", .104, .26, (.59, .38))

    for lane, y in LANES.items():
        # The shorter edge leaves a break next to the pump-room handoff.
        start = -2.71 if lane == "near" else -3.42
        stroke([(start, y+.40), (-.39, y+.40)], "zone", .082, dash=(.41, .25))
        stroke([(-3.42, y-.40), (-.39, y-.40)], "zone", .082, dash=(.41, .25))

    for y in FLIGHTS.values():
        stroke([(-5.07, y-.53), (-6.00, y-.53), (-6.00, y+.53), (-5.07, y+.53)],
               "aisle", .089, .11)

    for x in portals.values():
        for side in (-1, 1):
            stroke([(x+side*1.26, 1.26), (x+side*1.26, 3.72)], "aisle", .096)
        for offset in (-1.02, -.68, -.34, 0., .34, .68, 1.02):
            stroke([(x+offset, .92), (x+offset, 1.33)], "crossing", .15)

    for name, width, depth in (("dock", 1.12, .80), ("spare_dock", 1.28, .96), ("return_dock", 1.28, .96)):
        x, y, _ = stations[name]
        rectangle(x-width/2, y-depth/2, x+width/2, y+depth/2, "dock", .087, .047)
        west, south, east, north = x-width/2+.085, y-depth/2+.085, x+width/2-.085, y+depth/2-.085
        for offset in (-.55, -.30, -.05, .20, .45, .70, .95):
            a, b = west+max(0., offset), south+max(0., -offset)
            length = min(east-a, north-b)
            if length > .07:
                stroke([(a, b), (a+length, b+length)], "dock", .060)
    return marks
