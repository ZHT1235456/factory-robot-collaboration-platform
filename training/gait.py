"""Plan ground-relative footsteps and solve joint references for the plant robots."""

from dataclasses import dataclass
import math

import mujoco as mj
import numpy as np


GO2_FEET = ("FL", "FR", "RL", "RR")
G1_SOLES = ("left_sole", "right_sole")
# First harmonic of shoulder opposition from 35 phase-aligned CMU walks
# distributed by Masaryk University. The amplitude is scaled to G1.
HUMAN_ARM_WAVE = (.9897, .1432)


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))


def yaw_quat(yaw):
    return np.array((math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)))


def pelvis_quat(yaw, pitch, roll):
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    return np.array((cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
                     cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr))


def sole_rotation(yaw, pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return rotation(yaw) @ np.array(((c, 0., s), (0., 1., 0.), (-s, 0., c)))


@dataclass(frozen=True)
class StairFlight:
    edge: float = -5.10
    tread: float = .30
    riser: float = .175
    count: int = 16

    @property
    def edges(self):
        return self.edge + np.arange(self.count) * self.tread

    @property
    def centers(self):
        return self.edges + .5 * self.tread

    @property
    def upper(self):
        return self.riser * self.count

    def height(self, x):
        return int(np.searchsorted(self.edges, x, side="right")) * self.riser

    def clearance(self, heel):
        # The rear pad must clear the higher tread before the foot descends.
        blend = .032
        return float(self.riser * np.sum(smooth((heel - self.edges + blend) / blend)))


@dataclass
class Frame:
    ground: np.ndarray
    yaw: float
    feet: np.ndarray
    sole_yaw: np.ndarray
    support: np.ndarray
    lowering: float
    arm_swing: float
    sole_pitch: np.ndarray | None = None
    pelvis_pitch: float = 0.
    pelvis_roll: float = 0.
    pelvis_twist: float = 0.
    walking: float = 0.


class Footsteps:
    def __init__(self, robot, offsets, dt):
        self.g1 = robot == "g1"
        self.offsets = offsets
        self.groups = ((0,), (1,)) if self.g1 else ((0, 3), (1, 2))
        self.dt = dt

    def planted(self, ground, yaw):
        return ground + self.offsets @ rotation(yaw).T

    def turn(self, ground, yaw, angle):
        # Finish with both feet (or both diagonals) set to the new heading.
        increment = .65 if self.g1 else .34
        steps = max(1, math.ceil(abs(angle) / increment))
        dest = yaw + np.r_[np.arange(1, steps + 1) / steps, 1.] * angle
        period = .34 if self.g1 else .27
        lead = int(angle > 0.)
        return self._frames(ground, yaw, [ground] * len(dest), dest, period, lead=lead, turning=True)

    def walk(self, start, end, yaw, speed, stairs=None):
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        forward = (end[:2] - start[:2]) @ rotation(yaw)[:2, 0] > 0.
        rocking = 0.
        if stairs is not None:
            if not self.g1 or start[0] <= end[0]:
                raise ValueError("The factory stair route expects G1 to descend")
            bias_x = float((self.offsets @ rotation(yaw).T)[0, 0])
            xs = np.r_[-.12, stairs.centers[::-1], end[0] + bias_x]
            xs = [x for x in xs if end[0] + bias_x - 1.e-6 <= x < start[0] + bias_x]
            stations = [np.array((x - bias_x, end[1], stairs.height(x))) for x in xs]
            if len(stations) == 0 or np.linalg.norm(stations[-1] - end) > .004:
                stations.append(end)
            stations.append(end)
            period = max(.49, stairs.tread / speed)
        else:
            max_step = (.405 if forward else .30) if self.g1 else .207
            count = max(1, math.ceil(distance / max_step))
            stations = [start + (end - start) * fraction for fraction in np.arange(1, count + 1) / count]
            stations.append(end)
            period = max((.49 if forward else .45) if self.g1 else .27, distance / count / speed)
            if self.g1 and forward:
                rocking = min(1., distance / count / .34) * float(smooth((speed-.42)/.34))
        return self._frames(start, yaw, stations, [yaw] * len(stations), period, stairs=stairs, rocking=rocking)

    def _frames(self, start, yaw, stations, headings, period, stairs=None, lead=0, turning=False, rocking=0.):
        ticks = max(4, round(period / self.dt))
        count = len(stations)
        supports = np.repeat(start[None], 2, axis=0)
        support_yaw = np.full(2, yaw)
        centers = [np.r_[start, yaw]]
        for i, (point, heading) in enumerate(zip(stations, headings)):
            group = (lead + i) % 2
            supports[group] = point
            support_yaw[group] = heading
            center = supports.mean(0)
            if stairs is not None:
                center[2] = min(supports[:, 2])
            centers.append(np.r_[center, support_yaw.mean()])
        centers = np.stack(centers)
        tangent = np.zeros_like(centers)
        tangent[1:-1] = .5 * (centers[2:] - centers[:-2])
        feet = self.planted(start, yaw)
        sole_yaw = np.full(len(feet), yaw)
        lift = .064 if stairs is not None else .057 if self.g1 else .068
        drop = .088 if stairs is not None else .031 if self.g1 else .015
        beat = ticks*self.dt
        total = count*beat
        fade_in, fade_out = (.32, .34) if self.g1 else (.31, .31)
        for i, (point, heading) in enumerate(zip(stations, headings)):
            group = (lead + i) % 2
            swing = list(self.groups[group])
            initial = feet[swing].copy()
            destination = self.planted(point, heading)[swing]
            initial_yaw = sole_yaw[swing].copy()
            for j in range(1, ticks + 1):
                u = j / ticks
                h = u * u * (3. - 2. * u)
                pose = ((1. - h) * centers[i] + h * centers[i + 1]
                        + u * (1. - u) * ((1. - u) * tangent[i] - u * tangent[i + 1]))
                raise_at, land_at = (.105, .885) if rocking else (.10, .90)
                phase = float(np.clip((u - raise_at) / (land_at-raise_at), 0., 1.))
                margin = .035 if rocking else .13 if self.g1 else .19
                travel = float(smooth((phase - margin) / (1. - 2. * margin)))
                feet[swing] = initial + travel * (destination - initial)
                sole_yaw[swing] = initial_yaw + travel * (heading - initial_yaw)
                in_air = 0. < phase < 1.
                support = np.ones(len(feet), dtype=bool)
                if in_air:
                    support[swing] = False
                    if stairs is not None:
                        for foot in swing:
                            floor = stairs.clearance(feet[foot, 0] + .091)
                            feet[foot, 2] = max(feet[foot, 2], floor + self.offsets[foot, 2])
                    rise, fall = (.48, .52) if rocking else (.27, .27) if self.g1 else (.24, .24)
                    clearance = float(smooth(phase / rise) * smooth((1. - phase) / fall))
                    feet[swing, 2] += lift * clearance

                elapsed = i*beat + j*self.dt
                envelope = float(smooth(elapsed / fade_in) * smooth((total - elapsed) / fade_out))
                ground = pose[:3].copy()
                direction = rotation(float(pose[3]))
                torso_yaw = float(pose[3])
                if not self.g1:
                    bounce = .005 * math.sin(math.pi * u) ** 2 * envelope
                    yield Frame(ground, torso_yaw, feet.copy(), sole_yaw.copy(), support, drop*envelope-bounce, 0.)
                    continue

                parity = 2*group-1
                midstance = math.sin(math.pi*u)**2
                stride = rocking*envelope
                sway = .009 if turning else .018 if stairs is not None else .016+.009*rocking
                ground[:2] += parity*sway*midstance*envelope*direction[:2, 1]
                separation = float((feet[0]-feet[1]) @ direction[:, 0])
                arm = float(np.clip(separation*.63, -.24, .24)) * envelope
                shoulder_wave = HUMAN_ARM_WAVE[0]*math.cos(math.pi*u) + HUMAN_ARM_WAVE[1]*math.sin(math.pi*u)
                arm = arm*(1.-rocking) + parity*.22*shoulder_wave*stride
                twist = -.047 * float(np.clip(separation/.39, -1., 1.)) * stride
                pitch = (.043 if stairs is not None else .026*rocking) * envelope
                roll = parity*(.015+.024*rocking)*midstance*envelope
                rise = (.008 if stairs is not None else .010+.007*rocking)*midstance
                pitches = np.zeros(2)
                current = feet.copy()
                if rocking:
                    if u < raise_at:
                        toe = .27 * smooth(u/raise_at)
                    else:
                        toe = .27*(1.-smooth(phase/.30)) - .23*smooth((phase-.37)/.38)
                    pitches[group] = float(toe)*stride
                    if i:
                        # The heel stays planted as the forefoot lowers on the next beat.
                        boundary = float(smooth(i*beat/fade_in)*smooth((total-i*beat)/fade_out))
                        pitches[1-group] = -.23*rocking*boundary*(1.-smooth(u/.25))
                    for leg, angle in enumerate(pitches):
                        # Revolve about the toe or heel pad instead of the centre of the sole.
                        pivot = math.copysign(.085, angle)
                        correction = np.array((pivot*(1.-math.cos(angle)), 0., pivot*math.sin(angle)))
                        current[leg] += rotation(sole_yaw[leg]) @ correction
                yield Frame(ground, torso_yaw, current, sole_yaw.copy(), support, (drop-rise)*envelope, arm,
                            pitches, pitch, roll, twist, envelope)


class LegIK:
    """Solve the existing MJCF chains at the foot positions scheduled above."""

    def __init__(self, model, scratch, robots):
        self.m = model
        self.d = scratch
        self.human = robots["g1"]
        self.sites = [model.site(name).id for name in G1_SOLES]
        self.ground_pads = [model.geom("go_" + name).id for name in GO2_FEET]
        self.hips = np.array([model.body("go_" + name + "_hip").pos for name in GO2_FEET])
        self.lateral = np.array([model.body("go_" + name + "_thigh").pos[1] for name in GO2_FEET])
        self.thigh = -model.body("go_FL_calf").pos[2]
        pad = model.geom_pos[self.ground_pads[0]]
        self.calf = math.hypot(pad[0], pad[2])
        self.shin_offset = math.atan2(-pad[0], -pad[2])
        self.ranges = {}
        for key, robot in robots.items():
            actuator = robot["act"]
            joint = model.actuator_trnid[actuator:actuator+12, 0]
            self.ranges[key] = model.jnt_range[joint].copy()
        self.refs = np.tile([-.125, 0., 0., .25, -.125, 0.], 2)
        self.jp = np.empty((3, model.nv))
        self.jr = np.empty_like(self.jp)
        self.reg = .00013 * np.eye(6)
        self.human_input = None

    def positions(self, key, data):
        if key == "g1":
            return data.site_xpos[self.sites].copy()
        return data.geom_xpos[self.ground_pads].copy()

    def go2(self, root, yaw, feet):
        local = (feet - root) @ rotation(yaw) - self.hips
        x, y, z = local.T
        below = -np.sqrt(np.maximum(.0001, y * y + z * z - self.lateral**2))
        roll = (np.arctan2(z, y) - np.arctan2(below, self.lateral) + np.pi) % (2. * np.pi) - np.pi
        u, v = -x, -below
        thigh, calf = self.thigh, self.calf
        bend = -np.arccos(np.clip((u*u + v*v - thigh*thigh - calf*calf) / (2. * thigh * calf), -1., 1.))
        pitch = np.arctan2(u, v) - np.arctan2(calf * np.sin(bend), thigh + calf * np.cos(bend))
        angles = np.column_stack((roll, pitch, bend - self.shin_offset)).ravel()
        return np.clip(angles, self.ranges["go2"][:, 0] + .004, self.ranges["go2"][:, 1] - .004)

    def g1(self, root, attitude, feet, headings, pitches, state):
        inputs = np.r_[root, attitude, feet.ravel(), headings, pitches]
        if self.human_input is not None and np.array_equal(inputs, self.human_input):
            return self.refs.copy()
        self.human_input = inputs
        r, q, v = (self.human[k] for k in ("root", "q", "v"))
        data = self.d
        data.qpos[:] = state
        data.qpos[r:r+3] = root
        data.qpos[r+3:r+7] = attitude
        data.qpos[q:q+12] = self.refs
        limits = self.ranges["g1"]
        rotations = [sole_rotation(yaw, pitch) for yaw, pitch in zip(headings, pitches)]
        for _ in range(7):
            mj.mj_kinematics(self.m, data)
            mj.mj_comPos(self.m, data)
            max_error = 0.
            for leg, site in enumerate(self.sites):
                angles = slice(q + 6*leg, q + 6*(leg+1))
                dofs = slice(v + 6*leg, v + 6*(leg+1))
                actual = data.site_xmat[site].reshape(3, 3)
                attitude = .5 * np.cross(actual.T, rotations[leg].T).sum(0)
                task = np.r_[feet[leg] - data.site_xpos[site], .27 * attitude]
                residual = float(np.linalg.norm(task))
                max_error = max(max_error, residual)
                if residual < .00007:
                    continue
                mj.mj_jacSite(self.m, data, self.jp, self.jr, site)
                jac = np.vstack((self.jp[:, dofs], .27 * self.jr[:, dofs]))
                adjust = np.linalg.solve(jac.T @ jac + self.reg, jac.T @ task)
                limit = limits[6*leg:6*(leg+1)]
                lower = limit[:, 0].copy() + .004
                lower[3] = .13
                data.qpos[angles] = np.clip(data.qpos[angles] + np.clip(adjust, -.22, .22), lower, limit[:, 1] - .004)
            if max_error < .00007:
                break
        self.refs = data.qpos[q:q+12].copy()
        return self.refs.copy()
