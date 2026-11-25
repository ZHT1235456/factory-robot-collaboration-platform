import robotXml from '../model/go2.xml?raw';
import sceneXml from '../model/single-step.xml?raw';

const parse = text => new DOMParser().parseFromString(text, 'application/xml');
const model = parse(robotXml);
const scene = parse(sceneXml);
const values = (node, field) => node.getAttribute(field).trim().split(/\s+/).map(Number);
const body = name => model.querySelector('worldbody body[name="' + name + '"]');
const defaults = name => model.querySelector('default[class="' + name + '"]');
const footGeometry = defaults('foot').querySelector('geom');
const start = values(scene.querySelector('key[name="stair_start"]'), 'qpos');
const jointOrder = [...model.querySelectorAll('worldbody joint[name]')].map(node => node.getAttribute('name'));

export const basePosition = start.slice(0, 3);
export const legs = ['FL', 'FR', 'RL', 'RR'].map(id => {
  const parts = ['hip', 'thigh', 'calf'].map(part => body(id + '_' + part));
  const joints = parts.map(part => part.querySelector(':scope > joint'));
  return {
    id,
    hip: values(parts[0], 'pos'),
    thigh: values(parts[1], 'pos'),
    calf: values(parts[2], 'pos'),
    pad: values(footGeometry, 'pos'),
    radius: values(footGeometry, 'size')[0],
    collision: parts.map(part => [...part.querySelectorAll(':scope > geom[class="collision"]')].map(geom => ({
      type: geom.getAttribute('type'), size: values(geom, 'size'),
      pos: geom.hasAttribute('pos') ? values(geom, 'pos') : [0, 0, 0],
      quat: geom.hasAttribute('quat') ? values(geom, 'quat') : [1, 0, 0, 0],
    }))),
    limits: joints.map(joint => values(defaults(joint.getAttribute('class')).querySelector('joint'), 'range')),
    start: joints.map(joint => start[7 + jointOrder.indexOf(joint.getAttribute('name'))]),
  };
});
export const activeLeg = legs[0];

const box = scene.querySelector('geom[name="step"]');
const center = values(box, 'pos');
const half = values(box, 'size');
export const defaultStep = {
  edge: center[0] - half[0], y: center[1], width: 2 * half[1],
  depth: 2 * half[0], height: 2 * half[2],
};
export const toDegrees = radians => radians * 180 / Math.PI;
export const toRadians = degrees => degrees * Math.PI / 180;

export function footPosition(leg, angles) {
  const [roll, hip, knee] = angles;
  const bend = hip + knee;
  const [fx, , fz] = leg.pad;
  const x = leg.calf[2] * Math.sin(hip) + fx * Math.cos(bend) + fz * Math.sin(bend);
  const z = leg.calf[2] * Math.cos(hip) - fx * Math.sin(bend) + fz * Math.cos(bend);
  const y = leg.thigh[1];
  return [
    basePosition[0] + leg.hip[0] + x,
    basePosition[1] + leg.hip[1] + y * Math.cos(roll) - z * Math.sin(roll),
    basePosition[2] + leg.hip[2] + y * Math.sin(roll) + z * Math.cos(roll),
  ];
}

function smooth(t) {
  const s = Math.min(1, Math.max(0, t));
  return s * s * s * (10 + s * (-15 + 6 * s));
}

const origin = footPosition(activeLeg, activeLeg.start);
export function referencePoint(time, step) {
  const u = smooth(time), v = 1 - u;
  // A cubic swing with near-vertical takeoff and landing tangents.
  const advance = u*u*(3-2*u);
  const rise = 3*v*v*u*(step.height+.072) + 3*v*u*u*(step.height+.048) + u*u*u*step.height;
  return [
    origin[0] + (step.edge + Math.min(step.depth * 0.2, 0.055) - origin[0]) * advance,
    origin[1],
    origin[2] + rise,
  ];
}

// The reference swing is planar. The small offset of the MJCF foot sphere is part of the IK.
export function referenceAngles(time, step) {
  const [x, , z] = referencePoint(time, step);
  const leg = activeLeg;
  const u = basePosition[0] + leg.hip[0] - x;
  const v = basePosition[2] + leg.hip[2] - z;
  const l1 = -leg.calf[2];
  const l2 = Math.hypot(leg.pad[0], leg.pad[2]);
  const cosine = (u * u + v * v - l1 * l1 - l2 * l2) / (2 * l1 * l2);
  const bend = -Math.acos(Math.max(-1, Math.min(1, cosine)));
  return [
    0,
    Math.atan2(u, v) - Math.atan2(l2 * Math.sin(bend), l1 + l2 * Math.cos(bend)),
    bend - Math.atan2(-leg.pad[0], -leg.pad[2]),
  ];
}

// Signed distance from the spherical pad to the entire solid box, including side and front faces.
export function stepClearance(point, step, radius = activeLeg.radius) {
  const dx = Math.abs(point[0] - step.edge - step.depth / 2) - step.depth / 2;
  const dy = Math.abs(point[1] - step.y) - step.width / 2;
  const dz = Math.abs(point[2] - step.height / 2) - step.height / 2;
  return Math.hypot(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0))
    + Math.min(Math.max(dx, dy, dz), 0) - radius;
}

export const poses = [
  { name: '站立', time: 0 },
  { name: '抬脚', time: 0.28 },
  { name: '越棱', time: 0.51 },
  { name: '落脚', time: 1 },
];
