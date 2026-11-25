import {
  BoxGeometry, CanvasTexture, Color, CylinderGeometry, Matrix4, Mesh, MeshStandardMaterial, MirroredRepeatWrapping,
  Quaternion, RepeatWrapping, SphereGeometry, SRGBColorSpace, TextureLoader, TorusGeometry, Vector3,
} from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import concreteUrl from '../assets/imagegen/concrete-floor.webp?url';
import factoryFloorUrl from '../assets/imagegen/factory-floor.webp?url';

const units = {
  box: new BoxGeometry(2, 2, 2),
  cylinder: new CylinderGeometry(1, 1, 2, 24).rotateX(Math.PI/2),
  ellipsoid: new SphereGeometry(1, 32, 20),
};
const finish = {
  paint: [.57, .14], panel: [.51, .19], metal: [.41, .61], brushed: [.34, .68],
  tread: [.57, .59], concrete: [.88, .01], plastic: [.64, .03], rubber: [.91, .02],
  cardboard: [.91, 0], tape: [.47, .03], paper: [.82, 0], porcelain: [.31, .04],
  window: [.22, .34], display: [.38, .13], light: [.35, 0],
  'factory-floor': [.79, .02], 'floor-paint': [.73, .015],
};
const metersPerTile = { concrete: 2.4, tread: .143, 'factory-floor': 3.8 };
const nonShadowing = new Set(['concrete', 'factory-floor', 'paper', 'light', 'display']);

function treadMap() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#424242';
  ctx.fillRect(0, 0, 96, 96);
  for (const [x, y, angle] of [[24, 24, .65], [72, 24, -.65], [24, 72, -.65], [72, 72, .65]]) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    const raised = ctx.createLinearGradient(0, -6, 0, 6);
    raised.addColorStop(0, '#666');
    raised.addColorStop(.40, '#c9c9c9');
    raised.addColorStop(.72, '#a1a1a1');
    raised.addColorStop(1, '#5d5d5d');
    ctx.fillStyle = raised;
    ctx.beginPath();
    ctx.ellipse(0, 0, 17, 6, 0, 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
  }
  const map = new CanvasTexture(canvas);
  map.wrapS = map.wrapT = RepeatWrapping;
  return map;
}

export class PlantSurfaces {
  constructor(renderer, redraw) {
    this.materials = new Map();
    const anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
    const load = new TextureLoader();
    this.concrete = load.load(concreteUrl, redraw);
    this.factoryFloor = load.load(factoryFloorUrl, redraw);
    for (const tile of [this.concrete, this.factoryFloor]) {
      tile.colorSpace = SRGBColorSpace;
      tile.wrapS = tile.wrapT = RepeatWrapping;
      tile.anisotropy = anisotropy;
    }
    this.factoryFloor.wrapS = this.factoryFloor.wrapT = MirroredRepeatWrapping;
    this.tread = treadMap();
    this.tread.anisotropy = anisotropy;
  }

  material(surface, rgb) {
    const key = surface + '/' + rgb.join('/');
    const existing = this.materials.get(key);
    if (existing) return existing;
    const [roughness, metalness] = finish[surface] || finish.paint;
    const color = new Color().setRGB(rgb[0], rgb[1], rgb[2], SRGBColorSpace);
    const spec = { color, roughness, metalness };
    if (surface === 'concrete') {
      spec.color.lerp(new Color(0xffffff), .52);
      spec.map = this.concrete;
      spec.bumpMap = this.concrete;
      spec.bumpScale = .004;
    } else if (surface === 'factory-floor') {
      spec.color.setRGB(.91, .94, .96, SRGBColorSpace);
      spec.map = this.factoryFloor;
      spec.bumpMap = this.factoryFloor;
      spec.bumpScale = .006;
    } else if (surface === 'tread') {
      spec.bumpMap = this.tread;
      spec.bumpScale = .005;
    } else if (surface === 'display' || surface === 'light') {
      spec.emissive = color;
      spec.emissiveIntensity = surface === 'light' ? .85 : .26;
    }
    const material = new MeshStandardMaterial(spec);
    this.materials.set(key, material);
    return material;
  }

  batch(parent, shapes) {
    const collections = new Map();
    const position = new Vector3(), rotation = new Quaternion(), scale = new Vector3();
    const matrix = new Matrix4();
    for (const { type, p, s, color, q, surface } of shapes) {
      const material = this.material(surface, color);
      const cast = Math.max(...s) > .046 && !nonShadowing.has(surface);
      const key = material.id + '/' + cast;
      if (!collections.has(key)) collections.set(key, { material, cast, geometry: [] });
      const geometry = type === 'elbow' ? new TorusGeometry(s[0], s[1], 24, 16, Math.PI/2) : units[type].clone();
      position.set(...p);
      if (q) rotation.set(q[1], q[2], q[3], q[0]).normalize();
      else rotation.identity();
      if (type === 'elbow') scale.set(1, 1, 1);
      else scale.set(s[0], type === 'cylinder' ? s[0] : s[1], type === 'cylinder' ? s[1] : s[2]);
      geometry.applyMatrix4(matrix.compose(position, rotation, scale));
      if (metersPerTile[surface]) meterUV(geometry, metersPerTile[surface]);
      collections.get(key).geometry.push(geometry);
    }
    for (const { material, cast, geometry } of collections.values()) {
      const mesh = new Mesh(mergeGeometries(geometry), material);
      mesh.castShadow = cast;
      mesh.receiveShadow = true;
      parent.add(mesh);
      for (const source of geometry) source.dispose();
    }
  }
}

function meterUV(geometry, tile) {
  const pos = geometry.attributes.position, normal = geometry.attributes.normal, uv = geometry.attributes.uv;
  for (let i = 0; i < pos.count; i++) {
    const nx = Math.abs(normal.getX(i)), ny = Math.abs(normal.getY(i)), nz = Math.abs(normal.getZ(i));
    if (nz >= nx && nz >= ny) uv.setXY(i, pos.getX(i)/tile, pos.getY(i)/tile);
    else if (nx >= ny) uv.setXY(i, pos.getY(i)/tile, pos.getZ(i)/tile);
    else uv.setXY(i, pos.getX(i)/tile, pos.getZ(i)/tile);
  }
}
