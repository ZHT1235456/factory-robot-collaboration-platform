import * as T from 'three';

const capacity = 128;
const circle = 2*Math.PI;
const sweepColor = new T.Color(0x00a8b3);
const probeColor = new T.Color(0x008abf);
const echoColor = new T.Color(0xf6a82b);

function attribute(count, size) {
  return new T.BufferAttribute(new Float32Array(count*size), size).setUsage(T.DynamicDrawUsage);
}

function echoTexture() {
  const size = 32;
  const pixels = new Uint8Array(size*size*4).fill(255);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const radius = Math.hypot((x+.5)/size*2-1, (y+.5)/size*2-1);
      const core = 1-T.MathUtils.smoothstep(radius, .20, .43);
      const halo = .40*(1-T.MathUtils.smoothstep(radius, .35, 1));
      pixels[(y*size+x)*4+3] = 255*Math.max(core, halo);
    }
  }
  const texture = new T.DataTexture(pixels, size, size);
  texture.magFilter = texture.minFilter = T.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

export class Go2Detection {
  constructor() {
    this.group = new T.Group();
    this.group.visible = false;
    this.beamPositions = attribute(capacity*4, 3);
    this.beamColors = attribute(capacity*4, 4);
    this.echoPositions = attribute(capacity, 3);
    this.echoColors = attribute(capacity, 4);

    this.beams = new T.BufferGeometry();
    this.beams.setAttribute('position', this.beamPositions);
    this.beams.setAttribute('color', this.beamColors);
    const indices = [];
    for (let i = 0; i < capacity*4; i += 4)
      indices.push(i, i+1, i+2, i+1, i+3, i+2);
    this.beams.setIndex(indices);
    const beamMesh = new T.Mesh(this.beams, new T.MeshBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, toneMapped: false,
      side: T.DoubleSide, forceSinglePass: true,
    }));
    beamMesh.frustumCulled = false;

    this.echoes = new T.BufferGeometry();
    this.echoes.setAttribute('position', this.echoPositions);
    this.echoes.setAttribute('color', this.echoColors);
    const echoPoints = new T.Points(this.echoes, new T.PointsMaterial({
      map: echoTexture(), size: 11, sizeAttenuation: false, vertexColors: true,
      transparent: true, depthWrite: false, toneMapped: false,
    }));
    echoPoints.frustumCulled = false;
    echoPoints.renderOrder = 1;
    this.group.add(beamMesh, echoPoints);
  }

  addRay([x, y, z], [endX, endY, endZ], returned, color, power, width) {
    if (this.rayCount >= capacity || power <= .015) return;
    const dx = endX-x, dy = endY-y, dz = endZ-z;
    const length = Math.hypot(dx, dy, dz);
    if (length < .025) return;
    const horizontal = Math.hypot(dx, dy);
    const sideX = horizontal > 1.e-5 ? -width*dy/horizontal : width;
    const sideY = horizontal > 1.e-5 ? width*dx/horizontal : 0;
    // Keep the glow on the sensor's side of the measured surface.
    const end = returned ? 1-.018/length : 1;
    const u = x+end*dx, v = y+end*dy, w = z+end*dz;
    const i = 4*this.rayCount++;
    this.beamPositions.setXYZ(i, x-sideX*.19, y-sideY*.19, z);
    this.beamPositions.setXYZ(i+1, x+sideX*.19, y+sideY*.19, z);
    this.beamPositions.setXYZ(i+2, u-sideX, v-sideY, w);
    this.beamPositions.setXYZ(i+3, u+sideX, v+sideY, w);
    for (let vertex = 0; vertex < 4; vertex++)
      this.beamColors.setXYZW(i+vertex, color.r, color.g, color.b, power*(vertex < 2 ? .30 : .76));
    if (returned) {
      const dot = this.echoCount++;
      this.echoPositions.setXYZ(dot, u, v, w);
      this.echoColors.setXYZW(dot, echoColor.r, echoColor.g, echoColor.b, power);
    }
  }

  drawSweep({ origin, heading, ranges, progress }) {
    const front = heading/circle+.5+progress;
    const fade = 1-T.MathUtils.smoothstep(progress, .62, 1);
    const trail = .22;
    for (const [i, sample] of ranges.entries()) {
      const turn = i/ranges.length;
      const lag = T.MathUtils.euclideanModulo(front-turn, 1);
      if (sample === null || lag >= trail) continue;
      const power = fade*(.16+.84*(1-lag/trail)**2);
      const angle = turn*circle-Math.PI;
      const distance = Math.abs(sample);
      const end = [origin[0]+distance*Math.cos(angle), origin[1]+distance*Math.sin(angle), origin[2]];
      this.addRay(origin, end, sample >= 0, sweepColor, power, .015+.013*power);
    }
  }

  update(detection) {
    this.group.visible = false;
    if (!detection) return;
    this.rayCount = this.echoCount = 0;
    if (detection.sweep) this.drawSweep(detection.sweep);
    if (detection.probe) {
      const { origin, rays, progress } = detection.probe;
      const power = 1-T.MathUtils.smoothstep(progress, .38, 1);
      for (const [x, y, z, returned] of rays)
        this.addRay(origin, [x, y, z], returned, probeColor, power, .030);
    }
    this.beams.setDrawRange(0, this.rayCount*6);
    this.echoes.setDrawRange(0, this.echoCount);
    for (const buffer of [this.beamPositions, this.beamColors, this.echoPositions, this.echoColors])
      buffer.needsUpdate = true;
    this.group.visible = this.rayCount > 0;
  }
}
