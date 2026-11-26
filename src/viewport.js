import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { createRobot } from './robot.js';
import { loadHumanoid } from './humanoid.js';
import { PlantSurfaces } from './plant-surfaces.js';
import { addFloorMarkings } from './floor-markings.js';
import { Go2Detection } from './go2-detection.js';

const gripSizes = {
  pump: [.047, .032, .028, 0xdba338],
  cnc: [.063, .030, .022, 0x71808b],
  pack: [.046, .028, .034, 0xab8052],
  store: [.066, .038, .035, 0x356081],
};
const metal = new T.MeshStandardMaterial({ color: 0x687c8f, roughness: .62, metalness: .15 });
const ink = new T.MeshStandardMaterial({ color: 0x2c3943, roughness: .68 });
const safety = new T.MeshStandardMaterial({ color: 0xd6a63b, roughness: .58 });
const display = new T.MeshStandardMaterial({ color: 0x365c64, roughness: .25, metalness: .28 });
const anodized = new T.MeshStandardMaterial({ color: 0x829098, roughness: .36, metalness: .58 });
const softRubber = new T.MeshStandardMaterial({ color: 0x202628, roughness: .94 });
const poseVector = new T.Vector3();
const poseRotation = new T.Quaternion();
const mapUp = new T.Vector3(0, 0, 1);
const positionGap = new T.Vector3();

function piece(parent, sizes, at, material) {
  const mesh = new T.Mesh(new T.BoxGeometry(...sizes), material);
  mesh.position.set(...at);
  mesh.receiveShadow = mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}

function label(text, className = '') {
  const element = document.createElement('span');
  element.className = 'scene-label ' + className;
  element.textContent = text;
  return element;
}

function move(object, pose, amount) {
  poseVector.set(pose[0], pose[1], pose[2]);
  poseRotation.set(pose[4], pose[5], pose[6], pose[3]);
  object.position.lerp(poseVector, amount);
  object.quaternion.slerp(poseRotation, amount);
}

function pin(parent, from, to, radius, material) {
  const start = new T.Vector3(...from), end = new T.Vector3(...to);
  const axis = end.clone().sub(start);
  const mesh = new T.Mesh(new T.CylinderGeometry(radius, radius, axis.length(), 18), material);
  mesh.position.copy(start).lerp(end, .5);
  mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), axis.normalize());
  mesh.castShadow = mesh.receiveShadow = true;
  parent.add(mesh);
}

function makeGripper(wrist) {
  const hand = new T.Group();
  hand.position.x = .045;
  wrist.add(hand);
  pin(hand, [-.004, 0, 0], [.008, 0, 0], .035, anodized);
  piece(hand, [.060, .080, .040], [.026, 0, 0], metal);
  piece(hand, [.048, .050, .004], [.023, 0, .021], anodized);
  piece(hand, [.008, .067, .017], [.054, 0, .008], ink);
  for (const x of [.005, .044]) {
    for (const y of [-.025, .025])
      pin(hand, [x, y, .020], [x, y, .024], .004, ink);
  }
  return [1, -1].map(side => {
    const jaw = new T.Group();
    jaw.position.set(.106, side*.048, 0);
    hand.add(jaw);
    piece(jaw, [.070, .018, .046], [0, 0, 0], softRubber);
    piece(jaw, [.063, .009, .032], [-.003, side*.0035, .002], anodized);
    piece(jaw, [.019, .014, .016], [-.023, 0, .011], metal);
    for (const x of [-.010, .001, .012, .023])
      piece(jaw, [.003, .0018, .028], [x, -side*.0088, 0], ink);
    return jaw;
  });
}

function detailWorkpiece(part, sceneId, x, y, z, material, handleZ) {
  if (sceneId === 'cnc') {
    const tool = new T.Mesh(new T.CylinderGeometry(.016, .026, 2*x, 24).rotateZ(Math.PI/2), material);
    tool.position.z = -.031;
    tool.castShadow = tool.receiveShadow = true;
    part.add(tool);
    const collar = new T.Mesh(new T.CylinderGeometry(.029, .029, .022, 24).rotateZ(Math.PI/2), ink);
    collar.position.set(-.018, 0, -.031);
    part.add(collar);
  } else if (sceneId === 'store') {
    piece(part, [2*x, 2*y, .012], [0, 0, -.031-z+.006], material);
    for (const side of [-1, 1]) {
      piece(part, [2*x, .009, 2*z], [0, side*(y-.0045), -.031], material);
      piece(part, [.009, 2*y, 2*z], [side*(x-.0045), 0, -.031], material);
    }
    piece(part, [.041, .002, .018], [-.018, -y-.001, -.030], metal);
  } else {
    piece(part, [2*x, 2*y, 2*z], [0, 0, -.031], material);
    if (sceneId === 'pump') {
      piece(part, [.034, .036, .003], [.023, 0, z-.030], softRubber);
      piece(part, [.023, .026, .002], [.025, 0, z-.028], display);
      for (const a of [-1, 1]) {
        for (const b of [-1, 1])
          pin(part, [a*(x-.007), b*(y-.008), z-.034], [a*(x-.007), b*(y-.008), z-.029], .0025, anodized);
      }
    } else {
      piece(part, [.031, 2*y+.001, .002], [-.021, 0, z-.029], safety);
    }
  }
  const bridge = handleZ-z+.033;
  for (const side of [-1, 1])
    piece(part, [.008, .013, bridge], [side*.025, 0, z-.031+bridge/2], ink);
  piece(part, [.062, .046, .030], [0, 0, handleZ], ink);
}

export class WorkcellView {
  constructor(host, labels) {
    this.host = host;
    this.labelLayer = labels;
    this.scene = new T.Scene();
    this.scene.background = new T.Color(0xdce3e7);
    this.scene.add(new T.HemisphereLight(0xf4f8fa, 0x737774, 1.65));
    const key = new T.DirectionalLight(0xfff4e6, 2.0);
    this.sun = key;
    key.position.set(-5.8, -6.6, 10.9);
    key.castShadow = true;
    Object.assign(key.shadow.camera, { left: -10, right: 10, top: 10, bottom: -10, far: 27 });
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.normalBias = .008;
    key.shadow.bias = -.00015;
    this.scene.add(key);
    const fill = new T.DirectionalLight(0xd8eaff, .74);
    fill.position.set(8, 3, 8);
    this.scene.add(fill);

    this.renderer = new T.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = T.PCFShadowMap;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    const room = new RoomEnvironment();
    const reflection = new T.PMREMGenerator(this.renderer);
    this.scene.environment = reflection.fromScene(room, .065).texture;
    this.scene.environmentIntensity = .68;
    room.dispose();
    reflection.dispose();
    this.surfaces = new PlantSurfaces(this.renderer, () => { this.dirty = true; });
    this.host.prepend(this.renderer.domElement);
    this.camera = new T.OrthographicCamera(-8, 8, 5, -5, .03, 120);
    this.camera.up.copy(mapUp);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .14;
    this.controls.minZoom = .54;
    this.controls.maxZoom = 5.7;
    this.controls.maxPolarAngle = Math.PI*.495;
    this.controls.addEventListener('change', () => { this.dirty = true; });
    this.plant = new T.Group();
    this.scene.add(this.plant);
    this.go2Detection = new Go2Detection();
    this.scene.add(this.go2Detection.group);
    this.stationMarker = new T.Mesh(new T.RingGeometry(.115, .134, 36),
      new T.MeshBasicMaterial({ color: 0x2274c9, side: T.DoubleSide, transparent: true, opacity: .82 }));
    this.stationMarker.visible = false;
    this.scene.add(this.stationMarker);
    this.mode = 'overview';
    this.last = 0;
    this.geometryLabels = [];
    this.areaLabels = [];
    this.fixedMaterials = new Map();
    this.robotLabels = [];
    this.dirty = true;
    this.refreshSize = new ResizeObserver(() => this.resize());
    this.refreshSize.observe(host);
    this.setView('overview');
    window.requestAnimationFrame(this.draw);
  }

  async loadRobots() {
    const dog = createRobot();
    const [human] = await Promise.all([loadHumanoid(), dog.sensorsReady]);
    this.robots = { g1: human, go2: dog };
    this.angles = { g1: new Float64Array(29), go2: new Float64Array(12) };
    this.scene.add(human.root, dog.root);
    this.basket = new T.Group();
    dog.root.add(this.basket);
    this.jaws = makeGripper(human.root.getObjectByName('right_wrist_yaw_link'));
    this.ladderPalms = ['left', 'right'].map(side => {
      const hand = new T.Group();
      human.root.getObjectByName(side+'_wrist_yaw_link').add(hand);
      pin(hand, [.022, 0, 0], [.061, 0, 0], .012, anodized);
      const pad = new T.Mesh(new T.SphereGeometry(.021, 16, 12), softRubber);
      pad.position.x = .061;
      hand.add(pad);
      hand.visible = false;
      return hand;
    });
    this.robotLabels = [
      { key: 'g1', name: 'G1', object: human.root, height: .70 },
      { key: 'go2', name: 'Go2', object: dog.root, height: .31 },
    ].map(item => {
      const element = label(item.name, 'robot-label '+item.key);
      const bubble = document.createElement('div');
      const speaker = document.createElement('strong');
      const words = document.createElement('span');
      bubble.className = 'speech-bubble '+item.key;
      bubble.dataset.speaker = item.key;
      bubble.setAttribute('aria-hidden', 'true');
      bubble.hidden = true;
      speaker.textContent = item.name;
      words.className = 'speech-text';
      bubble.append(speaker, words);
      this.labelLayer.append(element, bubble);
      return { ...item, element, bubble, words };
    });
    this.dirty = true;
  }

  rebuild(layout, sceneId) {
    this.sceneId = sceneId;
    this.layout = layout;
    const outdoor = sceneId === 'yard';
    this.scene.background.setHex(outdoor ? 0xc8dfed : 0xdce3e7);
    this.sun.intensity = outdoor ? 2.6 : 2.0;
    this.packet = null;
    this.stationMarker.visible = false;
    this.go2Detection.update();
    for (const light of Object.values(this.indicatorMeshes ?? {})) light.material.dispose();
    for (const group of [this.plant, this.basket]) {
      group.traverse(node => { if (node.isMesh) node.geometry.dispose(); });
      group.clear();
    }
    for (const item of this.geometryLabels) item.element.remove();
    for (const item of this.areaLabels) item.element.remove();
    this.geometryLabels = [];
    this.areaLabels = [];
    this.part = this.spare = this.button = this.door = this.cart = this.lift = null;
    this.switchMeshes = {};
    this.indicatorMeshes = {};
    this.upperPlant = new T.Group();
    this.plant.add(this.upperPlant);
    const upper = shape => !!layout.upperDecks && shape.p[2] >= 2.62;
    this.surfaces.batch(this.plant, layout.shapes.filter(shape => !upper(shape)));
    this.surfaces.batch(this.upperPlant, layout.shapes.filter(upper));
    if (layout.floorMarkings) addFloorMarkings(this.plant, layout.floorMarkings, this.surfaces);
    this.basket.position.x = layout.basketX;
    this.surfaces.batch(this.basket, layout.basket);
    for (const raw of layout.labels) {
      const element = label(raw.name);
      this.labelLayer.append(element);
      this.geometryLabels.push({ element, position: new T.Vector3(...raw.p) });
    }
    const tooling = sceneId === 'factory' ? 'pump' : sceneId;
    if (tooling in gripSizes) {
      const [x, y, z, color] = gripSizes[tooling];
      this.part = new T.Group();
      this.plant.add(this.part);
      let mat = this.fixedMaterials.get('part-' + sceneId);
      if (!mat) {
        mat = new T.MeshStandardMaterial({ color, metalness: sceneId === 'cnc' ? .35 : .04, roughness: .54 });
        this.fixedMaterials.set('part-' + sceneId, mat);
      }
      detailWorkpiece(this.part, tooling, x, y, z, mat, layout.handleZ);
    }
    if (layout.factory) {
      if (layout.factory.lift) {
        this.lift = new T.Group();
        this.lift.position.set(...layout.factory.lift.position);
        this.surfaces.batch(this.lift, layout.factory.lift.shapes);
        this.plant.add(this.lift);
      }
      this.spare = new T.Group();
      this.plant.add(this.spare);
      const spareType = layout.factory.spareType;
      detailWorkpiece(this.spare, 'spare', ...gripSizes.pump.slice(0, 3),
        this.cachedMaterial('spare-'+spareType, spareType === 'seal' ? 0xd27338 : 0x315d83), layout.handleZ);
      for (const [key, point] of Object.entries(layout.factory.buttons)) {
        const mesh = new T.Mesh(new T.CylinderGeometry(.042, .042, .028, 24).rotateZ(Math.PI/2),
          this.cachedMaterial('button', 0x3ca76a));
        this.plant.add(mesh);
        mesh.position.set(...point);
        this.switchMeshes[key] = mesh;
      }
      for (const [key, [center, size, color]] of Object.entries(layout.factory.indicators)) {
        const material = this.surfaces.material('display', color).clone();
        this.indicatorMeshes[key] = piece(this.plant, size, center, material);
      }
      this.areaLabels = layout.orders.map(order => {
        const element = label(order.area, 'area-label');
        this.labelLayer.append(element);
        return { element, id: order.id, area: true,
          position: new T.Vector3(order.center[0], order.center[1]-1.5, .38) };
      });
      const shadow = this.sun.shadow.camera;
      Object.assign(shadow, { left: -33, right: 33, top: 31, bottom: -31, far: 85 });
      shadow.updateProjectionMatrix();
      this.sun.position.set(-17, -19, 32);
      this.sun.shadow.radius = 1.65;
      this.controls.minZoom = .14;
    }
    if (outdoor) {
      Object.assign(this.sun.shadow.camera, { left: -17, right: 17, top: 15, bottom: -15, far: 50 });
      this.sun.shadow.camera.updateProjectionMatrix();
      this.sun.position.set(-9, -9, 18);
      this.controls.minZoom = .22;
    }
    if (sceneId === 'pack') {
      this.button = new T.Mesh(new T.CylinderGeometry(.042, .042, .028, 24).rotateZ(Math.PI/2),
        this.cachedMaterial('button', 0x3ca76a));
      this.plant.add(this.button);
    }
    if (layout.cart?.length) {
      this.cart = new T.Group();
      this.surfaces.batch(this.cart, layout.cart);
      this.plant.add(this.cart);
    }
    if (layout.points.door) {
      this.door = new T.Group();
      const [x, y, z] = layout.points.door;
      this.door.position.set(x, y+.18, z);
      this.plant.add(this.door);
      const face = layout.factory ? this.surfaces.material('panel', [.22, .37, .53]) : display;
      for (const side of [-1, 1]) {
        piece(this.door, [.033, .41, .046], [0, -.205, side*.217], metal);
        piece(this.door, [.033, .047, .386], [0, -.205+side*.181, 0], metal);
        pin(this.door, [0, -.016, side*.134-.044], [0, -.016, side*.134+.044], .022, anodized);
      }
      piece(this.door, [.023, .316, .388], [0, -.205, 0], face);
      piece(this.door, [.043, .075, .055], [-.028, -.314, 0], safety);
      piece(this.door, [.004, .093, .029], [-.013, -.17, .137], anodized);
    }
    this.setView('overview');
    this.dirty = true;
  }

  cachedMaterial(key, color) {
    if (!this.fixedMaterials.has(key)) this.fixedMaterials.set(key, new T.MeshStandardMaterial({ color }));
    return this.fixedMaterials.get(key);
  }

  resize() {
    const width = this.host.clientWidth, height = this.host.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    const aspect = width/height;
    const spanY = Math.max(7.6, 14.85/aspect);
    this.camera.left = -spanY*aspect/2;
    this.camera.right = spanY*aspect/2;
    this.camera.top = spanY/2;
    this.camera.bottom = -spanY/2;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  setOrder(order, currentId) {
    const changed = this.order?.id !== order.id;
    this.order = order;
    if (changed && this.mode === 'zone') this.setView('zone');
    for (const area of this.areaLabels) area.element.classList.toggle('current', area.id === currentId);
  }

  setView(mode) {
    this.mode = mode;
    this.host.dataset.cameraView = mode;
    const target = this.controls.target;
    this.updateFloorVisibility();
    const isFactory = this.sceneId === 'factory';
    const outdoor = this.sceneId === 'yard';
    const center = this.sceneId === 'pump' ? 1.62 : isFactory ? 1.47 : .79;
    target.set(0, 0, center);
    if (mode === 'plan') {
      this.camera.position.set(0, -.012, 19.5);
      this.camera.zoom = isFactory ? .285 : outdoor ? .52 : 1.08;
    } else if (mode === 'overview') {
      this.camera.position.set(...(isFactory ? [17, -37, 31] : [7.8, -13.6, 9.4]));
      this.camera.zoom = isFactory ? .257 : outdoor ? .49 : this.sceneId === 'pump' ? .85 : .94;
    } else if (mode === 'zone' && this.order) {
      target.set(...this.order.center);
      this.camera.position.copy(target).add(new T.Vector3(7.8, -13.6, 10.7));
      this.camera.zoom = outdoor ? .67 : this.order.id === 'pump' ? .83 : .76;
    } else {
      const key = mode === 'g1' || mode === 'go2' ? mode : null;
      const focus = this.layout?.points[this.packet?.focus];
      if (key && this.robots) target.copy(this.robots[key].root.position);
      else if (focus) target.set(...focus);
      else if (this.order) target.set(...this.order.center);
      else if (this.part) target.copy(this.part.position);
      this.camera.position.copy(target).add(new T.Vector3(4.4, -6.4, 4.3));
      this.camera.zoom = mode === 'focus' ? 2.95 : 2.75;
    }
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.dirty = true;
  }

  setMapFloor(floor) {
    this.mapFloor = floor;
    this.updateFloorVisibility();
  }

  updateFloorVisibility() {
    if (!this.upperPlant) return;
    this.upperPlant.visible = this.mode !== 'go2' && !(this.mode === 'plan' && this.mapFloor === 'lower');
    this.dirty = true;
  }

  apply(packet, snap = false) {
    this.packet = packet;
    this.go2Detection.update(packet.survey?.go2Detection);
    const station = this.layout.points[packet.focus];
    this.stationMarker.visible = !!station && packet.status === 'running';
    if (station) this.stationMarker.position.set(station[0], station[1], station[2]+.055);
    this.blend = snap ? 0 : 10;
    this.interpolate(snap ? 1 : .4);
    this.dirty = true;
  }

  setSpeech(messages) {
    for (const item of this.robotLabels) {
      const message = messages[item.key];
      if (item.message === message) continue;
      item.message = message;
      item.speechSize = null;
      item.words.textContent = message?.text ?? '';
      item.bubble.hidden = !message;
      item.bubble.style.visibility = 'hidden';
      this.dirty = true;
    }
  }

  placeSpeech(anchors, width, height) {
    const inset = 10, gap = 12;
    let previous;
    for (const { item, x, y } of anchors.sort((a, b) => b.item.message.time-a.item.message.time)) {
      const bubble = item.bubble;
      if (item.speechSize?.viewport !== width)
        item.speechSize = { viewport: width, width: bubble.offsetWidth, height: bubble.offsetHeight };
      const { width: w, height: h } = item.speechSize;
      const left = T.MathUtils.clamp(x-w/2, inset, width-w-inset);
      const candidates = [y-h-gap, y+gap];
      const sharedWidth = previous && left+w+gap > previous.left && left < previous.right+gap;
      if (sharedWidth) candidates.push(previous.top-h-gap, previous.bottom+gap);
      const top = candidates.find(candidate => candidate >= inset && candidate+h <= height-inset &&
        (!sharedWidth || candidate+h+gap <= previous.top || candidate >= previous.bottom+gap)) ??
        T.MathUtils.clamp(y-h-gap, inset, height-h-inset);
      bubble.style.left = left+'px';
      bubble.style.top = top+'px';
      bubble.style.setProperty('--tail-x', T.MathUtils.clamp(x-left, 10, w-10)+'px');
      bubble.classList.toggle('below', top >= y);
      bubble.style.visibility = 'visible';
      previous = { left, right: left+w, top, bottom: top+h };
    }
  }

  interpolate(amount) {
    const p = this.packet;
    if (!p || !this.robots) return;
    for (const key of ['g1', 'go2']) {
      const robot = this.robots[key], state = p.actors[key], angles = this.angles[key];
      const blend = key === 'g1' && p.ladder ? 1 : amount;
      move(robot.root, state.pose, blend);
      state.joints.forEach((q, i) => { angles[i] += (q-angles[i])*blend; });
      if (key === 'g1') robot.apply([0, 0, 0, 0, 0, 0, 0, 0, ...angles]);
      else robot.limbs.forEach((limb, i) => limb.apply(angles.subarray(3*i, 3*i+3)));
    }
    this.jaws[0].parent.visible = !p.ladder;
    for (const palm of this.ladderPalms) palm.visible = Boolean(p.ladder);
    this.jaws[0].position.y = .048-p.grip[0];
    this.jaws[1].position.y = -.048+p.grip[1];
    if (this.part && p.part) move(this.part, p.part, amount);
    if (this.spare && p.spare) move(this.spare, p.spare, amount);
    if (p.devices) {
      for (const [key, stroke] of Object.entries(p.devices.buttons))
        this.switchMeshes[key].position.x = this.layout.factory.buttons[key][0]+stroke;
      for (const [key, color] of Object.entries(p.devices.indicators)) {
        const material = this.indicatorMeshes[key].material;
        material.color.setRGB(...color, T.SRGBColorSpace);
        material.emissive.copy(material.color);
      }
    }
    if (this.door) this.door.rotation.z += (p.door-this.door.rotation.z)*amount;
    if (this.cart && p.cart) {
      poseVector.set(...p.cart);
      this.cart.position.lerp(poseVector, amount);
    }
    if (this.lift) this.lift.position.z += (p.liftHeight-this.lift.position.z)*amount;
    if (this.button) {
      const [x, y, z] = this.layout.points.button;
      this.button.position.set(x+p.button, y, z);
    }
  }

  draw = (time) => {
    requestAnimationFrame(this.draw);
    const delta = Math.min((time-this.last)/1000, .05);
    this.last = time;
    if (this.blend > 0) {
      this.interpolate(1-Math.exp(-delta*40));
      this.blend--;
      this.dirty = true;
    }
    if (this.packet && !['overview', 'plan', 'zone'].includes(this.mode)) {
      const key = this.robots?.[this.mode];
      const point = this.layout.points[this.packet.focus];
      if (key) positionGap.copy(key.root.position);
      else if (point) positionGap.set(...point);
      else if (this.order) positionGap.set(...this.order.center);
      else if (this.part) positionGap.copy(this.part.position);
      else positionGap.copy(this.controls.target);
      positionGap.sub(this.controls.target);
      if (positionGap.lengthSq() > 1.e-4) {
        positionGap.multiplyScalar(.08);
        this.camera.position.add(positionGap);
        this.controls.target.add(positionGap);
        this.dirty = true;
      }
    }
    this.controls.update();
    if (!this.dirty || !this.host.clientHeight || document.hidden) return;
    this.renderer.render(this.scene, this.camera);
    const width = this.host.clientWidth, height = this.host.clientHeight;
    const anchors = [];
    const showAreas = this.areaLabels.length && ['overview', 'plan'].includes(this.mode);
    for (const item of [...this.geometryLabels, ...this.areaLabels, ...this.robotLabels]) {
      (item.object ? poseVector.copy(item.object.position).addScaledVector(mapUp, item.height)
        : poseVector.copy(item.position)).project(this.camera);
      const levelVisible = item.object || Boolean(item.area) === Boolean(showAreas);
      const visible = levelVisible && Math.abs(poseVector.x) < .97 && Math.abs(poseVector.y) < .96 && this.packet;
      item.element.hidden = !visible || Boolean(item.message);
      if (!visible) {
        if (item.message) item.bubble.style.visibility = 'hidden';
        continue;
      }
      const x = width*(poseVector.x+1)/2, y = height*(1-poseVector.y)/2;
      if (item.message) anchors.push({ item, x, y });
      else {
        item.element.style.left = x.toFixed(1) + 'px';
        item.element.style.top = y.toFixed(1) + 'px';
      }
    }
    this.placeSpeech(anchors, width, height);
    this.dirty = false;
  };
}
