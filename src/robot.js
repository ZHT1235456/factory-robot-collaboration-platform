import * as T from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { MeshBVH } from 'three-mesh-bvh';
import sensorUrl from '../model/front-sensors.glb?url';
import { legs, basePosition } from './rig.js';

const finish = (color, roughness, metalness = 0) => new T.MeshStandardMaterial({ color, roughness, metalness });
const paint = finish(0xd6d3c9, 0.34, 0.13);
const panel = finish(0xb5b6af, 0.41, 0.17);
const graphite = finish(0x333a40, 0.51, 0.28);
const rubber = finish(0x232829, 0.83);
const metal = finish(0x69747b, 0.32, 0.72);
const darkGlass = finish(0x172827, 0.13, 0.52);

// Object-space grain keeps the fine molded finish continuous across individual shell seams.
paint.onBeforeCompile = shader => {
  shader.vertexShader = 'varying vec3 vShellPosition;\n' + shader.vertexShader.replace(
    '#include <begin_vertex>', '#include <begin_vertex>\n vShellPosition = position;'
  );
  shader.fragmentShader = 'varying vec3 vShellPosition;\n' + shader.fragmentShader.replace(
    '#include <roughnessmap_fragment>',
    '#include <roughnessmap_fragment>\n' +
    'vec3 grainCell = vShellPosition * 1250.0;\n' +
    'float grain = fract(sin(dot(floor(grainCell),vec3(12.9898,78.233,36.72))) * 43758.5453) - 0.5;\n' +
    'grain *= 1.0 - smoothstep(0.45,1.8,length(fwidth(grainCell)));\n' +
    'roughnessFactor += grain * 0.13; diffuseColor.rgb *= 1.0 + grain * 0.055;'
  );
};

function install(parent, geometry, material, at = [0, 0, 0]) {
  const mesh = new T.Mesh(geometry, material);
  mesh.position.set(...at);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

// Smooth section lofts in the robot's local frame. Each row is station, two offsets and two half-widths.
function loft(rows, axis, exponent = 0.9, segments = 48) {
  const pos = [], indices = [];
  const shape = v => Math.sign(v) * Math.pow(Math.abs(v), exponent);
  for (const [s, a, b, ra, rb] of rows) {
    for (let j = 0; j < segments; j++) {
      const theta = 2 * Math.PI * j / segments;
      const p = ra * shape(Math.cos(theta)), q = rb * shape(Math.sin(theta));
      if (axis === 'body') pos.push(s, a + p, b + q);
      else pos.push(a + p, b - q, -s);
    }
  }
  for (let row = 0; row < rows.length - 1; row++) {
    for (let i = 0; i < segments; i++) {
      const a = row * segments + i, b = row * segments + (i + 1) % segments;
      indices.push(a, b, a + segments, b, b + segments, a + segments);
    }
  }
  for (const [row, first] of [[0, true], [rows.length - 1, false]]) {
    const [s, a, b] = rows[row];
    const center = pos.length / 3;
    if (axis === 'body') pos.push(s, a, b);
    else pos.push(a, b, -s);
    for (let i = 0; i < segments; i++) {
      const p = row * segments + i, q = row * segments + (i + 1) % segments;
      indices.push(center, first ? q : p, first ? p : q);
    }
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function tube(parent, points, radius, material) {
  const path = new T.CatmullRomCurve3(points.map(p => new T.Vector3(...p)));
  return install(parent, new T.TubeGeometry(path, 44, radius, 6, false), material);
}

function plate(parent, size, radius, at, material) {
  return install(parent, new RoundedBoxGeometry(...size, 3, radius), material, at);
}

function cylinder(parent, r, length, at, material) {
  return install(parent, new T.CylinderGeometry(r, r, length, 48), material, at);
}

function ellipsoid(parent, scale, at, mat) {
  const cap = install(parent, new T.SphereGeometry(1, 40, 24), mat, at);
  cap.scale.set(...scale);
  return cap;
}

function bodyShell(parent) {
  const bodyRows = [
    [-.227,0,.023,.040,.030],[-.218,0,.020,.061,.044],[-.196,0,.021,.073,.051],
    [-.161,0,.020,.081,.056],[-.125,0,.010,.087,.064],[-.073,0,.004,.090,.068],
    [.059,0,.003,.092,.071],[.115,0,.008,.092,.068],[.159,0,.020,.086,.060],
    [.192,0,.026,.080,.055],[.221,0,.022,.078,.055],[.252,0,.011,.078,.058],
    [.277,0,.003,.074,.055],[.303,0,.004,.066,.053],[.317,0,.006,.054,.051],
    [.326,0,.009,.034,.047],[.329,0,.010,.0225,.040],
  ];
  install(parent, loft(bodyRows,'body',.66,64), paint);
  install(parent, loft([
    [-.212,0,-.030,.047,.014],[-.182,0,-.034,.066,.022],[-.128,0,-.037,.074,.032],
    [-.07,0,-.038,.083,.036],[.063,0,-.038,.085,.036],[.15,0,-.032,.072,.032],
    [.209,0,-.029,.064,.022],[.275,0,-.029,.053,.013],[.297,0,-.029,.035,.009],
  ], 'body', .62, 52), graphite);
  const topHeight = (x,y) => {
    const j=bodyRows.findIndex(row=>row[0]>=x), a=bodyRows[j-1], b=bodyRows[j];
    const f=(x-a[0])/(b[0]-a[0]);
    const blend=i=>a[i]+(b[i]-a[i])*f;
    return blend(2)+blend(4)*Math.pow(1-Math.pow(Math.abs(y)/blend(3),2/.66),.66/2);
  };
  const lid=[];
  for(let n=0;n<=72;n++) {
    const t=2*Math.PI*n/72, c=Math.cos(t), s=Math.sin(t);
    const x=-.017+.114*Math.sign(c)*Math.pow(Math.abs(c),.33);
    const y=.059*Math.sign(s)*Math.pow(Math.abs(s),.33);
    lid.push([x,y,topHeight(x,y)+.0004]);
  }
  tube(parent,lid,.00085,graphite);
  for (let i = 0; i < 6; i++) {
    const x=-.097+i*.009;
    tube(parent,[[-.002+x,-.029,topHeight(x,-.029)],[x,0,topHeight(x,0)],[x+.002,.029,topHeight(x,.029)]],.0014,graphite);
  }
  for (const side of [-1, 1]) {
    tube(parent, [[-.178,side*.073,.048],[-.119,side*.085,.030],
      [-.046,side*.090,.023],[.052,side*.091,.024],[.115,side*.088,.035],
      [.146,side*.083,.047]], .0007, panel);
  }
  ellipsoid(parent,[.046,.042,.019],[.289,0,-.043],graphite);
  plate(parent,[.006,.040,.074],.0028,[.329,0,.011],graphite);
  plate(parent,[.003,.034,.065],.0014,[.3315,0,.011],darkGlass);
  for (const [z,r] of [[.022,.0108],[-.002,.0055]]) {
    const rim=cylinder(parent,r,.002,[.3335,0,z],metal);
    rim.rotation.z=-Math.PI/2;
    ellipsoid(parent,[.0021,r*.81,r*.81],[.3342,0,z],darkGlass);
  }
}

function legSurfaces() {
  const thigh = loft([
    [.000,-.002,0,.030,.025],[.012,-.003,0,.039,.030],
    [.035,-.005,0,.042,.031],[.059,-.005,0,.038,.029],
    [.097,-.008,0,.030,.024],[.140,-.009,0,.022,.020],
    [.168,-.010,0,.018,.018],[.179,-.009,0,.014,.0175],
    [.1835,-.008,0,.012,.0165],
  ], 'leg', .89);
  const vertices=thigh.getAttribute('position');
  for(let i=0;i<vertices.count;i++) {
    const station=-vertices.getZ(i);
    if(station<=.168) continue;
    const x=vertices.getX(i);
    const relief=.0295-Math.sqrt(Math.max(0,.0295**2-x*x));
    vertices.setZ(i,-station-relief*Math.min(1,(station-.168)/.0155));
  }
  thigh.computeVertexNormals();
  const shank = loft([
    [.012,-.008,0,.009,.013],[.024,-.009,0,.011,.015],[.044,-.009,0,.013,.0155],
    [.070,-.005,0,.015,.016],[.106,.003,0,.015,.015],
    [.143,.010,0,.0135,.014],[.167,.010,0,.0138,.014],
    [.188,.005,0,.0155,.015],[.203,.000,0,.019,.017],
    [.209,-.0015,0,.019,.017],
  ], 'leg', .92);
  const kneeHousing=new T.CylinderGeometry(.024,.024,.031,64);
  kneeHousing.deleteAttribute('uv');
  const calf=mergeGeometries([shank,kneeHousing]);
  shank.dispose(); kneeHousing.dispose();
  thigh.boundsTree = new MeshBVH(thigh);
  calf.boundsTree = new MeshBVH(calf);
  return { thigh, calf };
}

function buildLeg(root, leg, surfaces) {
  const side = Math.sign(leg.hip[1]);
  const hip = new T.Group(), thigh = new T.Group(), calf = new T.Group();
  root.add(hip);
  hip.add(thigh);
  thigh.add(calf);
  hip.position.set(...leg.hip);
  thigh.position.set(...leg.thigh);
  calf.position.set(...leg.calf);
  hip.name = leg.id + '_hip_joint'; thigh.name = leg.id + '_thigh_joint'; calf.name = leg.id + '_calf_joint';
  cylinder(hip,.052,.076,[0,side*.039,0],graphite);
  const hoodSection=new T.Shape();
  const end=Math.PI+.13, start=-.13;
  hoodSection.absarc(0,0,.060,start,end,false);
  hoodSection.lineTo(.053*Math.cos(end),.053*Math.sin(end));
  hoodSection.absarc(0,0,.053,end,start,true);
  hoodSection.closePath();
  const hood=new T.ExtrudeGeometry(hoodSection,{
    depth:.041,bevelEnabled:true,bevelSize:.0016,bevelThickness:.0018,bevelSegments:3,curveSegments:30,
  });
  hood.rotateX(Math.PI/2);
  install(hip,hood,paint,[0,side>0?.086:-.045,0]);
  for (const lateral of [.019,.035,.069]) {
    cylinder(hip,.0528,.002,[0,side*lateral,0],lateral === .035 ? metal : graphite);
  }
  ellipsoid(hip,[.051,.028,.053],[0,side*.065,0],panel);
  cylinder(thigh,.047,.026,[0,side*.002,0],graphite);
  cylinder(thigh,.0475,.003,[0,side*.0155,0],metal);
  ellipsoid(thigh,[.046,.017,.046],[0,side*.017,0],paint);
  const upper = install(thigh,surfaces.thigh,paint);
  const lower = install(calf,surfaces.calf,paint);
  cylinder(calf,.0235,.002,[0,side*.016,0],graphite);
  ellipsoid(calf,[.0227,.0035,.0227],[0,side*.0169,0],paint);
  const pad = install(calf,new T.SphereGeometry(leg.radius,32,24),rubber,leg.pad);
  const lipHeight=.008;
  const lipRadius=Math.sqrt(leg.radius*leg.radius-lipHeight*lipHeight)-.00035;
  install(pad,new T.TorusGeometry(lipRadius,.00038,6,48),graphite,[0,0,lipHeight]);
  const apply = angles => {
    hip.rotation.x = angles[0];
    thigh.rotation.y = angles[1];
    calf.rotation.y = angles[2];
  };
  apply(leg.start);
  return { hip, thigh, calf, upper, lower, pad, apply };
}

export function createRobot() {
  const root = new T.Group();
  root.position.set(...basePosition);
  bodyShell(root);
  const surfaces = legSurfaces();
  const limbs = legs.map(leg => buildLeg(root,leg,surfaces));
  const sensorsReady = new GLTFLoader().loadAsync(sensorUrl).then(({ scene }) => {
    scene.rotation.x = Math.PI / 2;
    scene.traverse(obj => {
      if (obj.isMesh) { obj.castShadow = true; obj.receiveShadow = true; }
    });
    root.add(scene);
  });
  const active = limbs[0];
  const relative = new T.Matrix4(), a = {}, b = {};
  let previousKnee, measured;
  function shellDistance() {
    if (previousKnee === active.calf.rotation.y) return measured;
    previousKnee = active.calf.rotation.y;
    root.updateMatrixWorld(true);
    relative.copy(active.upper.matrixWorld).invert().multiply(active.lower.matrixWorld);
    const best = surfaces.thigh.boundsTree.closestPointToGeometry(surfaces.calf,relative,a,b);
    measured = best.distance;
    return measured;
  }
  return { root, active, limbs, sensorsReady, shellDistance };
}
