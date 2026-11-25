import { Group, Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import modelUrl from '../model/g1.glb?url';

// The GLB stores the original MJCF body offsets and joint axes in the same Z-up frame.
export async function loadHumanoid() {
  const { scene } = await new GLTFLoader().loadAsync(modelUrl);
  const joints = new Array(29);
  scene.traverse(node => {
    if (node.isMesh) node.castShadow = node.receiveShadow = true;
    const { jointIndex, jointAxis } = node.userData;
    if (Number.isInteger(jointIndex)) {
      joints[jointIndex] = { node, axis: new Vector3(...jointAxis), rest: node.quaternion.clone() };
    }
  });
  const root = new Group();
  root.add(scene);
  const soles = ['left_sole', 'right_sole'].map(name => scene.getObjectByName(name));
  const knee = scene.getObjectByName('left_knee_link');
  if (joints.filter(Boolean).length !== 29 || !knee || soles.some(s => !s)) throw new Error('G1 关节模型不完整');
  const rotation = new Quaternion();
  function apply(frame) {
    joints.forEach(({ node, axis, rest }, i) => {
      node.quaternion.copy(rest).multiply(rotation.setFromAxisAngle(axis, frame[8+i]));
    });
  }
  return { root, soles, knee, apply };
}
