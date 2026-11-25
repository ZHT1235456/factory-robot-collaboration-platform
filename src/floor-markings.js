import { BufferGeometry, Float32BufferAttribute, Mesh, Vector2 } from 'three';

const styles = {
  joint: { color: [.315, .34, .35], height: .0014 },
  aisle: { color: [.96, .63, .073], height: .0028 },
  zone: { color: [.86, .57, .068], height: .0028 },
  dock: { color: [.94, .64, .080], height: .0034 },
  crossing: { color: [.87, .89, .85], height: .0034 },
};

function roundedPath({ points, radius, width, closed }) {
  const vertices = points.map(point => new Vector2(...point));
  const path = [];
  for (let i = 0; i < vertices.length; i++) {
    const corner = vertices[i];
    if (!radius || (!closed && (i === 0 || i === vertices.length-1))) {
      path.push(corner);
      continue;
    }
    const prev = vertices[(i+vertices.length-1)%vertices.length];
    const next = vertices[(i+1)%vertices.length];
    const trim = Math.min(Math.max(radius, width*1.08), corner.distanceTo(prev)*.45, corner.distanceTo(next)*.45);
    const inlet = prev.clone().sub(corner).setLength(trim).add(corner);
    const outlet = next.clone().sub(corner).setLength(trim).add(corner);
    for (let j = 0; j <= 10; j++) {
      const t = j/10;
      path.push(inlet.clone().multiplyScalar((1-t)**2)
        .addScaledVector(corner, 2*t*(1-t)).addScaledVector(outlet, t*t));
    }
  }
  if (closed) path.push(path[0]);
  return path;
}

function strokeEdges(path, width, closed) {
  return path.map((point, i) => {
    const first = path[i-1] ?? (closed ? path[path.length-2] : null);
    const last = path[i+1] ?? (closed ? path[1] : null);
    const outgoing = last ? last.clone().sub(point).normalize() : point.clone().sub(first).normalize();
    const incoming = first ? point.clone().sub(first).normalize() : outgoing;
    const forward = incoming.clone().add(outgoing).normalize();
    const half = width/(2*forward.dot(outgoing));
    return { point, normal: new Vector2(-forward.y, forward.x).multiplyScalar(half) };
  });
}

function dashPaths(points, dash) {
  if (!dash) return [points];
  const paths = [];
  let stroke = [], phase = 0, remaining = dash[0];
  for (let i = 1; i < points.length; i++) {
    const from = points[i-1], to = points[i];
    const length = from.point.distanceTo(to.point);
    const along = distance => ({
      point: from.point.clone().lerp(to.point, distance/length),
      normal: from.normal.clone().lerp(to.normal, distance/length),
    });
    let distance = 0;
    while (length-distance > 1.e-6) {
      const step = Math.min(remaining, length-distance);
      if (phase === 0) {
        if (!stroke.length) stroke.push(along(distance));
        stroke.push(along(distance+step));
      }
      distance += step;
      remaining -= step;
      if (remaining <= 1.e-6) {
        if (stroke.length) paths.push(stroke);
        stroke = [];
        phase = 1-phase;
        remaining = dash[phase];
      }
    }
  }
  if (stroke.length) paths.push(stroke);
  return paths;
}

function ribbon(batch, path, height) {
  const start = batch.positions.length/3;
  for (let i = 0; i < path.length; i++) {
    const { point, normal } = path[i];
    batch.positions.push(point.x+normal.x, point.y+normal.y, height,
                         point.x-normal.x, point.y-normal.y, height);
    batch.normals.push(0, 0, 1, 0, 0, 1);
    if (i) {
      const j = start+2*i;
      batch.indices.push(j-2, j-1, j, j-1, j+1, j);
    }
  }
}

export function addFloorMarkings(plant, markings, surfaces) {
  const batches = new Map();
  for (const mark of markings) {
    if (!batches.has(mark.kind)) batches.set(mark.kind, { positions: [], normals: [], indices: [] });
    const batch = batches.get(mark.kind);
    const paths = dashPaths(strokeEdges(roundedPath(mark), mark.width, mark.closed), mark.dash);
    for (const path of paths)
      ribbon(batch, path, styles[mark.kind].height);
  }
  for (const [kind, { positions, normals, indices }] of batches) {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3));
    geometry.setIndex(indices);
    const mesh = new Mesh(geometry, surfaces.material('floor-paint', styles[kind].color));
    mesh.name = 'floor-'+kind;
    mesh.receiveShadow = true;
    plant.add(mesh);
  }
}
