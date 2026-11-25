const cells = [[225, 231, 227], [198, 229, 220], [185, 97, 81]];
const actors = { g1: ['G1', '#2d5576'], go2: ['Go2', '#126961'] };

export class SurveyMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.layers = {};
    this.resize = new ResizeObserver(() => this.draw());
    this.resize.observe(canvas);
  }

  update(snapshot, layout, floor, order, received) {
    this.snapshot = snapshot;
    this.layout = layout;
    this.floor = floor;
    this.order = order;
    this.received = received;
    const survey = snapshot.survey;
    this.route = !received || survey.route?.revision === received.revision ? survey.route : null;
    const grid = this.layer(survey, floor);
    this.draw();
    if (!order) return Math.round(100*grid.known/grid.values.length);
    if (grid.coverage.has(order.id)) return grid.coverage.get(order.id);
    const [startX, startY, endX, endY] = order.bounds;
    const [ox, oy] = survey.origin;
    const [cols, rows] = survey.size;
    const res = survey.resolution;
    let known = 0, total = 0;
    for (let y = Math.max(0, Math.floor((startY-oy)/res)); y < Math.min(rows, Math.ceil((endY-oy)/res)); y++)
      for (let x = Math.max(0, Math.floor((startX-ox)/res)); x < Math.min(cols, Math.ceil((endX-ox)/res)); x++) {
        if (grid.values[y*cols+x]) known++;
        total++;
      }
    const percent = total ? Math.round(100*known/total) : 0;
    grid.coverage.set(order.id, percent);
    return percent;
  }

  layer(survey, floor) {
    const data = this.received?.cells ?? survey.floors[floor];
    const [cols, rows] = survey.size;
    const cached = this.layers[floor];
    if (cached?.data === data && cached.canvas.width === cols && cached.canvas.height === rows) return cached;
    const canvas = document.createElement('canvas');
    canvas.width = cols;
    canvas.height = rows;
    const values = new Uint8Array(cols*rows);
    if (survey.encoding === 'runs') {
      let offset = 0;
      for (const [, type, length] of data.matchAll(/([abc])([0-9A-F]+)/g)) {
        const end = offset + Number.parseInt(length, 16);
        values.fill(type.charCodeAt(0)-97, offset, end);
        offset = end;
      }
    } else {
      for (let i = 0; i < values.length; i++) values[i] = data.charCodeAt(i)-48;
    }
    const context = canvas.getContext('2d');
    const pixels = context.createImageData(cols, rows);
    let known = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const cell = values[row*cols+col];
        const i = ((rows-row-1)*cols+col)*4;
        const [r, g, b] = cells[cell];
        pixels.data[i] = r;
        pixels.data[i+1] = g;
        pixels.data[i+2] = b;
        pixels.data[i+3] = 255;
        if (cell) known++;
      }
    }
    context.putImageData(pixels, 0, 0);
    return this.layers[floor] = { data, canvas, values, known, coverage: new Map() };
  }

  draw() {
    const { canvas, ctx, snapshot, layout, floor, order } = this;
    const survey = snapshot?.survey;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!survey || w < 20 || h < 20) return;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const width = Math.round(w*ratio), height = Math.round(h*ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#f1f5f2';
    ctx.fillRect(0, 0, w, h);
    const [cols, rows] = survey.size;
    const [ox, oy] = survey.origin;
    const res = survey.resolution;
    const [west, south, east, north] = order?.bounds ?? [ox, oy, ox+cols*res, oy+rows*res];
    const scale = Math.min((w-16)/(east-west), (h-14)/(north-south));
    const mapWidth = (east-west)*scale, mapHeight = (north-south)*scale;
    const marginX = (w-mapWidth)/2, marginY = (h-mapHeight)/2;
    const pos = (x, y) => [marginX+(x-west)*scale, marginY+(north-y)*scale];
    const inside = (x, y) => x >= west && x <= east && y >= south && y <= north;
    ctx.save();
    ctx.beginPath();
    ctx.rect(marginX, marginY, mapWidth, mapHeight);
    ctx.clip();
    ctx.drawImage(this.layer(survey, floor).canvas, ...pos(ox, oy+rows*res), cols*res*scale, rows*res*scale);

    if (!order && layout.orders) {
      for (const [i, area] of layout.orders.entries()) {
        const [x1, y1, x2, y2] = area.bounds;
        const [x, y] = pos(x1, y2);
        ctx.lineWidth = i === snapshot.order ? 1.35 : .85;
        ctx.strokeStyle = i === snapshot.order ? '#237267' : '#a6b7ae';
        ctx.strokeRect(x, y, (x2-x1)*scale, (y2-y1)*scale);
      }
    }
    const route = this.route;
    if (floor === 'lower' && route) {
      ctx.beginPath();
      ctx.lineWidth = order ? 2.15 : 2.6;
      ctx.strokeStyle = '#2d5576';
      ctx.setLineDash(order ? [5, 3] : []);
      route.points.forEach(([x, y], i) => {
        const [px, py] = pos(x, y);
        if (i) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const landmarks = order
      ? (order.id === 'pump' ? layout.surveyLandmarks[floor] : layout.zoneLandmarks[order.id])
      : layout.orders?.map(area => [area.area, ...area.center]) ?? layout.surveyLandmarks[floor];
    ctx.font = '600 10px "Segoe UI", "Microsoft YaHei UI", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    for (const [label, x, y] of landmarks) {
      if (!inside(x, y)) continue;
      let [px, py] = pos(x, y);
      const half = ctx.measureText(label).width/2+4;
      px = Math.max(marginX+half, Math.min(marginX+mapWidth-half, px));
      py = Math.max(marginY+7, Math.min(marginY+mapHeight-7, py));
      ctx.fillStyle = '#fcfffbf0';
      ctx.fillRect(px-half, py-7, 2*half, 14);
      ctx.fillStyle = '#344a42';
      ctx.fillText(label, px, py);
    }
    if (floor === 'lower')
      for (const evidence of [survey.finding, survey.evidence]) {
        if (!evidence || !inside(...evidence.point)) continue;
        const [x, y] = pos(...evidence.point);
        ctx.beginPath();
        ctx.arc(x, y, order ? 4 : 2.8, 0, Math.PI*2);
        ctx.fillStyle = '#9d6216';
        ctx.fill();
      }
    ctx.font = '700 10px Consolas, monospace';
    for (const [key, [name, color]] of Object.entries(actors)) {
      const [x, y, z] = snapshot.actors[key].pose;
      if (!inside(x, y) || (z > 2.79 ? 'upper' : 'lower') !== floor) continue;
      const [px, py] = pos(x, y);
      ctx.beginPath();
      ctx.arc(px, py, 11, 0, Math.PI*2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.fillText(name, px, py+.4);
    }
    ctx.restore();
  }
}
