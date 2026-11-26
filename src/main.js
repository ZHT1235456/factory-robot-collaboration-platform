import { invoke, isTauri } from '@tauri-apps/api/core';
import factoryScene from '../scenes/factory.json';
import yardScene from '../scenes/yard.json';
import { WorkcellView } from './viewport.js';
import { SurveyMap } from './survey-map.js';
import { actionCameraActor, recentSpeech } from './action-camera.js';
import './style.css';

const $ = id => document.getElementById(id);
const isDesktop = isTauri();
const client = crypto.randomUUID();
const controls = ['run', 'pause', 'step', 'reset', 'new-seed'].map($);
const stageStates = { ready: '待运行', running: '作业中', paused: '已暂停', done: '已验收', fault: '需复核', review: '回看' };
const robotNames = { g1: 'G1', go2: 'Go2' };
const floorNames = { clear: '通', blocked: '阻', unknown: '待查' };
const laneNames = { near: '服务通道', far: '南侧回程通道' };
const portalNames = { west: '换热西门', east: '换热东门' };
const scenarios = { factory: factoryScene, yard: yardScene };
const ladderStates = { mounting: '登梯准备', ascending: '跨档上楼', landing: '转移至二层平台' };
let scene = factoryScene;

let view, ready, latest, displayed, history = [], journal = [], program = [];
let active = false, resetting = false, busy = false, generation = 0, timer = 0, replayIndex = null;
let virtualOrigin = 0, wallOrigin = 0, lastEventKey = '';
let floor = 'lower';
let scope = 'factory', mapSource = 'g1', inspectedOrder = null, programOrder = null;
let cameraMode = 'action', followedRobot = null;
const orderControls = [];
const surveyMap = new SurveyMap($('survey-map'));

const clock = seconds => {
  const milli = Math.round(seconds*1000);
  return String(Math.floor(milli/60000)).padStart(2, '0') + ':' +
    String(Math.floor(milli/1000) % 60).padStart(2, '0') + '.' + String(milli%1000).padStart(3, '0');
};

const element = (tag, text, className = '') => {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
};

function system(text, state) {
  $('engine-status').replaceChildren(document.createElement('span'), document.createTextNode(text));
  $('engine-status').dataset.state = state;
}

async function request(body) {
  const message = { client, ...body };
  let data;
  if (isDesktop) data = await invoke('inference', { message });
  else {
    const response = await fetch('/__inference', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(message),
    });
    data = await response.json();
  }
  if (!data.ok) throw new Error(data.error || '物理引擎请求失败');
  return data;
}

function showCatalogEntry() {
  $('scene-title').textContent = scene.name + ' · ' + scene.job;
  orderControls.length = 0;
  programOrder = null;
  for (const [index, order] of scene.orders.entries()) {
    const button = element('button', '');
    button.type = 'button';
    button.dataset.order = String(index);
    button.title = '定位到'+order.area;
    button.append(element('span', order.name), element('span', '待执行'));
    orderControls.push(button);
  }
  $('orders').replaceChildren(...orderControls);
  showOrders(null);
}

function showOrders(snapshot) {
  const activeIndex = snapshot?.order ?? 0;
  const index = inspectedOrder ?? activeIndex;
  const order = scene.orders[index];
  const building = snapshot?.mapping?.stage === 'surveying' || snapshot?.mapping?.stage === 'sending';
  const finished = snapshot?.completed ?? [];
  const complete = finished.includes(order.id);
  const fault = snapshot?.status === 'fault' && index === activeIndex;
  const stepsDone = scene.orders.filter(item => finished.includes(item.id))
    .reduce((count, item) => count+item.steps.length, 0) +
    (finished.includes(scene.orders[activeIndex].id) ? 0 : Math.max(0, snapshot?.step ?? 0));
  $('order-progress').textContent = finished.length+' / '+scene.orders.length+' 工单';
  const total = scene.orders.reduce((count, item) => count+item.steps.length, 0);
  $('checkpoint-progress').textContent = stepsDone+' / '+total+' 节点';
  $('zone').textContent = scene.zone+'  /  '+order.area;
  $('active-order').textContent = building && inspectedOrder === null
    ? (scene.id === 'yard' ? 'Go2 室外测绘 · G1 等待接图' : 'Go2 下层建图 · G1 等待接图')
    : order.area+' · '+order.name;
  for (const [offset, button] of orderControls.entries()) {
    const state = finished.includes(scene.orders[offset].id) ? 'complete' :
      (offset === activeIndex ? (snapshot?.status === 'fault' ? 'fault' : 'current') : 'pending');
    button.dataset.state = state;
    button.setAttribute('aria-pressed', String(offset === inspectedOrder));
    button.lastElementChild.textContent = { complete: '已验收', fault: '待复核', pending: '待执行',
      current: snapshot?.status === 'ready' || !snapshot ? '待运行' : building ? '待接图' : '进行中' }[state];
  }
  if (programOrder !== index) {
    programOrder = index;
    program = order.steps.map((text, offset) => {
      const li = element('li', '');
      const glyph = element('span', String(offset+1), 'step-symbol');
      glyph.setAttribute('aria-hidden', 'true');
      li.append(glyph, element('span', text));
      return li;
    });
    $('steps').replaceChildren(...program);
  }
  for (const [offset, li] of program.entries()) {
    const checked = complete || index === activeIndex && offset < snapshot?.step;
    const current = !checked && index === activeIndex && offset === snapshot?.step;
    li.className = checked ? 'done' : current ? (fault ? 'fault' : 'current') : '';
    li.firstChild.textContent = checked ? '✓' : current && fault ? '!' : String(offset+1);
    if (current) li.setAttribute('aria-current', 'step');
    else li.removeAttribute('aria-current');
  }
  view?.setOrder(order, scene.orders[activeIndex].id);
}

function pickCamera(mode) {
  cameraMode = mode;
  for (const button of $('cameras').children)
    button.setAttribute('aria-pressed', String(button.dataset.camera === mode));
  if (mode === 'action') followAutomaticCamera(displayed, true);
  else view?.setView(mode);
}

function followAutomaticCamera(snapshot, recenter = false) {
  const surveying = snapshot?.mapping?.stage === 'surveying' && snapshot.status === 'running';
  const label = surveying ? '随建图' : '随行动';
  const button = $('auto-camera');
  if (button.textContent !== label) button.textContent = label;
  if (cameraMode !== 'action') return;
  const robot = snapshot?.cameraActor ?? null;
  if (robot === followedRobot && !recenter) return;
  followedRobot = robot;
  view?.setView(robot ?? 'overview');
}

function controlsState() {
  const locked = !latest || resetting;
  for (const control of controls) control.disabled = locked;
  $('pause').disabled = locked || !active;
  $('run').disabled = locked || active || (latest.status === 'done' && replayIndex === null) || latest.status === 'fault';
  $('step').disabled = locked || active || busy || latest.status === 'fault' ||
    (latest.status === 'done' && replayIndex === null);
  $('seek').disabled = locked || active || history.length < 2;
  $('site-select').disabled = resetting;
}

function showEvents(snapshot) {
  const raw = journal.slice(0, snapshot.eventCount)
    .filter(info => info.sender || info.kind === 'wait' || info.kind === 'fault').reverse();
  const key = snapshot.eventCount;
  if (key === lastEventKey) return;
  lastEventKey = key;
  $('message-count').textContent = (snapshot.metrics.messages || 0) + ' 条';
  const rows = raw.map(info => {
    const row = element('li', '', info.kind);
    if (info.signal) row.dataset.signal = info.signal;
    const head = element('div', '', 'message-head');
    if (info.sender) head.append(element('span', robotNames[info.sender], 'message-tag '+info.sender),
      element('span', '→', 'message-arrow'), element('span', robotNames[info.receiver], 'message-tag '+info.receiver));
    else head.append(element('span', '工单', 'message-tag'));
    head.append(element('time', clock(info.time).slice(0, 8)));
    row.append(head, element('span', info.text, 'message-text'));
    return row;
  });
  $('events').replaceChildren(...(rows.length ? rows : [element('li', '等待 Go2 开始建图', 'message-empty')]));
}

function showMap(snapshot) {
  const map = snapshot.survey;
  if (!map) return;
  const upper = floor === 'upper';
  const mapping = snapshot.mapping;
  const delivery = mapping?.received;
  const received = !upper && mapSource === 'g1' && delivery
    ? journal[delivery.event]?.payload?.map : null;
  const sources = $('map-sources');
  sources.hidden = upper;
  for (const button of sources.children) {
    button.disabled = button.dataset.source === 'g1' && !delivery;
    button.setAttribute('aria-pressed', String(button.dataset.source === (received ? 'g1' : 'go2')));
  }
  const index = upper ? (inspectedOrder === 2 || snapshot.order === 2 ? 2 : 0) : inspectedOrder ?? snapshot.order;
  const order = scene.orders[index];
  const known = surveyMap.update(snapshot, view.layout, floor, scope === 'zone' ? order : null, received);
  $('map-version').textContent = 'V'+String(received?.revision ?? map.floorVersions?.[floor] ?? map.revision).padStart(2, '0');
  $('map-location').textContent = scope === 'zone' ? order.area : scene.extent[0]+' × '+scene.extent[1]+' m';
  $('map-coverage').textContent = '已观测 '+known+'%';
  const fields = upper && index === 0 ? { A: 'A 梯上口', B: 'B 梯上口' } : upper || scene.id === 'yard' ? {} :
    index === 0 ? laneNames : (index === 2 || scope === 'factory') ? portalNames : {};
  const states = upper ? map.gates : index === 0 ? map.lanes : snapshot.portals;
  const rows = Object.entries(fields).map(([key, label]) => {
    const line = element('div', label);
    const state = states[key] || 'unknown';
    line.append(element('span', floorNames[state], 'route-state '+state));
    return line;
  });
  if (!upper && mapping) {
    const stage = snapshot.status === 'fault' && !delivery ? 'fault' : mapping.stage;
    const message = {
      ready: '等待 Go2 开始下层建图',
      surveying: `Go2 建图 · ${mapping.stops}/${mapping.total} 个观测点`,
      sending: 'Go2 已发送地图 · 等待 G1 确认',
      received: 'G1 已接收下层地图 V'+delivery?.revision,
      fault: '下层地图待复核',
    }[stage];
    const handoff = element('div', message, 'map-handoff');
    handoff.id = 'map-handoff';
    handoff.dataset.stage = stage;
    rows.unshift(handoff);
  }
  if (upper && index === 0 && snapshot.metrics.selectedFlight)
    rows.push(element('div', 'G1 已选用 '+snapshot.metrics.selectedFlight+' 梯', 'route-picked'));
  const route = surveyMap.route;
  if (!upper && route && (scope === 'factory' || (route.area ?? 'pump') === order.id))
    rows.push(element('div', 'G1 已采纳 '+(route.description ?? laneNames[route.lane])+
      ' · V'+route.revision, 'route-picked'));
  $('map-status').replaceChildren(...rows);
}

function present(snapshot, snap = false) {
  displayed = snapshot;
  view.apply(snapshot, snap);
  view.setSpeech(snapshot.speech);
  showOrders(snapshot);
  followAutomaticCamera(snapshot, snap && !active);
  let state = snapshot.status;
  if (replayIndex !== null) state = 'review';
  else if (state === 'running' && !active) state = 'paused';
  const badge = $('job-status');
  badge.dataset.state = state;
  badge.textContent = stageStates[state] || '等待';
  $('clock').textContent = clock(snapshot.t);
  $('elapsed').textContent = clock(latest.t);
  $('seek').value = replayIndex === null ? String(history.length-1) : String(replayIndex);
  $('seek').setAttribute('aria-valuetext', clock(snapshot.t));
  for (const key of ['g1', 'go2']) {
    const moving = snapshot.actors[key].moving;
    const ladder = key === 'g1' && snapshot.ladder;
    const betweenFloors = ladder && ladder.stage !== 'mounting' || key === 'g1' && snapshot.liftMoving;
    const location = scene.id === 'yard' ? '室外' : betweenFloors ? '层间' :
      snapshot.actors[key].pose[2] > 2.79 ? '二层' : '一层';
    const carrying = key === 'g1' ? snapshot.holding : snapshot.cargo === 'spare' &&
      snapshot.metrics.cargoLoads > 0 && snapshot.metrics.basketPickups < 2;
    const detecting = key === 'go2' && snapshot.survey.go2Detection;
    const mapping = snapshot.mapping?.stage;
    const motion = ladder ? ladderStates[ladder.stage] : key === 'g1' && snapshot.liftMoving ? '乘升降台' :
      mapping === 'surveying' ? (key === 'go2' ? '建图' : '待接图')
      : mapping === 'sending' ? (key === 'go2' ? '交付地图' : '接图')
      : detecting ? '探测' : carrying ? (moving ? '携件' : '承重') : moving ? '移动' : '驻留';
    $(key + '-state').textContent = location+' · '+motion;
    $(key + '-speed').textContent = snapshot.actors[key].speed.toFixed(2) + ' m/s';
  }
  const telemetry = [];
  if (snapshot.reading) {
    const sample = snapshot.reading;
    const line = element('div', sample.name + ' ');
    line.append(element('strong', String(sample.value) + (sample.unit ? ' ' + sample.unit : ''), 'reading'));
    telemetry.push(line);
  }
  if (snapshot.gripN > .08 || snapshot.holding) {
    const grip = element('div', 'G1 夹爪法向力 ');
    grip.append(element('strong', snapshot.gripN.toFixed(1) + ' N'));
    telemetry.push(grip);
  }
  if (snapshot.order > 0 && snapshot.metrics.verifiedFilter)
    telemetry.push(element('div', '已核对过滤位号 · '+snapshot.metrics.verifiedFilter));
  else if (snapshot.metrics.verifiedPump)
    telemetry.push(element('div', '高位柜号与低位指示一致 · '+snapshot.metrics.verifiedPump));
  else if (snapshot.survey.finding)
    telemetry.push(element('div', '低位接液指示 · '+snapshot.survey.finding.pump));
  if (snapshot.order === 0 && snapshot.survey.evidence)
    telemetry.push(element('div', '盖内指示片 · 已看到'));
  if (snapshot.order >= 3 && snapshot.metrics.spareMassG)
    telemetry.push(element('div', '备件称重 · '+snapshot.metrics.spareMassG+' g'));
  if (snapshot.metrics.acceptedReturn)
    telemetry.push(element('div', '回水位绿灯已读回 · 托架稳定'));
  const stride = snapshot.ladder?.step;
  if (stride) telemetry.push(element('div', (stride.limb === 'left_toe' ? '左脚 ' : '右脚 ') +
    stride.from+' → '+stride.to+' 档'));
  if (snapshot.door > .025)
    telemetry.push(element('div', '观察盖开度 ' + (snapshot.door*180/Math.PI).toFixed(1) + '°'));
  if (snapshot.status === 'fault' && snapshot.eventCount)
    telemetry.push(element('div', journal[snapshot.eventCount-1].text, 'fault'));
  $('feedback').replaceChildren(...(telemetry.length ? telemetry : [element('div', '等待首次有效反馈')]));
  showMap(snapshot);
  showEvents(snapshot);
  controlsState();
}

function receive(snapshot, snap = false) {
  journal.push(...snapshot.events);
  // Record the automatic target for this frame, including while a manual view is selected.
  snapshot.speech = recentSpeech(snapshot, journal);
  snapshot.cameraActor = actionCameraActor(snapshot, latest, snapshot.speech);
  if (!snapshot.survey.floors) snapshot.survey.floors = latest.survey.floors;
  latest = snapshot;
  history.push(snapshot);
  $('seek').max = String(history.length-1);
  if (snapshot.status === 'done' || snapshot.status === 'fault') active = false;
  present(snapshot, snap);
}

function halt() {
  active = false;
  clearTimeout(timer);
  if (displayed && latest) present(displayed);
  controlsState();
}

function showError(error) {
  halt();
  system('本地引擎异常', 'fault');
  $('viewport-message').hidden = false;
  $('viewport-message').textContent = String(error.message || error);
}

async function reset(nextScene = scene) {
  if (resetting) return;
  const input = $('seed');
  if (!input.reportValidity() || input.value === '') return;
  halt();
  scene = nextScene;
  const token = ++generation;
  resetting = true;
  latest = displayed = null;
  replayIndex = null;
  history = [];
  journal = [];
  lastEventKey = '';
  inspectedOrder = null;
  mapSource = 'g1';
  showCatalogEntry();
  setMapView('factory', 'lower');
  showOrders(null);
  $('events').replaceChildren();
  $('viewport-message').hidden = false;
  $('viewport-message').textContent = '正在装配 ' + scene.name;
  controlsState();
  try {
    const [, packet] = await Promise.all([ready, request({ op: 'reset', scene: scene.id, seed: Number(input.value) })]);
    if (token !== generation) return;
    view.rebuild(packet.layout, scene.id);
    $('floors').querySelector('[data-floor="upper"]').hidden = scene.id !== 'factory';
    pickCamera('action');
    $('viewport-message').hidden = true;
    system('离线仿真 · ONNX 已加载', 'ok');
    receive(packet, true);
  } catch (error) {
    showError(error);
  } finally {
    resetting = false;
    controlsState();
  }
}

async function advance(count, token) {
  if (!latest || busy) return;
  busy = true;
  try {
    const packet = await request({ op: 'advance', count, events: latest.eventCount, mapRevision: latest.survey.revision });
    if (token === generation) receive(packet);
  } catch (error) {
    if (token === generation) showError(error);
  } finally {
    busy = false;
    controlsState();
  }
}

function origin() {
  virtualOrigin = displayed?.t ?? 0;
  wallOrigin = performance.now();
}

async function nextFrame() {
  if (!active || !latest) return;
  const token = generation;
  const target = virtualOrigin + (performance.now()-wallOrigin)/1000 * Number($('rate').value);
  if (replayIndex !== null) {
    while (replayIndex < history.length-1 && history[replayIndex+1].t <= target) replayIndex++;
    if (replayIndex === history.length-1) {
      replayIndex = null;
      if (latest.status === 'done') active = false;
    }
    present(replayIndex === null ? latest : history[replayIndex]);
  } else {
    const due = Math.floor((target-latest.t)/.02 + 1.e-6);
    if (due > 0) await advance(Math.min(due, Number($('rate').value) >= 16 ? 32 : 8), token);
  }
  if (active && generation === token) timer = window.setTimeout(nextFrame, 14);
}

async function toggleFullscreen() {
  try {
    let full;
    if (isDesktop) full = await invoke('toggle_fullscreen');
    else if (document.fullscreenElement) { await document.exitFullscreen(); full = false; }
    else { await document.documentElement.requestFullscreen(); full = true; }
    $('fullscreen').lastElementChild.textContent = full ? '恢复窗口' : '切换全屏';
  } catch (error) {
    system(String(error.message || error), 'fault');
  }
}

$('fullscreen').lastElementChild.textContent = isDesktop ? '恢复窗口' : '切换全屏';
$('fullscreen').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => {
  if (!isDesktop) $('fullscreen').lastElementChild.textContent = document.fullscreenElement ? '恢复窗口' : '切换全屏';
});
$('cameras').addEventListener('click', event => {
  const mode = event.target.closest('button')?.dataset.camera;
  if (!mode) return;
  if (mode === 'zone' || mode === 'overview' || mode === 'action') {
    inspectedOrder = null;
    if (displayed) {
      showOrders(displayed);
      showMap(displayed);
    }
  }
  pickCamera(mode);
});
$('orders').addEventListener('click', event => {
  const button = event.target.closest('button[data-order]');
  if (!button || !displayed) return;
  inspectedOrder = Number(button.dataset.order);
  setMapView('zone', scene.orders[inspectedOrder].id === 'heat' && displayed.actors.g1.pose[2] > 2.79 ?
    'upper' : 'lower');
  showOrders(displayed);
  pickCamera('zone');
  showMap(displayed);
});
function setMapView(mapScope, mapFloor) {
  scope = mapScope;
  floor = mapFloor;
  view?.setMapFloor(mapFloor);
  for (const button of $('floors').querySelectorAll('button'))
    button.setAttribute('aria-pressed', String(button.dataset.floor === floor || button.dataset.scope === scope));
}
$('floors').addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  setMapView(button.dataset.scope ?? scope, button.dataset.floor ?? floor);
  if (displayed) showMap(displayed);
});
$('map-sources').addEventListener('click', event => {
  const button = event.target.closest('button[data-source]');
  if (!button || button.disabled || !displayed) return;
  mapSource = button.dataset.source;
  showMap(displayed);
});
$('run').addEventListener('click', () => {
  if (active || !latest || resetting) return;
  active = true;
  origin();
  controlsState();
  nextFrame();
});
$('pause').addEventListener('click', halt);
$('step').addEventListener('click', () => {
  if (replayIndex !== null) {
    const next = replayIndex + 1;
    replayIndex = next >= history.length-1 ? null : next;
    present(history[next], true);
  } else advance(1, generation);
});
$('reset').addEventListener('click', () => reset());
$('site-select').addEventListener('change', event => {
  const input = $('seed');
  if (!input.reportValidity() || input.value === '') {
    event.target.value = scene.id;
    return;
  }
  reset(scenarios[event.target.value]);
});
$('new-seed').addEventListener('click', () => {
  $('seed').value = String(crypto.getRandomValues(new Uint32Array(1))[0] % 2147483648);
  reset();
});
$('seed').addEventListener('keydown', event => { if (event.key === 'Enter') reset(); });
$('rate').addEventListener('change', origin);
$('seek').addEventListener('input', event => {
  const index = Number(event.target.value);
  replayIndex = index >= history.length-1 ? null : index;
  if (history[index]) present(history[index], true);
});
window.addEventListener('keydown', event => {
  if (event.key === 'F11') {
    event.preventDefault();
    if (!event.repeat) toggleFullscreen();
  }
  if (event.code === 'Space' && !['BUTTON', 'INPUT', 'SELECT'].includes(document.activeElement?.tagName)) {
    event.preventDefault();
    if (active) halt();
    else if (!$('run').disabled) $('run').click();
  }
});
window.addEventListener('pagehide', () => {
  if (isDesktop) return;
  void fetch('/__inference', { method: 'POST', keepalive: true,
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ client, op: 'close' }) }).catch(() => {});
});

controlsState();
try {
  view = new WorkcellView($('viewport'), $('scene-labels'));
  ready = view.loadRobots();
  reset();
} catch (error) {
  showError(error);
}
