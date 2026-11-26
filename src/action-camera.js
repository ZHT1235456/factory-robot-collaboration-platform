const speechSeconds = 6;

export function recentSpeech({ actors, t, eventCount }, journal) {
  const messages = {};
  for (let i = eventCount-1; i >= 0; i--) {
    const event = journal[i];
    if (t-event.time >= speechSeconds) break;
    if (Object.hasOwn(actors, event.sender) && !messages[event.sender])
      messages[event.sender] = event;
  }
  return messages;
}

function isMoving(snapshot, robot) {
  return snapshot?.status === 'running' &&
    (snapshot.actors[robot].moving || (robot === 'g1' && snapshot.liftMoving));
}

export function actionCameraActor(snapshot, previous, speech) {
  const current = previous?.cameraActor ?? null;
  const message = snapshot.events.findLast(event => Object.hasOwn(snapshot.actors, event.sender));
  if (message) return message.sender;

  const moving = Object.keys(snapshot.actors).filter(robot => isMoving(snapshot, robot));
  const started = moving.filter(robot => !isMoving(previous, robot));
  if (started.length) return started.includes(current) ? current : started[0];
  if (moving.includes(current) || speech[current]) return current;
  if (moving.length) return moving[0];
  return snapshot.status === 'running' && snapshot.mapping?.stage === 'surveying' ? 'go2' : current;
}
