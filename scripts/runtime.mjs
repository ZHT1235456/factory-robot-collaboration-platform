import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

export function physicsPython() {
  const executable = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  const candidates = [resolve('.inference-env', executable), resolve('../机器狗/.inference-env', executable)];
  return process.env.COLLAB_PYTHON || candidates.find(path => existsSync(path)) || 'python';
}
