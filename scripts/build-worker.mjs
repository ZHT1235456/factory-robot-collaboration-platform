import { spawnSync } from 'node:child_process';
import { physicsPython } from './runtime.mjs';

const result = spawnSync(physicsPython(), ['scripts/build_worker.py'], { stdio: 'inherit', windowsHide: true });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
