use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::Manager;

struct Worker {
    process: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

#[cfg(not(debug_assertions))]
const WORKER_NAME: &str = concat!("robot-engine-", env!("COLLAB_ENGINE_HASH"), ".exe");

#[cfg(not(debug_assertions))]
fn command(cache: &Path) -> std::io::Result<Command> {
    const IMAGE: &[u8] = include_bytes!("../target/inference/robot-engine.exe");
    let path = cache.join(WORKER_NAME);
    let matches = || fs::read(&path).is_ok_and(|bytes| bytes == IMAGE);
    if !matches() {
        let staging = path.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&staging, IMAGE)?;
        if let Err(e) = fs::rename(&staging, &path) {
            let _ = fs::remove_file(&staging);
            if !matches() { return Err(e); }
        }
    }
    Ok(Command::new(path))
}

#[cfg(not(debug_assertions))]
fn prune_old_workers(directory: &Path) {
    let Ok(entries) = fs::read_dir(directory) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let digest = name.strip_prefix("robot-engine-").and_then(|v| v.strip_suffix(".exe"));
        if name != WORKER_NAME && digest.is_some_and(|v| v.len() == 64 && v.bytes().all(|c| c.is_ascii_hexdigit()))
            && entry.file_type().is_ok_and(|t| t.is_file())
        {
            // Windows keeps an executing image locked. Try again on the next launch.
            let _ = fs::remove_file(entry.path());
        }
    }
}

#[cfg(debug_assertions)]
fn command(_cache: &Path) -> std::io::Result<Command> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let local = root.join(".inference-env/Scripts/python.exe");
    let reference = root.join("../机器狗/.inference-env/Scripts/python.exe");
    let python = std::env::var_os("COLLAB_PYTHON")
        .or_else(|| local.exists().then(|| local.into_os_string()))
        .or_else(|| reference.exists().then(|| reference.into_os_string()))
        .unwrap_or_else(|| "python".into());
    let mut cmd = Command::new(python);
    cmd.arg(root.join("training/live.py"));
    Ok(cmd)
}

impl Worker {
    fn launch(cache: &Path) -> std::io::Result<Self> {
        fs::create_dir_all(cache)?;
        let log = fs::File::create(cache.join(format!("engine-{}.log", std::process::id())))?;
        #[cfg(not(debug_assertions))]
        let directory = cache.join("workers");
        #[cfg(debug_assertions)]
        let directory = cache.to_path_buf();
        #[cfg(not(debug_assertions))]
        let _launch_lock = {
            fs::create_dir_all(&directory)?;
            let lock = fs::OpenOptions::new().create(true).truncate(false).read(true).write(true)
                .open(directory.join("launch.lock"))?;
            // Serialize installation, spawning and pruning across desktop instances.
            lock.lock()?;
            lock
        };
        let mut process = command(&directory)?.creation_flags(0x08000000)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(log).spawn()?;
        #[cfg(not(debug_assertions))]
        prune_old_workers(&directory);
        Ok(Self { input: process.stdin.take(), output: BufReader::new(process.stdout.take().unwrap()), process })
    }

    fn request(&mut self, message: &Value) -> std::io::Result<Value> {
        let input = self.input.as_mut().unwrap();
        serde_json::to_writer(&mut *input, message)?;
        input.write_all(b"\n")?;
        input.flush()?;
        let mut reply = String::new();
        if self.output.read_line(&mut reply)? == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "物理引擎已退出"));
        }
        Ok(serde_json::from_str(&reply)?)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.input.take());
        let _ = self.process.wait();
    }
}

#[derive(Default)]
pub struct Engine(Arc<Mutex<Option<Worker>>>);

#[tauri::command]
pub async fn inference(app: tauri::AppHandle, engine: tauri::State<'_, Engine>, message: Value) -> Result<Value, String> {
    let inner = engine.0.clone();
    let cache = app.path().app_cache_dir().map_err(|e| e.to_string())?.join("physics");
    tauri::async_runtime::spawn_blocking(move || {
        let mut active = inner.lock().map_err(|_| "物理引擎状态无效".to_string())?;
        if active.is_none() {
            *active = Some(Worker::launch(&cache).map_err(|e| format!("启动本地物理引擎失败：{e}"))?);
        }
        match active.as_mut().unwrap().request(&message) {
            Ok(reply) => Ok(reply),
            Err(e) => {
                active.take();
                Err(format!("本地物理引擎通信失败：{e}"))
            }
        }
    }).await.map_err(|e| e.to_string())?
}
