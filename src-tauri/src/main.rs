#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicBool, Ordering};
mod inference;

#[tauri::command]
fn toggle_fullscreen(
    window: tauri::WebviewWindow,
    sized: tauri::State<AtomicBool>,
) -> Result<bool, String> {
    let next = !window.is_fullscreen().map_err(|err| err.to_string())?;
    window.set_fullscreen(next).map_err(|err| err.to_string())?;
    if !next && !sized.load(Ordering::Relaxed) {
        if let Some(monitor) = window.current_monitor().map_err(|err| err.to_string())? {
            let area = monitor.work_area();
            let size =
                tauri::PhysicalSize::new(area.size.width * 91 / 100, area.size.height * 87 / 100);
            window.set_size(size).map_err(|err| err.to_string())?;
            window.center().map_err(|err| err.to_string())?;
            sized.store(true, Ordering::Relaxed);
        }
    }
    Ok(next)
}

fn main() {
    tauri::Builder::default()
        .manage(AtomicBool::new(false))
        .manage(inference::Engine::default())
        .invoke_handler(tauri::generate_handler![toggle_fullscreen, inference::inference])
        .run(tauri::generate_context!())
        .expect("工厂机器人协同作业平台启动失败");
}
