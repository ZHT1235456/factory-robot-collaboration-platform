fn main() {
    if std::env::var("PROFILE").as_deref() == Ok("release") {
        use sha2::{Digest, Sha256};
        let engine = "target/inference/robot-engine.exe";
        println!("cargo:rerun-if-changed={engine}");
        let bytes = std::fs::read(engine).expect("Build the CPU worker with scripts/build_worker.py first");
        println!("cargo:rustc-env=COLLAB_ENGINE_HASH={:x}", Sha256::digest(bytes));
    }
    tauri_build::build()
}
