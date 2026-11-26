"""Package the exact local inference inputs into one compressed worker."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src-tauri/target/inference"
INPUTS = ["model/g1.xml", "model/go2.xml", "model/G1-LICENSE", "model/LICENSE",
          "scenes/catalog.json", "scenes/factory.json", "scenes/yard.json", "results/g1-contact-policy.onnx",
          "results/foot-tracking-policy.onnx", "results/g1-ladder-skip-policy.onnx"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256()
    for name in [*INPUTS, "training/live.py", "training/sim.py", "training/plant.py", "training/navigation.py",
                 "training/gait.py", "training/gait_policy.py", "training/pump_survey.py", "training/occupancy.py",
                 "training/factory_layout.py", "training/factory_markings.py", "training/factory_mission.py",
                 "training/ladder.py", "training/yard_layout.py", "training/yard_mission.py",
                 "training/inference-requirements.txt", "scripts/build_worker.py",
                 "scripts/pyinstaller-hooks/hook-onnxruntime.py"]:
        signature.update((ROOT / name).read_bytes())
    key = signature.hexdigest()
    stamp = OUT / "source.sha256"
    if stamp.exists() and stamp.read_text() == key and (OUT / "robot-engine.exe").is_file():
        return
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--console", "--noupx",
               "--name", "robot-engine", "--distpath", str(OUT), "--workpath", str(OUT / "build"),
               "--specpath", str(OUT), "--additional-hooks-dir", str(ROOT / "scripts/pyinstaller-hooks"),
               "--collect-binaries", "mujoco"]
    for library in ("mujoco", "numpy", "onnxruntime"):
        command += ["--copy-metadata", library]
    for library in ("PyQt5", "PyQt6", "torch", "matplotlib", "IPython", "tkinter", "glfw", "OpenGL",
                    "numpy.testing", "numpy.f2py"):
        command += ["--exclude-module", library]
    for name in INPUTS:
        command += ["--add-data", f"{ROOT / name}{os.pathsep}{Path(name).parent}"]
    command.append(str(ROOT / "training/live.py"))
    subprocess.run(command, cwd=ROOT, check=True)
    stamp.write_text(key, encoding="ascii")


if __name__ == "__main__":
    main()
