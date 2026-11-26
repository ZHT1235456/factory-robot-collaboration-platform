"""Render a recorded MuJoCo climbing rollout as an MP4."""

import argparse
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np

from ladder import Setup, build_model


def render(trace_path, output, azimuth):
    with np.load(trace_path) as trace:
        setup = Setup(spacing=float(trace["spacing"]))
        poses, dt = trace["poses"], float(trace["dt"])
    width, height = 896, 672
    model = build_model(setup)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.lookat[:] = [.12, 0., 1.10]
    camera.distance = 2.97
    camera.azimuth = azimuth
    camera.elevation = -10.
    option = mj.MjvOption()
    option.geomgroup[3] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
               "-framerate", str(1/(dt*2)), "-i", "-", "-an", "-c:v", "libx264",
               "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with mj.Renderer(model, height=height, width=width) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for pose in poses[::2]:
                data.qpos[:] = pose
                mj.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                process.stdin.write(renderer.render().tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the climbing rollout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--azimuth", type=float, default=65.)
    args = parser.parse_args()
    render(args.trace, args.output, args.azimuth)


if __name__ == "__main__":
    main()
