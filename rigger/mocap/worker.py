# SPDX-License-Identifier: GPL-3.0-or-later
"""Background worker: a headless Blender that turns a video into a motion file.

    blender -b --factory-startup --python worker.py -- job.json

The job file names the video, the skeleton, the output and the settings.
Progress goes to stdout as JSON lines prefixed with RIGGER_JSON, so the
operator in the open Blender can follow it while the interface stays usable.
"""

import importlib
import json
import os
import sys
import traceback

PREFIX = "RIGGER_JSON "


def say(**msg):
    print(PREFIX + json.dumps(msg), flush=True)


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    if job.get("ai_dir"):
        os.environ["RIGGER_AI_DIR"] = job["ai_dir"]
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(package_dir))
    pkg = importlib.import_module(os.path.basename(package_dir))
    pipeline = importlib.import_module(pkg.__name__ + ".mocap.pipeline")

    def progress(stage, done, total):
        if done == total or done % 5 == 0:
            say(stage=stage, done=done, total=total)
    try:
        meta, _sk, _arrays = pipeline.run(job["video"], job["skeleton"], out=job["out"], focal_mm=job["focal_mm"],
                                          start=job.get("start", 0), count=job.get("count"),
                                          max_side=job.get("max_side", 1280), cache_dir=job.get("cache_dir"),
                                          settings=job.get("settings"), progress=progress)
        say(done_file=job["out"], frames=meta["frames"], found=meta["found"])
    except Exception as exc:  # reported to the operator, which shows it to the user
        say(error=str(exc) or type(exc).__name__, trace=traceback.format_exc())
        sys.exit(1)


main()
