"""Package the add-on into an installable zip.

    python build.py            -> dist/rigger-<version>.zip

The zip has blender_manifest.toml at its root, so it installs as an
extension in Blender 4.2+ (Edit > Preferences > Get Extensions >
Install from Disk).
"""
import os
import re
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "rigger")


def version():
    with open(os.path.join(SRC, "blender_manifest.toml"), encoding="utf-8") as fh:
        return re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M).group(1)


def build():
    out_dir = os.path.join(ROOT, "dist")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"rigger-{version()}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, dirs, files in os.walk(SRC):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if name.endswith(".pyc"):
                    continue
                path = os.path.join(folder, name)
                zf.write(path, os.path.relpath(path, SRC))
    print(out)
    return out


if __name__ == "__main__":
    build()
