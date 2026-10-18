"""Download the benchmark bodies (Blender Studio "Human Base Meshes", CC0).

    python bench/fetch_assets.py

The bundle (~50 MB) goes to bench/assets/, which git ignores. It is only
downloaded once and checked against a SHA-256.
"""
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = ("https://download.blender.org/demo/asset-bundles/human-base-meshes/"
       "human-base-meshes-bundle-v1.4.1.zip")
SHA256 = "811f43accbb31a88266d932f8f5563b2d13586fca0ba2693aad1f5fe582b3515"
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BLEND = os.path.join(ASSETS, "human-base-meshes-bundle-v1.4.1", "human_base_meshes_bundle.blend")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch():
    if os.path.exists(BLEND):
        return BLEND
    os.makedirs(ASSETS, exist_ok=True)
    archive = os.path.join(ASSETS, os.path.basename(URL))
    if not os.path.exists(archive) or sha256(archive) != SHA256:
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, archive)
    if sha256(archive) != SHA256:
        sys.exit(f"Checksum mismatch for {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(ASSETS)
    return BLEND


if __name__ == "__main__":
    print(fetch())
