"""Explicit one-time download; normal inference never calls this script."""
import hashlib
import json
import os
from pathlib import Path
import zipfile
import requests
import fcntl
import shutil

ROOT = Path(os.getenv("MOTION_MODEL_DIR", "/models/mdm"))
REVISION = "ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778"
# Official 50-step encoder archive, HumanML3D statistics, OpenAI ViT-B/32.
EXPECTED = {
    "model.pt": "0fbdc8547c8f262b8838645586790b55f983d90db3bb7ed58e4b5d49429587ca",
    "args.json": "924adeca8b21dbf2779101ae2dd896f2187715d4a184449230f74508ceffbeae",
    "Mean.npy": "26e136555dab04c94a129d446c26e6b9939cbf045fbf77bcf5462c1fb5a2001c",
    "Std.npy": "6565a65ed9b31e23c328829a309e1c482be8b85fd23b43d65451a9b19a917f40",
    "clip/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def valid(name):
    path = ROOT / name
    return path.is_file() and digest(path) == EXPECTED[name]


def atomic_file(name, stream):
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("wb") as output:
        shutil.copyfileobj(stream, output)
    if digest(temporary) != EXPECTED[name]:
        raise RuntimeError(f"Unexpected SHA256: {name}; retry preparation")
    temporary.replace(target)


def prepare():
    import gdown
    import clip
    if not all(valid(name) for name in EXPECTED):
        (ROOT / "ready.json").unlink(missing_ok=True)
    if not all(valid(name) for name in ("model.pt", "args.json")):
        archive = ROOT / "humanml_enc_512_50steps.zip"
        archive_valid = False
        if zipfile.is_zipfile(archive):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    archive_valid = bundle.testzip() is None
            except (OSError, zipfile.BadZipFile):
                pass
        if not archive_valid:
            temporary = archive.with_suffix(".zip.part")
            gdown.download(id="1cfadR1eZ116TIdXK7qDX1RugAerEiJXr", output=str(temporary), quiet=False)
            with zipfile.ZipFile(temporary) as bundle:
                if bundle.testzip() is not None: raise RuntimeError("Checkpoint ZIP is corrupt; retry preparation")
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            names = [n for n in bundle.namelist() if n.endswith(".pt") and "opt" not in Path(n).name]
            if len(names) != 1: raise RuntimeError(f"Unexpected checkpoint files: {names}")
            # Copy named members only, never extract arbitrary archive paths.
            for source, name in [(names[0], "model.pt"), (next(n for n in bundle.namelist() if n.endswith("args.json")), "args.json")]:
                if not valid(name):
                    with bundle.open(source) as stream: atomic_file(name, stream)
    for name in ("Mean.npy", "Std.npy"):
        if not valid(name):
            with requests.get(f"https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/HumanML3D/{name}", timeout=60, stream=True) as response:
                response.raise_for_status(); atomic_file(name, response.raw)
    if not valid("clip/ViT-B-32.pt"):
        # Download into a staging directory; a partial file is never published.
        staging = ROOT / "clip-staging"
        clip.load("ViT-B/32", device="cpu", download_root=str(staging))
        with (staging / "ViT-B-32.pt").open("rb") as stream: atomic_file("clip/ViT-B-32.pt", stream)
    if not all(valid(name) for name in EXPECTED): raise RuntimeError("Model preparation is incomplete")
    manifest = {"model": "MDM HumanML3D 50 steps", "revision": REVISION,
                "source": "https://github.com/GuyTevet/motion-diffusion-model", "files": EXPECTED}
    temporary = ROOT / "ready.json.tmp"; temporary.write_text(json.dumps(manifest, indent=2)); temporary.replace(ROOT / "ready.json")
    print("Local model files verified and ready", flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare()

if __name__ == "__main__": main()
