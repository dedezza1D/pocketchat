"""Download the published weights and their tokenizer, without a GitHub account.

    uv run python scripts/download_weights.py chat         # checkpoints/sft_reinit.pt (CC BY-SA 3.0)
    uv run python scripts/download_weights.py pretrained   # checkpoints/pretrain_final.pt (MIT)
    uv run python scripts/download_weights.py all

Files come from the GitHub releases and are checked against their SHA-256. A file that is already
in place with the right hash is kept; one with a different hash is left alone and reported.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/dedezza1D/pocketchat/releases/download"
TOKENIZER = ("data/tokenizer_full.json", "6f27f5978dd437aa40c41eb8bd43d06715418804c584272bdebadfe4044d2b66")
MODELS = {
    "pretrained": ("v0.1.0", "checkpoints/pretrain_final.pt",
                   "ee74d221cdd74760348923759715e09a452dff6fc38395c8b1ea77b45c254429"),
    "chat": ("v0.2.0", "checkpoints/sft_reinit.pt",
             "e86f3e4db5944929806aeab7254c44c81f410b4fa61a464288f8cfee92aa695c"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str) -> str:
    """Download `url` to `dest` unless it is already there; returns what happened."""
    if dest.exists():
        if sha256(dest) == expected:
            return "already there"
        raise SystemExit(f"{dest} exists but is not the published file (different SHA-256); "
                         "move it elsewhere and run again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while block := response.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r{dest.name}: {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if sha256(part) != expected:
        part.unlink()
        raise SystemExit(f"{url} did not match its SHA-256; nothing was written to {dest}")
    part.replace(dest)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=[*MODELS, "all"])
    args = ap.parse_args()

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        tag, path, digest = MODELS[name]
        print(f"{path}: {fetch(f'{RELEASES}/{tag}/{Path(path).name}', Path(path), digest)}")
    # both releases ship the same tokenizer file
    tag = MODELS[names[0]][0]
    print(f"{TOKENIZER[0]}: {fetch(f'{RELEASES}/{tag}/{Path(TOKENIZER[0]).name}', Path(TOKENIZER[0]), TOKENIZER[1])}")


if __name__ == "__main__":
    main()
