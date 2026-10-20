"""Resume, verify and extract the official PushT dataset."""
import argparse
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"
SIZE = 13_136_247_974
SHA256 = "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path(__file__).resolve().parents[1] / "storage")
    args = parser.parse_args()
    target = args.storage / "datasets" / "pusht_expert_train.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Dataset already exists: {target}", flush=True)
        return
    archive = target.with_suffix(".h5.zst")
    for attempt in range(10):
        offset = archive.stat().st_size if archive.exists() else 0
        if offset == SIZE:
            break
        if offset > SIZE:
            raise RuntimeError("Archive exceeds expected size; check file manually")
        request = urllib.request.Request(
            URL + f"?download=true&attempt={time.time_ns()}",
            headers={"Range": f"bytes={offset}-", "User-Agent": "Flow-JEPA-setup"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Invalid resume response; archive preserved")
                with archive.open("ab" if offset else "wb") as out:
                    last_report = time.monotonic()
                    while chunk := response.read(8 * 1024 * 1024):
                        out.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report >= 15:
                            print(f"Downloaded {offset / 1e9:.2f}/{SIZE / 1e9:.2f} GB ({100 * offset / SIZE:.1f}%)", flush=True)
                            last_report = time.monotonic()
        except (OSError, urllib.error.URLError) as exc:
            print(f"Download interrupted ({exc}); retry {attempt + 1}/10", flush=True)
            time.sleep(3)
    if not archive.exists() or archive.stat().st_size != SIZE:
        raise RuntimeError("Download incomplete; rerun to resume")
    print("Verifying SHA-256...", flush=True)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("Archive SHA-256 mismatch; extraction cancelled")
    import zstandard
    partial = target.with_suffix(".h5.partial")
    print("SHA-256 OK. Decompressing dataset...", flush=True)
    with archive.open("rb") as source, partial.open("wb") as output:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            last_report = time.monotonic()
            while chunk := reader.read(8 * 1024 * 1024):
                output.write(chunk)
                if time.monotonic() - last_report >= 20:
                    print(f"Extracted {output.tell() / 1e9:.2f} GB", flush=True)
                    last_report = time.monotonic()
    partial.replace(target)
    print(f"Dataset ready: {target} ({target.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
