from datetime import datetime, timedelta

import requests
from pathlib import Path

def fetch_manifest(url: str) -> dict:
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"manifest request failed: {response.status_code}")
    manifest = response.json()
    if "match_player" not in manifest.get("tables",{}):
        raise RuntimeError("manifest has no tables.match_player")
    return manifest

def select_files(manifest: dict, *, now: datetime, lookback_hours: int) -> list[str]:
    window_start = (now - timedelta(hours=lookback_hours)).timestamp()
    window_end = now.timestamp()
    selected = []
    for entry in manifest["tables"]["match_player"]["files"]:
        if entry.get("kind") not in {"delta", "residual"}:
            continue
        lo = entry.get("lo")
        hi = entry.get("hi")
        if lo is None or hi is None:
            continue
        if lo < window_end and hi > window_start:
            selected.append(entry)
    selected.sort(key=lambda entry: (entry["lo"], entry["hi"]))
    return selected

def file_url(manifest: dict, entry: dict) -> str:
    base = str(manifest["public_url"]).rstrip("/")
    key = str(entry["key"]).lstrip("/")
    return f"{base}/{key}"

def download_files(entries: list[dict], raw_dir: Path, manifest: dict) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = [Path(entry["key"]).name for entry in entries]
    flat_names = len(names) == len(set(names))
    saved = []
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        key = str(entry["key"])
        relative = Path(key).name if flat_names else Path(key)
        destination = raw_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = int(entry["bytes"])
        if destination.exists() and destination.stat().st_size == expected:
            print(f"[{index}/{total}] skip {key} ({expected} bytes)")
            saved.append(destination)
            continue
        temporary = destination.with_name(destination.name + ".part")
        url = file_url(manifest, entry)
        print(f"[{index}/{total}] {key} ({expected} bytes)")
        with requests.get(url, stream=True, timeout=120) as response:
            if response.status_code != 200:
                raise RuntimeError(f"download failed: {response.status_code} {url}")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
        saved.append(destination)
    return saved