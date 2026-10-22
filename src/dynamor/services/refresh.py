import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
from scipy import sparse

from dynamor.catalog.assets import hero_close_categories, hero_rows, item_rows, signature_class_names
from dynamor.data.client import download_files, fetch_manifest, select_files
from dynamor.data.extract import build_rows, scan_raw, write_interim
from dynamor.data.filter import filter_matches
from dynamor.features.catalog import hero_ids, item_ids, read_ids, write_ids
from dynamor.features.drafts import draft_matrix
from dynamor.features.targets import item_time_matrix, target_matrix

STAGE_MANIFEST = "Manifest"
STAGE_DOWNLOAD = "Download"
STAGE_PRUNE = "Remove old dumps"
STAGE_MATCHES = "Match table"
STAGE_FEATURES = "Features"
STAGE_INDEX = "Index and names"
STAGES = (
    STAGE_MANIFEST,
    STAGE_DOWNLOAD,
    STAGE_PRUNE,
    STAGE_MATCHES,
    STAGE_FEATURES,
    STAGE_INDEX,
)
DUE_AFTER = timedelta(days=7)

HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes?only_active=true"
ITEMS_URL = "https://api.deadlock-api.com/v1/assets/items"


@dataclass(frozen=True)
class StageProgress:
    stage: str
    state: str
    fraction: float = 0.0
    error: str = ""


class RefreshError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def prune_raw(raw_dir: Path, keep: list[Path]) -> list[Path]:
    kept = {path.name for path in keep}
    removed: list[Path] = []
    for path in list(raw_dir.glob("*.parquet")) + list(raw_dir.glob("*.part")):
        if path.suffix == ".parquet" and path.name in kept:
            continue
        path.unlink()
        removed.append(path)
    return removed


def refresh_stamp_path(settings) -> Path:
    return settings.index_path.parent.parent / "refresh.json"


def write_refresh_stamp(settings, finished_at: datetime) -> None:
    path = refresh_stamp_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"finished_at": finished_at.isoformat()}),
        encoding="utf-8",
    )


def read_refresh_stamp(settings) -> datetime | None:
    path = refresh_stamp_path(settings)
    if not path.exists():
        return None
    finished = datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["finished_at"])
    if finished.tzinfo is None:
        return finished.replace(tzinfo=timezone.utc)
    return finished


def refresh_is_due(finished_at: datetime | None, *, now: datetime) -> bool:
    if finished_at is None:
        return False
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now - finished_at > DUE_AFTER


def build_match_table(settings):
    paths = sorted(settings.raw_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files in {settings.raw_dir}")
    frame = build_rows(filter_matches(scan_raw(paths), settings), settings)
    write_interim(frame, settings.interim_path)
    return frame


def build_feature_matrices(settings) -> dict:
    if not settings.interim_path.exists():
        raise SystemExit(f"interim table is missing: {settings.interim_path}")
    frame = pl.read_parquet(settings.interim_path)
    heroes = hero_ids(frame)
    items = item_ids(frame)
    write_ids(heroes, settings.hero_catalog_path)
    write_ids(items, settings.item_catalog_path)
    drafts = draft_matrix(frame, heroes)
    targets = target_matrix(frame, items)
    item_times = item_time_matrix(frame, items)
    settings.rows_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["match_id", "hero_id_target", "average_badge", "start_time"]
    if "upgrade_child" in frame.columns and "upgrade_parent" in frame.columns:
        columns.extend(["upgrade_child", "upgrade_parent"])
    if {"purchase_items", "purchase_times", "purchase_sold"} <= set(frame.columns):
        columns.extend(["purchase_items", "purchase_times", "purchase_sold"])
    frame.select(columns).write_parquet(settings.rows_path)
    sparse.save_npz(settings.drafts_path, drafts)
    sparse.save_npz(settings.targets_path, targets)
    sparse.save_npz(settings.item_times_path, item_times)
    return {
        "n_rows": frame.height,
        "heroes": len(heroes),
        "items": len(items),
        "draft_ones": int(drafts.nnz),
        "target_ones": int(targets.nnz),
    }


def build_cosine_index(settings):
    if not settings.drafts_path.exists():
        raise SystemExit(f"draft matrix is missing: {settings.drafts_path}")
    drafts = sparse.load_npz(settings.drafts_path)
    enemies = drafts[:, drafts.shape[1] // 2 :]
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(settings.index_path, enemies)
    return enemies


def build_name_catalog(settings) -> dict:
    if not settings.hero_catalog_path.exists():
        raise SystemExit(f"hero catalog is missing: {settings.hero_catalog_path}")
    if not settings.item_catalog_path.exists():
        raise SystemExit(f"item catalog is missing: {settings.item_catalog_path}")
    hero_catalog = read_ids(settings.hero_catalog_path)
    item_catalog = read_ids(settings.item_catalog_path)
    heroes_payload = _fetch_json(HEROES_URL)
    items_payload = _fetch_json(ITEMS_URL)
    heroes = hero_rows(heroes_payload, hero_catalog)
    close = hero_close_categories(heroes_payload, items_payload)
    for row in heroes:
        row["scales"] = ",".join(close.get(row["id"], ()))
    items = item_rows(items_payload, item_catalog, signature_class_names(heroes_payload))
    _write_table(heroes, settings.hero_names_path)
    _write_table(items, settings.item_names_path)
    slots = sorted(
        {
            str(item.get("item_slot_type"))
            for item in items_payload
            if item.get("type") == "upgrade"
        }
    )
    counts: dict[str, int] = {}
    for row in items:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return {"heroes": len(heroes), "items": len(items), "counts": counts, "slots": slots}


def run_refresh(settings, on_progress, *, now: datetime | None = None) -> None:
    moment = now or datetime.now(timezone.utc)
    manifest, entries = _run_stage(
        STAGE_MANIFEST,
        lambda: _select_dump(settings, moment),
        on_progress,
    )
    saved = _run_stage(
        STAGE_DOWNLOAD,
        lambda: _download_dump(settings, manifest, entries, on_progress),
        on_progress,
    )
    _run_stage(STAGE_PRUNE, lambda: prune_raw(settings.raw_dir, saved), on_progress)
    _run_stage(STAGE_MATCHES, lambda: build_match_table(settings), on_progress)
    _run_stage(STAGE_FEATURES, lambda: build_feature_matrices(settings), on_progress)
    _run_stage(
        STAGE_INDEX,
        lambda: (build_cosine_index(settings), build_name_catalog(settings)),
        on_progress,
    )
    write_refresh_stamp(settings, datetime.now(timezone.utc))


def _select_dump(settings, now: datetime):
    manifest = fetch_manifest(settings.manifest_url)
    entries = select_files(manifest, now=now, lookback_hours=int(settings.lookback_hours))
    return manifest, entries


def _download_dump(settings, manifest, entries, on_progress) -> list[Path]:
    def on_file(index: int, total: int) -> None:
        fraction = 0.0 if total == 0 else index / total
        on_progress(StageProgress(STAGE_DOWNLOAD, "running", fraction))

    return download_files(entries, settings.raw_dir, manifest, on_file=on_file)


def _run_stage(name: str, action, on_progress):
    on_progress(StageProgress(name, "running", 0.0))
    try:
        result = action()
    except RefreshError:
        raise
    except SystemExit as exc:
        message = str(exc) if exc.code is None else str(exc.code)
        on_progress(StageProgress(name, "failed", 0.0, message))
        raise RefreshError(name, message) from exc
    except Exception as exc:
        message = str(exc)
        on_progress(StageProgress(name, "failed", 0.0, message))
        raise RefreshError(name, message) from exc
    on_progress(StageProgress(name, "done", 1.0))
    return result


def _fetch_json(url: str) -> list:
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise SystemExit(f"assets request failed: {response.status_code} {url}")
    return response.json()


def _write_table(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows)
    casts = [pl.col("id").cast(pl.UInt32)]
    if "cost" in frame.columns:
        casts.append(pl.col("cost").cast(pl.UInt32, strict=False))
    if "owner" in frame.columns:
        casts.append(pl.col("owner").cast(pl.UInt32, strict=False))
    frame.with_columns(casts).write_parquet(path)
