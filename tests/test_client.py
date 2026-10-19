from datetime import datetime, timedelta, timezone

from dynamor.data.client import file_url, select_files

NOW = datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)
START = NOW - timedelta(hours=6)

def entry(key, kind, lo, hi):
    return {"key": key, "kind": kind, "lo": lo, "hi": hi, "bytes": 10, "rows": 1}

def test_select_files_keeps_overlap_and_drops_base():
    inside = entry("delta/inside.parquet", "delta", START.timestamp() + 60, NOW.timestamp() - 60)
    outside = entry("delta/outside.parquet", "delta", START.timestamp() - 10_000, START.timestamp() - 60)
    residual = entry("residual/partial.parquet", "residual", START.timestamp() - 60, START.timestamp() + 60)
    base = entry("base/old.parquet", "base", START.timestamp() + 60, NOW.timestamp() - 60)
    manifest = {
        "public_url": "https://data.deadlock-api.com",
        "tables": {"match_player": {"files": [outside, base, residual, inside]}},
    }
    selected = select_files(manifest, now=NOW, lookback_hours=6)
    assert [item["key"] for item in selected] == [
        "residual/partial.parquet",
        "delta/inside.parquet",
    ]
    assert file_url(manifest, selected[0]) == "https://data.deadlock-api.com/residual/partial.parquet"