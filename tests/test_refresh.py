from datetime import datetime, timedelta, timezone

from dynamor.services.refresh import prune_raw, refresh_is_due


def test_prune_raw_keeps_selected_files_and_drops_the_rest(tmp_path):
    keep = tmp_path / "keep.parquet"
    old = tmp_path / "old.parquet"
    part = tmp_path / "keep.parquet.part"
    keep.write_bytes(b"a")
    old.write_bytes(b"b")
    part.write_bytes(b"c")

    removed = prune_raw(tmp_path, [keep])

    assert keep.exists()
    assert not old.exists()
    assert not part.exists()
    assert {path.name for path in removed} == {"old.parquet", "keep.parquet.part"}


def test_refresh_is_due_after_seven_days():
    now = datetime(2026, 10, 22, tzinfo=timezone.utc)

    assert refresh_is_due(now - timedelta(days=8), now=now) is True
    assert refresh_is_due(now - timedelta(days=7), now=now) is False
    assert refresh_is_due(now - timedelta(days=1), now=now) is False
    assert refresh_is_due(None, now=now) is False
