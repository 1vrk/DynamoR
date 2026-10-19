from datetime import datetime, timezone

from dynamor.data.client import download_files, fetch_manifest, select_files
from dynamor.settings import settings

def main() -> None:
    now = datetime.now(timezone.utc)
    manifest = fetch_manifest(settings.manifest_url)
    entries = select_files(
        manifest,
        now=now,
        lookback_hours = settings.lookback_hours
    )
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    #тут душним ненмного
    print(f"{len(entries)} files, {total_bytes / 1_048_576:.1f}MB")
    download_files(entries, settings.raw_dir, manifest)

if __name__ == "__main__":
    main()