from dataclasses import dataclass
from pathlib import Path

import yaml
 #как я помню тоже самое что и parent.parent -> на второй лвл апается
ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    lookback_hours: int
    min_average_badge: int
    players_per_team: int
    match_mode: str
    game_mode: str
    manifest_url: str
    raw_dir: Path
    interim_path: Path

def load_settings() -> Settings:
    mvp = yaml.safe_load((ROOT / "configs" / "mvp.yaml").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    return Settings(
        lookback_hours=mvp["lookback_hours"],
        min_average_badge=mvp["min_average_badge"],
        players_per_team=mvp["players_per_team"],
        match_mode=mvp["match_mode"],
        game_mode=mvp["game_mode"],
        manifest_url=paths["manifest_url"],
        raw_dir=Path(ROOT / paths["raw_dir"]).resolve(),
        interim_path=Path(ROOT / paths["interim_path"]).resolve()
    )

#разок прочитает конфиги и обхект вернет
settings = load_settings()