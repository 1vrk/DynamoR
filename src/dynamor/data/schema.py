#тут крч бредос, шлефуем чутка
SOURCE_COLUMNS = [
    "match_id",
    "start_time",
    "match_mode",
    "game_mode",
    "match_outcome",
    "average_badge",
    "team",
    "hero_id",
    "won",
    "items.item_id",
    "items.sold_time_s",
    "items.game_time_s",
    "items.upgrade_id",
    "items.flags",
]

OUTPUT_COLUMNS = [
    "match_id",
    "hero_id_target",
    "ally_heroes",
    "enemy_heroes",
    "winning_items",
    "winning_item_times",
    "upgrade_child",
    "upgrade_parent",
    "purchase_items",
    "purchase_times",
    "purchase_sold",
    "average_badge",
    "start_time",
]

RANKED = "Ranked"
NORMAL = "Normal"
TEAM_WIN = "TeamWin"
TEAM0 = "Team0"
TEAM1 = "Team1"
SPECTATOR = "Spectator"

MATCH_MODE = {4: RANKED, "4": RANKED, RANKED: RANKED}
GAME_MODE = {1: NORMAL, "1": NORMAL, NORMAL: NORMAL}
MATCH_OUTCOME = {0: TEAM_WIN, "0": TEAM_WIN, TEAM_WIN: TEAM_WIN}
TEAM = {
    0: TEAM0,
    "0": TEAM0,
    TEAM0: TEAM0,
    1: TEAM1,
    "1": TEAM1,
    TEAM1: TEAM1,
    16: SPECTATOR,
    "16": SPECTATOR,
    SPECTATOR: SPECTATOR,
}

def normalize_enum(value, mapping: dict) -> str:
    if value in mapping:
        return mapping[value]
    try:
        code = int(value)
    except(TypeError, ValueError):
        code = None
    if code in mapping:
        return mapping[code]
    raise ValueError(f"Unknown enum value: {value!r}")