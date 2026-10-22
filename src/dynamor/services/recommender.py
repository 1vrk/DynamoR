from dataclasses import dataclass, replace

import numpy as np
import polars as pl
from scipy import sparse

from dynamor.features.catalog import index_of, read_ids
from dynamor.models.base import Recommendation
from dynamor.models.cosine import enemy_vector, nearest_rows, rank_items


@dataclass(frozen=True)
class ShownItem:
    item_id: int
    name: str
    category: str
    confidence_pct: int
    tf: int
    score: float
    cost: int | None
    purchase_s: int | None
    image: str | None
    upgrade_from: int | None = None


@dataclass(frozen=True)
class DraftBuild:
    items: list[ShownItem]
    skills: list[ShownItem]


@dataclass(frozen=True)
class Context:
    index: sparse.csr_matrix
    targets: sparse.csr_matrix
    hero_ids: np.ndarray
    hero_catalog: list[int]
    item_catalog: list[int]
    heroes: list[dict]
    item_times: sparse.csr_matrix
    names: dict[int, tuple[str, str, int | None, str | None]]
    skills: dict[int, list[int]]
    upgrades: list[list[tuple[int, int]]]
    scales: dict[int, tuple[str, ...]]
    purchases: list[list[tuple[int, int, int]]]
    scale_weights: dict[int, dict[str, float]]
    item_stats: dict[int, dict[str, int]]
    hero_tags: dict[int, tuple[str, ...]]
    item_answers: dict[int, tuple[str, ...]]
    item_effects: dict[int, dict[str, float]] | None


def suggest(
    hero_id: int,
    enemy_ids,
    index,
    targets,
    hero_ids: np.ndarray,
    hero_catalog: list[int],
    item_catalog: list[int],
    names: dict[int, tuple[str, str, int | None, str | None]],
    item_times,
    top_k: int,
    top_items: int,
    skill_ids=(),
    upgrades=(),
    purchases=(),
    scale_weights=None,
    item_stats=None,
    hero_tags=None,
    item_answers=None,
    item_effects=None,
) -> DraftBuild:
    query = enemy_vector(enemy_ids, hero_catalog)
    if query.nnz == 0:
        return DraftBuild([], [])
    mask = np.asarray(hero_ids) == int(hero_id)
    neighbor_rows, _scores = nearest_rows(index, mask, query, top_k)
    if len(neighbor_rows) == 0:
        return DraftBuild([], [])
    positions = index_of(item_catalog)
    weights = {} if scale_weights is None else scale_weights
    stats = {} if item_stats is None else item_stats
    threats = _enemy_threats(enemy_ids, {} if hero_tags is None else hero_tags)
    answers = {} if item_answers is None else item_answers
    ranked = rank_items(targets, neighbor_rows, np.flatnonzero(mask), item_catalog, len(item_catalog))
    kept: list[ShownItem] = []
    for item in ranked:
        shown = _shown_item(item, names, item_times, neighbor_rows, positions)
        if shown.category == "skill":
            continue
        kept.append(shown)
    timeline = sorted(kept, key=lambda row: (row.purchase_s is None, row.purchase_s or 0, row.item_id))
    linked = _attach_upgrades(timeline, _modal_parents(upgrades, neighbor_rows))
    fractions = _curve_fractions(linked)
    use_effects = item_effects is not None
    effects = {} if item_effects is None else item_effects
    scored: list[tuple[float, ShownItem]] = []
    for shown in kept:
        owned = effects.get(shown.item_id, {}) if use_effects else None
        factor = _selection_multiplier(
            shown,
            weights,
            stats.get(shown.item_id, {}),
            _threat_bonus(answers.get(shown.item_id, ()), threats, owned),
            fractions.get(shown.item_id, 0.0),
            owned,
        )
        scored.append((shown.score * factor, shown))
    scored.sort(key=lambda row: (-row[0], row[1].item_id))
    shop = [shown for _factor, shown in scored[:top_items]]
    shop.extend(_lane_items(purchases, neighbor_rows, {item.item_id for item in shop}, names))
    shop.sort(key=lambda row: (row.purchase_s is None, row.purchase_s or 0, row.item_id))
    shop = _attach_upgrades(shop, _modal_parents(upgrades, neighbor_rows))
    skills = [
        _shown_item(item, names, item_times, neighbor_rows, positions)
        for item in _skill_recommendations(skill_ids, targets, neighbor_rows, item_catalog)
    ]
    skills.sort(key=lambda row: (row.purchase_s is None, row.purchase_s or 0, row.item_id))
    return DraftBuild(shop, skills)


def _skill_recommendations(skill_ids, targets, neighbor_rows, item_catalog):
    positions = index_of(item_catalog)
    rows = []
    kept = np.asarray(neighbor_rows)
    for skill_id in skill_ids:
        column = positions.get(int(skill_id))
        if column is None:
            continue
        present = np.asarray(targets[kept].getcol(column).data).ravel()
        support = int((present > 0).sum()) if present.size else 0
        rows.append(
            Recommendation(
                item_id=int(skill_id),
                score=0.0,
                tf=support,
                confidence=0.0 if len(kept) == 0 else support / len(kept),
            )
        )
    return rows


def _modal_parents(upgrades, neighbor_rows) -> dict[int, int]:
    votes: dict[int, dict[int, int]] = {}
    for row in neighbor_rows:
        if row >= len(upgrades):
            continue
        for child, parent in upgrades[row]:
            counts = votes.setdefault(int(child), {})
            counts[int(parent)] = counts.get(int(parent), 0) + 1
    return {
        child: min(counts, key=lambda parent: (-counts[parent], parent))
        for child, counts in votes.items()
    }


def _attach_upgrades(items: list[ShownItem], parents: dict[int, int]) -> list[ShownItem]:
    order = {item.item_id: index for index, item in enumerate(items)}
    by_id = {item.item_id: item for item in items}
    linked = []
    for item in items:
        parent = by_id.get(parents.get(item.item_id))
        upgrade_from = None
        if (
            parent is not None
            and order[parent.item_id] < order[item.item_id]
            and parent.category == item.category
            and parent.cost is not None
            and item.cost is not None
            and parent.cost < item.cost
        ):
            upgrade_from = parent.item_id
        linked.append(replace(item, upgrade_from=upgrade_from))
    return linked


INVEST_CATEGORIES = ("weapon", "vitality", "spirit")
INVEST_TARGET = 4800
LANE_SHARE = 0.4
LANE_LIMIT = 8
INVEST_ALPHA = 0.25
STAT_ALPHA = 0.1
MAX_BONUS = 0.5
SPEED_WEIGHT = 0.25
TAG_ALPHA = 0.2
THREAT_MIN_ENEMIES = 2
LINE_UNIT = 0.30
CURVE_SOULS = (800, 1600, 2400, 3200, 4800, 6400, 8000, 11200, 16000, 22400, 28800)
CURVE_BONUS = {
    "weapon": (9, 12, 15, 18, 46, 54, 62, 74, 86, 100, 115),
    "vitality": (9, 12, 15, 20, 38, 42, 46, 50, 54, 60, 66),
    "spirit": (7, 11, 15, 19, 38, 45, 52, 59, 66, 75, 100),
}
CURVE_AT_4800 = {category: float(steps[4]) for category, steps in CURVE_BONUS.items()}
_ANSWER_OF = {
    "weapon": "defense",
    "spirit": "defense",
    "control": "defense",
    "heal": "antiheal",
}
_EFFECT_SLOT = {
    "weapon_damage": "weapon",
    "fire_rate": "weapon",
    "melee": "weapon",
    "spirit": "spirit",
    "health": "vitality",
    "heal": "vitality",
    "lifesteal": "vitality",
}
_OPEN_EFFECTS = frozenset({"move_speed", "stamina"})
_RESIST_THREAT = {
    "bullet_resist": "weapon",
    "spirit_resist": "spirit",
    "debuff_resist": "control",
}


def _enemy_threats(enemy_ids, hero_tags: dict[int, tuple[str, ...]]) -> dict[str, float]:
    picked = [int(enemy) for enemy in enemy_ids if enemy is not None]
    total = len(picked)
    if total == 0 or not hero_tags:
        return {}
    counts: dict[str, int] = {}
    for enemy in picked:
        for tag in hero_tags.get(enemy, ()):
            if tag in _ANSWER_OF:
                counts[tag] = counts.get(tag, 0) + 1
    return {tag: count / total for tag, count in counts.items() if count >= THREAT_MIN_ENEMIES}


def curve_bonus(category: str, souls: int) -> float:
    steps = CURVE_BONUS.get(category)
    if not steps:
        return 0.0
    reached = 0.0
    for threshold, value in zip(CURVE_SOULS, steps):
        if souls < threshold:
            break
        reached = float(value)
    return reached


def _curve_fractions(items: list[ShownItem]) -> dict[int, float]:
    totals = {category: 0 for category in INVEST_CATEGORIES}
    replaced: set[int] = set()
    by_id = {item.item_id: item for item in items}
    fractions: dict[int, float] = {}
    for item in items:
        full = CURVE_AT_4800.get(item.category)
        if full is None or item.cost is None:
            fractions[item.item_id] = 0.0
            continue
        added = item.cost
        parent = by_id.get(item.upgrade_from) if item.upgrade_from is not None else None
        if (
            parent is not None
            and parent.item_id not in replaced
            and parent.cost is not None
            and parent.cost < item.cost
        ):
            added = item.cost - parent.cost
            replaced.add(parent.item_id)
        before = totals[item.category]
        totals[item.category] = before + added
        delta = curve_bonus(item.category, totals[item.category]) - curve_bonus(item.category, before)
        fractions[item.item_id] = delta / full
    return fractions


def _effect_stat_score(effects: dict[str, float], weights: dict[str, float]) -> float:
    score = 0.0
    for key, value in effects.items():
        if key in _OPEN_EFFECTS:
            score += value / LINE_UNIT
        elif key in _EFFECT_SLOT:
            score += (value / LINE_UNIT) * weights.get(_EFFECT_SLOT[key], 0.0)
    return score


def _threat_bonus(answers, threats: dict[str, float], effects: dict[str, float] | None = None) -> float:
    if not threats:
        return 0.0
    if effects is None:
        if not answers:
            return 0.0
        owned = set(answers)
        return TAG_ALPHA * sum(share for threat, share in threats.items() if _ANSWER_OF[threat] in owned)
    bonus = 0.0
    if answers and "antiheal" in set(answers):
        bonus += TAG_ALPHA * threats.get("heal", 0.0)
    for key, threat in _RESIST_THREAT.items():
        value = float(effects.get(key, 0.0) or 0.0)
        share = threats.get(threat, 0.0)
        if value and share:
            bonus += TAG_ALPHA * share * (value / LINE_UNIT)
    return bonus


def _selection_multiplier(
    item: ShownItem,
    weights: dict[str, float],
    stats: dict[str, int],
    threat_bonus: float = 0.0,
    invest_fraction: float = 0.0,
    effects: dict[str, float] | None = None,
) -> float:
    slot_weight = weights.get(item.category, 0.0)
    invest = slot_weight * invest_fraction
    if effects is None:
        stat_score = sum(stats.get(bucket, 0) * weights.get(bucket, 0.0) for bucket in INVEST_CATEGORIES)
        stat_score += stats.get("speed", 0) * SPEED_WEIGHT
    else:
        stat_score = _effect_stat_score(effects, weights)
    bonus = min(MAX_BONUS, INVEST_ALPHA * invest + STAT_ALPHA * stat_score + threat_bonus)
    return 1.0 + bonus


def _lane_items(purchases, neighbor_rows, already: set[int], names) -> list[ShownItem]:
    buyers: dict[int, list[int]] = {}
    for row in neighbor_rows:
        if row >= len(purchases):
            continue
        seen: set[int] = set()
        for item_id, bought_at, _sold_at in purchases[row]:
            item_id = int(item_id)
            if item_id in seen:
                continue
            seen.add(item_id)
            buyers.setdefault(item_id, []).append(int(bought_at))
    neighbor_count = len(neighbor_rows)
    if neighbor_count == 0:
        return []
    chosen = []
    for item_id, times in buyers.items():
        if item_id in already:
            continue
        share = len(times) / neighbor_count
        if share < LANE_SHARE:
            continue
        name, category, cost, image = names.get(item_id, (str(item_id), "other", None, None))
        if category not in INVEST_CATEGORIES:
            continue
        chosen.append((share, item_id, times, name, category, cost, image))
    chosen.sort(key=lambda row: (-row[0], row[1]))
    items = []
    for share, item_id, times, name, category, cost, image in chosen[:LANE_LIMIT]:
        items.append(
            ShownItem(
                item_id=item_id,
                name=name,
                category=category,
                confidence_pct=int(round(share * 100)),
                tf=len(times),
                score=0.0,
                cost=None if cost is None else int(cost),
                purchase_s=int(np.median(times)),
                image=None if image is None else str(image),
            )
        )
    return items


@dataclass(frozen=True)
class Investment:
    totals: dict[str, int]
    crossed: dict[int, str]


def investment_progress(items: list[ShownItem], close_categories) -> Investment:
    totals = {category: 0 for category in INVEST_CATEGORIES}
    crossed: dict[int, str] = {}
    close = set(close_categories)
    replaced: set[int] = set()
    by_id = {item.item_id: item for item in items}
    for item in items:
        if item.category not in totals or item.cost is None:
            continue
        added = item.cost
        parent = by_id.get(item.upgrade_from) if item.upgrade_from is not None else None
        if (
            parent is not None
            and parent.item_id not in replaced
            and parent.cost is not None
            and parent.cost < item.cost
        ):
            added = item.cost - parent.cost
            replaced.add(parent.item_id)
        before = totals[item.category]
        totals[item.category] = before + added
        if item.category in close and before < INVEST_TARGET <= totals[item.category]:
            crossed[item.item_id] = item.category
    return Investment(totals, crossed)


def _shown_item(item, names, item_times, neighbor_rows, positions) -> ShownItem:
    name, category, cost, image = names.get(item.item_id, (str(item.item_id), "other", None, None))
    return ShownItem(
        item_id=item.item_id,
        name=name,
        category=category,
        confidence_pct=int(round(item.confidence * 100)),
        tf=item.tf,
        score=item.score,
        cost=None if cost is None else int(cost),
        purchase_s=_median_purchase_s(item_times, neighbor_rows, positions.get(item.item_id)),
        image=None if image is None else str(image),
    )


def format_item(item: ShownItem) -> str:
    parts = [item.name, f"{item.confidence_pct}%"]
    if item.cost is not None:
        parts.append(str(item.cost))
    if item.purchase_s is not None:
        parts.append(f"{item.purchase_s // 60}:{item.purchase_s % 60:02d}")
    return " — ".join(parts)


def _median_purchase_s(item_times, neighbor_rows, column: int | None) -> int | None:
    if column is None or len(neighbor_rows) == 0:
        return None
    stored = np.asarray(item_times[np.asarray(neighbor_rows)].getcol(column).data).ravel()
    if stored.size == 0:
        return None
    return int(np.median(stored))


EARLY_S = 10 * 60
MID_S = 20 * 60


def group_phases(items: list[ShownItem]) -> dict[str, list[ShownItem]]:
    grouped = {"early": [], "mid": [], "late": []}
    for item in items:
        if item.purchase_s is None or item.purchase_s >= MID_S:
            grouped["late"].append(item)
        elif item.purchase_s >= EARLY_S:
            grouped["mid"].append(item)
        else:
            grouped["early"].append(item)
    return grouped


def group_items(items: list[ShownItem]) -> dict[str, list[ShownItem]]:
    grouped = {"weapon": [], "vitality": [], "spirit": [], "other": []}
    for item in items:
        grouped.get(item.category, grouped["other"]).append(item)
    return grouped


def _upgrade_pairs(frame: pl.DataFrame) -> list[list[tuple[int, int]]]:
    if "upgrade_child" not in frame.columns or "upgrade_parent" not in frame.columns:
        return [[] for _ in range(frame.height)]
    pairs = []
    children = frame.get_column("upgrade_child").to_list()
    parents = frame.get_column("upgrade_parent").to_list()
    for child_ids, parent_ids in zip(children, parents):
        pairs.append([
            (int(child), int(parent))
            for child, parent in zip(child_ids or [], parent_ids or [])
        ])
    return pairs


def _purchases(frame: pl.DataFrame) -> list[list[tuple[int, int, int]]]:
    needed = {"purchase_items", "purchase_times", "purchase_sold"}
    if not needed <= set(frame.columns):
        return [[] for _ in range(frame.height)]
    logs = []
    items = frame.get_column("purchase_items").to_list()
    times = frame.get_column("purchase_times").to_list()
    sold = frame.get_column("purchase_sold").to_list()
    for item_ids, bought_at, sold_at in zip(items, times, sold):
        logs.append([
            (int(item_id), int(bought), int(sold_time))
            for item_id, bought, sold_time in zip(item_ids or [], bought_at or [], sold_at or [])
        ])
    return logs


def _parse_numbered(text, cast):
    parsed = {}
    for part in str("" if text is None else text).split(","):
        if ":" not in part:
            continue
        name, raw = part.split(":", 1)
        name = name.strip()
        if name == "":
            continue
        try:
            parsed[name] = cast(raw.strip())
        except ValueError:
            continue
    return parsed


def _scale_weights_by_hero(frame: pl.DataFrame) -> dict[int, dict[str, float]]:
    if "scale_weights" not in frame.columns:
        return {}
    return {
        int(row["id"]): _parse_numbered(row["scale_weights"], float)
        for row in frame.iter_rows(named=True)
    }


def _tags_by_id(frame: pl.DataFrame, column: str) -> dict[int, tuple[str, ...]]:
    if column not in frame.columns:
        return {}
    parsed = {}
    for row in frame.iter_rows(named=True):
        tags = tuple(part.strip() for part in str(row[column] or "").split(",") if part.strip())
        if tags:
            parsed[int(row["id"])] = tags
    return parsed


def _effects_by_item(frame: pl.DataFrame) -> dict[int, dict[str, float]] | None:
    if "effects" not in frame.columns:
        return None
    effects = {}
    for row in frame.iter_rows(named=True):
        parsed = _parse_numbered(row["effects"], float)
        if parsed:
            effects[int(row["id"])] = parsed
    return effects


def _stats_by_item(frame: pl.DataFrame) -> dict[int, dict[str, int]]:
    if "stats" not in frame.columns:
        return {}
    stats = {}
    for row in frame.iter_rows(named=True):
        parsed = _parse_numbered(row["stats"], int)
        if parsed:
            stats[int(row["id"])] = parsed
    return stats


def _scales_by_hero(frame: pl.DataFrame) -> dict[int, tuple[str, ...]]:
    if "scales" not in frame.columns:
        return {}
    scales = {}
    for row in frame.iter_rows(named=True):
        text = "" if row["scales"] is None else str(row["scales"])
        scales[int(row["id"])] = tuple(part for part in text.split(",") if part)
    return scales


def _skills_by_hero(frame: pl.DataFrame) -> dict[int, list[int]]:
    if "owner" not in frame.columns:
        return {}
    skills: dict[int, list[int]] = {}
    for row in frame.iter_rows(named=True):
        if row["category"] != "skill" or row["owner"] is None:
            continue
        skills.setdefault(int(row["owner"]), []).append(int(row["id"]))
    return skills


def validate_draft(hero_id: int, enemy_ids, known_ids) -> str | None:
    enemies = list(enemy_ids)
    if len(enemies) != 6 or any(enemy is None for enemy in enemies):
        return "Pick six enemies"
    if len(set(enemies)) != 6:
        return "Enemy heroes must be different"
    if hero_id in enemies:
        return "Your hero cannot also be an enemy"
    known = {int(item_id) for item_id in known_ids}
    if int(hero_id) not in known or any(int(enemy) not in known for enemy in enemies):
        return "Unknown hero"
    return None


def load_context(settings) -> Context:
    if not settings.index_path.exists():
        raise FileNotFoundError(settings.index_path)
    if not settings.item_times_path.exists():
        raise FileNotFoundError(settings.item_times_path)
    item_frame = pl.read_parquet(settings.item_names_path)
    hero_frame = pl.read_parquet(settings.hero_names_path)
    rows = pl.read_parquet(settings.rows_path)
    return Context(
        index=sparse.load_npz(settings.index_path),
        targets=sparse.load_npz(settings.targets_path),
        hero_ids=rows.get_column("hero_id_target").to_numpy(),
        hero_catalog=read_ids(settings.hero_catalog_path),
        item_catalog=read_ids(settings.item_catalog_path),
        heroes=[
            {"id": int(row["id"]), "name": row["name"]}
            for row in hero_frame.iter_rows(named=True)
        ],
        item_times=sparse.load_npz(settings.item_times_path),
        skills=_skills_by_hero(item_frame),
        upgrades=_upgrade_pairs(rows),
        scales=_scales_by_hero(hero_frame),
        purchases=_purchases(rows),
        scale_weights=_scale_weights_by_hero(hero_frame),
        item_stats=_stats_by_item(item_frame),
        hero_tags=_tags_by_id(hero_frame, "tags"),
        item_answers=_tags_by_id(item_frame, "answers"),
        item_effects=_effects_by_item(item_frame),
        names={
            int(row["id"]): (
                row["name"],
                row["category"],
                None if row["cost"] is None else int(row["cost"]),
                None if row["image"] is None else str(row["image"]),
            )
            for row in item_frame.iter_rows(named=True)
        },
    )