from dataclasses import dataclass

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
) -> DraftBuild:
    query = enemy_vector(enemy_ids, hero_catalog)
    if query.nnz == 0:
        return DraftBuild([], [])
    mask = np.asarray(hero_ids) == int(hero_id)
    neighbor_rows, _scores = nearest_rows(index, mask, query, top_k)
    if len(neighbor_rows) == 0:
        return DraftBuild([], [])
    positions = index_of(item_catalog)
    ranked = rank_items(targets, neighbor_rows, np.flatnonzero(mask), item_catalog, len(item_catalog))
    shop: list[ShownItem] = []
    for item in ranked:
        shown = _shown_item(item, names, item_times, neighbor_rows, positions)
        if shown.category == "skill":
            continue
        shop.append(shown)
        if len(shop) == top_items:
            break
    shop.sort(key=lambda row: (row.purchase_s is None, row.purchase_s or 0, row.item_id))
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
    return Context(
        index=sparse.load_npz(settings.index_path),
        targets=sparse.load_npz(settings.targets_path),
        hero_ids=pl.read_parquet(settings.rows_path).get_column("hero_id_target").to_numpy(),
        hero_catalog=read_ids(settings.hero_catalog_path),
        item_catalog=read_ids(settings.item_catalog_path),
        heroes=[
            {"id": int(row["id"]), "name": row["name"]}
            for row in hero_frame.iter_rows(named=True)
        ],
        item_times=sparse.load_npz(settings.item_times_path),
        skills=_skills_by_hero(item_frame),
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