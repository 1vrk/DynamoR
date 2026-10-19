from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from dynamor.features.catalog import read_ids
from dynamor.models.cosine import recommend


@dataclass(frozen=True)
class ShownItem:
    item_id: int
    name: str
    category: str
    confidence_pct: int
    tf: int
    score: float


@dataclass(frozen=True)
class Context:
    index: sparse.csr_matrix
    targets: sparse.csr_matrix
    hero_ids: np.ndarray
    hero_catalog: list[int]
    item_catalog: list[int]
    heroes: list[dict]
    names: dict[int, tuple[str, str]]


def suggest(
    hero_id: int,
    enemy_ids,
    index,
    targets,
    hero_ids: np.ndarray,
    hero_catalog: list[int],
    item_catalog: list[int],
    names: dict[int, tuple[str, str]],
    top_k: int,
    top_items: int,
) -> list[ShownItem]:
    recommendations = recommend(
        hero_id,
        enemy_ids,
        index,
        targets,
        hero_ids,
        hero_catalog,
        item_catalog,
        top_k,
        top_items,
    )
    shown: list[ShownItem] = []
    for item in recommendations:
        name, category = names.get(item.item_id, (str(item.item_id), "other"))
        shown.append(
            ShownItem(
                item_id=item.item_id,
                name=name,
                category=category,
                confidence_pct=int(round(item.confidence * 100)),
                tf=item.tf,
                score=item.score,
            )
        )
    return shown


def group_items(items: list[ShownItem]) -> dict[str, list[ShownItem]]:
    grouped = {"weapon": [], "vitality": [], "spirit": [], "other": []}
    for item in items:
        grouped.get(item.category, grouped["other"]).append(item)
    return grouped


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
        names={
            int(row["id"]): (row["name"], row["category"])
            for row in item_frame.iter_rows(named=True)
        },
    )