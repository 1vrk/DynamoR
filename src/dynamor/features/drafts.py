import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from dynamor.features.catalog import index_of


def draft_matrix(frame: pl.DataFrame, hero_catalog: list[int]) -> csr_matrix:
    positions = index_of(hero_catalog)
    hero_count = len(hero_catalog)
    row_index: list[int] = []
    columns: list[int] = []
    for index, record in enumerate(
        frame.select("hero_id_target", "ally_heroes", "enemy_heroes").iter_rows(named=True)
    ):
        allies = _unique_ids([record["hero_id_target"], *(record["ally_heroes"] or [])])
        enemies = _unique_ids(record["enemy_heroes"] or [])
        row_index.extend([index] * (len(allies) + len(enemies)))
        columns.extend(positions[hero_id] for hero_id in allies)
        columns.extend(hero_count + positions[hero_id] for hero_id in enemies)
    values = np.ones(len(row_index), dtype=np.uint8)
    matrix = csr_matrix(
        (values, (row_index, columns)),
        shape=(frame.height, 2 * hero_count),
        dtype=np.uint8,
    )
    matrix.eliminate_zeros()
    return matrix


def _unique_ids(hero_ids) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for hero_id in hero_ids:
        if hero_id is None or hero_id in seen:
            continue
        seen.add(int(hero_id))
        unique.append(int(hero_id))
    return unique
