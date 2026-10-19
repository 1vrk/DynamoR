import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from dynamor.features.catalog import index_of


def target_matrix(frame: pl.DataFrame, item_catalog: list[int]) -> csr_matrix:
    positions = index_of(item_catalog)
    row_index: list[int] = []
    columns: list[int] = []
    for index, items in enumerate(frame.get_column("winning_items").to_list()):
        for item_id in _unique_ids(items or []):
            row_index.append(index)
            columns.append(positions[item_id])
    values = np.ones(len(row_index), dtype=np.uint8)
    matrix = csr_matrix(
        (values, (row_index, columns)),
        shape=(frame.height, len(item_catalog)),
        dtype=np.uint8,
    )
    matrix.eliminate_zeros()
    return matrix


def _unique_ids(item_ids) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for item_id in item_ids:
        if item_id is None or item_id in seen:
            continue
        seen.add(int(item_id))
        unique.append(int(item_id))
    return unique
