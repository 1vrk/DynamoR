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


def item_time_matrix(frame: pl.DataFrame, item_catalog: list[int]) -> csr_matrix:
    positions = index_of(item_catalog)
    row_index: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    item_lists = frame.get_column("winning_items").to_list()
    time_lists = frame.get_column("winning_item_times").to_list()
    for index, (items, times) in enumerate(zip(item_lists, time_lists)):
        for item_id, bought_at in zip(items or [], times or []):
            if item_id is None:
                continue
            row_index.append(index)
            columns.append(positions[int(item_id)])
            values.append(int(bought_at))
    return csr_matrix(
        (np.array(values, dtype=np.uint32), (row_index, columns)),
        shape=(frame.height, len(item_catalog)),
        dtype=np.uint32,
    )


def _unique_ids(item_ids) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for item_id in item_ids:
        if item_id is None or item_id in seen:
            continue
        seen.add(int(item_id))
        unique.append(int(item_id))
    return unique
