import numpy as np
import polars as pl
import pytest

from dynamor.features.targets import item_time_matrix, target_matrix


def test_target_matrix_marks_each_item_once():
    frame = pl.DataFrame({"winning_items": [[3190916303, 10, 10]]})

    matrix = target_matrix(frame, [10, 3190916303])

    assert matrix.shape == (1, 2)
    assert matrix.toarray().tolist() == [[1, 1]]
    assert matrix.nnz == 2


def test_target_matrix_rejects_unknown_item():
    frame = pl.DataFrame({"winning_items": [[99]]})

    with pytest.raises(KeyError):
        target_matrix(frame, [10, 3190916303])


def test_item_time_matrix_keeps_a_purchase_at_zero_seconds():
    frame = pl.DataFrame(
        {
            "winning_items": [[10, 20]],
            "winning_item_times": [[0, 75]],
        }
    )

    matrix = item_time_matrix(frame, [10, 20])

    assert matrix.dtype == np.uint32
    assert 0 in matrix.getrow(0).indices
    assert matrix.getrow(0).data.tolist() == [0, 75]
