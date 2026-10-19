import polars as pl
import pytest

from dynamor.features.targets import target_matrix


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
