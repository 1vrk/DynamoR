import polars as pl
import pytest

from dynamor.features.drafts import draft_matrix


def test_draft_matrix_marks_allies_and_enemies_once():
    frame = pl.DataFrame(
        {
            "hero_id_target": [1],
            "ally_heroes": [[7, 7]],
            "enemy_heroes": [[60]],
        }
    )

    matrix = draft_matrix(frame, [1, 7, 60])

    assert matrix.shape == (1, 6)
    assert matrix.toarray().tolist() == [[1, 1, 0, 0, 0, 1]]
    assert matrix.nnz == 3


def test_draft_matrix_rejects_unknown_hero():
    frame = pl.DataFrame(
        {
            "hero_id_target": [99],
            "ally_heroes": [[1]],
            "enemy_heroes": [[7]],
        }
    )

    with pytest.raises(KeyError):
        draft_matrix(frame, [1, 7, 60])
