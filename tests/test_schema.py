from dynamor.data.schema import normalize_enum, MATCH_MODE, RANKED

def test_schema():
    assert normalize_enum(4, MATCH_MODE) == RANKED
    assert normalize_enum("Ranked", MATCH_MODE) == RANKED