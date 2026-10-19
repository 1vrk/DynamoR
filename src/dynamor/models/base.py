from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    score: float
    tf: int
    confidence: float
