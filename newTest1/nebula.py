"""Nebula micro-simulator generating whimsical star lanes."""
import random
from typing import List, Tuple

STAR_COLORS = ["cerulean", "amber", "violet", "jade", "scarlet"]


def emit_lane(seed: int) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    hops = rng.randint(4, 9)
    return [
        (rng.choice(STAR_COLORS), rng.randint(1, 42))
        for _ in range(hops)
    ]


def describe_lane(lane: List[Tuple[str, int]]) -> str:
    return ", ".join(f"{length} light-mins of {color}" for color, length in lane)


if __name__ == "__main__":
    route = emit_lane(2025)
    print("Projected star lane:")
    print(describe_lane(route))
