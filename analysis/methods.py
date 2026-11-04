"""Registry of the compared scheduling methods.

Every figure takes the label, color and marker of a method from here, so a
method looks the same everywhere. Colors are a subset of the Okabe-Ito
palette, validated for colour-vision deficiencies on a white page (worst
all-pairs deutan Delta E 7.6, hence the mandatory secondary encoding: one
marker per method).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    key: str
    label: str  # legend label
    short: str  # compact label (tables, tick labels)
    color: str
    marker: str
    marker_scale: float = 1.0  # some glyphs (star) look smaller at equal size


QLEARNING = Method("qlearning", "Tabular Q-learning", "Q-learning", "#0072B2", "o")
BCQ = Method("bcq", "Discrete BCQ (offline)", "BCQ", "#009E73", "s")
TD3 = Method("td3", "Continuous TD3 (offline)", "TD3", "#D55E00", "^")
GNN = Method("gnn", "EdgeGAT GNN (offline)", "GNN", "#CC79A7", "D")
HEURISTIC = Method("heuristic", "AIMD heuristic", "Heuristic", "#E69F00", "*", marker_scale=1.4)
FIXED = Method("fixed", "Fixed period", "Fixed", "#555555", "X")
# Reactive baselines of the learning campaign (they do not learn and ignore beta); they replace its heuristic.
TRICKLE = Method("trickle", "Trickle (RFC 6206)", "Trickle", "#56B4E9", "v", marker_scale=1.2)
GRADUAL = Method("gradual", "Gradual controller (AIMD)", "Gradual", "#E69F00", "P", marker_scale=1.2)

LEARNED: tuple[Method, ...] = (QLEARNING, BCQ, TD3, GNN)
ALL: tuple[Method, ...] = (*LEARNED, HEURISTIC, FIXED)
BY_KEY: dict[str, Method] = {m.key: m for m in ALL}

# Neutral greys for the individual fixed periods (1 s lightest ... 5 s darkest).
FIXED_PERIOD_GREYS = {1.0: "#b0b0b0", 2.0: "#919191", 3.0: "#737373", 4.0: "#555555", 5.0: "#333333"}

# File-name prefix of the main experiment exports -> method key.
EXPERIMENT_TO_METHOD = {
    "experimentTable": QLEARNING.key,
    "experimentDeepOffline": BCQ.key,
    "experimentDeepOfflineContinuous": TD3.key,
    "experimentDeepOfflineGNN": GNN.key,
    "experimentBaseline": FIXED.key,
}

# algoId of the generalization experiment -> (method key, fixed period or None).
GENERALIZATION_ALGORITHMS: dict[int, tuple[str, float | None]] = {
    0: (TD3.key, None),
    1: (BCQ.key, None),
    2: (QLEARNING.key, None),
    3: (HEURISTIC.key, None),
    4: (FIXED.key, 1.0),
    5: (FIXED.key, 2.0),
    6: (FIXED.key, 3.0),
    7: (FIXED.key, 4.0),
    8: (FIXED.key, 5.0),
    9: (GNN.key, None),
}

# Methods whose policy takes beta as an input at inference time.
BETA_CONDITIONED = {BCQ.key, TD3.key, GNN.key, HEURISTIC.key}


def get(key: str) -> Method:
    return BY_KEY[key]
