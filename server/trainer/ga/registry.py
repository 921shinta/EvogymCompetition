from __future__ import annotations
from typing import Dict
from server.trainer.ga.operators.mutations import DefaultMutation
from server.trainer.ga.operators.crossovers import (
    NoCrossover,
    TorsoLegCrossover,
    RandomHorizontalSplitCrossover,
)
from server.trainer.ga.operators.selections import TruncationSelection

_mutations: Dict[str, object] = { "default": DefaultMutation() }

# ここに新しい交叉を登録
_crossovers: Dict[str, object] = { 
    "none": NoCrossover(),
    "torso_leg": TorsoLegCrossover(),
    "random_horizontal": RandomHorizontalSplitCrossover(),
}

_selections: Dict[str, object] = { "truncation": TruncationSelection() }

def get_mutation(name: str):  return _mutations[name]
def get_crossover(name: str): return _crossovers[name]
def get_selection(name: str): return _selections[name]