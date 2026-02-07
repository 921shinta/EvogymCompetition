from __future__ import annotations
from abc import ABC, abstractmethod
import random
from typing import Optional
from server.trainer.ga.base import Individual
from server.trainer.utils.algo_utils import mutate as evogym_mutate

class BaseMutation(ABC):
    @abstractmethod
    def __call__(self, parent: Individual, new_label: int) -> Optional[Individual]:
        ...

class DefaultMutation(BaseMutation):
    def __call__(self, parent: Individual, new_label: int) -> Optional[Individual]:
        # 構造の変異
        child_res = evogym_mutate(parent.body.copy(), mutation_rate=0.1, num_attempts=50)
        if child_res is None:
            return None
        body_c, conn_c = child_res

        # パラメータの変異 (タプルの展開)
        f, a, p = parent.controller_params
        
        # 確率的に微調整
        if random.random() < 0.2:
            f = max(0.001, min(f * random.uniform(0.8, 1.2), 0.2))
        if random.random() < 0.3:
            a = max(0.2,  min(a * random.uniform(0.8, 1.2), 2.0))
        if random.random() < 0.3:
            p += random.uniform(-0.5, 0.5)

        return Individual(body_c, conn_c, new_label, (f, a, p))