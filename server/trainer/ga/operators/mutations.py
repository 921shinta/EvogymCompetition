from __future__ import annotations
from abc import ABC, abstractmethod
import random
from typing import Optional
import numpy as np

from server.trainer.ga.base import Individual
from server.trainer.utils.algo_utils import mutate as evogym_mutate

class BaseMutation(ABC):
    @abstractmethod
    def __call__(self, parent: Individual, new_label: int) -> Optional[Individual]:
        """
        親1体 -> 子1体（失敗時 None）
        """
        ...

class DefaultMutation(BaseMutation):
    """
    これまで MuLambdaES で使っていた突然変異ロジックをクラス化。
    【修正】振幅の上限リミッター(2.0)を撤廃。交叉で得たパワーを維持できるように変更。
    """
    def __call__(self, parent: Individual, new_label: int) -> Optional[Individual]:
        # evogym標準の形状変異
        # 突然変異確率を 3% (0.03) に設定
        child = evogym_mutate(parent.body.copy(), mutation_rate=0.03, num_attempts=50)
        if child is None:
            return None
        body_c, conn_c = child

        # コントローラパラメータの変異
        params = parent.controller_params
        
        # パラメータが3つ（Sin波）の場合のみ摂動を加える
        if isinstance(params, (tuple, list, np.ndarray)) and len(params) == 3:
            f, a, p = params
            
            # 周波数(f): 少し変動させる
            if random.random() < 0.2:
                f = max(0.001, f * random.uniform(0.8, 1.2))
            
            # 振幅(a): ★重要★ 上限 2.0 を撤廃！
            # 交叉で得た数百倍のパワーを維持しつつ、少し変動させるだけに留める
            if random.random() < 0.3:
                a = a * random.uniform(0.8, 1.2) 
            
            # 位相(p): 少しずらす
            if random.random() < 0.3:
                p += random.uniform(-0.5, 0.5)
                
            new_params = (f, a, p)
        else:
            new_params = params

        return Individual(body_c, conn_c, new_label, new_params)

class RotationMutation(BaseMutation):
    """
    ロボットの構造を90度単位で回転させる突然変異オペレータ。
    """
    def __call__(self, parent: Individual, new_label: int) -> Optional[Individual]:
        k = np.random.randint(1, 4)
        new_body = np.rot90(parent.body, k=k)
        
        return Individual(
            body=new_body,
            connections=None,
            label=new_label,
            controller_params=parent.controller_params
        )