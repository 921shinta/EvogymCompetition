from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from evogym import is_connected, get_full_connectivity
from server.trainer.ga.base import Individual

class BaseCrossover(ABC):
    @abstractmethod
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """親2体 -> 子2体"""
        ...

class NoCrossover(BaseCrossover):
    """交叉を行わずコピーするだけ。"""
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        c1 = Individual(p1.body.copy(), p1.connections.copy(), p1.label, p1.controller_params)
        c2 = Individual(p2.body.copy(), p2.connections.copy(), p2.label, p2.controller_params)
        return c1, c2

class SinglePointCrossover(BaseCrossover):
    """
    一点交叉を行うクラス。
    
    【今回の戦略：パワー特化 (Amplitude Focused)・調整版】
    1. 構造: 縦横ランダムに分割し、多様なボディバランスを探索する。
    2. 制御: 「周波数」や「位相」の小細工は廃止し、親の値を完全継承する。
       振幅(Amplitude)を **5〜10倍** にブーストし、
       物理崩壊を防ぎつつ強力な跳躍力を付与する（200倍はやりすぎだったため修正）。
    """
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        structure1 = p1.body
        structure2 = p2.body
        rows, cols = structure1.shape

        child1_body = structure1.copy()
        child2_body = structure2.copy()

        # 50%の確率で「縦切り」か「横切り」か
        if np.random.random() < 0.5:
            # --- 縦に切る (Columns split: 左右に分ける) ---
            if cols > 1:
                split_point = np.random.randint(1, cols)
                c1_cand = np.hstack((structure1[:, :split_point], structure2[:, split_point:]))
                c2_cand = np.hstack((structure2[:, :split_point], structure1[:, split_point:]))
                child1_body = c1_cand
                child2_body = c2_cand
        else:
            # --- 横に切る (Rows split: 上下に分ける) ---
            if rows > 1:
                split_point = np.random.randint(1, rows)
                c1_cand = np.vstack((structure1[:split_point, :], structure2[split_point:, :]))
                c2_cand = np.vstack((structure2[:split_point, :], structure1[split_point:, :]))
                child1_body = c1_cand
                child2_body = c2_cand

        # --- 連結チェック ---
        if not is_connected(child1_body):
            child1_body = structure1.copy()
        if not is_connected(child2_body):
            child2_body = structure2.copy()

        # --- コントローラの調整（振幅ブースト・マイルド版） ---
        params1 = p1.controller_params
        params2 = p2.controller_params
        
        new_params1 = params1
        new_params2 = params2

        # Sin波コントローラ (f, a, p) の場合のみ調整を行う
        if isinstance(params1, (tuple, list, np.ndarray)) and len(params1) == 3:
            f1, a1, p1_val = params1
            
            # 1. 振幅ブースト (Power Boost)
            # 物理崩壊を避けるため、5.0倍〜10.0倍程度に落ち着かせる (上限50.0)
            a1_boosted = min(a1 * np.random.uniform(5.0, 10.0), 50.0)
            
            # 周波数と位相はそのまま継承
            new_params1 = (f1, a1_boosted, p1_val)

        if isinstance(params2, (tuple, list, np.ndarray)) and len(params2) == 3:
            f2, a2, p2_val = params2
            
            # 親2側も同様
            a2_boosted = min(a2 * np.random.uniform(5.0, 10.0), 50.0)
            new_params2 = (f2, a2_boosted, p2_val)

        # 個体生成
        conn1 = get_full_connectivity(child1_body)
        conn2 = get_full_connectivity(child2_body)

        c1 = Individual(body=child1_body, connections=conn1, label=p1.label, controller_params=new_params1)
        c2 = Individual(body=child2_body, connections=conn2, label=p2.label, controller_params=new_params2)

        return c1, c2