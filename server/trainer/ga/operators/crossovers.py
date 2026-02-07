from __future__ import annotations
import random
import numpy as np
from typing import Tuple
from evogym import is_connected, get_full_connectivity
from server.trainer.ga.base import Individual
from copy import deepcopy


class BaseCrossover:
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        ...


class NoCrossover(BaseCrossover):
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        # そのままコピー
        return deepcopy(p1), deepcopy(p2)


class TorsoLegCrossover(BaseCrossover):
    """上2行（胴体）・下3行（脚）で固定水平分割する交叉"""
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        split = 2
        
        # ボディだけを混ぜる
        c1_body = np.vstack([p1.body[:split, :], p2.body[split:, :]])
        c2_body = np.vstack([p2.body[:split, :], p1.body[split:, :]])

        # 連結性チェック
        if not is_connected(c1_body):
            c1_body = p1.body.copy()
        if not is_connected(c2_body):
            c2_body = p2.body.copy()

        # パラメータは混ぜずに親のものを継承 (可視化ツールとの互換性のため)
        # 構造が変わるので connections は再計算 (-1 ではないが None で渡して再計算させる)
        c1 = Individual(c1_body, get_full_connectivity(c1_body), -1, p1.controller_params)
        c2 = Individual(c2_body, get_full_connectivity(c2_body), -1, p2.controller_params)
        return c1, c2


class RandomHorizontalSplitCrossover(BaseCrossover):
    """ランダムな高さで水平分割する交叉"""
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        h, _ = p1.body.shape
        split = random.randint(1, h - 1)

        c1_body = np.vstack([p1.body[:split, :], p2.body[split:, :]])
        c2_body = np.vstack([p2.body[:split, :], p1.body[split:, :]])

        if not is_connected(c1_body):
            c1_body = p1.body.copy()
        if not is_connected(c2_body):
            c2_body = p2.body.copy()

        c1 = Individual(c1_body, get_full_connectivity(c1_body), -1, p1.controller_params)
        c2 = Individual(c2_body, get_full_connectivity(c2_body), -1, p2.controller_params)
        return c1, c2