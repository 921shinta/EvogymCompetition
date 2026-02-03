from __future__ import annotations
import math
from pathlib import Path
import importlib
import sys

import gymnasium as gym
import numpy as np

# --- マルチプロセス時の環境登録 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAPPED = False

def _mp_bootstrap_register() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    importlib.invalidate_caches()
    import evogym.envs  # noqa: F401
    import server.custom_env.register  # noqa: F401
    _BOOTSTRAPPED = True

# --- コントローラ ---
def periodic_controller(step: int, n_act: int, params: tuple[float, float, float]) -> np.ndarray:
    f, a, p = params
    val = a * math.sin(2 * math.pi * f * step + p)
    return np.full((n_act,), val, dtype=np.float32)

# --- フォールバック判定 ---
def _should_fallback(e: Exception) -> bool:
    msg = str(e).lower()
    return ("unexpected keyword" in msg) or ("got an unexpected keyword" in msg)

# --- 評価本体 ---
def evaluate_structure(
    body: np.ndarray,
    connections: np.ndarray,
    controller_params: tuple[float, float, float],
    env_name: str,
    max_steps: int,
) -> float:
    _mp_bootstrap_register()

    try:
        gym.spec(env_name)
    except Exception:
        from server.custom_env import ensure_registered
        ensure_registered(env_name)

    try:
        env = gym.make(env_name, body=body, connections=connections, render_mode=None)
    except (TypeError, gym.error.Error) as e:
        if _should_fallback(e):
            env = gym.make(env_name, render_mode=None)
        else:
            raise

    shape = getattr(env.action_space, "shape", None)
    if not shape or shape[0] <= 0:
        env.close()
        raise RuntimeError(f"action_space が不正（shape={shape}）。")

    params = controller_params
    obs, _ = env.reset()
    total = 0.0
    n_act = shape[0]

    # --- 高さ計測 ---
    max_height = 0.0
    start_height = 0.0
    try:
        start_pos = env.object_pos_at_time(env.get_time(), "robot")
        start_height = start_pos[1]
        max_height = start_height
    except Exception:
        pass

    for t in range(max_steps):
        action = periodic_controller(t, n_act, params)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)

        try:
            pos = env.object_pos_at_time(env.get_time(), "robot")
            current_height = pos[1]
            if current_height > max_height:
                max_height = current_height
        except Exception:
            pass

        if terminated or truncated:
            break

    env.close()

    # --- ボーナス計算と足切り ---
    jump_gain = max(0.0, max_height - start_height)
    
    # 【復活】足切り処理
    # 0.5m (50cm) も飛べない個体は、移動スコアを 1/100 に減点して事実上の「落第」とします。
    if jump_gain < 0.5:
        total = total * 0.01
    
    # ジャンプボーナス（係数 100.0）
    # 高く飛べば飛ぶほど圧倒的なスコアになります。
    jump_bonus = jump_gain * 100.0
    
    total += jump_bonus

    return total