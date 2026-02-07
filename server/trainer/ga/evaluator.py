from __future__ import annotations
import math
from pathlib import Path
import importlib
import sys
import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAPPED = False

def _mp_bootstrap_register() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    importlib.invalidate_caches()
    import evogym.envs
    import server.custom_env.register
    _BOOTSTRAPPED = True

# --- 元のコントローラ (グローバルパラメータ版) ---
def periodic_controller(step: int, n_act: int, params: tuple) -> np.ndarray:
    f, a, p = params
    val = a * math.sin(2 * math.pi * f * step + p)
    # 全てのアクチュエータに同じ値を返す
    return np.full((n_act,), val, dtype=np.float32)

def _should_fallback(e: Exception) -> bool:
    msg = str(e).lower()
    return ("unexpected keyword" in msg) or ("got an unexpected keyword" in msg)

def evaluate_structure(
    body: np.ndarray,
    connections: np.ndarray,
    controller_params: tuple, # タプルを受け取る
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

    # アクション空間のチェック
    shape = getattr(env.action_space, "shape", None)
    if not shape or shape[0] <= 0:
        env.close()
        # アクチュエータがない場合は報酬0
        return 0.0

    n_act = shape[0]
    obs, _ = env.reset()
    total = 0.0
    
    for t in range(max_steps):
        # controller_params (f,a,p) をそのまま渡す
        action = periodic_controller(t, n_act, controller_params)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break

    env.close()
    return total