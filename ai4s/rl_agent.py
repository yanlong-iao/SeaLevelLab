"""
rl_agent.py
===========
**Stable-Baselines3 PPO** on the very same `SeaLevelLabEnv`.

BO vs RL, in one sentence: BO is a *myopic, per-episode* planner (refit GP, maximise a one-step
acquisition); RL learns an *amortised* policy pi(a | belief) across many episodes with different
hidden truths -- i.e. it learns its own acquisition function, including non-myopic behaviour
("spend early experiments spreading out, late ones exploiting").  Both consume the same belief
state, so the lab can swap them without touching hardware code.
[HKQAI JD: intelligence layer for self-driving labs; RL]
"""
from __future__ import annotations

import warnings
import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sea_level_env import SeaLevelLabEnv
from sea_level_twin import DigitalTwin

warnings.filterwarnings("ignore", module="torch")


def _symmetric(env: SeaLevelLabEnv):
    """SB3 gotcha: its Gaussian policy starts centred on 0 and clips to the action box, so with a
    [0,1] box half the early proposals pile up on the corner (140E, 25S).  A symmetric [-1,1] box,
    rescaled back to the lab's [0,1] design space by the wrapper, removes that bias."""
    return RescaleAction(env, min_action=-1.0, max_action=1.0)


def train_ppo(twin: DigitalTwin, mode: str, budget: int = 15, total_timesteps: int = 15_000, seed: int = 0, verbose: int = 0) -> PPO:
    env = Monitor(_symmetric(SeaLevelLabEnv(twin, mode=mode, budget=budget)))
    model = PPO(
        "MlpPolicy", env,
        n_steps=budget * 16, batch_size=budget * 4, n_epochs=10, learning_rate=3e-4,
        gamma=1.0,                      # finite horizon, budget is in the state -> undiscounted return is the natural objective
        ent_coef=0.01, seed=seed, verbose=verbose,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def run_ppo(model: PPO, env: SeaLevelLabEnv, seed: int = 0) -> dict:
    env = _symmetric(env)
    obs, info = env.reset(seed=seed)
    hist, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        hist.append(info); done = term or trunc
    last = hist[-1]
    return dict(agent="SB3-PPO", mode=env.unwrapped.mode, total_reward=float(sum(h["reward"] for h in hist)),
                ipv_fraction=last["ipv_fraction"], simple_regret=last["simple_regret"], best_y=last["best_y"], n=len(hist))


if __name__ == "__main__":
    tw = DigitalTwin()
    m = train_ppo(tw, "active_learning", budget=10, total_timesteps=4_000, verbose=0)
    print(run_ppo(m, SeaLevelLabEnv(tw, mode="active_learning", budget=10), seed=0))
