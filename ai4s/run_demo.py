"""
run_demo.py -- end-to-end demonstration of the AI4S stack built on flood_final.qmd's model.

    python run_demo.py                 # twin -> Gym env -> Random / BoTorch / Optuna / PPO -> LLM agent
    python run_demo.py --live          # additionally lets Claude drive the lab (needs API credentials)

Outputs land in ./outputs: results.json, belief_map.png, agent_transcript.txt
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sea_level_twin import DigitalTwin
from sea_level_env import SeaLevelLabEnv
from bo_agent import run_bo, run_optuna, run_random
from rl_agent import run_ppo, train_ppo
from llm_agent import LabToolbox, MockScientistAgent

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def benchmark(twin, mode, budget, episodes, ppo_model):
    env = SeaLevelLabEnv(twin, mode=mode, budget=budget)
    rows = []
    for s in range(episodes):
        rows.append(run_random(env, s))
        rows.append(run_bo(env, s))
        if mode == "optimization":
            rows.append(run_optuna(env, s))
        if ppo_model is not None:
            rows.append(run_ppo(ppo_model, env, s))
    return rows


def table(rows, mode):
    metric = "ipv_fraction" if mode == "active_learning" else "simple_regret"
    agents = sorted({r["agent"] for r in rows}, key=lambda a: ["Random", "Optuna-GP", "SB3-PPO", "BoTorch"].index(a))
    lines = [f"\n=== mode = {mode}   (metric: {metric}, lower is better; mean +/- sd over episodes) ==="]
    lines.append(f"{'agent':<12}{metric:>16}{'total reward':>16}{'best y (mm/yr)':>16}")
    out = {}
    for a in agents:
        v = np.array([r[metric] for r in rows if r["agent"] == a])
        tr = np.array([r["total_reward"] for r in rows if r["agent"] == a])
        by = np.array([r["best_y"] for r in rows if r["agent"] == a])
        lines.append(f"{a:<12}{v.mean():>10.3f} +/- {v.std():<5.3f}{tr.mean():>10.3f}{by.mean():>16.2f}")
        out[a] = dict(metric=metric, mean=float(v.mean()), sd=float(v.std()), total_reward=float(tr.mean()), best_y=float(by.mean()))
    print("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--rl-steps", type=int, default=40_000)
    ap.add_argument("--skip-rl", action="store_true")
    ap.add_argument("--live", action="store_true", help="let Claude drive the lab via the Anthropic SDK")
    a = ap.parse_args()

    t0 = time.time()
    twin = DigitalTwin()
    print("Digital twin fitted from Processed_Final_data.csv:")
    print(json.dumps(twin.describe()["gp_hyper"]))

    results = {}
    for mode in ("active_learning", "optimization"):
        ppo = None
        if not a.skip_rl:
            print(f"\n[RL] training PPO for mode={mode} ({a.rl_steps} steps) ...", flush=True)
            ppo = train_ppo(twin, mode, budget=a.budget, total_timesteps=a.rl_steps)
        rows = benchmark(twin, mode, a.budget, a.episodes, ppo)
        results[mode] = table(rows, mode)

    # Figure: one BoTorch active-learning episode
    env = SeaLevelLabEnv(twin, mode="active_learning", budget=a.budget)
    run_bo(env, seed=0)
    fig = env.render(); fig.savefig(OUT / "belief_map.png", dpi=130)
    print(f"\nsaved {OUT/'belief_map.png'}")

    # Agentic layer (offline mock)
    print("\n=== LLM agent (mock ReAct) driving the lab ===")
    env = SeaLevelLabEnv(twin, mode="active_learning", budget=a.budget); env.reset(seed=1)
    tb = LabToolbox(env)
    trace = MockScientistAgent(tb).run(f"Characterise the Pacific sea-level-rise field with {a.budget} experiments and locate the highest-risk site.")
    (OUT / "agent_transcript.txt").write_text("\n".join(trace))

    if a.live:
        from llm_agent import ClaudeScientistAgent
        print("\n=== LLM agent (Claude, live) driving the lab ===")
        env = SeaLevelLabEnv(twin, mode="active_learning", budget=a.budget); env.reset(seed=2)
        ClaudeScientistAgent(LabToolbox(env)).run(f"You have {a.budget} experiments. Characterise the field, then find the highest-risk site.")

    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {OUT/'results.json'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
