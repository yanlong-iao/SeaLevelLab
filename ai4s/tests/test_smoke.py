"""Smoke tests: the lab must satisfy the Gymnasium contract and every agent must close the loop."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from sea_level_twin import DigitalTwin, to_degrees, to_unit  # noqa: E402
from sea_level_env import SeaLevelLabEnv  # noqa: E402


@pytest.fixture(scope="module")
def twin():
    return DigitalTwin()


def test_twin_is_fitted_to_the_13_gauges(twin):
    d = twin.describe()
    assert d["n_stations"] == 13
    assert 0.05 < d["gp_hyper"]["lengthscale_unit"] < 1.0
    assert 1 < d["gp_hyper"]["mean_mm_yr"] < 10          # global-mean rise rate is a few mm/yr
    x = np.array([[0.3, 0.6]])
    assert np.allclose(to_unit(*to_degrees(x).T), x)


def test_env_passes_gymnasium_checker(twin):
    from gymnasium.utils.env_checker import check_env
    check_env(SeaLevelLabEnv(twin, mode="active_learning", budget=4), skip_render_check=True)


@pytest.mark.parametrize("mode", ["active_learning", "optimization"])
def test_episode_terminates_at_budget_and_belief_sharpens(twin, mode):
    env = SeaLevelLabEnv(twin, mode=mode, budget=5)
    env.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert term and not trunc and info["n_experiments"] == 5
    assert info["ipv_fraction"] < 1.0                     # experiments can only reduce integrated variance


def test_botorch_loop_beats_random_on_the_same_truth(twin):
    from bo_agent import run_bo, run_random
    env = SeaLevelLabEnv(twin, mode="active_learning", budget=8)
    assert run_bo(env, seed=1)["ipv_fraction"] < run_random(env, seed=1)["ipv_fraction"]


def test_ppo_trains_and_rolls_out(twin):
    from rl_agent import run_ppo, train_ppo
    model = train_ppo(twin, "optimization", budget=4, total_timesteps=256)
    out = run_ppo(model, SeaLevelLabEnv(twin, mode="optimization", budget=4), seed=0)
    assert out["n"] == 4


def test_llm_toolbox_refuses_to_overspend(twin):
    import json
    from llm_agent import LabToolbox
    env = SeaLevelLabEnv(twin, mode="active_learning", budget=3); env.reset(seed=0)
    tb = LabToolbox(env)
    assert "error" in json.loads(tb.call("run_bo_campaign", mode="optimization", n_experiments=99))
    assert json.loads(tb.call("run_bo_campaign", mode="active_learning", n_experiments=3))["ran"] == 3
    assert "error" in json.loads(tb.call("run_experiment", lon_deg=170.0, lat_deg=-5.0))
