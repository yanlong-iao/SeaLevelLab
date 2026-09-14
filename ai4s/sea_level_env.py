"""
sea_level_env.py
================
The Pacific tide-gauge network re-cast as a **self-driving laboratory** with a Gymnasium API.

The MDP (strictly in scientific-discovery terms)
-----------------------------------------------
* **Hidden physical state**  f : the true sea-level-rise field (mm/yr).  Never observed directly,
  so the raw problem is a POMDP.
* **State s_t (belief)**     Under a GP, the posterior (mu_t, Sigma_t) is a *sufficient statistic*
  of the whole experiment history (X_1..t, y_1..t).  Exposing it as the observation turns the POMDP
  into a *belief-MDP* whose transition kernel is Bayes' rule.  This is the same trick as a Kalman
  filter / state-space model: the RW2 model in the older `flood project.qmd` *is* a linear-Gaussian
  state-space model, and INLA's latent field update is the spatial analogue of a Kalman update.
* **Action a_t in [0,1]^2**  WHERE to run the next experiment (deploy a gauge / run a campaign).
* **Reward r_t**
    - mode="active_learning": (IPV_{t-1} - IPV_t) / IPV_0  - cost   (integrated posterior variance
      reduction = a-optimal design gain; a proxy for expected information gain)
    - mode="optimization":    max(0, y_t - best_{t-1}) / sigma_f - cost   (improvement of the best
      measured rise rate = hot-spot search; cumulative reward tracks the "best found" curve that BO
      papers report)
* **Episode**                ends when the experiment budget is spent.  Budget is part of the state,
  so this is a genuine terminal state (`terminated=True`), not a time-limit truncation.

[HKQAI JD: "designing closed-loop optimisation systems"] -- `step()` *is* the loop:
    propose x  ->  physical experiment (twin.measure)  ->  computational update (GP)  ->  reward.
Any agent (BoTorch, Optuna, SB3-PPO, an LLM) only has to speak Gymnasium.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sea_level_twin import DigitalTwin, MaternGP, to_degrees, to_unit


class SeaLevelLabEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, twin: DigitalTwin | None = None, mode: str = "active_learning", budget: int = 15,
                 grid_res: int = 12, cost_per_experiment: float = 0.01, warm_start: bool | None = None,
                 render_mode: str | None = None):
        assert mode in ("active_learning", "optimization")
        self.twin = twin if twin is not None else DigitalTwin()
        self.mode, self.budget, self.G, self.cost = mode, int(budget), int(grid_res), float(cost_per_experiment)
        # Active learning = "where should the 14th gauge go?"  -> the 13 real gauges are prior data.
        # Optimisation   = "find the hot-spot with a fresh campaign" -> cold start (stations only set the prior).
        self.warm_start = (mode == "active_learning") if warm_start is None else bool(warm_start)
        self.render_mode = render_mode

        g = np.linspace(0.0, 1.0, self.G)
        GX, GY = np.meshgrid(g, g, indexing="ij")
        self.grid = np.c_[GX.ravel(), GY.ravel()]                      # design points of the belief map

        # [JD: intelligence layer] the action IS the experimental design decision.
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        # observation = [mu grid (G^2), sd grid (G^2), budget fraction, best-so-far], all standardised.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * self.G**2 + 2,), dtype=np.float32)

        self.belief: MaternGP | None = None
        self.X_run: list[np.ndarray] = []
        self.y_run: list[float] = []
        self.history: list[dict] = []
        self.t = 0
        self.ipv0 = self.ipv = 1.0
        self.best_y = 0.0
        self.best_x: np.ndarray | None = None

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.twin.sample_truth(self.np_random)            # a NEW hidden truth every episode -> policies must generalise
        self.belief = MaternGP(self.twin.hyper)           # the lab's analysis pipeline (= the INLA model, ported)
        if self.warm_start:
            self.belief.set_data(self.twin.X_st, self.twin.y_st)
        self.X_run, self.y_run, self.history, self.t = [], [], [], 0
        mu, sd = self.belief.posterior(self.grid)
        self.ipv0 = self.ipv = float((sd**2).mean())
        self.best_y, self.best_x = float(self.twin.hyper.mean), None   # prior expectation is the baseline to beat
        return self._obs(mu, sd), self._info(mu, sd)

    def step(self, action):
        x = np.clip(np.asarray(action, dtype=float).reshape(2), 0.0, 1.0)
        y = self.twin.measure(x, self.np_random)          # (1) PHYSICAL experiment  [JD: coordinate physical experiments]
        self.belief.add(x, y)                             # (2) COMPUTATIONAL update  [JD: coordinate computational experiments]
        self.X_run.append(x); self.y_run.append(y); self.t += 1
        mu, sd = self.belief.posterior(self.grid)
        ipv_new = float((sd**2).mean())
        if self.mode == "active_learning":
            reward = (self.ipv - ipv_new) / self.ipv0 - self.cost
        else:
            reward = max(0.0, y - self.best_y) / self.twin.hyper.signal_sd - self.cost
        self.ipv = ipv_new
        if y > self.best_y:
            self.best_y, self.best_x = y, x
        terminated = self.t >= self.budget
        info = self._info(mu, sd)
        info.update(x=x, y=float(y), x_deg=to_degrees(x)[0].round(2).tolist(), reward=float(reward))
        self.history.append(info)
        return self._obs(mu, sd), float(reward), terminated, False, info

    # ------------------------------------------------------------------ helpers used by every agent
    def observed_data(self) -> tuple[np.ndarray, np.ndarray]:
        """All (x, y) pairs currently in the belief -- what BoTorch refits its surrogate on."""
        return self.belief.X.copy(), self.belief.y.copy()

    def _obs(self, mu, sd) -> np.ndarray:
        h = self.twin.hyper
        return np.concatenate([(mu - h.mean) / h.signal_sd, sd / h.signal_sd,
                               [1.0 - self.t / self.budget, (self.best_y - h.mean) / h.signal_sd]]).astype(np.float32)

    def _info(self, mu, sd) -> dict:
        i = int(np.argmax(sd))
        info = dict(n_experiments=self.t, budget_left=self.budget - self.t,
                    ipv_fraction=float(self.ipv / self.ipv0), max_sd=float(sd[i]),
                    max_sd_at_deg=to_degrees(self.grid[i])[0].round(2).tolist(),
                    best_y=float(self.best_y),
                    best_x_deg=None if self.best_x is None else to_degrees(self.best_x)[0].round(2).tolist())
        tmax, _ = self.twin.true_max
        f_best = float(self.twin.f_true(self.best_x)[0]) if self.best_x is not None else float(self.twin.hyper.mean)
        info["simple_regret"] = float(tmax - f_best)       # oracle metric: only the simulator knows it
        return info

    def belief_summary(self, k: int = 3) -> dict:
        """Compact, LLM-readable state (an LLM should never be fed 288 floats)."""
        mu, sd = self.belief.posterior(self.grid)
        top = np.argsort(-sd)[:k]
        peak = np.argsort(-mu)[:k]
        return dict(mode=self.mode, n_experiments=self.t, budget_left=self.budget - self.t,
                    ipv_fraction_remaining=round(float(self.ipv / self.ipv0), 3),
                    uncertainty_hotspots=[dict(lon=float(to_degrees(self.grid[j])[0][0].round(1)),
                                               lat=float(to_degrees(self.grid[j])[0][1].round(1)),
                                               posterior_sd=round(float(sd[j]), 2)) for j in top],
                    predicted_peaks=[dict(lon=float(to_degrees(self.grid[j])[0][0].round(1)),
                                          lat=float(to_degrees(self.grid[j])[0][1].round(1)),
                                          posterior_mean=round(float(mu[j]), 2), posterior_sd=round(float(sd[j]), 2)) for j in peak],
                    best_measured_mm_yr=round(float(self.best_y), 2),
                    best_measured_at_deg=None if self.best_x is None else to_degrees(self.best_x)[0].round(1).tolist())

    def render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 40
        g = np.linspace(0, 1, n); GX, GY = np.meshgrid(g, g, indexing="ij")
        P = np.c_[GX.ravel(), GY.ravel()]
        mu, sd = self.belief.posterior(P)
        lon = np.linspace(*to_degrees(np.array([[0, 0], [1, 1]]))[:, 0], n)
        lat = np.linspace(*to_degrees(np.array([[0, 0], [1, 1]]))[:, 1], n)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        panels = [(self.twin.f_true(P).reshape(n, n), "Hidden truth (twin)", "viridis"),
                  (mu.reshape(n, n), "Belief: posterior mean (mm/yr)", "viridis"),
                  (sd.reshape(n, n), "Belief: posterior sd  (acquisition fuel)", "magma")]
        for ax, (Zi, title, cmap) in zip(axes, panels):
            im = ax.pcolormesh(lon, lat, Zi.T, cmap=cmap, shading="auto")
            fig.colorbar(im, ax=ax, shrink=0.85)
            st = to_degrees(self.twin.X_st)
            ax.scatter(st[:, 0], st[:, 1], marker="^", s=45, c="white", edgecolor="k", label="13 real gauges")
            if self.X_run:
                xr = to_degrees(np.array(self.X_run))
                ax.scatter(xr[:, 0], xr[:, 1], s=40, c="red", edgecolor="k", label="agent experiments")
                for k, (a, b) in enumerate(xr):
                    ax.annotate(str(k + 1), (a, b), fontsize=7, color="w", ha="center", va="center")
            ax.set_title(title, fontsize=10); ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude")
        axes[0].legend(loc="lower left", fontsize=7)
        fig.suptitle(f"SeaLevelLab-v0 | mode={self.mode} | experiments={self.t}/{self.budget} | IPV left={self.ipv/self.ipv0:.2f}", fontsize=11)
        fig.tight_layout()
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig); return arr
        return fig


class DiscreteGridActionWrapper(gym.ActionWrapper):
    """Discrete(G*G) -> cell centre.  Lets value-based agents (DQN) or a tabular baseline use the lab."""
    def __init__(self, env: SeaLevelLabEnv, n: int = 8):
        super().__init__(env)
        self.n = n
        g = (np.arange(n) + 0.5) / n
        GX, GY = np.meshgrid(g, g, indexing="ij")
        self.cells = np.c_[GX.ravel(), GY.ravel()]
        self.action_space = spaces.Discrete(n * n)

    def action(self, a):
        return self.cells[int(a)].astype(np.float32)


# Register so `gym.make("SeaLevelLab-v0", mode="optimization")` works anywhere in the lab stack.
try:
    gym.register(id="SeaLevelLab-v0", entry_point="sea_level_env:SeaLevelLabEnv", max_episode_steps=None)
except gym.error.Error:
    pass


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = SeaLevelLabEnv(mode="active_learning", budget=5)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    print("obs dim", obs.shape, "| initial IPV", round(env.ipv0, 3), "| max sd at", info["max_sd_at_deg"])
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        print(f"t={info['n_experiments']} y={info['y']:.2f} r={r:+.3f} IPV left={info['ipv_fraction']:.3f}")
    print("Gymnasium check passed.")
