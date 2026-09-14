"""
bo_agent.py
===========
The closed loop: **BoTorch** (and, as a second driver, **Optuna**) talking to `SeaLevelLabEnv`.

How BO "operates seamlessly" with the environment
-------------------------------------------------
    for each step until the budget is spent:
        X, y   = env.observed_data()                 # everything the lab has measured so far
        model  = SingleTaskGP(X, y) ; fit_gpytorch_mll   # surrogate refit  (= INLA refit, type-II ML)
        acq    = qNegIntegratedPosteriorVariance | qLogExpectedImprovement
        x_next = optimize_acqf(acq)                  # inner optimisation over the design space
        env.step(x_next)                             # physical experiment + Bayesian belief update

The surrogate is the *same mathematical object* as the SPDE field in flood_final.qmd (Matern GP),
just with hyper-parameters refit every iteration on the growing data set instead of once.
The posterior sd map the R code plotted is what the acquisition function integrates.
[HKQAI JD: closed-loop optimisation systems; ML x scientific computing]
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from sea_level_env import SeaLevelLabEnv
from sea_level_twin import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

# NB: do NOT call torch.set_default_dtype(float64) globally -- it silently breaks SB3's float32
# policy networks when both agents live in the same process.  Keep float64 local to BoTorch tensors.
DT = torch.float64
warnings.filterwarnings("ignore", module="botorch")
warnings.filterwarnings("ignore", module="linear_operator")
warnings.filterwarnings("ignore", module="torch")
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)


def fit_surrogate(X: np.ndarray, y: np.ndarray) -> SingleTaskGP:
    """Matern-3/2 GP with ARD -- separate east-west / north-south ranges, an anisotropy the
    isotropic INLA SPDE in the R code could not express.

    The log-normal priors are not decoration: with ~15-30 noisy points the marginal likelihood
    happily collapses to "length-scale 0.3 deg, noise 0" (pure interpolation).  R-INLA guards
    against this with PC priors on (range, sigma); here the same guard is MAP-II with LogNormal
    priors, so fit_gpytorch_mll maximises log-likelihood + log-prior.  A statistician recognises
    this instantly; many ML engineers learn it the hard way.
    """
    train_X = torch.as_tensor(np.asarray(X, float), dtype=DT)
    train_Y = torch.as_tensor(np.asarray(y, float), dtype=DT).reshape(-1, 1)
    covar = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2, lengthscale_prior=LogNormalPrior(np.log(0.25), 0.5)),
                        outputscale_prior=LogNormalPrior(0.0, 1.0))
    lik = GaussianLikelihood(noise_prior=LogNormalPrior(-2.0, 1.0), noise_constraint=GreaterThan(1e-3))
    model = SingleTaskGP(train_X, train_Y, covar_module=covar, likelihood=lik, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))   # type-II MAP == INLA "eb" + PC priors
    return model


def surrogate_report(model: SingleTaskGP) -> dict:
    """Hyper-parameters in the units a statistician reports (compare with inla.spde2.result)."""
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    return dict(lengthscale_unit=ls.round(3).tolist(),
                lengthscale_deg=[round(float(ls[0] * (LON_MAX - LON_MIN)), 1), round(float(ls[1] * (LAT_MAX - LAT_MIN)), 1)],
                outputscale=round(float(model.covar_module.outputscale.detach()), 3),
                noise=round(float(model.likelihood.noise.detach().mean()), 4))


def propose(model: SingleTaskGP, mode: str, mc_points: torch.Tensor, best_f: float) -> np.ndarray:
    if mode == "active_learning":
        # a-optimal design: pick x minimising the *integrated* posterior variance over the domain
        acq = qNegIntegratedPosteriorVariance(model, mc_points=mc_points)
    else:
        # measurements are noisy (gauge noise ~0.6 mm/yr), so the "best observed" is itself uncertain:
        # noisy-EI integrates over that instead of trusting a lucky draw (Letham et al. 2019)
        acq = qLogNoisyExpectedImprovement(model, X_baseline=model.train_inputs[0], prune_baseline=True)
    cand, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=8, raw_samples=128)
    return cand.detach().numpy().reshape(2)


def bo_steps(env: SeaLevelLabEnv, n_steps: int, mode: str | None = None, n_init: int = 3, verbose: bool = False) -> list[dict]:
    """Continue the CURRENT episode for n_steps with BoTorch.  Called by the LLM agent as a tool,
    so the objective (`mode`) may be switched mid-episode: same lab, same belief, new goal."""
    if mode is not None:
        env.mode = mode
    mc_points = torch.as_tensor(env.grid, dtype=DT)
    sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=int(env.np_random.integers(1 << 30)))
    out = []
    for _ in range(n_steps):
        if env.t >= env.budget:
            break
        X, y = env.observed_data()
        if len(y) < n_init:                                   # cold start -> space-filling design
            x = sobol.draw(1, dtype=DT).numpy().reshape(2)
        else:
            model = fit_surrogate(X, y)
            x = propose(model, env.mode, mc_points, best_f=float(np.max(y)))
        obs, r, term, trunc, info = env.step(x)
        out.append(info)
        if verbose:
            print(f"  BO t={info['n_experiments']:2d} x={info['x_deg']} y={info['y']:.2f} r={r:+.3f} IPV={info['ipv_fraction']:.3f} regret={info['simple_regret']:.2f}")
    return out


def run_bo(env: SeaLevelLabEnv, seed: int = 0, verbose: bool = False) -> dict:
    env.reset(seed=seed)
    hist = bo_steps(env, env.budget, verbose=verbose)
    return _summary("BoTorch", env, hist)


def run_random(env: SeaLevelLabEnv, seed: int = 0) -> dict:
    """Baseline every AI4S result must beat: uniformly random experiments."""
    env.reset(seed=seed)
    rng = np.random.default_rng(10_000 + seed)                 # separate stream -> same truth & noise as BO
    hist = [env.step(rng.uniform(size=2))[4] for _ in range(env.budget)]
    return _summary("Random", env, hist)


def run_optuna(env: SeaLevelLabEnv, seed: int = 0) -> dict:
    """Second BO driver.  Optuna's GPSampler is a black-box optimiser: fine for the optimisation
    objective, but it cannot do active learning because the information-gain objective lives
    *inside* the surrogate, which Optuna hides -- this is exactly why BoTorch is the primary tool."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    assert env.mode == "optimization"
    env.reset(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=3))
    hist = []

    def objective(trial):
        x = np.array([trial.suggest_float("lon", 0, 1), trial.suggest_float("lat", 0, 1)])
        obs, r, term, trunc, info = env.step(x)
        hist.append(info)
        return info["y"]

    study.optimize(objective, n_trials=env.budget)
    return _summary("Optuna-GP", env, hist)


def _summary(agent: str, env: SeaLevelLabEnv, hist: list[dict]) -> dict:
    last = hist[-1]
    return dict(agent=agent, mode=env.mode, total_reward=float(sum(h["reward"] for h in hist)),
                ipv_fraction=last["ipv_fraction"], simple_regret=last["simple_regret"],
                best_y=last["best_y"], n=len(hist))


if __name__ == "__main__":
    from sea_level_twin import DigitalTwin
    tw = DigitalTwin()
    for mode in ("active_learning", "optimization"):
        env = SeaLevelLabEnv(tw, mode=mode, budget=10)
        print(mode, "random :", run_random(env, 0))
        print(mode, "botorch:", run_bo(env, 0, verbose=True))
        if mode == "optimization":
            print(mode, "optuna :", run_optuna(env, 0))
