# SeaLevelLab — the `flood_final.qmd` model as a self-driving-lab intelligence layer

`flood_final.qmd` fits an R-INLA SPDE Matérn field to monthly sea level at 13 Pacific tide gauges.
This folder keeps the *same statistical object* (a Matérn Gaussian process) and wraps it in the
stack an AI4S team uses to run a self-driving laboratory:

| Layer | File | Standard | What it does |
|---|---|---|---|
| Simulator / digital twin | `sea_level_twin.py` | numpy + scipy | Ground-truth rise-rate field drawn from a GP fitted (MAP-II) to the 13 real gauges; `measure(x)` is the physical experiment |
| Environment | `sea_level_env.py` | **Gymnasium** (`SeaLevelLab-v0`) | Belief-MDP: state = GP posterior grid + budget, action = next location, reward = variance reduction *or* improvement |
| Closed loop | `bo_agent.py` | **BoTorch** (primary) + **Optuna** | Refit surrogate → acquisition (`qNegIntegratedPosteriorVariance` / `qLogNoisyExpectedImprovement`) → `env.step` |
| Learned policy | `rl_agent.py` | **Stable-Baselines3** PPO | Amortised experimental-design policy trained across many hidden truths |
| Agentic layer | `llm_agent.py` | `@tool` toolbox + OpenAI-compatible function calling | LLM scientist reasons (ReAct) and calls the lab as tools; offline mock + live agent (OpenAI, Azure, vLLM, Ollama) |
| Demo | `run_demo.py` | — | Random vs BoTorch vs Optuna vs PPO benchmark, belief map figure, agent transcript |

> **Live demo in RStudio:** open `../flood_ai4s.qmd` and press *Render*. It re-implements this stack in R (Matérn GP from scratch, R6 Gymnasium-style env, closed-form acquisitions, `ellmer` agent) with 3D plotly surfaces, and calls this Python folder from one `reticulate` chunk.

## Run it

```bash
cd ai4s
source .venv/bin/activate          # created with: python3 -m venv .venv && pip install -r requirements.txt
python run_demo.py                 # ~3 min on a laptop (PPO training dominates); --skip-rl for ~30 s
python run_demo.py --live          # also lets an LLM drive the lab (OPENAI_API_KEY, or OPENAI_BASE_URL for a local model)
```

Outputs in `outputs/`: `results.json` (benchmark), `belief_map.png` (truth / posterior mean / posterior sd with
the agent's experiments), `agent_transcript.txt` (Thought → Action → Observation trace).

Individual modules are runnable for unit checks: `python sea_level_env.py` runs Gymnasium's `check_env`.

## The math bridge in one table

| In `flood_final.qmd` (R-INLA) | In this stack | Why it is the same thing |
|---|---|---|
| `f(field, model = spde)`, `alpha = 2` | Matérn GP (`MaternGP`, `SingleTaskGP` + `MaternKernel`) | SPDE `(κ²−Δ)^{α/2}(τu)=W` has a Matérn-covariance GP solution, ν = α − d/2 |
| `range0`, `kappa0 = √8/range0` | `lengthscale`; practical range ρ = √(8ν)/κ | Same parameter, different name |
| `sigma0`, `tau0` | `outputscale` / `signal_sd` | τ is the SPDE scaling of σ |
| Gaussian family precision | `noise_sd` / `likelihood.noise` | Nugget |
| `theta.prior.mean/prec` (PC priors) | `LogNormalPrior` on lengthscale & noise | MAP-II regularisation, needed because 13 sites cannot identify range vs nugget |
| `int.strategy = "eb"` | `fit_gpytorch_mll` / `fit_hyperparameters` | Both maximise the (log-)marginal likelihood: type-II ML |
| `summary.fitted.values$sd` map | `env.belief.posterior(grid)[1]` | The posterior sd is what every acquisition function integrates |
| "Prediction error vs posterior SD" plot | calibration check for BO | If sd is not calibrated, UCB/EI/IPV acquisitions are mis-priced |
| `rw2` time model (older `flood project.qmd`) | state-space / belief-MDP transition | Integrated random walk ≡ linear-Gaussian state-space model; Kalman update ≡ Bayesian belief update in the env |

## Benchmark (`python run_demo.py --budget 15 --episodes 6 --rl-steps 40000`)

Same hidden truths and measurement noise for every agent (seeded). Lower metric is better.

**Active learning** (fraction of prior map variance remaining after 15 experiments):

| agent | IPV fraction remaining | episode return | best measured (mm/yr) |
|---|---|---|---|
| Random | 0.794 ± 0.035 | +0.056 | 8.13 |
| SB3-PPO | 0.917 ± 0.012 | -0.067 | 6.24 |
| BoTorch | 0.684 ± 0.008 | +0.166 | 7.95 |

**Optimisation** (simple regret in mm/yr against the twin's true maximum):

| agent | simple regret | episode return | best measured (mm/yr) |
|---|---|---|---|
| Random | 1.838 ± 1.292 | +1.577 | 8.13 |
| Optuna-GP | 1.344 ± 0.840 | +1.903 | 8.69 |
| SB3-PPO | 2.603 ± 1.262 | +1.206 | 7.49 |
| BoTorch | 1.961 ± 1.068 | +1.672 | 8.30 |

See [`docs/report.html`](../docs/report.html) (live: GitHub Pages) for the five-phase write-up (comprehension, math bridge, MDP design, agentic workflow, code, interview pitch).
