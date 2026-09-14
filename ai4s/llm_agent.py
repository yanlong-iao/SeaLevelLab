"""
llm_agent.py
============
The **agentic layer**: an LLM "scientist" that reasons in natural language and calls the RL/BO
lab code as *tools* (ReAct: Thought -> Action -> Observation -> ...).

Two implementations share one toolbox:
  * `MockScientistAgent`   -- deterministic ReAct loop, no API key, used in tests/CI and the demo.
  * `ClaudeScientistAgent` -- the real thing: Anthropic SDK tool runner (`@beta_tool`), Claude does
                              the reasoning; each tool call executes a computational/physical experiment.

Design rules that matter in a real lab  [HKQAI JD: AI agents that coordinate experiments]
  1. Tools return *compact JSON summaries*, never raw arrays -- the LLM reasons over statistics
     (max posterior sd, IPV fraction, best value), not over 288 floats.
  2. Tools are *idempotent and budget-aware*: a call that would overspend the budget is refused
     with an explanatory error string, so the model can re-plan instead of crashing the loop.
  3. The LLM never touches hardware directly; it can only call `run_experiment` / `run_bo_campaign`,
     which go through `SeaLevelLabEnv.step` -- the single audited path to the physical world.
"""
from __future__ import annotations

import inspect
import json
from typing import Callable

import numpy as np

from bo_agent import bo_steps, fit_surrogate, surrogate_report
from sea_level_env import SeaLevelLabEnv
from sea_level_twin import to_unit

_PY2JSON = {int: "integer", float: "number", str: "string", bool: "boolean"}


# --------------------------------------------------------------------------- @tool (LangChain-style)
def tool(fn: Callable) -> Callable:
    """Minimal `@tool`: derives the JSON schema Anthropic / LangChain / OpenAI all expect from the
    signature + docstring.  Same shape as `anthropic.beta_tool`, so the toolbox is provider-agnostic."""
    sig = inspect.signature(fn, eval_str=True)          # eval_str: resolve `from __future__ import annotations`
    props, required = {}, []
    for name, p in sig.parameters.items():
        ann = p.annotation if p.annotation is not inspect._empty else str
        props[name] = {"type": _PY2JSON.get(ann, "string")}
        if p.default is inspect._empty:
            required.append(name)
    fn.tool_spec = {"name": fn.__name__, "description": inspect.getdoc(fn) or "",
                    "input_schema": {"type": "object", "properties": props, "required": required}}
    return fn


class LabToolbox:
    """Binds the lab (env + BO code) to a set of callable tools.  `functions` is what you hand to
    an LLM framework; `call()` is what the framework invokes."""

    def __init__(self, env: SeaLevelLabEnv):
        self.env = env
        self.calls: list[dict] = []

        @tool
        def describe_lab() -> str:
            """Describe the self-driving sea-level lab: target property, design-space bounds, the 13 existing
            tide gauges, fitted surrogate hyper-parameters, remaining experiment budget and the current belief.
            """
            return json.dumps(dict(twin=env.twin.describe(), belief=env.belief_summary()), indent=1)

        @tool
        def run_experiment(lon_deg: float, lat_deg: float) -> str:
            """Run ONE physical experiment (measure the sea-level-rise rate, mm/yr) at a location.

            Args:
                lon_deg: Longitude in degrees East, 140-210 (use 190 for 170 W).
                lat_deg: Latitude in degrees, -25 to 12.
            """
            if env.t >= env.budget:
                return json.dumps({"error": "experiment budget exhausted; call final_report"})
            x = to_unit(np.array([lon_deg]), np.array([lat_deg]))[0]
            obs, r, term, trunc, info = env.step(x)
            return json.dumps(dict(measured_mm_yr=round(info["y"], 2), at_deg=info["x_deg"], reward=round(r, 3),
                                   belief=env.belief_summary()))

        @tool
        def run_bo_campaign(mode: str, n_experiments: int) -> str:
            """Run a closed-loop Bayesian-optimisation campaign of n experiments with BoTorch.

            Args:
                mode: "active_learning" (reduce map uncertainty / choose gauge sites) or "optimization" (find the highest rise-rate hot-spot).
                n_experiments: Number of physical experiments to spend (must not exceed remaining budget).
            """
            if mode not in ("active_learning", "optimization"):
                return json.dumps({"error": "mode must be 'active_learning' or 'optimization'"})
            if n_experiments > env.budget - env.t:
                return json.dumps({"error": f"only {env.budget - env.t} experiments left"})
            before = env.belief_summary()
            hist = bo_steps(env, int(n_experiments), mode=mode)
            return json.dumps(dict(ran=len(hist), mode=mode,
                                   ipv_fraction_before=before["ipv_fraction_remaining"],
                                   ipv_fraction_after=round(hist[-1]["ipv_fraction"], 3),
                                   experiments=[dict(at_deg=h["x_deg"], mm_yr=round(h["y"], 2)) for h in hist],
                                   belief=env.belief_summary()))

        @tool
        def surrogate_hyperparameters() -> str:
            """Refit the BoTorch surrogate on everything measured so far and report its Matern length-scales
            (degrees), signal variance and noise -- comparable to INLA's range / sigma / precision.
            """
            X, y = env.observed_data()
            if len(y) < 3:
                return json.dumps({"error": "need >= 3 observations to fit a surrogate"})
            return json.dumps(surrogate_report(fit_surrogate(X, y)))

        @tool
        def final_report() -> str:
            """Summarise the campaign: experiments run, uncertainty removed, best site found."""
            s = env.belief_summary()
            s["experiments"] = [dict(at_deg=h["x_deg"], mm_yr=round(h["y"], 2)) for h in env.history]
            return json.dumps(s, indent=1)

        self.functions = [describe_lab, run_experiment, run_bo_campaign, surrogate_hyperparameters, final_report]
        self._by_name = {f.__name__: f for f in self.functions}

    @property
    def specs(self) -> list[dict]:
        return [f.tool_spec for f in self.functions]

    def call(self, name: str, **kwargs) -> str:
        out = self._by_name[name](**kwargs)
        self.calls.append(dict(tool=name, args=kwargs, result=out))
        return out


# --------------------------------------------------------------------------- Mock ReAct agent
class MockScientistAgent:
    """A scripted stand-in for the LLM so the agentic loop is testable offline.  The *reasoning*
    is a faithful sketch of what the real prompt asks Claude to do: characterise first (reduce
    uncertainty), exploit second (find the hot-spot), then report -- re-planning on tool errors."""

    def __init__(self, toolbox: LabToolbox, min_gain_per_experiment: float = 0.012, reserve: int = 5):
        # keep characterising while each experiment still removes > 1.2% of the initial map variance
        self.tb, self.min_gain, self.reserve = toolbox, min_gain_per_experiment, reserve
        self.trace: list[str] = []

    def _log(self, kind: str, text: str):
        line = f"{kind:<12}{text}"
        self.trace.append(line); print(line)

    @staticmethod
    def _compact(out: dict) -> dict:
        """What the (mock) LLM actually reads back: a handful of numbers, not the full JSON."""
        if "twin" in out:
            b = out["belief"]
            return dict(gp_hyper=out["twin"]["gp_hyper"], budget_left=b["budget_left"], top_uncertainty=b["uncertainty_hotspots"][0])
        d = {k: v for k, v in out.items() if k not in ("belief", "experiments")}
        if "belief" in out:
            d["belief"] = {k: out["belief"][k] for k in ("budget_left", "ipv_fraction_remaining", "best_measured_mm_yr", "best_measured_at_deg")}
        return d

    def act(self, name: str, **kw) -> dict:
        self._log("Action:", f"{name}({', '.join(f'{k}={v!r}' for k, v in kw.items())})")
        out = json.loads(self.tb.call(name, **kw))
        self._log("Observation:", json.dumps(self._compact(out))[:400])
        return out

    def run(self, goal: str) -> list[str]:
        self._log("Goal:", goal)
        self._log("Thought:", "I need the lab's state before spending budget: bounds, existing gauges, current uncertainty.")
        lab = self.act("describe_lab")
        sig = lab["twin"]["gp_hyper"]["signal_sd"]
        belief = lab["belief"]
        rounds, gain_per_exp = 0, 1.0
        while belief["budget_left"] > self.reserve and gain_per_exp > self.min_gain and rounds < 3:
            hs = belief["uncertainty_hotspots"][0]
            self._log("Thought:", f"IPV is still {belief['ipv_fraction_remaining']:.3f} of the prior and the least-known cell is ({hs['lon']}E, {hs['lat']}) "
                                  f"with sd {hs['posterior_sd']} mm/yr ({hs['posterior_sd']/sig:.2f} signal-sd). Spend 3 experiments on active learning.")
            res = self.act("run_bo_campaign", mode="active_learning", n_experiments=3)
            gain_per_exp = (res["ipv_fraction_before"] - res["ipv_fraction_after"]) / res["ran"]
            self._log("Thought:", f"IPV fell {res['ipv_fraction_before']:.3f} -> {res['ipv_fraction_after']:.3f}: {100*gain_per_exp:.1f}% of prior variance per experiment"
                                  + (" -- diminishing returns, stop characterising." if gain_per_exp <= self.min_gain else " -- still worth it."))
            belief = res["belief"]; rounds += 1
        left = belief["budget_left"]
        pk = belief["predicted_peaks"][0]
        self._log("Thought:", f"Uncertainty is acceptable; the belief predicts the highest rise rate near ({pk['lon']}E, {pk['lat']}) "
                              f"({pk['posterior_mean']} +/- {pk['posterior_sd']} mm/yr). Switch objective to hot-spot search with the remaining {left} experiments.")
        res = self.act("run_bo_campaign", mode="optimization", n_experiments=left)
        b = res["belief"]
        self._log("Thought:", f"Best measured rate {b['best_measured_mm_yr']} mm/yr at {b['best_measured_at_deg']}. Check surrogate physics before reporting.")
        hp = self.act("surrogate_hyperparameters")
        prior_ls = lab["twin"]["gp_hyper"]["lengthscale_deg_lon"]
        verdict = ("consistent with the basin-scale coherence (ENSO/PDO) assumed by the twin" if min(hp["lengthscale_deg"]) >= 0.4 * prior_ls
                   else "much shorter than the twin's prior range -> the hot-spot may be a local (tectonic) effect; flag for a follow-up campaign")
        self._log("Thought:", f"Refit length-scales {hp['lengthscale_deg']} deg vs prior {prior_ls} deg: {verdict}.")
        self.act("final_report")
        self._log("Answer:", f"Campaign complete: {b['n_experiments']} experiments, IPV reduced to {b['ipv_fraction_remaining']} of start, "
                             f"hot-spot {b['best_measured_mm_yr']} mm/yr at {b['best_measured_at_deg']}.")
        return self.trace


# --------------------------------------------------------------------------- Live Claude agent (optional)
SYSTEM_PROMPT = """You are the scientist-in-the-loop of a self-driving sea-level laboratory in the tropical Pacific.
The lab measures the sea-level-rise rate (mm/yr) at any location, but every experiment costs budget.
Work in a plan -> experiment -> interpret loop using the tools. Characterise uncertainty before exploiting.
Explain each decision in one or two sentences before calling a tool, quote the numbers the tools return,
and finish with final_report and a short written conclusion for a coastal-adaptation planner."""


class ClaudeScientistAgent:
    """Anthropic SDK tool runner around the same toolbox.  Requires credentials
    (ANTHROPIC_API_KEY or `ant auth login`).  Model: claude-opus-5 with server-side fallbacks."""

    def __init__(self, toolbox: LabToolbox, model: str = "claude-opus-5"):
        import anthropic
        from anthropic import beta_tool
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools = [beta_tool(f) for f in toolbox.functions]      # schema generated from signature + docstring

    def run(self, goal: str) -> list[str]:
        trace = []
        runner = self.client.beta.messages.tool_runner(
            model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, tools=self.tools,
            messages=[{"role": "user", "content": goal}],
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        for message in runner:                                       # SDK executes tools and loops until end_turn
            for block in message.content:
                if block.type == "text":
                    trace.append(f"Claude:      {block.text}"); print(trace[-1])
                elif block.type == "tool_use":
                    trace.append(f"Action:      {block.name}({json.dumps(block.input)})"); print(trace[-1])
        return trace


if __name__ == "__main__":
    from sea_level_twin import DigitalTwin
    env = SeaLevelLabEnv(DigitalTwin(), mode="active_learning", budget=15)
    env.reset(seed=0)
    tb = LabToolbox(env)
    print(json.dumps(tb.specs[2], indent=1))
    MockScientistAgent(tb).run("Characterise the sea-level-rise field with 15 experiments and locate the highest-risk site.")
