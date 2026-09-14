"""
sea_level_twin.py
=================
A *digital twin* of the Pacific sea-level-rise field, built from the SAME CSV that
`flood_final.qmd` analyses (`Processed_Final_data.csv`, 13 tide-gauge stations, 1992-2025).

Statistical lineage (the "Math Bridge")
---------------------------------------
`flood_final.qmd` fits, with R-INLA,

        y_i = beta_0 + beta_1 * covar_i + u(s_i) + eps_i,        u ~ SPDE-Matern(alpha = 2)

The SPDE  (kappa^2 - Laplacian)^(alpha/2) (tau u) = W  has, by Lindgren, Rue & Lindstrom (2011),
a stationary solution that is a Gaussian process with Matern covariance

        C(h) = sigma^2 * 2^(1-nu) / Gamma(nu) * (kappa h)^nu * K_nu(kappa h),   nu = alpha - d/2

so alpha = 2 in d = 2 gives nu = 1, and the INLA "range"  rho = sqrt(8 nu) / kappa.
INLA represents u through a *sparse precision matrix Q* (GMRF on the mesh) and integrates
hyper-parameters with a Laplace approximation (empirical Bayes when int.strategy = "eb").

Here the same GP is written with a *dense covariance matrix K* and its hyper-parameters are
fitted by maximising the log marginal likelihood -- i.e. type-II maximum likelihood, which is
exactly INLA's `int.strategy = "eb"` and exactly what BoTorch's `fit_gpytorch_mll` does.
That is the whole bridge: the object the statistician calls a "latent Matern field" is the
object the AI4S engineer calls a "GP surrogate" inside a Bayesian-optimisation loop.

Why a twin at all?  A self-driving lab needs a simulator to develop and unit-test its
decision layer *before* touching hardware.  This module is that simulator: the physical
"experiment" is `DigitalTwin.measure(x)` (deploy a gauge / run a campaign at location x and
read back a noisy sea-level-rise rate).  [HKQAI JD: "integrate ML with scientific computing"]
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.special import gamma as _gamma, kv as _kv

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "Processed_Final_data.csv"

# Study domain (degrees).  Longitudes are mapped to [0, 360) because the data straddle the
# antimeridian (Cook Islands -157.8 -> 202.2).  Same box the INLA mesh covered.
LON_MIN, LON_MAX = 140.0, 210.0
LAT_MIN, LAT_MAX = -25.0, 12.0


# ----------------------------------------------------------------------------------------
# 1. Data: the identical cleaning the R code does, plus the derived "target property"
# ----------------------------------------------------------------------------------------
def load_station_table(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """One row per tide gauge with the quantities a materials scientist would call
    'measured properties': datum-relative mean level, linear rise rate (mm/yr), and the
    residual noise level after removing trend + annual cycle.

    The rise rate is the scientifically comparable target: monthly `Mean` is relative to each
    gauge's own datum (Fiji 1.28 m vs Solomon 0.71 m is a datum offset, not physics), which is
    one honest limitation of regressing raw `Mean` on space in `flood_final.qmd`.
    """
    df = pd.read_csv(csv_path)
    for c in ["Month", "Year", "Mean", "Std_Dev"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Month", "Year", "Mean", "Latitude", "Longitude"])
    df = df[(df.Year.between(1990, 2030)) & (df.Month.between(1, 12))]
    df = df.drop_duplicates(["Location", "Year", "Month"])
    df["t"] = df.Year + (df.Month - 0.5) / 12.0
    df["lon360"] = np.where(df.Longitude < 0, df.Longitude + 360.0, df.Longitude)

    rows = []
    for loc, g in df.groupby("Location"):
        # y = a + b (t - 2000) + annual harmonic  -> b is the rise rate
        X = np.c_[np.ones(len(g)), g.t - 2000.0, np.sin(2 * np.pi * g.t), np.cos(2 * np.pi * g.t)]
        beta, *_ = np.linalg.lstsq(X, g.Mean.values, rcond=None)
        resid = g.Mean.values - X @ beta
        rows.append(dict(
            station=loc, n_months=len(g),
            lat=float(g.Latitude.iloc[0]), lon360=float(g.lon360.iloc[0]),
            mean_level_m=float(g.Mean.mean()),
            trend_mm_yr=float(beta[1] * 1000.0),
            resid_sd_m=float(resid.std()),
            tidal_sd_m=float(g.Std_Dev.mean()),
            year_min=int(g.Year.min()), year_max=int(g.Year.max()),
        ))
    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


def to_unit(lon360: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Degrees -> normalised design space [0,1]^2 (what BoTorch/Gymnasium expect)."""
    return np.c_[(np.asarray(lon360) - LON_MIN) / (LON_MAX - LON_MIN),
                 (np.asarray(lat) - LAT_MIN) / (LAT_MAX - LAT_MIN)]


def to_degrees(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.c_[LON_MIN + x[:, 0] * (LON_MAX - LON_MIN), LAT_MIN + x[:, 1] * (LAT_MAX - LAT_MIN)]


# ----------------------------------------------------------------------------------------
# 2. The Matern GP, written out explicitly (dense twin of the INLA SPDE field)
# ----------------------------------------------------------------------------------------
def matern(d: np.ndarray, lengthscale: float, nu: float) -> np.ndarray:
    """Matern correlation for distance matrix d.  nu=1 is the SPDE alpha=2 field of the R code;
    nu=1.5 / 2.5 are the closed forms BoTorch uses by default."""
    r = np.asarray(d) / lengthscale
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        s = np.sqrt(3.0) * r
        return (1.0 + s) * np.exp(-s)
    if nu == 2.5:
        s = np.sqrt(5.0) * r
        return (1.0 + s + s**2 / 3.0) * np.exp(-s)
    s = np.sqrt(2.0 * nu) * r
    out = np.ones_like(s)
    m = s > 0
    out[m] = (2 ** (1 - nu) / _gamma(nu)) * s[m] ** nu * _kv(nu, s[m])
    return out


@dataclasses.dataclass
class GPHyper:
    lengthscale: float   # in unit-square coordinates; INLA range rho = lengthscale*sqrt(8nu)/sqrt(2nu)... see README
    signal_sd: float     # sigma  (INLA: sigma0)
    noise_sd: float      # nugget (INLA: 1/sqrt(precision of the Gaussian family))
    mean: float          # constant mean (INLA: the `intercept` fixed effect)
    nu: float = 1.5


class MaternGP:
    """Exact GP regression with a constant mean -- the dense-covariance twin of
    `y ~ -1 + intercept + f(field, model = spde)` in flood_final.qmd."""

    def __init__(self, hyper: GPHyper):
        self.h = hyper
        self.X = np.zeros((0, 2))
        self.y = np.zeros(0)
        self._L = None
        self._alpha = None

    # --- the same three quantities INLA's inla.spde2.result reports ---------------------
    @property
    def inla_range(self) -> float:
        """Lindgren's practical range rho = sqrt(8 nu)/kappa, expressed in unit coordinates."""
        return self.h.lengthscale * np.sqrt(8 * self.h.nu) / np.sqrt(2 * self.h.nu)

    def kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        return self.h.signal_sd**2 * matern(d, self.h.lengthscale, self.h.nu)

    def set_data(self, X: np.ndarray, y: np.ndarray) -> "MaternGP":
        self.X = np.asarray(X, float).reshape(-1, 2)
        self.y = np.asarray(y, float).reshape(-1)
        n = len(self.y)
        if n == 0:
            self._L = self._alpha = None
            return self
        K = self.kernel(self.X, self.X) + (self.h.noise_sd**2 + 1e-8) * np.eye(n)
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, self.y - self.h.mean))
        return self

    def add(self, x: np.ndarray, y: float) -> "MaternGP":
        return self.set_data(np.vstack([self.X, np.atleast_2d(x)]), np.append(self.y, y))

    def posterior(self, Xs: np.ndarray, full_cov: bool = False):
        """Posterior mean and sd -- the two maps `summary.fitted.values$mean / $sd` the R code
        plots.  The sd map is the raw material of every acquisition function below."""
        Xs = np.atleast_2d(Xs)
        Kss = self.kernel(Xs, Xs)
        if self._L is None:
            mu = np.full(len(Xs), self.h.mean)
            return (mu, Kss) if full_cov else (mu, np.sqrt(np.diag(Kss)))
        Ks = self.kernel(Xs, self.X)
        mu = self.h.mean + Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        cov = Kss - v.T @ v
        if full_cov:
            return mu, cov
        return mu, np.sqrt(np.clip(np.diag(cov), 1e-12, None))

    def log_marginal_likelihood(self) -> float:
        if self._L is None:
            return 0.0
        r = self.y - self.h.mean
        return float(-0.5 * r @ self._alpha - np.log(np.diag(self._L)).sum() - 0.5 * len(r) * np.log(2 * np.pi))


# Weakly-informative priors on log-hyper-parameters.  With only 13 gauges the marginal
# likelihood cannot separate "short range + big nugget" from "long range + small nugget"
# (range/sigma are only jointly identifiable: Zhang 2004).  R-INLA solves this with PC priors
# on (range, sigma) -- the `theta.prior.mean / theta.prior.prec` lines in flood_final.qmd.
# We do the same with log-normal priors, i.e. MAP-II instead of ML-II.  Basin-scale sea-level
# trend patterns (ENSO / PDO) are coherent over ~15-30 deg of longitude, hence the prior mode
# of 0.25 (unit box) = 17.5 deg.  Samoa's 9.8 mm/yr post-2009-earthquake subsidence is local
# tectonics, which the nugget absorbs instead of bending the whole field.
DEFAULT_PRIORS = dict(log_lengthscale=(np.log(0.25), 0.5), log_noise_sd=(np.log(0.8), 0.5))


def fit_hyperparameters(X: np.ndarray, y: np.ndarray, nu: float = 1.5, seed: int = 0,
                        priors: dict | None = DEFAULT_PRIORS) -> GPHyper:
    """Type-II maximum a-posteriori (priors=None -> type-II maximum likelihood, which is
    INLA `int.strategy="eb"` and BoTorch `fit_gpytorch_mll`).  Optimised in log-space from
    random restarts because the surface of a 13-point spatial GP is multimodal."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)

    def nll(theta):
        ls, ssd, nsd = np.exp(theta)
        gp = MaternGP(GPHyper(ls, ssd, nsd, y.mean(), nu)).set_data(X, y)
        val = -gp.log_marginal_likelihood()
        if priors:
            m, s = priors["log_lengthscale"]; val += 0.5 * ((theta[0] - m) / s) ** 2
            m, s = priors["log_noise_sd"];    val += 0.5 * ((theta[2] - m) / s) ** 2
        return val

    best = None
    for _ in range(12):
        x0 = np.log([rng.uniform(0.1, 0.8), y.std() * rng.uniform(0.5, 1.5), y.std() * rng.uniform(0.05, 0.5)])
        res = minimize(nll, x0, method="L-BFGS-B",
                       bounds=[(np.log(0.05), np.log(2.0)), (np.log(1e-2), np.log(50)), (np.log(1e-2), np.log(10))])
        if best is None or res.fun < best.fun:
            best = res
    ls, ssd, nsd = np.exp(best.x)
    return GPHyper(float(ls), float(ssd), float(nsd), float(y.mean()), nu)


# ----------------------------------------------------------------------------------------
# 3. The digital twin: a hidden ground-truth field + a noisy "experiment" oracle
# ----------------------------------------------------------------------------------------
class DigitalTwin:
    """Ground truth = one posterior draw of the station-fitted GP (so it honours the 13 real
    gauges but has unknown structure between them, like the real ocean).  Each `reset` of the
    Gymnasium env draws a new truth -> the RL policy must generalise, not memorise.

    `measure(x)` is the *physical experiment*: it is the only channel through which an agent
    learns about the truth, and each call costs budget.  [HKQAI JD: coordinate physical and
    computational experiments -- the twin is the physical side, the GP belief is the
    computational side]
    """

    def __init__(self, csv_path: str | Path = DEFAULT_CSV, nu: float = 1.5, grid: int = 41,
                 noise_sd_mm_yr: float = 0.6, seed: int = 0):
        self.stations = load_station_table(csv_path)
        self.X_st = to_unit(self.stations.lon360.values, self.stations.lat.values)
        self.y_st = self.stations.trend_mm_yr.values.copy()
        self.hyper = fit_hyperparameters(self.X_st, self.y_st, nu=nu, seed=seed)
        self.prior_gp = MaternGP(self.hyper).set_data(self.X_st, self.y_st)
        self.noise_sd = noise_sd_mm_yr
        self.grid_n = grid
        g = np.linspace(0, 1, grid)
        self.gx, self.gy = g, g
        GX, GY = np.meshgrid(g, g, indexing="ij")
        self.grid_pts = np.c_[GX.ravel(), GY.ravel()]
        self._mu_g, self._cov_g = self.prior_gp.posterior(self.grid_pts, full_cov=True)
        self._chol = np.linalg.cholesky(self._cov_g + 1e-6 * np.eye(len(self._mu_g)))
        self.truth_grid = None
        self._interp = None
        self.sample_truth(np.random.default_rng(seed))

    def sample_truth(self, rng: np.random.Generator) -> None:
        z = rng.standard_normal(len(self._mu_g))
        self.truth_grid = (self._mu_g + self._chol @ z).reshape(self.grid_n, self.grid_n)
        self._interp = RegularGridInterpolator((self.gx, self.gy), self.truth_grid, bounds_error=False, fill_value=None)

    def f_true(self, x: np.ndarray) -> np.ndarray:
        return self._interp(np.clip(np.atleast_2d(x), 0, 1))

    def measure(self, x: np.ndarray, rng: np.random.Generator) -> float:
        """Run one experiment at location x (unit coords).  Returns a noisy rise rate in mm/yr."""
        return float(self.f_true(x)[0] + rng.normal(0.0, self.noise_sd))

    @property
    def true_max(self) -> tuple[float, np.ndarray]:
        i = int(np.argmax(self.truth_grid))
        return float(self.truth_grid.ravel()[i]), self.grid_pts[i]

    def describe(self) -> dict:
        h = self.hyper
        return dict(
            n_stations=int(len(self.stations)),
            target="sea-level rise rate (mm/yr) from 1992-2025 tide gauges",
            domain_deg=dict(lon=[LON_MIN, LON_MAX], lat=[LAT_MIN, LAT_MAX]),
            gp_hyper=dict(lengthscale_unit=round(h.lengthscale, 3),
                          lengthscale_deg_lon=round(h.lengthscale * (LON_MAX - LON_MIN), 1),
                          inla_practical_range_unit=round(self.prior_gp.inla_range, 3),
                          signal_sd=round(h.signal_sd, 3), noise_sd=round(h.noise_sd, 3),
                          mean_mm_yr=round(h.mean, 3), nu=h.nu),
            station_trends_mm_yr={r.station: round(r.trend_mm_yr, 2) for r in self.stations.itertuples()},
        )


if __name__ == "__main__":
    tw = DigitalTwin()
    import json
    print(json.dumps(tw.describe(), indent=2))
    print("true max rise rate on this draw: %.2f mm/yr at %s deg" % (tw.true_max[0], to_degrees(tw.true_max[1])[0].round(1)))
