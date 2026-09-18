#!/usr/bin/env Rscript
# app/build_app.R -- builds docs/app.html, an interactive 3D query app for
#
#     P( annual maximum sea level exceeds this station's own 1-in-N baseline level )
#
# The quantity is defined per tide gauge in its OWN datum: the project has no DEM and no absolute
# elevations, so nothing here is a probability that any land is flooded, and nothing must be described that way.
#
# Statistics: a non-stationary GEV on de-trended annual maxima of the `Maximum` column
# (shared shape xi, station location mu0_s and scale sigma_s), with the trend beta(x) and
# log sigma(x) carried across space by the Matern GP shared with flood_ai4s.qmd (R/gp_core.R).
# Everything is computed here from Processed_Final_data.csv; the qmd is NOT re-rendered.
#
# Output: one self-contained page (plotly.js embedded by the R package, zero network requests).
#
#     Rscript app/build_app.R

suppressPackageStartupMessages({ library(dplyr); library(plotly); library(htmlwidgets); library(jsonlite) })
if (!nzchar(Sys.getenv("RSTUDIO_PANDOC")) && !rmarkdown::pandoc_available()) {   # selfcontained = TRUE needs pandoc
  cand <- Sys.glob("/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/*/pandoc")
  if (length(cand)) Sys.setenv(RSTUDIO_PANDOC = dirname(cand[1]))
}
setwd(normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])), "..")))
source("R/gp_core.R")
set.seed(2026)

# ======================================================================================
# 1. Data: identical cleaning to the `data` chunk of flood_ai4s.qmd
# ======================================================================================
raw <- read.csv("Processed_Final_data.csv")
data <- raw |>
  mutate(across(c(Month, Year, Mean, Std_Dev), ~ suppressWarnings(as.numeric(.x)))) |>
  filter(!is.na(Mean), !is.na(Std_Dev), !is.na(Latitude), !is.na(Longitude),
         between(Year, 1990, 2030), between(Month, 1, 12)) |>
  distinct(Location, Year, Month, .keep_all = TRUE) |>
  mutate(t = Year + (Month - 0.5) / 12,
         lon360 = ifelse(Longitude < 0, Longitude + 360, Longitude),
         Maximum = suppressWarnings(as.numeric(Maximum)))
cat(sprintf("data: %d raw rows -> %d clean rows, %d stations\n", nrow(raw), nrow(data), n_distinct(data$Location)))

# ======================================================================================
# 2. Rise rate per station (trend + annual cycle), as in the `rates` chunk
# ======================================================================================
rates <- data |> group_by(Location) |> group_modify(~ {
  m <- lm(Mean ~ I(t - 2000) + sin(2 * pi * t) + cos(2 * pi * t), data = .x)
  tibble(lon = first(.x$lon360), lat = first(.x$Latitude), n_months = nrow(.x), trend_mm_yr = 1000 * coef(m)[[2]])
}) |> ungroup() |> arrange(Location)

# ======================================================================================
# 3. Annual maxima of `Maximum`, de-trended to the year-2000 baseline of each station
# ======================================================================================
annual <- data |> filter(!is.na(Maximum)) |>
  inner_join(rates |> select(Location, trend_mm_yr), by = "Location") |>
  mutate(M_star = Maximum - (trend_mm_yr / 1000) * (t - 2000)) |>
  group_by(Location, Year) |> summarise(n_months = n(), amax = max(M_star), .groups = "drop") |>
  filter(n_months >= 8) |> arrange(Location, Year)
amax_by_station <- split(annual$amax, factor(annual$Location, levels = rates$Location))   # same station order as `rates`
stopifnot(identical(names(amax_by_station), rates$Location))
S <- nrow(rates)
cat(sprintf("annual maxima: %d station-years (%d-%d per station)\n", nrow(annual), min(table(annual$Location)), max(table(annual$Location))))

# ======================================================================================
# 4. Joint GEV maximum likelihood: shared xi, per-station mu0_s and sigma_s  (27 parameters)
#    Hand-written: no extreme-value package is required.
# ======================================================================================
gev_nll <- function(par) {
  mu <- par[1:S]; sigma <- par[S + 1:S]; xi <- par[2 * S + 1]
  nll <- 0
  for (s in seq_len(S)) {
    w <- (amax_by_station[[s]] - mu[s]) / sigma[s]
    if (abs(xi) < 1e-6) { nll <- nll + sum(log(sigma[s]) + w + exp(-w)); next }      # Gumbel limit
    z <- 1 + xi * w
    if (any(z <= 0)) return(1e10)                                                    # outside the GEV support
    nll <- nll + sum(log(sigma[s]) + (1 + 1 / xi) * log(z) + z^(-1 / xi))
  }
  nll
}
sigma0 <- sapply(amax_by_station, sd) * sqrt(6) / pi                                 # moment matching (Gumbel)
mu0    <- sapply(amax_by_station, mean) - 0.5772 * sigma0
par0   <- c(mu0, sigma0, 0.05)
gev <- optim(par0, gev_nll, method = "L-BFGS-B",
             lower = c(rep(-Inf, S), rep(1e-3, S), -0.4), upper = c(rep(Inf, S), rep(Inf, S), 0.4),
             control = list(parscale = c(rep(1, S), rep(0.1, S), 0.1), maxit = 2000))
if (gev$convergence != 0) stop("GEV joint MLE did not converge: ", gev$message)
xi      <- gev$par[2 * S + 1]
mu0_s   <- gev$par[1:S]                # location at the year-2000 baseline, in each station's OWN datum
sigma_s <- gev$par[S + 1:S]            # scale (m)
cat(sprintf("GEV: converged, nll = %.2f, shared xi = %.3f, sigma range %.3f-%.3f m\n", gev$value, xi, min(sigma_s), max(sigma_s)))

# ======================================================================================
# 5. Spatial fields with the shared Matern GP (R/gp_core.R)
#    beta(x)      : rise rate, mm/yr           -> mu_beta, sd_beta
#    log sigma(x) : log GEV scale              -> mu_ls, sd_ls
#    mu0_s is deliberately NOT interpolated: it contains each gauge's datum offset (Fiji 1.28 m vs
#    Solomon 0.71 m is a benchmark difference, not oceanography), so it is not comparable across
#    stations. The closed form below only ever needs beta(x), sigma(x) and xi.
# ======================================================================================
NX <- 160; NY <- 120
g_lon <- seq(LON[1], LON[2], length.out = NX); g_lat <- seq(LAT[1], LAT[2], length.out = NY)
grid_deg <- as.matrix(expand.grid(lon = g_lon, lat = g_lat))       # lon varies fastest: index = (i_lat-1)*NX + j_lon
grid     <- to_unit(grid_deg[, 1], grid_deg[, 2])
X_st     <- to_unit(rates$lon, rates$lat)

hyp_beta <- gp_fit(X_st, rates$trend_mm_yr)                        # same call as the qmd (default priors)
f_beta   <- gp_post_diag(X_st, rates$trend_mm_yr, grid, hyp_beta)
y_ls     <- log(sigma_s)
hyp_ls   <- gp_fit(X_st, y_ls, prior = list(ls = c(log(0.25), 0.5), noise = c(log(0.4 * sd(y_ls)), 0.5)))  # noise prior rescaled to the units of log sigma
f_ls     <- gp_post_diag(X_st, y_ls, grid, hyp_ls)
support  <- pmin(pmax(1 - f_beta$sd / hyp_beta$sig, 0), 1)
cat(sprintf("GP beta: lengthscale %.3f, sig %.2f mm/yr, noise %.2f | GP log-sigma: lengthscale %.3f, sig %.3f, noise %.3f\n",
            hyp_beta$ls, hyp_beta$sig, hyp_beta$noise, hyp_ls$ls, hyp_ls$sig, hyp_ls$noise))
cat(sprintf("support: %.0f%% of the grid >= 0.25\n", 100 * mean(support >= 0.25)))

# ======================================================================================
# 6. Closed-form exceedance probability (the JS in the page uses the identical formula)
# ======================================================================================
p_exceed <- function(beta_mm_yr, sigma_m, xi, Y, N) {
  z_N   <- ((-log(1 - 1 / N))^(-xi) - 1) / xi           # standardised baseline return level, same everywhere
  delta <- (beta_mm_yr / 1000) * (Y - 2000) / sigma_m    # dimensionless shift of the location
  1 - exp(-(pmax(0, 1 + xi * (z_N - delta)))^(-1 / xi))
}
# P is monotone increasing in beta and decreasing in sigma, so the 90% band is obtained by plugging
# the 5% / 95% quantiles of beta and the opposite quantiles of sigma into the same formula.
p_band <- function(Y, N) list(
  med = p_exceed(f_beta$mu,                    exp(f_ls$mu),                  xi, Y, N),
  lo  = p_exceed(f_beta$mu - 1.645 * f_beta$sd, exp(f_ls$mu + 1.645 * f_ls$sd), xi, Y, N),
  hi  = p_exceed(f_beta$mu + 1.645 * f_beta$sd, exp(f_ls$mu - 1.645 * f_ls$sd), xi, Y, N))

# ======================================================================================
# 7. "Where should the lab measure next?" -- the IPV criterion of acq_ipv() in flood_ai4s.qmd,
#    evaluated on this grid in chunks (the full posterior covariance would be 19 200^2).
#    gain(c) = mean_g Cov(g, c)^2 / (Var(c) + noise^2), for the beta(x) GP.
# ======================================================================================
acq_ipv_chunked <- function(X, y, G, h, chunk = 800) {
  K <- k_matern(X, X, h$ls, h$sig) + diag(h$noise^2 + 1e-8, nrow(X)); L <- chol(K)
  Vg <- forwardsolve(t(L), t(k_matern(G, X, h$ls, h$sig)))          # nrow(X) x nG
  var_c <- gp_post_diag(X, y, G, h)$sd^2
  gain <- numeric(nrow(G))
  for (s in seq(1, nrow(G), by = chunk)) {
    idx <- s:min(s + chunk - 1, nrow(G))
    Cov <- k_matern(G, G[idx, , drop = FALSE], h$ls, h$sig) - crossprod(Vg, Vg[, idx, drop = FALSE])   # nG x chunk
    gain[idx] <- colMeans(Cov^2) / (var_c[idx] + h$noise^2)
  }
  gain
}
ipv_gain <- acq_ipv_chunked(X_st, rates$trend_mm_yr, grid, hyp_beta)
next_idx <- which.max(ipv_gain)
cat(sprintf("next measurement (max IPV reduction): lon %.1f, lat %.1f\n", grid_deg[next_idx, 1], grid_deg[next_idx, 2]))

# ======================================================================================
# 8. Station table at the initial view (Y = 2050, N = 10) with each station's own beta and sigma
# ======================================================================================
Y0 <- 2050; N0 <- 10
stations <- rates |> mutate(sigma_m = sigma_s, mu0_m = mu0_s,
                            P_2050_N10 = p_exceed(trend_mm_yr, sigma_s, xi, Y0, N0)) |>
  select(Location, lon, lat, trend_mm_yr, sigma_m, mu0_m, P_2050_N10)
cat("\nP(annual max > own 1-in-10 baseline level) in 2050, per station:\n")
print(as.data.frame(stations |> mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)

# ======================================================================================
# 9. Self-checks (any failure stops the build)
# ======================================================================================
P0 <- p_exceed(f_beta$mu, exp(f_ls$mu), xi, 2000, 10)
if (any(abs(P0 - 0.1) > 0.01)) stop("check 1: P(Y = 2000, N = 10) must equal 1/N everywhere")
prev <- p_exceed(f_beta$mu, exp(f_ls$mu), xi, 2000, 10)
for (Y in seq(2001, 2100, by = 1)) { cur <- p_exceed(f_beta$mu, exp(f_ls$mu), xi, Y, 10); if (any(cur < prev - 1e-12)) stop("check 2: P not monotone in Y"); prev <- cur }
prev <- p_exceed(f_beta$mu, exp(f_ls$mu), xi, 2050, 2)
for (N in c(5, 10, 20, 50)) { cur <- p_exceed(f_beta$mu, exp(f_ls$mu), xi, 2050, N); if (any(cur > prev + 1e-12)) stop("check 2: P not monotone in N"); prev <- cur }
for (Y in c(2000, 2025, 2050, 2075, 2100)) for (N in c(2, 5, 10, 20, 50)) {
  b <- p_band(Y, N)
  for (v in b) if (any(!is.finite(v)) || any(v < 0) || any(v > 1)) stop("check 3: P outside [0, 1] or NA at Y=", Y, " N=", N)
}
hi_st <- stations$P_2050_N10[which.max(stations$trend_mm_yr)]; lo_st <- stations$P_2050_N10[which.min(stations$trend_mm_yr)]
if (!(hi_st > lo_st)) stop("check 4: station with the highest trend must have the highest P")
if (gev$convergence != 0 || !(xi > -0.4 && xi < 0.4)) stop("check 5: GEV not converged or xi on the boundary")
cat("\nself-checks 1-5: all passed\n")

# ======================================================================================
# 10. The page: one plotly surface (height = P, colour = support) + gauges + next-measurement marker,
#     all interaction injected as JS through htmlwidgets::onRender. Zero network requests.
# ======================================================================================
sig4 <- function(v) signif(v, 4)
b0 <- p_band(Y0, N0)
z0 <- matrix(sig4(b0$med), nrow = NY, ncol = NX, byrow = TRUE)          # rows = lat, cols = lon (plotly convention)
c0 <- matrix(sig4(support), nrow = NY, ncol = NX, byrow = TRUE)
support_scale <- list(c(0, "#ececec"), c(0.2499, "#ececec"), c(0.25, "#bfe3ea"), c(0.6, "#2b9db0"), c(1, "#083a4d"))  # < 0.25 fades to grey

payload <- toJSON(list(
  nx = NX, ny = NY, lon = sig4(g_lon), lat = sig4(g_lat), xi = sig4(xi), sig_beta = sig4(hyp_beta$sig),
  mu_beta = sig4(f_beta$mu), sd_beta = sig4(f_beta$sd), mu_ls = sig4(f_ls$mu), sd_ls = sig4(f_ls$sd), support = sig4(support),
  stations = stations |> transmute(name = Location, lon = sig4(lon), lat = sig4(lat), beta = sig4(trend_mm_yr), sigma = sig4(sigma_m)),
  next_pt = list(idx = next_idx - 1L, lon = sig4(grid_deg[next_idx, 1]), lat = sig4(grid_deg[next_idx, 2])),
  init = list(Y = Y0, N = N0)
), digits = I(4), auto_unbox = TRUE)

js <- sprintf('function(el, x) {
  var D = %s;
  var NX = D.nx, NY = D.ny, XI = D.xi;

  /* ---- the same closed form as p_exceed() in app/build_app.R ---- */
  function pEx(beta, sigma, Y, N) {
    var zN = (Math.pow(-Math.log(1 - 1 / N), -XI) - 1) / XI;
    var delta = (beta / 1000) * (Y - 2000) / sigma;
    var s = Math.max(0, 1 + XI * (zN - delta));
    return 1 - Math.exp(-Math.pow(s, -1 / XI));
  }
  function fieldP(Y, N, which) {           /* which: 0 = median, -1 = lower 5%%, +1 = upper 95%% */
    var z = new Array(NY);
    for (var i = 0; i < NY; i++) {
      var row = new Array(NX);
      for (var j = 0; j < NX; j++) {
        var k = i * NX + j;
        var b = D.mu_beta[k] + which * 1.645 * D.sd_beta[k];
        var s = Math.exp(D.mu_ls[k] - which * 1.645 * D.sd_ls[k]);
        row[j] = pEx(b, s, Y, N);
      }
      z[i] = row;
    }
    return z;
  }
  function stationP(Y, N) { return D.stations.map(function(s) { return pEx(s.beta, s.sigma, Y, N); }); }
  function nextP(Y, N) { var k = D.next_pt.idx; return [pEx(D.mu_beta[k], Math.exp(D.mu_ls[k]), Y, N)]; }

  /* ---- layout: control bar on top, plot left, query panel right ---- */
  var css = document.createElement("style");
  css.textContent = "html,body{margin:0;height:100%%;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1b2a33;background:#f7f9fa}" +
    "#sl-wrap{display:flex;flex-direction:column;height:100vh}" +
    "#sl-bar{display:flex;flex-wrap:wrap;gap:18px;align-items:center;padding:10px 16px;border-bottom:1px solid #d5dde3;background:#fff}" +
    "#sl-bar h1{font-size:15px;font-weight:600;margin:0 16px 0 0}" +
    "#sl-bar label{font-size:13px;display:flex;align-items:center;gap:8px}" +
    "#sl-bar input[type=range]{width:260px}" +
    "#sl-main{display:flex;flex:1;min-height:0}" +
    "#sl-plot{flex:1;min-width:0}" +
    "#sl-panel{width:330px;flex:none;border-left:1px solid #d5dde3;background:#fff;display:flex;flex-direction:column}" +
    "#sl-panel .body{padding:14px 16px;overflow:auto;flex:1;font-size:13.5px;line-height:1.5}" +
    "#sl-panel h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#5a6b78;margin:0 0 10px}" +
    "#sl-panel dl{display:grid;grid-template-columns:118px 1fr;gap:4px 10px;margin:0}" +
    "#sl-panel dt{color:#5a6b78}#sl-panel dd{margin:0;font-variant-numeric:tabular-nums}" +
    "#sl-panel .big{font-size:26px;font-weight:600;color:#0e6f82;margin:6px 0 2px}" +
    "#sl-panel .warn{background:#fdecea;border-left:3px solid #c0392b;padding:8px 10px;margin:12px 0;font-size:12.5px}" +
    "#sl-panel .hint{color:#5a6b78;font-size:12.5px}" +
    "#sl-panel .disc{border-top:1px solid #d5dde3;padding:10px 16px;font-size:11.5px;color:#5a6b78;background:#f3f6f8}" +
    "@media (max-width:820px){#sl-main{flex-direction:column}#sl-panel{width:auto;border-left:none;border-top:1px solid #d5dde3;max-height:45vh}}";
  document.head.appendChild(css);

  var wrap = document.createElement("div"); wrap.id = "sl-wrap";
  wrap.innerHTML =
    "<div id=\\"sl-bar\\"><h1>P(annual maximum sea level exceeds this station\'s own 1-in-N baseline level)</h1>" +
    "<label>Year <input id=\\"sl-year\\" type=\\"range\\" min=\\"2000\\" max=\\"2100\\" step=\\"1\\" value=\\"" + D.init.Y + "\\"> <b id=\\"sl-year-v\\">" + D.init.Y + "</b></label>" +
    "<label>Baseline return period <select id=\\"sl-N\\"><option>2</option><option>5</option><option selected>10</option><option>20</option><option>50</option></select> years</label></div>" +
    "<div id=\\"sl-main\\"><div id=\\"sl-plot\\"></div><div id=\\"sl-panel\\"><div class=\\"body\\" id=\\"sl-body\\"></div>" +
    "<div class=\\"disc\\"><b>What this is.</b> The probability that a station\'s annual maximum sea level exceeds the level that had a 1-in-N chance per year around 2000, in that station\'s own datum. " +
    "The project holds no DEM and no absolute elevations, so this is not a probability that any land is flooded. Trend and scale between gauges come from a Gaussian-process interpolation; xi is a shared GEV shape.</div></div></div>";
  document.body.appendChild(wrap);
  document.getElementById("sl-plot").appendChild(el);
  el.style.width = "100%%"; el.style.height = "100%%";
  var empties = document.querySelectorAll("body > div:not(#sl-wrap)");
  for (var q = 0; q < empties.length; q++) if (!empties[q].textContent.trim()) empties[q].remove();
  Plotly.Plots.resize(el);
  window.addEventListener("resize", function() { Plotly.Plots.resize(el); });

  /* ---- controls ---- */
  var yearEl = document.getElementById("sl-year"), nEl = document.getElementById("sl-N");
  function cur() { return { Y: +yearEl.value, N: +nEl.value }; }
  var last = null;
  function refresh() {
    var c = cur(); document.getElementById("sl-year-v").textContent = c.Y;
    Plotly.restyle(el, { z: [fieldP(c.Y, c.N, 0)] }, [0]);
    Plotly.restyle(el, { z: [stationP(c.Y, c.N)] }, [1]);
    Plotly.restyle(el, { z: [nextP(c.Y, c.N)] }, [2]);
    if (last) showPoint(last.lon, last.lat);
  }
  yearEl.addEventListener("input", refresh); nEl.addEventListener("change", refresh);

  /* ---- click -> query panel ---- */
  function haversineKm(lon1, lat1, lon2, lat2) {
    var r = Math.PI / 180, dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
    var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * 6371 * Math.asin(Math.sqrt(a));
  }
  function fmtLon(lon) { return lon > 180 ? (360 - lon).toFixed(1) + "&deg;W" : lon.toFixed(1) + "&deg;E"; }
  function pct(v) { return (100 * v).toFixed(1) + "%%"; }
  function showPoint(lon, lat) {
    last = { lon: lon, lat: lat };
    var j = Math.round((lon - D.lon[0]) / (D.lon[NX - 1] - D.lon[0]) * (NX - 1));
    var i = Math.round((lat - D.lat[0]) / (D.lat[NY - 1] - D.lat[0]) * (NY - 1));
    j = Math.min(Math.max(j, 0), NX - 1); i = Math.min(Math.max(i, 0), NY - 1);
    var k = i * NX + j, c = cur();
    var med = pEx(D.mu_beta[k], Math.exp(D.mu_ls[k]), c.Y, c.N);
    var lo  = pEx(D.mu_beta[k] - 1.645 * D.sd_beta[k], Math.exp(D.mu_ls[k] + 1.645 * D.sd_ls[k]), c.Y, c.N);
    var hi  = pEx(D.mu_beta[k] + 1.645 * D.sd_beta[k], Math.exp(D.mu_ls[k] - 1.645 * D.sd_ls[k]), c.Y, c.N);
    var sup = D.support[k], best = null;
    D.stations.forEach(function(s) { var d = haversineKm(lon, lat, s.lon, s.lat); if (!best || d < best.d) best = { d: d, name: s.name }; });
    var html = "<h2>Query</h2>" +
      "<dl><dt>Location</dt><dd>" + fmtLon(D.lon[j]) + ", " + D.lat[i].toFixed(1) + "&deg;</dd>" +
      "<dt>Year / baseline</dt><dd>" + c.Y + " / 1-in-" + c.N + "</dd></dl>" +
      "<div class=\\"big\\">" + pct(med) + "</div><div class=\\"hint\\">P median &middot; 90%% credible interval " + pct(lo) + " &ndash; " + pct(hi) + "</div>" +
      "<dl style=\\"margin-top:12px\\"><dt>Rise rate</dt><dd>" + D.mu_beta[k].toFixed(2) + " &plusmn; " + D.sd_beta[k].toFixed(2) + " mm/yr</dd>" +
      "<dt>GEV scale</dt><dd>" + Math.exp(D.mu_ls[k]).toFixed(3) + " m</dd>" +
      "<dt>Data support</dt><dd>" + pct(sup) + "</dd>" +
      "<dt>Nearest gauge</dt><dd>" + best.name + ", " + Math.round(best.d) + " km</dd></dl>" +
      (sup < 0.25 ? "<div class=\\"warn\\"><b>Weak data support</b> &mdash; posterior has reverted to the prior. The estimate here is the basin-wide mean rate, not local evidence; treat the number as a placeholder until a gauge is deployed nearby.</div>" : "") +
      (k === D.next_pt.idx ? "<div class=\\"hint\\">This is where the lab should measure next (largest expected reduction of map uncertainty).</div>" : "");
    document.getElementById("sl-body").innerHTML = html;
  }
  el.on("plotly_click", function(d) { var p = d.points[0]; showPoint(+p.x, +p.y); });
  document.getElementById("sl-body").innerHTML = "<h2>Query</h2><div class=\\"hint\\">Click anywhere on the surface to read the probability, its 90%% credible interval, the data support and the nearest gauge. Move the year slider or change the baseline return period to recompute the whole field in the browser.</div>";
}', payload)

p <- plot_ly() |>
  add_surface(x = g_lon, y = g_lat, z = z0, surfacecolor = c0, cmin = 0, cmax = 1, colorscale = support_scale,
              colorbar = list(title = list(text = "data support"), len = 0.5), name = "P field",
              hovertemplate = "lon %{x:.1f}, lat %{y:.1f}<br>P = %{z:.3f}<extra>click for details</extra>") |>
  add_markers(x = stations$lon, y = stations$lat, z = sig4(stations$P_2050_N10), text = stations$Location, name = "tide gauge (own beta, sigma)",
              marker = list(color = "white", size = 5, line = list(color = "#1b2a33", width = 1.5)),
              hovertemplate = "%{text}<br>P = %{z:.3f}<extra></extra>") |>
  add_markers(x = grid_deg[next_idx, 1], y = grid_deg[next_idx, 2], z = sig4(b0$med[next_idx]), name = "where the lab should measure next (max IPV reduction)",
              marker = list(color = "#d2573a", size = 8, symbol = "diamond", line = list(color = "#1b2a33", width = 1)),
              hovertemplate = "next measurement<br>lon %{x:.1f}, lat %{y:.1f}<extra></extra>") |>
  layout(showlegend = TRUE, legend = list(orientation = "h", y = 0.02, x = 0.02, bgcolor = "rgba(255,255,255,.7)"),
         margin = list(l = 0, r = 0, t = 10, b = 0),
         scene = list(xaxis = list(title = "longitude (deg E)"), yaxis = list(title = "latitude"),
                      zaxis = list(title = "P(exceed own 1-in-N baseline)", range = c(0, 1)),
                      aspectratio = list(x = 1.6, y = 1, z = 0.6), camera = list(eye = list(x = 1.5, y = -1.7, z = 0.9)))) |>
  config(displaylogo = FALSE, responsive = TRUE) |>
  onRender(js)

dir.create("docs", showWarnings = FALSE)
out <- normalizePath("docs", mustWork = TRUE)
saveWidget(p, file.path(out, "app.html"), selfcontained = TRUE, title = "SeaLevelLab query", libdir = NULL)
unlink(file.path(out, "app_files"), recursive = TRUE)
cat(sprintf("\nwrote docs/app.html  (%.2f MB)\n", file.size("docs/app.html") / 1e6))
