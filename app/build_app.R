#!/usr/bin/env Rscript
# app/build_app.R -- computes the data behind docs/app.html, an interactive query app for
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
# Architecture: R only computes. Sections 1-9 fit the model and check it; section 10 writes
#   docs/app_data.js   one assignment  window.APP_DATA = {...}  (a <script>, so file:// works without fetch/CORS)
#   docs/vendor/       leaflet.js, leaflet.css, plotly.min.js copied from the installed R packages
# The page itself (docs/app.html) and the formulas (docs/app_math.js) are hand-written and committed.
#
#     Rscript app/build_app.R && node app/test_app_math.js

suppressPackageStartupMessages({ library(dplyr); library(jsonlite) })
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
# 10. Data export for the hand-written page + vendor assets + build-time assertions
# ======================================================================================
sig4 <- function(v) signif(v, 4)

## 10a. vendor: leaflet.js / leaflet.css / plotly.min.js  (installed R packages -> install -> cdnjs)
dir.create("docs/vendor", showWarnings = FALSE, recursive = TRUE)
vendor_from_packages <- function() {
  found <- list()
  for (pkg in c("leaflet", "plotly")) {
    if (!requireNamespace(pkg, quietly = TRUE)) next
    lib <- system.file("htmlwidgets/lib", package = pkg)
    files <- list.files(lib, recursive = TRUE, full.names = TRUE, pattern = "leaflet\\.(js|css)$|^plotly.*\\.min\\.js$")
    files <- files[basename(files) %in% c("leaflet.js", "leaflet.css") | grepl("^plotly.*\\.min\\.js$", basename(files))]
    for (f in files) found[[if (grepl("^plotly", basename(f))) "plotly.min.js" else basename(f)]] <- f
    if (pkg == "leaflet") { img <- file.path(lib, "leaflet", "images"); if (dir.exists(img)) found[["images"]] <- img }
  }
  found
}
vendor_targets <- c("leaflet.js", "leaflet.css", "plotly.min.js")
vf <- vendor_from_packages()
if (!all(vendor_targets %in% names(vf))) {
  install.packages(c("leaflet", "plotly"), repos = "https://cloud.r-project.org", quiet = TRUE)
  vf <- vendor_from_packages()
}
vendor_source <- "installed R packages"
if (all(vendor_targets %in% names(vf))) {
  for (t in vendor_targets) file.copy(vf[[t]], file.path("docs/vendor", t), overwrite = TRUE)
  if (!is.null(vf$images)) { dir.create("docs/vendor/images", showWarnings = FALSE); invisible(file.copy(list.files(vf$images, full.names = TRUE), "docs/vendor/images", overwrite = TRUE)) }
} else {
  vendor_source <- "cdnjs download"
  cdn <- c("leaflet.js" = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js",
           "leaflet.css" = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css",
           "plotly.min.js" = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js")
  for (t in vendor_targets) if (download.file(cdn[[t]], file.path("docs/vendor", t), quiet = TRUE, mode = "wb") != 0) stop("vendor: could not obtain ", t)
}
cat(sprintf("vendor: %s -> %s\n", vendor_source, paste(sprintf("%s (%.0f KB)", vendor_targets, file.size(file.path("docs/vendor", vendor_targets)) / 1e3), collapse = ", ")))

## 10b. coastline for the offline fallback layer (longitudes stay 0-360)
coastline <- tryCatch({
  db <- if (requireNamespace("mapdata", quietly = TRUE)) { suppressPackageStartupMessages(library(mapdata)); "world2Hires" } else "world2"
  m  <- maps::map(db, xlim = c(LON[1], LON[2]), ylim = c(LAT[1], LAT[2]), plot = FALSE)
  xs <- sig4(m$x); ys <- sig4(m$y)
  segs <- split(seq_along(xs), cumsum(is.na(xs)))
  out <- lapply(segs, function(idx) { idx <- idx[!is.na(xs[idx])]; if (length(idx) < 2) return(NULL)
    pts <- cbind(xs[idx], ys[idx]); keep <- c(TRUE, rowSums(abs(diff(pts))) > 0); pts[keep, , drop = FALSE] })
  out <- Filter(function(z) !is.null(z) && nrow(z) >= 2, out)
  cat(sprintf("coastline: %s, %d segments, %d points\n", db, length(out), sum(sapply(out, nrow))))
  unname(out)
}, error = function(e) { cat("WARNING: coastline unavailable (", conditionMessage(e), ") -- exporting an empty array; the offline basemap layer will be empty\n"); list() })

## 10c. window.APP_DATA
amax_series <- lapply(rates$Location, function(l) { a <- annual[annual$Location == l, ]; unname(cbind(a$Year, sig4(a$amax))) })
app_data <- list(
  # field arrays are flattened with LONGITUDE VARYING FASTEST: index k = i_lat * 160 + j_lon
  # (lat[0] = -25 is the southernmost row; the page flips this when it paints the raster).
  lon = sig4(g_lon), lat = sig4(g_lat), nx = NX, ny = NY,
  mu_beta = sig4(f_beta$mu), sd_beta = sig4(f_beta$sd), mu_ls = sig4(f_ls$mu), sd_ls = sig4(f_ls$sd), support = sig4(support),
  xi = sig4(xi), sig_beta = sig4(hyp_beta$sig),
  stations = lapply(seq_len(S), function(s) list(
    Location = rates$Location[s], lon360 = sig4(rates$lon[s]), lat = sig4(rates$lat[s]),
    beta = sig4(rates$trend_mm_yr[s]), sigma = sig4(sigma_s[s]), mu0 = sig4(mu0_s[s]), amax = amax_series[[s]])),
  next_site = list(lon360 = sig4(grid_deg[next_idx, 1]), lat = sig4(grid_deg[next_idx, 2])),
  coastline = coastline,
  built = format(Sys.time(), "%Y-%m-%d")
)
json <- toJSON(app_data, digits = I(4), auto_unbox = TRUE, null = "null")
writeLines(paste0("/* generated by app/build_app.R -- do not edit. Field arrays are flattened lon-fastest: k = i_lat*160 + j_lon; lat[0] is the SOUTH edge. */\n",
                  "window.APP_DATA = ", json, ";"), "docs/app_data.js")
if (!file.exists("docs/app_data.js") || file.size("docs/app_data.js") < 1e5) stop("docs/app_data.js was not written")
cat(sprintf("wrote docs/app_data.js (%.2f MB)\n", file.size("docs/app_data.js") / 1e6))

## 10d. build-time assertions for the export
for (nm in c("mu_beta", "sd_beta", "mu_ls", "sd_ls", "support")) if (length(app_data[[nm]]) != NX * NY) stop("check: ", nm, " must have 160 x 120 values")
if (!all(rates$lon >= LON[1] & rates$lon <= LON[2])) stop("check: every station lon360 must lie in [140, 210]")
for (t in vendor_targets) { f <- file.path("docs/vendor", t); if (!file.exists(f) || file.size(f) <= 10e3) stop("check: vendor file missing or too small: ", t) }
cat("export checks: all passed\n")
