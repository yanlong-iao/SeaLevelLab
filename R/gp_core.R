# R/gp_core.R -- shared statistical core of SeaLevelLab
# Sourced by flood_ai4s.qmd (setup chunk) and by app/build_app.R.
# Everything below except gp_post_diag() is moved verbatim from flood_ai4s.qmd.

# Study box (degrees).  Longitudes mapped to [0, 360) because the data straddle the antimeridian.
LON <- c(140, 210); LAT <- c(-25, 12)
to_unit    <- function(lon, lat) cbind((lon - LON[1]) / diff(LON), (lat - LAT[1]) / diff(LAT))
to_degrees <- function(x) cbind(LON[1] + x[, 1] * diff(LON), LAT[1] + x[, 2] * diff(LAT))

###########################
## 5. A Matern-3/2 Gaussian process, from scratch (~40 lines)
###########################
k_matern <- function(A, B, ls, sig) {              # Matern nu = 3/2 (closed form)
  d <- sqrt(outer(A[, 1], B[, 1], "-")^2 + outer(A[, 2], B[, 2], "-")^2)
  s <- sqrt(3) * d / ls
  sig^2 * (1 + s) * exp(-s)
}
gp_post <- function(X, y, Xs, h, full = FALSE) {  # posterior mean / sd (/ cov) at Xs
  if (nrow(X) == 0) { K <- k_matern(Xs, Xs, h$ls, h$sig); mu <- rep(h$mean, nrow(Xs))
    return(if (full) list(mu = mu, cov = K) else list(mu = mu, sd = sqrt(diag(K)))) }
  K  <- k_matern(X, X, h$ls, h$sig) + diag(h$noise^2 + 1e-8, nrow(X))
  L  <- chol(K)
  a  <- backsolve(L, forwardsolve(t(L), y - h$mean))
  Ks <- k_matern(Xs, X, h$ls, h$sig)
  mu <- h$mean + Ks %*% a
  V  <- forwardsolve(t(L), t(Ks))
  cov <- k_matern(Xs, Xs, h$ls, h$sig) - crossprod(V)
  if (full) list(mu = as.vector(mu), cov = cov) else list(mu = as.vector(mu), sd = sqrt(pmax(diag(cov), 1e-12)))
}
gp_lml <- function(X, y, h) {                      # log marginal likelihood
  K <- k_matern(X, X, h$ls, h$sig) + diag(h$noise^2 + 1e-8, nrow(X)); L <- chol(K)
  r <- y - h$mean; a <- backsolve(L, forwardsolve(t(L), r))
  -0.5 * sum(r * a) - sum(log(diag(L))) - 0.5 * length(y) * log(2 * pi)
}
gp_fit <- function(X, y, prior = list(ls = c(log(0.25), 0.5), noise = c(log(0.8), 0.5))) {
  # Type-II MAP: maximise log-likelihood + log-prior.  With 13 sites the range and the nugget are only
  # jointly identifiable (Zhang 2004); INLA guards this with PC priors, we use log-normal priors.
  obj <- function(th) { h <- list(ls = exp(th[1]), sig = exp(th[2]), noise = exp(th[3]), mean = mean(y))
    -gp_lml(X, y, h) + 0.5 * ((th[1] - prior$ls[1]) / prior$ls[2])^2 + 0.5 * ((th[3] - prior$noise[1]) / prior$noise[2])^2 }
  best <- NULL
  for (i in 1:10) { th0 <- c(log(runif(1, .1, .6)), log(sd(y) * runif(1, .5, 1.5)), log(sd(y) * runif(1, .1, .6)))
    o <- optim(th0, obj, method = "L-BFGS-B", lower = log(c(.05, .01, .01)), upper = log(c(2, 50, 10)))
    if (is.null(best) || o$value < best$value) best <- o }
  list(ls = exp(best$par[1]), sig = exp(best$par[2]), noise = exp(best$par[3]), mean = mean(y))
}

gp_post_diag <- function(X, y, Xs, h, chunk = 2000) {
  # Mathematically identical to gp_post(), but returns only mu and sd and works in chunks of
  # query points, so it never allocates the nrow(Xs) x nrow(Xs) posterior covariance.
  # (A 160 x 120 grid has 19 200 points; gp_post() would need a 2.9 GB matrix for it.)
  n <- nrow(Xs); mu <- numeric(n); sd <- numeric(n)
  if (nrow(X) == 0) return(list(mu = rep(h$mean, n), sd = rep(h$sig, n)))   # prior: k(x, x) = sig^2
  K <- k_matern(X, X, h$ls, h$sig) + diag(h$noise^2 + 1e-8, nrow(X))
  L <- chol(K)
  a <- backsolve(L, forwardsolve(t(L), y - h$mean))
  for (s in seq(1, n, by = chunk)) {
    idx <- s:min(s + chunk - 1, n)
    Ks  <- k_matern(Xs[idx, , drop = FALSE], X, h$ls, h$sig)
    mu[idx] <- h$mean + as.vector(Ks %*% a)
    V <- forwardsolve(t(L), t(Ks))                       # nrow(X) x length(idx)
    sd[idx] <- sqrt(pmax(h$sig^2 - colSums(V^2), 1e-12)) # diag(k(Xs,Xs)) - diag(V'V)
  }
  list(mu = mu, sd = sd)
}
