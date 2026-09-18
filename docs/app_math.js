/* app_math.js -- pure functions shared by docs/app.html and app/test_app_math.js.
 * No DOM access here. The formulas are the JavaScript twin of p_exceed() / p_band() in app/build_app.R:
 *
 *   z_N   = ((-log(1 - 1/N))^(-xi) - 1) / xi           standardised baseline return level (same everywhere)
 *   delta = (beta/1000) * (Y - 2000) / sigma            dimensionless shift of the GEV location since 2000
 *   P     = 1 - exp( -(max(0, 1 + xi*(z_N - delta)))^(-1/xi) )
 *
 * P is the probability that a station's annual maximum sea level exceeds the level that had a
 * 1-in-N chance per year around 2000, in that station's own datum. It is not a probability that land floods.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.AppMath = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var GUMBEL_EPS = 1e-9;

  function zReturn(xi, N) {
    var y = -Math.log(1 - 1 / N);
    if (Math.abs(xi) < GUMBEL_EPS) return -Math.log(y);          // Gumbel limit (never reached for the fitted xi)
    return (Math.pow(y, -xi) - 1) / xi;
  }

  function pExceed(betaMmYr, sigmaM, xi, Y, N) {
    var zN = zReturn(xi, N);
    var delta = (betaMmYr / 1000) * (Y - 2000) / sigmaM;
    if (Math.abs(xi) < GUMBEL_EPS) return 1 - Math.exp(-Math.exp(-(zN - delta)));
    var s = Math.max(0, 1 + xi * (zN - delta));
    return 1 - Math.exp(-Math.pow(s, -1 / xi));
  }

  /* 90% band: P is increasing in beta and decreasing in sigma, so plug the 5% / 95% quantiles of
   * beta and the opposite quantiles of sigma into the same formula (no Monte Carlo needed). */
  function pBand(muBeta, sdBeta, muLs, sdLs, xi, Y, N) {
    var med = pExceed(muBeta, Math.exp(muLs), xi, Y, N);
    var lo = pExceed(muBeta - 1.645 * sdBeta, Math.exp(muLs + 1.645 * sdLs), xi, Y, N);
    var hi = pExceed(muBeta + 1.645 * sdBeta, Math.exp(muLs - 1.645 * sdLs), xi, Y, N);
    return { lo: Math.min(lo, med), med: med, hi: Math.max(hi, med) };
  }

  /* GEV return level of a station in its own datum: x_T = mu0 + sigma * z_T */
  function returnLevel(mu0, sigma, xi, T) { return mu0 + sigma * zReturn(xi, T); }

  function haversineKm(lon1, lat1, lon2, lat2) {
    var r = Math.PI / 180, dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
    var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(a)));
  }

  /* Canvas row for a latitude. APP_DATA.lat is ascending (index 0 = southernmost, -25), but an image's
   * row 0 is its TOP edge (northernmost, +12). This is the one place the y axis is flipped. */
  function rowForLat(lat, latAxis) {
    var n = latAxis.length, lo = latAxis[0], hi = latAxis[n - 1];
    var row = Math.round((hi - lat) / (hi - lo) * (n - 1));
    return Math.min(Math.max(row, 0), n - 1);
  }

  /* Nearest grid column / latitude index for a coordinate (both axes are uniform). */
  function nearestIndex(v, axis) {
    var n = axis.length, i = Math.round((v - axis[0]) / (axis[n - 1] - axis[0]) * (n - 1));
    return Math.min(Math.max(i, 0), n - 1);
  }

  return { zReturn: zReturn, pExceed: pExceed, pBand: pBand, returnLevel: returnLevel,
           haversineKm: haversineKm, rowForLat: rowForLat, nearestIndex: nearestIndex };
});
