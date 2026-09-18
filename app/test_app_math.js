/* node app/test_app_math.js -- hard gate: the browser-side formulas must satisfy the same checks as R. */
"use strict";
const path = require("path");
global.window = global;                                    // app_data.js assigns window.APP_DATA
require(path.join(__dirname, "..", "docs", "app_data.js"));
const M = require(path.join(__dirname, "..", "docs", "app_math.js"));
const D = window.APP_DATA;

let failed = 0;
function check(name, ok, detail) {
  console.log((ok ? "PASS " : "FAIL ") + name + (ok || detail === undefined ? "" : "  -> " + detail));
  if (!ok) failed++;
}
const NX = D.lon.length, NY = D.lat.length, n = NX * NY;
check("grid is 160 x 120 and every field has 19200 values",
  NX === 160 && NY === 120 && [D.mu_beta, D.sd_beta, D.mu_ls, D.sd_ls, D.support].every(a => a.length === n));

// 1. delta = 0 must give back the baseline return period everywhere
let maxDev = 0;
for (let k = 0; k < n; k++) maxDev = Math.max(maxDev, Math.abs(M.pExceed(D.mu_beta[k], Math.exp(D.mu_ls[k]), D.xi, 2000, 10) - 0.1));
check("pExceed(Y = 2000, N = 10) == 0.1 on every grid point (tol 0.01)", maxDev < 0.01, "max deviation " + maxDev);

// 2. monotone in Y (non-decreasing) and in N (non-increasing); range [0, 1]; no NaN
let monoY = true, monoN = true, inRange = true;
for (let k = 0; k < n; k += 7) {                             // every 7th point: 2743 points x 101 years
  const b = D.mu_beta[k], s = Math.exp(D.mu_ls[k]);
  let prev = M.pExceed(b, s, D.xi, 2000, 10);
  for (let Y = 2001; Y <= 2100; Y++) { const p = M.pExceed(b, s, D.xi, Y, 10); if (p < prev - 1e-12) monoY = false; prev = p; }
  prev = M.pExceed(b, s, D.xi, 2050, 2);
  for (const N of [5, 10, 20, 50]) { const p = M.pExceed(b, s, D.xi, 2050, N); if (p > prev + 1e-12) monoN = false; prev = p; }
}
for (let k = 0; k < n; k++) for (const Y of [2000, 2050, 2100]) for (const N of [2, 10, 50]) {
  const p = M.pExceed(D.mu_beta[k], Math.exp(D.mu_ls[k]), D.xi, Y, N);
  if (!(p >= 0 && p <= 1) || Number.isNaN(p)) inRange = false;
}
check("P non-decreasing in Y", monoY);
check("P non-increasing in N", monoN);
check("P in [0, 1] with no NaN over the whole field", inRange);

// 3. band ordering
let bandOk = true;
for (let k = 0; k < n; k += 3) for (const Y of [2000, 2030, 2070, 2100]) for (const N of [2, 10, 50]) {
  const b = M.pBand(D.mu_beta[k], D.sd_beta[k], D.mu_ls[k], D.sd_ls[k], D.xi, Y, N);
  if (!(b.lo <= b.med && b.med <= b.hi)) bandOk = false;
}
check("pBand: lo <= med <= hi everywhere", bandOk);

// 4. the y-axis flip used when painting the raster
check("rowForLat(max lat) === 0", M.rowForLat(Math.max(...D.lat), D.lat) === 0);
check("rowForLat(min lat) === 119", M.rowForLat(Math.min(...D.lat), D.lat) === 119);

// 5. station ordering by trend
const st = D.stations;
const hi = st.reduce((a, b) => (b.beta > a.beta ? b : a)), lo = st.reduce((a, b) => (b.beta < a.beta ? b : a));
const pHi = M.pExceed(hi.beta, hi.sigma, D.xi, 2050, 10), pLo = M.pExceed(lo.beta, lo.sigma, D.xi, 2050, 10);
check("Y=2050, N=10: highest-trend station (" + hi.Location + ") has P > lowest-trend station (" + lo.Location + ")", pHi > pLo, pHi + " vs " + pLo);
check("13 stations with lon360 in [140, 210]", st.length === 13 && st.every(s => s.lon360 >= 140 && s.lon360 <= 210));

console.log(failed === 0 ? "\nall checks passed" : "\n" + failed + " check(s) FAILED");
process.exit(failed === 0 ? 0 : 1);
