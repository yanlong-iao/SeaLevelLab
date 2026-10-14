/* node app/test_live.js -- the live-data front end: decoding, validation (same rules as the Python ETL), chart, status. */
"use strict";
const path = require("path");
const fs = require("fs");
const L = require(path.join(__dirname, "..", "docs", "live.js"));

let failed = 0;
function check(name, ok, detail) {
  console.log((ok ? "PASS " : "FAIL ") + name + (ok || detail === undefined ? "" : "  -> " + detail));
  if (!ok) failed++;
}
function throws(fn) { try { fn(); return false; } catch (e) { return true; } }

const om = (n, mut) => {
  const d = { utc_offset_seconds: 0, hourly_units: { sea_level_height_msl: "m" },
              hourly: { time: Array.from({ length: n }, (_, h) => new Date(Date.UTC(2026, 9, 14, h)).toISOString().slice(0, 16)),
                        sea_level_height_msl: Array.from({ length: n }, (_, h) => 0.5 * Math.sin(h / 2)) } };
  if (mut) mut(d);
  return d;
};

// 1. parseOpenMeteo accepts a good response and returns anomalies centred on zero, in mm
const a = L.parseOpenMeteo(om(168));
const mean = a.series.reduce((s, p) => s + p[1], 0) / a.series.length;
check("parseOpenMeteo: 168 points, anomaly mean ~ 0, amplitude in mm", a.series.length === 168 && Math.abs(mean) < 0.1 && Math.max(...a.series.map(p => p[1])) > 300);
check("parseOpenMeteo: times are UTC", a.series[1][0] - a.series[0][0] === 3600e3 && new Date(a.series[0][0]).getUTCHours() === 0);

// 2. it rejects what the Python ETL rejects
check("rejects error payload", throws(() => L.parseOpenMeteo({ error: true, reason: "x" })));
check("rejects wrong unit", throws(() => L.parseOpenMeteo(om(4, d => { d.hourly_units.sea_level_height_msl = "ft"; }))));
check("rejects non-UTC", throws(() => L.parseOpenMeteo(om(4, d => { d.utc_offset_seconds = 3600; }))));
check("rejects length mismatch", throws(() => L.parseOpenMeteo(om(4, d => { d.hourly.time.pop(); }))));
check("rejects implausible value", throws(() => L.parseOpenMeteo(om(4, d => { d.hourly.sea_level_height_msl[2] = 99; }))));

// 3. the committed export decodes and matches the station list of the map
global.window = global;
require(path.join(__dirname, "..", "docs", "app_data.js"));
const live = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "docs", "live", "live_data.json"), "utf8"));
const names = window.APP_DATA.stations.map(s => s.Location);
check("every map gauge has a live mapping", names.every(n => L.stationFor(live, n)), names.filter(n => !L.stationFor(live, n)).join(", "));
const s0 = live.stations.find(s => s.obs);
const dec = L.decode(s0.obs);
check("decode: lengths and strictly increasing times", dec.length === s0.obs.n && dec.every((p, i) => i === 0 || p[0] > dec[i - 1][0]));

// 4. chart: valid SVG, line breaks at gaps
const svg = L.chart(dec, { label: "t" });
check("chart: one svg with a path", /^<svg[\s\S]*<path d="M[\d.]+,[\d.]+L/.test(svg) && (svg.match(/<svg/g) || []).length === 1);
const gap = [[0, 1], [3600e3, 2], [10 * 3600e3, 3], [11 * 3600e3, 4]];
check("chart: a >3 h gap starts a new subpath", (L.chart(gap).match(/M/g) || []).length === 2);
check("chart: empty input is handled", /No data/.test(L.chart([])));

// 5. status line
const st = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "docs", "live", "status.json"), "utf8"));
const line = L.statusLine(st, Date.parse(st.last_success || st.generated) + 3600e3);
check("statusLine: ok flag and relative time", line.ok === (st.last_run.status !== "failed" && !!st.last_success) && /h ago|min ago|just now/.test(line.run));
check("statusLine: failed run is not ok", L.statusLine({ last_run: { status: "failed" }, last_success: null, stations_with_obs: 0, stations_total: 13 }).ok === false);

// 6. escaping
check("esc() neutralises HTML", L.esc('<img src=x onerror="1">') === "&lt;img src=x onerror=&quot;1&quot;&gt;");

console.log(failed ? `\n${failed} check(s) FAILED` : "\nall live-data checks passed");
process.exit(failed ? 1 : 0);
