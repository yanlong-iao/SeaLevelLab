/* live.js -- the "Live data" layer of the map app and the status badge on the home page.
 *
 * Data path (see integrations.html for the diagram):
 *   UHSLC ERDDAP (hourly gauges)   -> daily GitHub Actions ETL -> SQLite -> docs/live/live_data.json -> this file
 *   Open-Meteo Marine (7-day fcst) -> same ETL snapshot, AND fetched live from the browser on demand
 *                                     (Open-Meteo sends Access-Control-Allow-Origin: *; UHSLC does not, which is
 *                                     why the gauges can only come through the ETL).
 * Every value is shown as an anomaly (series minus its own mean over the plotted window). The two sources use
 * different datums (gauge datum vs global mean sea level), so absolute levels are never compared.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SLLive = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var OPENMETEO = "https://marine-api.open-meteo.com/v1/marine";
  var LIVE_TIMEOUT_MS = 8000;

  function openMeteoUrl(lat, lon, days) {
    return OPENMETEO + "?latitude=" + lat.toFixed(4) + "&longitude=" + lon.toFixed(4) +
      "&hourly=sea_level_height_msl&forecast_days=" + (days || 7) + "&timezone=GMT";
  }

  /* exported series {t0, h[], anom_mm[]} -> [[ms, mm], ...] */
  function decode(part) {
    if (!part || !part.h) return [];
    var t0 = Date.parse(part.t0);
    return part.h.map(function (h, i) { return [t0 + h * 3600e3, part.anom_mm[i]]; });
  }

  /* Same checks as etl/live_sealevel.py parse_openmeteo: units, UTC, equal lengths, plausible values. */
  function parseOpenMeteo(doc) {
    if (!doc || doc.error) throw new Error("Open-Meteo: " + ((doc && doc.reason) || "empty response"));
    var h = doc.hourly, u = doc.hourly_units || {};
    if (!h || !h.time || !h.sea_level_height_msl) throw new Error("Open-Meteo: missing hourly fields");
    if ((u.sea_level_height_msl || "m") !== "m") throw new Error("Open-Meteo: unexpected unit " + u.sea_level_height_msl);
    if (h.time.length !== h.sea_level_height_msl.length) throw new Error("Open-Meteo: arrays differ in length");
    if ((doc.utc_offset_seconds || 0) !== 0) throw new Error("Open-Meteo: response not in UTC");
    var pts = [];
    for (var i = 0; i < h.time.length; i++) {
      var v = h.sea_level_height_msl[i];
      if (v === null || v === undefined) continue;
      if (typeof v !== "number" || !(v > -10 && v < 10)) throw new Error("Open-Meteo: implausible value " + v);
      var t = Date.parse(h.time[i] + (h.time[i].length === 16 ? ":00Z" : ""));
      if (isNaN(t)) throw new Error("Open-Meteo: bad time " + h.time[i]);
      pts.push([t, 1000 * v]);
    }
    if (!pts.length) throw new Error("Open-Meteo: no values");
    return toAnomaly(pts);
  }

  function toAnomaly(pts) {
    var m = 0; pts.forEach(function (p) { m += p[1]; }); m /= pts.length;
    return { series: pts.map(function (p) { return [p[0], Math.round((p[1] - m) * 10) / 10]; }), mean: m };
  }

  /* fetch with a hard timeout; resolves {series, mean, url, ms} or rejects */
  function fetchForecastLive(lat, lon, fetchImpl) {
    var f = fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
    if (!f) return Promise.reject(new Error("fetch unavailable"));
    var url = openMeteoUrl(lat, lon), t0 = Date.now();
    var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctl) ctl.abort(); }, LIVE_TIMEOUT_MS);
    return f(url, ctl ? { signal: ctl.signal } : {}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (doc) {
      clearTimeout(timer);
      var a = parseOpenMeteo(doc);
      return { series: a.series, mean: a.mean, url: url, ms: Date.now() - t0 };
    }, function (e) {
      clearTimeout(timer);
      throw (e && e.name === "AbortError") ? new Error("timed out after " + LIVE_TIMEOUT_MS / 1000 + " s") : e;
    });
  }

  /* ---------------------------------------------------------------- formatting */
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function fmtDate(ms, withTime) {
    var d = new Date(ms);
    var s = d.getUTCDate() + " " + MON[d.getUTCMonth()] + " " + d.getUTCFullYear();
    if (withTime) s += ", " + ("0" + d.getUTCHours()).slice(-2) + ":" + ("0" + d.getUTCMinutes()).slice(-2) + " UTC";
    return s;
  }
  function fmtAgo(ms, now) {
    var s = Math.max(0, ((now || Date.now()) - ms) / 1000);
    if (s < 90) return "just now";
    if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 129600) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " days ago";
  }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }

  /* ---------------------------------------------------------------- tiny SVG line chart (no library) */
  function niceStep(range, target) {
    var raw = range / target, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  }
  function chart(series, o) {
    o = o || {};
    var W = 600, H = o.height || 190, L = 44, R = 10, T = 10, B = 26;
    if (!series.length) return '<div class="lv-empty">No data</div>';
    var x0 = series[0][0], x1 = series[series.length - 1][0], ys = series.map(function (p) { return p[1]; });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    if (hi - lo < 1) { lo -= 1; hi += 1; }
    var step = niceStep(hi - lo, 4); lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    var sx = function (t) { return L + (W - L - R) * (t - x0) / Math.max(1, x1 - x0); };
    var sy = function (v) { return T + (H - T - B) * (hi - v) / (hi - lo); };
    var g = [];
    for (var v = lo; v <= hi + 1e-9; v += step) {
      g.push('<line x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(v).toFixed(1) + '" y2="' + sy(v).toFixed(1) + '" class="' + (Math.abs(v) < 1e-9 ? "lv-zero" : "lv-grid") + '"/>' +
             '<text x="' + (L - 6) + '" y="' + (sy(v) + 3.5).toFixed(1) + '" text-anchor="end" class="lv-tick">' + Math.round(v) + '</text>');
    }
    var days = (x1 - x0) / 864e5, every = days > 20 ? 7 : days > 5 ? 2 : 1;
    var d0 = new Date(x0); d0.setUTCHours(0, 0, 0, 0);
    for (var t = d0.getTime() + 864e5, k = 0; t < x1; t += 864e5, k++) {
      if (k % every) continue;
      var d = new Date(t);
      g.push('<line x1="' + sx(t).toFixed(1) + '" x2="' + sx(t).toFixed(1) + '" y1="' + T + '" y2="' + (H - B) + '" class="lv-grid"/>' +
             '<text x="' + sx(t).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle" class="lv-tick">' + d.getUTCDate() + " " + MON[d.getUTCMonth()] + '</text>');
    }
    // break the line at gaps longer than 3 hours
    var path = "", prev = null;
    series.forEach(function (p) {
      path += (prev === null || p[0] - prev > 3 * 3600e3 ? "M" : "L") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1);
      prev = p[0];
    });
    return '<svg class="lv-chart" viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none" role="img" aria-label="' + esc(o.label || "chart") + '">' +
      g.join("") + '<text x="' + (L - 38) + '" y="' + (T + 2) + '" class="lv-tick" transform="rotate(-90 ' + (L - 38) + "," + (T + 2) + ')" text-anchor="end">mm</text>' +
      '<path d="' + path + '" class="lv-line ' + (o.cls || "") + '"/></svg>';
  }

  function stats(series) {
    var ys = series.map(function (p) { return p[1]; });
    return { min: Math.min.apply(null, ys), max: Math.max.apply(null, ys) };
  }

  /* ---------------------------------------------------------------- data loading */
  var cache = null;
  function load(base) {
    if (!cache) {
      cache = fetch((base || "") + "live/live_data.json", { cache: "no-cache" }).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      cache.catch(function () { cache = null; });
    }
    return cache;
  }
  function stationFor(data, location) {
    for (var i = 0; i < data.stations.length; i++) if (data.stations[i].location === location) return data.stations[i];
    return null;
  }

  /* ---------------------------------------------------------------- the map-app pane */
  function renderPane(el, data, location, now) {
    var s = stationFor(data, location);
    if (!s) { el.innerHTML = '<p class="hint">No live mapping for this gauge.</p>'; return; }
    now = now || Date.now();
    var o = s.obs, f = s.forecast, obs = decode(o), fc = decode(f), h = [];
    h.push('<div class="lv-head"><b>' + esc(s.uhslc_name) + '</b> · UHSLC station ' + s.uhslc_id +
           ' <span class="lv-conf lv-' + esc(s.confidence) + '" title="' + esc(s.basis) + '">match: ' + esc(s.confidence) + '</span></div>');
    h.push('<h3>Latest observations · last ' + data.window_days + ' days, hourly</h3>');
    if (obs.length) {
      var so = stats(obs);
      h.push(chart(obs, { label: "Observed sea-level anomaly", cls: "lv-obs" }));
      h.push('<div class="lv-meta">Range ' + Math.round(so.min) + ' to +' + Math.round(so.max) + ' mm around the 30-day mean. ' +
             'Latest value <b>' + fmtDate(Date.parse(o.last), true) + '</b> (' + o.lag_days + ' days behind today: Fast Delivery data arrive with a 4–6 week lag).' +
             (o.stale ? ' <span class="lv-stale">Last refresh failed; showing the previous good export.</span>' : '') +
             '<br>Source: <a href="https://uhslc.soest.hawaii.edu/" target="_blank" rel="noopener">University of Hawaii Sea Level Center</a>, Fast Delivery, via ERDDAP · retrieved ' +
             fmtDate(Date.parse(o.retrieved), true) + ' · <a href="' + esc(o.query) + '" target="_blank" rel="noopener">exact API query</a></div>');
    } else h.push('<p class="hint">No recent observations for this gauge.</p>');

    h.push('<h3>Next 7 days · modelled forecast</h3><div class="lv-fc">');
    h.push(fc.length ? chart(fc, { label: "Forecast sea-level anomaly", cls: "lv-fc-line" }) : '<p class="hint">No forecast in the snapshot.</p>');
    h.push('</div><div class="lv-meta lv-fc-meta">' + (f ? fcMeta(f.issued, f.query, "snapshot", null, f.stale) : "") + '</div>');
    h.push('<div class="lv-note"><b>How to read this.</b> Both curves are anomalies: each series minus its own mean over the plotted window. ' +
           'Gauge levels are relative to the gauge\'s datum and the forecast to global mean sea level, so only the shape (tides, surges) is comparable, never the absolute height. ' +
           'Open-Meteo\'s marine model runs on a ~8 km grid and its accuracy near coasts is limited. Not for navigation or legal use.</div>');
    h.push('<div class="lv-pipe">Pipeline: daily <a href="https://github.com/yanlong-iao/SeaLevelLab/actions/workflows/live-data.yml" target="_blank" rel="noopener">GitHub Actions job</a> → ' +
           'validate → SQLite → <a href="live/live_data.json" target="_blank">live_data.json</a> · snapshot built ' + fmtDate(Date.parse(data.generated), true) +
           ' · <a href="integrations.html">how this works</a></div>');
    el.innerHTML = h.join("");

    // live refresh of the forecast straight from the browser; the snapshot stays if this fails
    if (!f) return;
    var fcEl = el.querySelector(".lv-fc"), metaEl = el.querySelector(".lv-fc-meta");
    metaEl.innerHTML = fcMeta(f.issued, f.query, "loading") ;
    fetchForecastLive(s.lat, s.lon).then(function (r) {
      if (el.getAttribute("data-loc") !== location) return;              // user moved on
      fcEl.innerHTML = chart(r.series, { label: "Forecast sea-level anomaly (live)", cls: "lv-fc-line" });
      metaEl.innerHTML = fcMeta(null, r.url, "live", r.ms);
    }, function (e) {
      if (el.getAttribute("data-loc") !== location) return;
      metaEl.innerHTML = fcMeta(f.issued, f.query, "fallback", null, f.stale, e.message);
    });
  }
  function fcMeta(issued, url, mode, ms, stale, err) {
    var src = 'Source: <a href="https://open-meteo.com/en/docs/marine-weather-api" target="_blank" rel="noopener">Open-Meteo Marine API</a>, sea_level_height_msl (tides + surge model) · ';
    var tag = { live: '<span class="lv-tag lv-live">● live</span> fetched from the browser just now (' + ms + ' ms)',
                loading: '<span class="lv-tag">refreshing…</span> showing the daily snapshot issued ' + (issued ? fmtDate(Date.parse(issued), true) : ""),
                snapshot: '<span class="lv-tag">snapshot</span> issued ' + (issued ? fmtDate(Date.parse(issued), true) : ""),
                fallback: '<span class="lv-tag lv-warn">snapshot</span> live refresh failed (' + esc(err || "") + '), showing the daily snapshot issued ' + (issued ? fmtDate(Date.parse(issued), true) : "") }[mode];
    return tag + (stale ? ' <span class="lv-stale">stale</span>' : '') + '<br>' + src + '<a href="' + esc(url) + '" target="_blank" rel="noopener">exact API query</a>';
  }

  /* ---------------------------------------------------------------- one-line status (map controls, home badge) */
  function statusLine(st, now) {
    if (!st || !st.last_run) return "Live data status unavailable";
    now = now || Date.now();
    var ok = st.last_success ? Date.parse(st.last_success) : null;
    return { ok: !!ok && st.last_run.status !== "failed",
             run: ok ? fmtAgo(ok, now) : "never",
             runAbs: ok ? fmtDate(ok, true) : "",
             obsTo: st.latest_observation ? fmtDate(Date.parse(st.latest_observation)) : "–",
             stations: st.stations_with_obs + "/" + st.stations_total,
             lastStatus: st.last_run.status };
  }

  return { openMeteoUrl: openMeteoUrl, decode: decode, parseOpenMeteo: parseOpenMeteo, toAnomaly: toAnomaly,
           fetchForecastLive: fetchForecastLive, chart: chart, fmtDate: fmtDate, fmtAgo: fmtAgo, esc: esc,
           load: load, stationFor: stationFor, renderPane: renderPane, statusLine: statusLine, LIVE_TIMEOUT_MS: LIVE_TIMEOUT_MS };
});
