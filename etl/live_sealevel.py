"""Live sea-level ETL: two public APIs -> SQLite -> small JSON files for the static site.

    python -m etl.live_sealevel                 # fetch, upsert, export (what the daily workflow runs)
    python -m etl.live_sealevel --export-only   # rebuild the JSON from the database without calling the APIs

Sources
  UHSLC Fast Delivery, hourly tide-gauge sea level, via ERDDAP (University of Hawaii Sea Level Center).
    No key. Data lag is roughly 4-6 weeks. The server sends no CORS header, so a browser cannot call it
    directly; that is one reason this runs as a scheduled job and the site reads an exported file.
  Open-Meteo Marine API, modelled sea_level_height_msl (tides + surge, relative to global mean sea level,
    ~8 km grid), 7-day forecast. No key for non-commercial use; CORS is open, so the page also refreshes it live.

Storage (data/live/sealevel_live.sqlite, standard-library sqlite3)
  stations      the 13 gauges and their UHSLC mapping
  observations  (uhslc_id, time) primary key -> sea_level_mm          upsert, so re-runs are idempotent
  forecasts     (location, issued, time) primary key -> sea_level_m   one row set per forecast issue
  etl_runs      one row per run: start, duration, status, rows per source, errors

Exports (docs/live/, read by the website)
  live_data.json  per station: last 30 days of hourly observations and the latest 7-day forecast
  status.json     last run, last successful run, per-source counts, a short run history

Failure policy: every HTTP call has a timeout and retries with exponential back-off; each station is fetched
independently; a response that fails schema validation is rejected, not stored. If a station gets no new data,
its previous export is kept. Exports are written to a temp file and renamed, so the site never reads half a file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATION_MAP = Path(__file__).with_name("station_map.json")
DB_PATH = ROOT / "data" / "live" / "sealevel_live.sqlite"
EXPORT_DIR = ROOT / "docs" / "live"

UHSLC_BASE = "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_fast.json"
OPENMETEO_BASE = "https://marine-api.open-meteo.com/v1/marine"
UHSLC_FILL = -32767
OBS_WINDOW_DAYS = 30          # what the site shows
OBS_FETCH_DAYS = 75           # what we ask for: the fast-delivery lag is ~4-6 weeks, so 30 days back from today is often empty
FORECAST_DAYS = 7
HISTORY_KEEP = 30             # run-history entries kept in status.json
USER_AGENT = "SeaLevelLab-ETL/1.0 (+https://github.com/yanlong-iao/SeaLevelLab)"


class SchemaError(ValueError):
    """The API answered, but not in the shape we expect. Nothing from that response is stored."""


# ----------------------------------------------------------------------------------------------- HTTP
def http_get_json(url: str, *, timeout: float = 30.0, retries: int = 3, backoff: float = 2.0, opener=None) -> dict:
    """GET a JSON document with a timeout and exponential back-off. 4xx other than 429 is not retried."""
    opener = opener or urllib.request.urlopen
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with opener(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code < 500 and e.code != 429:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"GET failed after {retries + 1} attempts: {last}") from last


# ----------------------------------------------------------------------------------------------- URLs
def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def uhslc_url(uhslc_id: int, since: datetime) -> str:
    # ERDDAP constraints are written literally (time>=...), with only the value percent-encoded
    q = f"time,sea_level&uhslc_id={int(uhslc_id)}&time>={urllib.parse.quote(iso(since))}"
    return f"{UHSLC_BASE}?{q}"


def openmeteo_url(lat: float, lon: float, days: int = FORECAST_DAYS) -> str:
    q = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                                "hourly": "sea_level_height_msl", "forecast_days": days, "timezone": "GMT"})
    return f"{OPENMETEO_BASE}?{q}"


# ----------------------------------------------------------------------------------------------- parsing + validation
def parse_uhslc(doc: dict) -> list[tuple[str, int]]:
    """ERDDAP .json table -> [(time ISO UTC, sea_level mm)], fill values and nulls dropped."""
    try:
        table = doc["table"]
        names, units, rows = table["columnNames"], table.get("columnUnits", []), table["rows"]
    except (KeyError, TypeError) as e:
        raise SchemaError(f"UHSLC: missing table fields ({e})") from e
    if names[:2] != ["time", "sea_level"]:
        raise SchemaError(f"UHSLC: unexpected columns {names}")
    if len(units) >= 2 and units[1] not in ("millimeters", "mm"):
        raise SchemaError(f"UHSLC: sea_level unit is {units[1]!r}, expected millimeters")
    out = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 2:
            raise SchemaError(f"UHSLC: malformed row {r!r}")
        t, v = r[0], r[1]
        if v is None or v == UHSLC_FILL:
            continue
        try:
            datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError) as e:
            raise SchemaError(f"UHSLC: bad time {t!r}") from e
        if not isinstance(v, (int, float)) or not -10000 < v < 15000:
            raise SchemaError(f"UHSLC: implausible sea_level {v!r}")
        out.append((t, int(v)))
    return out


def parse_openmeteo(doc: dict) -> list[tuple[str, float]]:
    """Open-Meteo hourly -> [(time ISO UTC, sea level m)], nulls dropped."""
    if isinstance(doc, dict) and doc.get("error"):
        raise SchemaError(f"Open-Meteo: {doc.get('reason', 'error')}")
    try:
        hourly, units = doc["hourly"], doc.get("hourly_units", {})
        times, vals = hourly["time"], hourly["sea_level_height_msl"]
    except (KeyError, TypeError) as e:
        raise SchemaError(f"Open-Meteo: missing hourly fields ({e})") from e
    if units.get("sea_level_height_msl", "m") != "m":
        raise SchemaError(f"Open-Meteo: unit is {units.get('sea_level_height_msl')!r}, expected m")
    if len(times) != len(vals):
        raise SchemaError("Open-Meteo: time and value arrays differ in length")
    if doc.get("utc_offset_seconds", 0) != 0:
        raise SchemaError("Open-Meteo: response is not in UTC")
    out = []
    for t, v in zip(times, vals):
        if v is None:
            continue
        if not isinstance(v, (int, float)) or not -10 < v < 10:
            raise SchemaError(f"Open-Meteo: implausible value {v!r}")
        if isinstance(t, str) and len(t) == 16:   # Open-Meteo gives "2026-10-14T00:00" in GMT
            t += ":00Z"
        try:
            datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError) as e:
            raise SchemaError(f"Open-Meteo: bad time {t!r}") from e
        out.append((t, float(v)))
    return out


# ----------------------------------------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
  location TEXT PRIMARY KEY, uhslc_id INTEGER NOT NULL, uhslc_name TEXT, lat REAL, lon REAL, basis TEXT, confidence TEXT);
CREATE TABLE IF NOT EXISTS observations (
  uhslc_id INTEGER NOT NULL, time TEXT NOT NULL, sea_level_mm INTEGER NOT NULL, retrieved TEXT NOT NULL,
  PRIMARY KEY (uhslc_id, time));
CREATE TABLE IF NOT EXISTS forecasts (
  location TEXT NOT NULL, issued TEXT NOT NULL, time TEXT NOT NULL, sea_level_m REAL NOT NULL,
  PRIMARY KEY (location, issued, time));
CREATE TABLE IF NOT EXISTS etl_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, started TEXT NOT NULL, duration_s REAL, status TEXT NOT NULL,
  obs_rows INTEGER, fc_rows INTEGER, stations_ok INTEGER, stations_total INTEGER, errors TEXT);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    return con


def upsert_stations(con: sqlite3.Connection, stations: list[dict]) -> None:
    con.executemany(
        "INSERT INTO stations VALUES (:location,:uhslc_id,:uhslc_name,:lat,:lon,:basis,:confidence) "
        "ON CONFLICT(location) DO UPDATE SET uhslc_id=excluded.uhslc_id, uhslc_name=excluded.uhslc_name, "
        "lat=excluded.lat, lon=excluded.lon, basis=excluded.basis, confidence=excluded.confidence", stations)


def upsert_observations(con: sqlite3.Connection, uhslc_id: int, rows: list[tuple[str, int]], retrieved: str) -> int:
    """Insert or update hourly values; returns how many rows were new or changed."""
    before = con.total_changes
    con.executemany(
        "INSERT INTO observations VALUES (?,?,?,?) ON CONFLICT(uhslc_id, time) DO UPDATE SET "
        "sea_level_mm=excluded.sea_level_mm, retrieved=excluded.retrieved WHERE sea_level_mm != excluded.sea_level_mm",
        [(uhslc_id, t, v, retrieved) for t, v in rows])
    return con.total_changes - before


def upsert_forecast(con: sqlite3.Connection, location: str, issued: str, rows: list[tuple[str, float]]) -> int:
    before = con.total_changes
    con.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?)", [(location, issued, t, v) for t, v in rows])
    # keep only the last 14 issues per station so the file stays small
    con.execute("DELETE FROM forecasts WHERE location=? AND issued NOT IN "
                "(SELECT DISTINCT issued FROM forecasts WHERE location=? ORDER BY issued DESC LIMIT 14)", (location, location))
    return con.total_changes - before


# ----------------------------------------------------------------------------------------------- export
def anomaly(values: list[float]) -> tuple[list[float], float]:
    """Each series is shown against its own mean over the plotted window; datums are never mixed."""
    if not values:
        return [], float("nan")
    m = sum(values) / len(values)
    return [v - m for v in values], m


def _ts(t: str) -> datetime:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hours_since(t0: str, times: list[str]) -> list[int]:
    """Times as whole hours after t0: a compact, gap-preserving encoding for the browser."""
    base = _ts(t0)
    return [round((_ts(t) - base).total_seconds() / 3600) for t in times]


def build_export(con: sqlite3.Connection, stations: list[dict], now: datetime, previous: dict | None = None) -> dict:
    previous = previous or {}
    prev_st = {s["location"]: s for s in previous.get("stations", [])}
    out = []
    for s in stations:
        loc, sid = s["location"], s["uhslc_id"]
        last = con.execute("SELECT MAX(time) FROM observations WHERE uhslc_id=?", (sid,)).fetchone()[0]
        obs = None
        if last:
            start = iso(datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) - timedelta(days=OBS_WINDOW_DAYS))
            rows = con.execute("SELECT time, sea_level_mm, retrieved FROM observations WHERE uhslc_id=? AND time>? ORDER BY time",
                               (sid, start)).fetchall()
            mm = [r[1] for r in rows]
            an, mean = anomaly(mm)
            obs = {"t0": rows[0][0], "h": hours_since(rows[0][0], [r[0] for r in rows]), "anom_mm": [round(a, 1) for a in an],
                   "mean_mm": round(mean, 1), "n": len(rows), "last": last, "retrieved": max(r[2] for r in rows),
                   "lag_days": round((now - datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds() / 86400, 1),
                   "query": uhslc_url(sid, datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))}
        elif prev_st.get(loc, {}).get("obs"):
            obs = dict(prev_st[loc]["obs"], stale=True)
        issued = con.execute("SELECT MAX(issued) FROM forecasts WHERE location=?", (loc,)).fetchone()[0]
        fc = None
        if issued:
            rows = con.execute("SELECT time, sea_level_m FROM forecasts WHERE location=? AND issued=? ORDER BY time", (loc, issued)).fetchall()
            mm = [1000 * r[1] for r in rows]
            an, mean = anomaly(mm)
            fc = {"t0": rows[0][0], "h": hours_since(rows[0][0], [r[0] for r in rows]), "anom_mm": [round(a, 1) for a in an], "mean_mm": round(mean, 1),
                  "n": len(rows), "issued": issued, "query": openmeteo_url(s["lat"], s["lon"])}
        elif prev_st.get(loc, {}).get("forecast"):
            fc = dict(prev_st[loc]["forecast"], stale=True)
        out.append({"location": loc, "uhslc_id": sid, "uhslc_name": s["uhslc_name"], "lat": s["lat"], "lon": s["lon"],
                    "basis": s["basis"], "confidence": s["confidence"], "obs": obs, "forecast": fc})
    return {"generated": iso(now), "window_days": OBS_WINDOW_DAYS, "forecast_days": FORECAST_DAYS,
            "note": "Values are anomalies: each series minus its own mean over the plotted window, in mm. "
                    "UHSLC levels are relative to each gauge's datum and Open-Meteo levels to global mean sea level, "
                    "so only the shape (tides, surges) is comparable, never the absolute offset.",
            "sources": {"uhslc": {"name": "UHSLC Fast Delivery (hourly)", "provider": "University of Hawaii Sea Level Center",
                                  "endpoint": UHSLC_BASE, "licence": "free use and redistribution, not for legal use"},
                        "openmeteo": {"name": "Open-Meteo Marine API, sea_level_height_msl", "provider": "Open-Meteo",
                                      "endpoint": OPENMETEO_BASE, "licence": "CC BY 4.0, free for non-commercial use",
                                      "caveat": "Modelled on a ~8 km grid; Open-Meteo notes limited accuracy in coastal areas."}},
            "stations": out}


def validate_export(doc: dict) -> None:
    """The contract the front end relies on. Raises SchemaError; nothing is written if it fails."""
    if not isinstance(doc.get("stations"), list) or not doc["stations"]:
        raise SchemaError("export: no stations")
    for s in doc["stations"]:
        for k in ("location", "uhslc_id", "lat", "lon"):
            if k not in s:
                raise SchemaError(f"export: station missing {k}")
        for part in ("obs", "forecast"):
            p = s.get(part)
            if p is None:
                continue
            if len(p["h"]) != len(p["anom_mm"]) or p["n"] != len(p["h"]) or p["h"] != sorted(p["h"]):
                raise SchemaError(f"export: {s['location']} {part} arrays inconsistent")
            if any(not math.isfinite(v) for v in p["anom_mm"]):
                raise SchemaError(f"export: {s['location']} {part} has non-finite values")


def write_json_atomic(path: Path, doc: dict) -> bool:
    """Write only if the content changed (ignoring 'generated'), via temp file + rename. Returns True if written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    if path.exists():
        try:
            old = json.loads(path.read_text("utf-8"))
            if {k: v for k, v in old.items() if k != "generated"} == {k: v for k, v in doc.items() if k != "generated"}:
                return False
        except json.JSONDecodeError:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, "utf-8")
    os.replace(tmp, path)
    return True


# ----------------------------------------------------------------------------------------------- run
def load_stations() -> list[dict]:
    return json.loads(STATION_MAP.read_text("utf-8"))["stations"]


def run(*, fetch: bool = True, db_path: Path | str = DB_PATH, export_dir: Path = EXPORT_DIR, get=http_get_json,
        now: datetime | None = None, log=print) -> dict:
    now = now or datetime.now(timezone.utc)
    t_start = time.monotonic()
    stations = load_stations()
    con = connect(db_path)
    upsert_stations(con, stations)
    errors, obs_rows, fc_rows, ok = [], 0, 0, 0
    if fetch:
        retrieved = iso(now)
        issued = now.strftime("%Y-%m-%dT%H:00Z")
        for s in stations:
            got = 0
            try:
                n = upsert_observations(con, s["uhslc_id"], parse_uhslc(get(uhslc_url(s["uhslc_id"], now - timedelta(days=OBS_FETCH_DAYS)))), retrieved)
                obs_rows += n; got += 1
                log(f"  UHSLC {s['uhslc_name']:<11} +{n} rows")
            except Exception as e:  # one station failing must not stop the others
                errors.append(f"UHSLC {s['location']}: {e}")
                log(f"  UHSLC {s['uhslc_name']:<11} FAILED {e}")
            try:
                n = upsert_forecast(con, s["location"], issued, parse_openmeteo(get(openmeteo_url(s["lat"], s["lon"]))))
                fc_rows += n; got += 1
                log(f"  Open-Meteo {s['uhslc_name']:<11} +{n} rows")
            except Exception as e:
                errors.append(f"Open-Meteo {s['location']}: {e}")
                log(f"  Open-Meteo {s['uhslc_name']:<11} FAILED {e}")
            ok += got == 2
            con.commit()
    status = "ok" if not errors else ("partial" if ok or obs_rows or fc_rows else "failed")
    duration = round(time.monotonic() - t_start, 2)
    con.execute("INSERT INTO etl_runs (started, duration_s, status, obs_rows, fc_rows, stations_ok, stations_total, errors) VALUES (?,?,?,?,?,?,?,?)",
                (iso(now), duration, status if fetch else "export-only", obs_rows, fc_rows, ok, len(stations), json.dumps(errors)))
    con.commit()

    live_path, status_path = export_dir / "live_data.json", export_dir / "status.json"
    previous = json.loads(live_path.read_text("utf-8")) if live_path.exists() else None
    prev_status = json.loads(status_path.read_text("utf-8")) if status_path.exists() else {}
    export = build_export(con, stations, now, previous)
    validate_export(export)
    changed = write_json_atomic(live_path, export)

    run_rec = {"started": iso(now), "duration_s": duration, "status": status if fetch else "export-only",
               "obs_rows_changed": obs_rows, "forecast_rows": fc_rows, "stations_ok": ok, "stations_total": len(stations),
               "errors": errors[:10]}
    history = ([run_rec] + prev_status.get("history", []))[:HISTORY_KEEP]
    last_ok = iso(now) if fetch and status in ("ok", "partial") else prev_status.get("last_success")
    obs_last = max((s["obs"]["last"] for s in export["stations"] if s.get("obs")), default=None)
    status_doc = {"generated": iso(now), "last_run": run_rec, "last_success": last_ok,
                  "stations_with_obs": sum(1 for s in export["stations"] if s.get("obs") and not s["obs"].get("stale")),
                  "stations_with_forecast": sum(1 for s in export["stations"] if s.get("forecast") and not s["forecast"].get("stale")),
                  "stations_total": len(stations), "latest_observation": obs_last, "data_changed": changed,
                  "db_rows": {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("observations", "forecasts", "etl_runs")},
                  "history": history}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status_doc, separators=(",", ":")), "utf-8")
    os.replace(tmp, status_path)
    con.close()
    log(f"{status}: {ok}/{len(stations)} stations, +{obs_rows} obs rows, {fc_rows} forecast rows, {duration}s, export changed={changed}")
    return status_doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export-only", action="store_true", help="rebuild docs/live/*.json from the database, no API calls")
    ap.add_argument("--db", default=str(DB_PATH))
    a = ap.parse_args(argv)
    st = run(fetch=not a.export_only, db_path=a.db)
    # exit non-zero only if nothing at all came back, so the workflow shows red but the last good export stays live
    return 1 if st["last_run"]["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
