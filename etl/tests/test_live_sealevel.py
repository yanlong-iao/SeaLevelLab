"""Offline tests for the live ETL: parsing, schema validation, idempotent upsert, export contract, failure policy."""
import json
import urllib.error
from datetime import datetime, timezone

import pytest

from etl import live_sealevel as L

NOW = datetime(2026, 10, 14, 6, 0, tzinfo=timezone.utc)


def uhslc_doc(rows, units=("UTC", "millimeters")):
    return {"table": {"columnNames": ["time", "sea_level"], "columnTypes": ["String", "short"],
                      "columnUnits": list(units), "rows": rows}}


def om_doc(n=4, start_val=0.1):
    from datetime import timedelta
    times = [(NOW.replace(hour=0) + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(n)]
    return {"latitude": -21.2, "longitude": -159.8, "utc_offset_seconds": 0,
            "hourly_units": {"time": "iso8601", "sea_level_height_msl": "m"},
            "hourly": {"time": times,
                       "sea_level_height_msl": [start_val + 0.01 * h for h in range(n)]}}


# ---------------------------------------------------------------- parsing + schema
def test_parse_uhslc_drops_fill_and_null():
    rows = [["2026-08-30T00:00:00Z", 499], ["2026-08-30T01:00:00Z", -32767], ["2026-08-30T02:00:00Z", None],
            ["2026-08-30T03:00:00Z", 272]]
    assert L.parse_uhslc(uhslc_doc(rows)) == [("2026-08-30T00:00:00Z", 499), ("2026-08-30T03:00:00Z", 272)]


@pytest.mark.parametrize("doc", [
    {},                                                                    # no table
    {"table": {"columnNames": ["time", "wind"], "rows": []}},              # wrong column
    uhslc_doc([["2026-08-30T00:00:00Z", 1]], units=("UTC", "metres")),     # wrong unit
    uhslc_doc([["30/08/2026", 1]]),                                        # bad time
    uhslc_doc([["2026-08-30T00:00:00Z", 999999]]),                         # implausible value
])
def test_parse_uhslc_rejects_bad_schema(doc):
    with pytest.raises(L.SchemaError):
        L.parse_uhslc(doc)


def test_parse_openmeteo_normalises_time_to_utc_iso():
    out = L.parse_openmeteo(om_doc(2))
    assert out[0] == ("2026-10-14T00:00:00Z", pytest.approx(0.1))
    assert len(out) == 2


@pytest.mark.parametrize("mutate", [
    lambda d: d["hourly"]["time"].__setitem__(0, "2026-10-14T100:00"),
    lambda d: d.update(error=True, reason="bad coordinates"),
    lambda d: d["hourly"].pop("sea_level_height_msl"),
    lambda d: d["hourly_units"].update(sea_level_height_msl="ft"),
    lambda d: d["hourly"]["time"].pop(),
    lambda d: d.update(utc_offset_seconds=3600),
])
def test_parse_openmeteo_rejects_bad_schema(mutate):
    d = om_doc(3); mutate(d)
    with pytest.raises(L.SchemaError):
        L.parse_openmeteo(d)


def test_urls_are_exact_api_queries():
    u = L.uhslc_url(23, NOW)
    assert u.startswith(L.UHSLC_BASE + "?time,sea_level&uhslc_id=23&time>=2026-10-14T06")
    o = L.openmeteo_url(-21.2, -159.7867)
    assert "hourly=sea_level_height_msl" in o and "forecast_days=7" in o and "latitude=-21.2000" in o


# ---------------------------------------------------------------- database
def test_upsert_observations_is_idempotent():
    con = L.connect(":memory:")
    rows = [("2026-08-30T00:00:00Z", 10), ("2026-08-30T01:00:00Z", 20)]
    assert L.upsert_observations(con, 23, rows, "r1") == 2
    assert L.upsert_observations(con, 23, rows, "r2") == 0                     # same data: nothing changes
    assert L.upsert_observations(con, 23, [("2026-08-30T01:00:00Z", 25)], "r3") == 1   # a revised value updates in place
    assert con.execute("SELECT COUNT(*), SUM(sea_level_mm) FROM observations").fetchone() == (2, 35)


def test_forecast_keeps_last_14_issues():
    con = L.connect(":memory:")
    for d in range(20):
        L.upsert_forecast(con, "TONGA", f"2026-10-{d + 1:02d}T06:00Z", [("2026-10-20T00:00:00Z", 0.1)])
    assert con.execute("SELECT COUNT(DISTINCT issued) FROM forecasts").fetchone()[0] == 14


# ---------------------------------------------------------------- end to end with a fake HTTP layer
def fake_get(fail_uhslc=(), fail_om=()):
    def get(url):
        if url.startswith(L.UHSLC_BASE):
            sid = int(url.split("uhslc_id=")[1].split("&")[0])
            if sid in fail_uhslc:
                raise RuntimeError("GET failed after 4 attempts: timed out")
            return uhslc_doc([[f"2026-08-{d:02d}T{h:02d}:00:00Z", 1000 + 300 * (h % 12 > 5)] for d in range(1, 31) for h in range(24)])
        if any(str(f) in url for f in fail_om):
            raise RuntimeError("GET failed after 4 attempts: 503")
        return om_doc(168)
    return get


def test_run_exports_valid_files(tmp_path):
    st = L.run(db_path=tmp_path / "db.sqlite", export_dir=tmp_path, get=fake_get(), now=NOW, log=lambda *_: None)
    assert st["last_run"]["status"] == "ok" and st["stations_with_obs"] == 13 and st["stations_with_forecast"] == 13
    doc = json.loads((tmp_path / "live_data.json").read_text())
    L.validate_export(doc)
    s = doc["stations"][0]
    assert s["obs"]["n"] == len(s["obs"]["h"]) and s["forecast"]["n"] == 168
    assert abs(sum(s["obs"]["anom_mm"]) / s["obs"]["n"]) < 0.5                # anomalies are centred on the window mean
    assert s["obs"]["query"].startswith(L.UHSLC_BASE) and s["forecast"]["query"].startswith(L.OPENMETEO_BASE)


def test_rerun_with_same_data_does_not_rewrite_export(tmp_path):
    kw = dict(db_path=tmp_path / "db.sqlite", export_dir=tmp_path, get=fake_get(), now=NOW, log=lambda *_: None)
    L.run(**kw)
    st = L.run(**kw)
    assert st["data_changed"] is False and st["last_run"]["obs_rows_changed"] == 0
    assert len(st["history"]) == 2


def test_failed_station_keeps_last_good_export(tmp_path):
    L.run(db_path=tmp_path / "a.sqlite", export_dir=tmp_path, get=fake_get(), now=NOW, log=lambda *_: None)
    # fresh database (e.g. the runner cache was evicted) and UHSLC down for Rarotonga (id 23)
    st = L.run(db_path=tmp_path / "b.sqlite", export_dir=tmp_path, get=fake_get(fail_uhslc=(23,)), now=NOW, log=lambda *_: None)
    assert st["last_run"]["status"] == "partial" and st["last_run"]["errors"]
    doc = json.loads((tmp_path / "live_data.json").read_text())
    cook = next(s for s in doc["stations"] if s["uhslc_id"] == 23)
    assert cook["obs"]["stale"] is True and cook["obs"]["n"] > 0              # previous data kept, marked stale


def test_total_failure_exits_nonzero_but_keeps_files(tmp_path, monkeypatch):
    L.run(db_path=tmp_path / "a.sqlite", export_dir=tmp_path, get=fake_get(), now=NOW, log=lambda *_: None)
    before = (tmp_path / "live_data.json").read_text()
    def down(url): raise RuntimeError("network unreachable")
    st = L.run(db_path=tmp_path / "c.sqlite", export_dir=tmp_path, get=down, now=NOW, log=lambda *_: None)
    assert st["last_run"]["status"] == "failed" and st["last_success"] == L.iso(NOW)  # last success is the earlier run
    assert json.loads((tmp_path / "live_data.json").read_text())["stations"][0]["obs"]["n"] == json.loads(before)["stations"][0]["obs"]["n"]


def test_http_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": 1}'
    def opener(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("timed out")
        return Resp()
    assert L.http_get_json("https://x.test", opener=opener, backoff=0) == {"ok": 1}
    assert calls["n"] == 3


def test_http_does_not_retry_client_errors():
    calls = {"n": 0}
    def opener(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError("https://x.test", 404, "not found", None, None)
    with pytest.raises(urllib.error.HTTPError):
        L.http_get_json("https://x.test", opener=opener, backoff=0)
    assert calls["n"] == 1
