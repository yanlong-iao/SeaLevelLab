"""The Data & Integrations page must not drift from the code: every referenced file and symbol has to exist."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = json.loads((ROOT / "docs" / "data" / "integrations.json").read_text("utf-8"))
REQUIRED = ["name", "type", "provider", "obtained", "auth", "frequency", "format", "transform", "stored", "used_by", "failure", "code"]
TYPES = {"open dataset", "REST API", "tile service", "SDK", "LLM API", "file store", "database", "platform"}


def refs():
    for x in DOC["integrations"]:
        for c in x["code"]:
            yield x["name"], c
    for s in DOC["request_flow"]["steps"]:
        if s.get("code"):
            yield "request_flow", s["code"]


def test_every_integration_has_all_fields():
    for x in DOC["integrations"]:
        missing = [k for k in REQUIRED if not x.get(k)]
        assert not missing, f"{x.get('name')}: missing {missing}"
        assert x["type"] in TYPES, f"{x['name']}: unknown type {x['type']}"


def test_code_references_exist():
    for owner, c in refs():
        p = ROOT / c["path"]
        assert p.exists(), f"{owner}: {c['path']} does not exist"
        if c.get("symbol"):
            assert c["symbol"] in p.read_text("utf-8"), f"{owner}: '{c['symbol']}' not found in {c['path']}"


def test_station_map_covers_the_dataset():
    import csv
    locs = {r["Location"] for r in csv.DictReader(open(ROOT / "Processed_Final_data.csv", encoding="utf-8"))}
    mapped = {s["location"] for s in json.loads((ROOT / "etl" / "station_map.json").read_text("utf-8"))["stations"]}
    assert locs == mapped, f"unmapped: {locs - mapped}, extra: {mapped - locs}"
