"""Databento batch-pull orchestrator.

Three modes:
  --quote      Print estimated USD cost for every pull. No charge. No network writes.
  --sample     Submit small (5-day) batch jobs to validate parsing + schema fields.
  --confirm    Submit the full batch jobs as defined in the YAML.

The DATABENTO_API_KEY env var must be set. Sample/confirm runs append entries to
data/raw/manifest.json with job_id, dataset, schema, symbols, start, end, cost, sha256.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
CHUNK_DIR = INTERIM_DIR / "id_list_chunks"
SAMPLE_DAYS = 5
STAGE2_DATASET = "OPRA.PILLAR"
STAGE2_SCHEMA = "statistics"


@dataclass
class PullSpec:
    name: str
    dataset: str
    schema: str
    symbols: list[str]
    stype_in: str
    start: str
    end: str
    description: str = ""
    expected_cost_usd: float | None = None
    time_window_et: list[str] | None = None


def load_specs(config_path: Path) -> list[PullSpec]:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    start, end = cfg["start"], cfg["end"]
    specs = []
    for p in cfg["pulls"]:
        if p.get("enabled", True) is False:
            continue
        specs.append(PullSpec(
            name=p["name"],
            dataset=p["dataset"],
            schema=p["schema"],
            symbols=p["symbols"],
            stype_in=p.get("stype_in", "raw_symbol"),
            start=p.get("start", start),
            end=p.get("end", end),
            description=p.get("description", ""),
            expected_cost_usd=p.get("expected_cost_usd"),
            time_window_et=p.get("time_window_et"),
        ))
    return specs


def get_client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY not set. Export it before running.")
    return db.Historical(key)


def quote(specs: list[PullSpec]) -> None:
    """Print estimated cost from Databento metadata API. No charge."""
    client = get_client()
    total = 0.0
    n_ok = 0
    n_err = 0
    errors_seen: set[str] = set()
    print(f"{'name':<28} {'dataset':<14} {'schema':<12} {'metadata $':>11}")
    print("-" * 70)
    for s in specs:
        try:
            est = client.metadata.get_cost(
                dataset=s.dataset,
                schema=s.schema,
                symbols=s.symbols,
                stype_in=s.stype_in,
                start=s.start,
                end=s.end,
            )
        except Exception as exc:
            n_err += 1
            err = str(exc).splitlines()[0][:60]
            errors_seen.add(err)
            print(f"{s.name:<28} {s.dataset:<14} {s.schema:<12} {'ERROR':>11}   {err}")
            continue
        total += est
        n_ok += 1
        print(f"{s.name:<28} {s.dataset:<14} {s.schema:<12} {est:>11.2f}")
    print("-" * 70)
    print(f"TOTAL (metadata)  : {total:>11.2f}   (specs ok={n_ok}, err={n_err})")
    if errors_seen:
        print("\nUnique error messages seen:")
        for e in sorted(errors_seen):
            print(f"  - {e}")


def _shrink_to_sample(spec: PullSpec) -> PullSpec:
    start_dt = dt.date.fromisoformat(spec.start)
    sample_end = start_dt + dt.timedelta(days=SAMPLE_DAYS)
    return PullSpec(**{**asdict(spec), "end": sample_end.isoformat()})


def submit(specs: list[PullSpec], sample: bool) -> None:
    """Submit batch jobs. Caller has already confirmed they want to spend."""
    client = get_client()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    for raw in specs:
        s = _shrink_to_sample(raw) if sample else raw
        out_dir = RAW_DIR / s.name / ("sample" if sample else "full")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[submit] {s.name} {s.dataset}/{s.schema} {s.start}->{s.end}")
        job = client.batch.submit_job(
            dataset=s.dataset,
            symbols=s.symbols,
            stype_in=s.stype_in,
            schema=s.schema,
            start=s.start,
            end=s.end,
            encoding="dbn",
            compression="zstd",
            split_duration="day",
        )
        job_id = job["id"]
        print(f"  job_id={job_id} state={job.get('state')}")
        _wait_for_job(client, job_id)
        files = client.batch.download(job_id=job_id, output_dir=str(out_dir))
        sha = _hash_dir(out_dir)
        manifest.setdefault("jobs", []).append({
            "name": s.name,
            "dataset": s.dataset,
            "schema": s.schema,
            "symbols": s.symbols,
            "start": s.start,
            "end": s.end,
            "sample": sample,
            "job_id": job_id,
            "out_dir": str(out_dir.relative_to(REPO_ROOT)),
            "n_files": len(files),
            "sha256": sha,
            "submitted_at": dt.datetime.utcnow().isoformat() + "Z",
        })
        _save_manifest(manifest)
        print(f"  downloaded {len(files)} files -> {out_dir.relative_to(REPO_ROOT)}")


VALID_STATES = ["queued", "processing", "done", "expired"]


def _get_job_state(client, job_id: str) -> str | None:
    """Return state for a single job. Databento SDK only exposes list_jobs."""
    jobs = client.batch.list_jobs(states=VALID_STATES)
    for j in jobs:
        if j.get("id") == job_id:
            return j.get("state")
    return None


def _wait_for_job(client, job_id: str, poll_seconds: int = 15, timeout_seconds: int = 3600) -> None:
    start = time.time()
    last_state = None
    while True:
        state = _get_job_state(client, job_id)
        if state == "done":
            return
        if state == "expired":
            raise RuntimeError(f"job {job_id} ended in state {state}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"job {job_id} not done after {timeout_seconds}s; last state={state}")
        if state != last_state:
            print(f"  waiting... state={state}")
            last_state = state
        time.sleep(poll_seconds)


def resume(job_id: str, name_hint: str | None = None) -> None:
    """Download an already-submitted batch job. No new charge."""
    client = get_client()
    out_dir = RAW_DIR / (name_hint or job_id) / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resume] job_id={job_id}, polling state...")
    _wait_for_job(client, job_id)
    files = client.batch.download(job_id=job_id, output_dir=str(out_dir))
    sha = _hash_dir(out_dir)
    manifest = _load_manifest()
    manifest.setdefault("jobs", []).append({
        "name": name_hint or job_id,
        "job_id": job_id,
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
        "n_files": len(files),
        "sha256": sha,
        "resumed_at": dt.datetime.utcnow().isoformat() + "Z",
    })
    _save_manifest(manifest)
    print(f"  downloaded {len(files)} files -> {out_dir.relative_to(REPO_ROOT)}")


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"jobs": []}


def _save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def _load_chunk_id_specs(start: str, end: str) -> list[PullSpec]:
    """Read data/interim/id_list_chunks/*.json into PullSpec list.

    Each chunk file represents one (month, sub-chunk) batch staying under
    Databento's 2000-symbols-per-request cap.
    """
    if not CHUNK_DIR.exists():
        sys.exit(f"missing {CHUNK_DIR} — run `python -m ingest.build_id_list` first")
    specs = []
    for p in sorted(CHUNK_DIR.glob("*.json")):
        info = json.loads(p.read_text())
        c_start = max(info["start"], start)
        c_end = min(info["end"], end)
        if c_start > c_end:
            continue
        specs.append(PullSpec(
            name=f"opra_statistics_{info['label']}",
            dataset=STAGE2_DATASET,
            schema=STAGE2_SCHEMA,
            symbols=[str(i) for i in info["instrument_ids"]],
            stype_in="instrument_id",
            start=c_start,
            end=c_end,
            description=f"Filtered SPY OI statistics, {info['label']}, {info['n_ids']} ids.",
            expected_cost_usd=None,
        ))
    return specs


def _load_window(config_path: Path) -> tuple[str, str]:
    cfg = yaml.safe_load(config_path.read_text())
    return cfg["start"], cfg["end"]


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quote", action="store_true",
                   help="Print cost estimate only for stage-1 pulls in the YAML.")
    g.add_argument("--quote-stage2", action="store_true",
                   help="Print cost estimate for stage-2 filtered statistics pulls.")
    g.add_argument("--sample", action="store_true", help="Pull a 5-day stage-1 sample.")
    g.add_argument("--confirm", action="store_true", help="Pull stage-1 in full.")
    g.add_argument("--confirm-stage2", action="store_true",
                   help="Pull stage-2 filtered statistics in full.")
    g.add_argument("--resume", metavar="JOB_ID",
                   help="Download an already-submitted job (no new charge).")
    ap.add_argument("--name", help="Optional name hint for --resume; used as subfolder name.")
    args = ap.parse_args()

    if args.resume:
        resume(args.resume, name_hint=args.name)
        return

    if args.quote_stage2 or args.confirm_stage2:
        start, end = _load_window(args.config)
        specs = _load_chunk_id_specs(start, end)
        if not specs:
            sys.exit("no stage-2 specs found.")
    else:
        specs = load_specs(args.config)

    if args.quote or args.quote_stage2:
        quote(specs)
    elif args.sample:
        submit(specs, sample=True)
    elif args.confirm:
        print("CONFIRM: stage-1 batch jobs may incur real charges.")
        print(f"  budget target ~$80-100 of credit (see {args.config})")
        ans = input("Type 'yes' to proceed: ").strip().lower()
        if ans != "yes":
            sys.exit("aborted.")
        submit(specs, sample=False)
    elif args.confirm_stage2:
        n = len(specs)
        print(f"CONFIRM: {n} stage-2 batch jobs (one per month-chunk), filtered statistics.")
        print("  Last --quote-stage2: ~$95 for the monthly-only universe "
              "(2024-08→2026-05). Re-quote first if the config has changed.")
        ans = input("Type 'yes' to proceed: ").strip().lower()
        if ans != "yes":
            sys.exit("aborted.")
        submit(specs, sample=False)


# Imported here to keep the optional dependency out of the top-of-module surface
# for code paths that don't use it (e.g. quote-only).


if __name__ == "__main__":
    main()
