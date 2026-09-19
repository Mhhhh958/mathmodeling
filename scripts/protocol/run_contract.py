#!/usr/bin/env python3
"""STEP 00-B run/version contract utilities. No model training is performed."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "00B-1.0"

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_head(repo_root: Path) -> str:
    return subprocess.check_output(["git","-C",str(repo_root),"rev-parse","HEAD"],text=True).strip()

def collect_environment() -> dict[str, Any]:
    gpu = None
    cuda = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            cuda = torch.version.cuda
    except Exception:
        pass
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count_logical": os.cpu_count(),
        "gpu": gpu,
        "cuda": cuda,
    }

def binding_object(code_version_id: str, data_version_id: str, run_config_sha256: str,
                   seed: int, environment_sha256: str) -> dict[str, Any]:
    return {
        "code_version_id": code_version_id,
        "data_version_id": data_version_id,
        "run_config_sha256": run_config_sha256,
        "seed": int(seed),
        "environment_sha256": environment_sha256,
    }

def binding_digest(binding: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(binding).encode("utf-8"))

def make_run_id(step_id: str, digest: str) -> str:
    step = "".join(c if c.isalnum() else "-" for c in step_id).strip("-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{step}_{ts}_{digest[:8]}_{uuid.uuid4().hex[:8]}"

def output_inventory(root: Path):
    records=[]
    if root.exists():
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            records.append({"relative_path":p.relative_to(root).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    return records, sha256_bytes(canonical_json(records).encode("utf-8"))

def checkpoint_digest(completed, pending, checkpoint_files) -> str:
    return sha256_bytes(canonical_json({"completed_batches":completed,"pending_batches":pending,"files":checkpoint_files}).encode("utf-8"))

def make_resume_token(run_id: str, batch_id: str, checkpoint_digest_hex: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in batch_id)
    return f"RESUME_{run_id}_{safe}_{checkpoint_digest_hex[:12]}"

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest",action="store_true")
    args=ap.parse_args()
    if args.selftest:
        b=binding_object("git:test","DATA-DRAFT","a"*64,20260919,"b"*64)
        d=binding_digest(b)
        r1,r2=make_run_id("00-B",d),make_run_id("00-B",d)
        assert r1 != r2 and d[:8] in r1 and d[:8] in r2
        assert binding_digest(binding_object("git:test","DATA-DRAFT","a"*64,20260920,"b"*64)) != d
        tok=make_resume_token(r1,"batch-001","c"*64)
        assert tok.startswith("RESUME_")
        print(canonical_json({"ok":True,"binding_digest":d,"run_id_1":r1,"run_id_2":r2,"resume_token":tok}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
