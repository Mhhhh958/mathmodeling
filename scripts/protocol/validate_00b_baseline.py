#!/usr/bin/env python3
"""Validate STEP 00-B baseline contracts without training or plotting."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_contract import sha256_file,binding_object,binding_digest,make_run_id,make_resume_token

REQUIRED=[
    ROOT/"protocol/00B/environment_manifest.json",
    ROOT/"protocol/00B/data_manifest.draft.json",
    ROOT/"protocol/00B/run_manifest.template.json",
    ROOT/"protocol/00B/run_config.yaml",
    ROOT/"protocol/00B/code_manifest.json",
    ROOT/"protocol/00B/checkpoint_protocol.md",
    ROOT/"protocol/00B/requirements-lock-00b.txt",
]

def main() -> int:
    missing=[p.as_posix() for p in REQUIRED if not p.exists()]
    if missing:
        raise SystemExit("Missing baseline files: "+", ".join(missing))
    env_path=ROOT/"protocol/00B/environment_manifest.json"
    data_path=ROOT/"protocol/00B/data_manifest.draft.json"
    cfg_path=ROOT/"protocol/00B/run_config.yaml"
    code_path=ROOT/"protocol/00B/code_manifest.json"
    env_sha=sha256_file(env_path)
    data_sha=sha256_file(data_path)
    cfg_sha=sha256_file(cfg_path)
    code_manifest=json.loads(code_path.read_text(encoding="utf-8"))
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    binding=binding_object(f"git:{head}",f"DATA-DRAFT:{data_sha[:16]}",cfg_sha,20260919,env_sha)
    digest=binding_digest(binding)
    run1,run2=make_run_id("00-B",digest),make_run_id("00-B",digest)
    checks={
        "run_ids_unique":run1 != run2,
        "run_ids_embed_binding":digest[:8] in run1 and digest[:8] in run2,
        "binding_changes_if_seed_changes":binding_digest(binding_object(f"git:{head}",f"DATA-DRAFT:{data_sha[:16]}",cfg_sha,20260920,env_sha)) != digest,
        "raw_data_not_frozen_before_01A":json.loads(data_path.read_text(encoding="utf-8"))["raw_data_version_id"] is None,
        "output_rule_run_scoped":"outputs/runs/<run_id>" in (ROOT/"protocol/00B/README.md").read_text(encoding="utf-8"),
        "code_manifest_lists_protocol_code":len(code_manifest.get("tracked_protocol_code",[])) >= 2,
        "resume_token_works":make_resume_token(run1,"batch-001","f"*64).startswith("RESUME_"),
    }
    ok=all(checks.values())
    result={
        "ok":ok,
        "code_version_id":f"git:{head}",
        "environment_sha256":env_sha,
        "data_manifest_sha256":data_sha,
        "run_config_sha256":cfg_sha,
        "binding_digest":digest,
        "sample_run_ids":[run1,run2],
        "checks":checks,
    }
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())
