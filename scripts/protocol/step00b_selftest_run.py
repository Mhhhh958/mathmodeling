#!/usr/bin/env python3
"""Create a real STEP 00-B protocol validation run. No model training or plotting."""
from __future__ import annotations
import json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_contract import (
    canonical_json, sha256_bytes, sha256_file, binding_object, binding_digest,
    make_run_id, make_resume_token, output_inventory, collect_environment
)

SEED=20260919

def cpu_model():
    p=Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":",1)[1].strip()
    return platform.processor() or None

def main() -> int:
    env_path=ROOT/"protocol/00B/environment_manifest.json"
    data_path=ROOT/"protocol/00B/data_manifest.draft.json"
    cfg_path=ROOT/"protocol/00B/run_config.yaml"
    code_path=ROOT/"protocol/00B/code_manifest.json"

    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    env_sha=sha256_file(env_path)
    data_sha=sha256_file(data_path)
    cfg_sha=sha256_file(cfg_path)
    code_sha=sha256_file(code_path)
    data_id=f"DATA-DRAFT:{data_sha[:16]}"
    binding=binding_object(f"git:{head}",data_id,cfg_sha,SEED,env_sha)
    digest=binding_digest(binding)
    run_id=make_run_id("00-B",digest)
    out=ROOT/"outputs/runs"/run_id
    (out/"manifests").mkdir(parents=True,exist_ok=False)
    (out/"logs").mkdir()
    (out/"checkpoints/batch-001").mkdir(parents=True)

    runtime=collect_environment()
    runtime["cpu_model"]=cpu_model()
    runtime["github_actions"]=os.getenv("GITHUB_ACTIONS")=="true"
    runtime["github_runner_os"]=os.getenv("RUNNER_OS")
    runtime["github_runner_arch"]=os.getenv("RUNNER_ARCH")
    runtime["git_commit"]=head

    pip_freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (out/"manifests/pip_freeze.txt").write_text(pip_freeze,encoding="utf-8")
    runtime["pip_freeze_sha256"]=sha256_file(out/"manifests/pip_freeze.txt")
    (out/"manifests/environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    # Binding uniqueness/stability validation.
    run2=make_run_id("00-B",digest)
    changed=binding_digest(binding_object(f"git:{head}",data_id,cfg_sha,SEED+1,env_sha))
    checks={
        "run_id_unique":run_id != run2,
        "run_id_embeds_binding":digest[:8] in run_id and digest[:8] in run2,
        "same_binding_same_digest":binding_digest(binding)==digest,
        "seed_change_changes_binding":changed != digest,
        "git_code_version_bound":binding["code_version_id"]==f"git:{head}",
        "raw_data_version_not_frozen":json.loads(data_path.read_text(encoding="utf-8"))["raw_data_version_id"] is None,
        "config_hash_bound":binding["run_config_sha256"]==cfg_sha,
        "environment_hash_bound":binding["environment_sha256"]==env_sha,
    }

    cp=(out/"checkpoints/batch-001/checkpoint_state.json")
    cp.write_text(json.dumps({"completed_batches":["batch-001"],"pending_batches":[],"note":"protocol selftest only"},indent=2)+"\n",encoding="utf-8")
    cp_digest=sha256_file(cp)
    resume=make_resume_token(run_id,"batch-001",cp_digest)
    (out/"checkpoints/batch-001/RESUME_TOKEN.txt").write_text(resume+"\n",encoding="utf-8")
    checks["resume_token_created"]=resume.startswith("RESUME_")
    checks["output_root_is_run_scoped"]=out.parent.name=="runs" and out.name==run_id

    validation={"ok":all(checks.values()),"checks":checks,"binding_digest":digest,"sample_second_run_id":run2,"resume_token":resume}
    (out/"logs/validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    files,tree_sha=output_inventory(out)
    # Exclude run_manifest itself from its output inventory to avoid self-hash recursion.
    manifest={
        "schema_version":"00B-1.0",
        "manifest_type":"run_manifest",
        "run_id":run_id,
        "step_id":"00-B",
        "status":"completed" if validation["ok"] else "failed",
        "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "execution_mode":"chat_plus_git",
        "binding":{**binding,"binding_digest":digest},
        "code":{"repository":"Mhhhh958/mathmodeling","commit":head,"code_manifest":"protocol/00B/code_manifest.json","code_manifest_sha256":code_sha},
        "data":{"data_manifest":"protocol/00B/data_manifest.draft.json","data_manifest_sha256":data_sha,"raw_data_version_id":None,"provisional_data_version_id":data_id},
        "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":env_sha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
        "config":{"path":"protocol/00B/run_config.yaml","sha256":cfg_sha},
        "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":tree_sha},
        "checkpoint":{"resume_token":resume,"completed_batches":["batch-001"],"pending_batches":[]},
        "parent_run_ids":[]
    }
    (out/"manifests/run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"ok":validation["ok"],"run_id":run_id,"binding_digest":digest,"output_root":str(out.relative_to(ROOT))},ensure_ascii=False))
    return 0 if validation["ok"] else 2

if __name__=="__main__":
    raise SystemExit(main())
