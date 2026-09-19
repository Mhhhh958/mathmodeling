#!/usr/bin/env python3
"""Validate STEP 00-C global rules/state machine. No training, Word generation, or plotting."""
from __future__ import annotations
import csv, hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/protocol"))
from run_contract import canonical_json, sha256_file, binding_object, binding_digest, make_run_id, make_resume_token, output_inventory

SEED=20260919
PASS_TOKEN="00-C-V2026.09.19-r1-通过"

EXPECTED_HEADERS={
"version_ledger.csv":["record_id","timestamp_utc","step_id","artifact_type","artifact_id","version_id","parent_version_id","status","run_id","code_version_id","data_version_id","config_sha256","environment_sha256","sha256","path","notes"],
"evidence_issue_ledger.csv":["item_id","timestamp_utc","step_id","item_type","severity","claim_or_problem","evidence_status","evidence_refs","owner_status","resolution","affects_steps","opened_run_id","closed_run_id","notes"],
"result_formula_ledger.csv":["item_id","timestamp_utc","step_id","question_id","item_type","name","symbol_or_metric","definition_or_formula","input_refs","output_value","unit","scope","run_id","evidence_refs","status","notes"],
"figure_table_ledger.csv":["item_id","timestamp_utc","step_id","question_id","item_type","filename","caption","claim_supported","data_source_id","run_id","sha256","status","word_location","notes"],
"citation_ledger.csv":["item_id","timestamp_utc","step_id","citation_key","source_type","title_or_name","authors_or_org","year","url_or_identifier","accessed_utc","claim_supported","word_location","status","notes"],
}

def j(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))

def main():
    rules=j(Path("protocol/00C/global_rules.json"))
    state=j(Path("protocol/00C/process_state_card.json"))
    at=j(Path("protocol/00C/templates/A_freeze_package.template.json"))
    bt=j(Path("protocol/00C/templates/B_delivery.template.json"))
    bp=j(Path("protocol/00C/templates/question_argument_blueprint.template.json"))
    cfg_path=ROOT/"protocol/00C/run_config.json"
    env_path=ROOT/"protocol/00B/environment_manifest.json"
    data_path=ROOT/"protocol/00B/data_manifest.draft.json"
    code_manifest_path=ROOT/"protocol/00C/code_manifest.json"

    checks={}
    # A/B boundary
    checks["A_forbids_word"]= "edit_or_rebuild_paper_word" in rules["ab_boundary"]["A"]["forbidden"]
    checks["B_forbids_training_tuning"]= all(x in rules["ab_boundary"]["B"]["forbidden"] for x in ["retrain","retune","rerun_model_selection"])
    checks["B_uses_passed_A_and_word"]= "use_only_passed_A_results" in rules["ab_boundary"]["B"]["allowed"] and "incrementally_edit_unique_mother_word" in rules["ab_boundary"]["B"]["allowed"]

    # Five ledgers are empty header-only tables.
    led_dir=ROOT/"protocol/00C/ledgers"
    for name,expected in EXPECTED_HEADERS.items():
        p=led_dir/name
        rows=list(csv.reader(p.open(encoding="utf-8",newline="")))
        checks[f"ledger_{name}_header"]= bool(rows) and rows[0]==expected
        checks[f"ledger_{name}_empty"]= len(rows)==1

    # Templates
    checks["A_template_required"]= all(k in at for k in ["run_id","versions","parameters","real_results","validation_evidence","anomalies_and_failures","b_handoff","unresolved_issues"])
    checks["B_template_required"]= all(k in bt for k in ["source_A_freeze_package_id","latest_word_before","new_word_after","change_list","affected_ledger_items","figures_and_captions","validation"])
    checks["B_template_no_training"]= bt["training_performed"] is False and bt["tuning_performed"] is False and bt["model_selection_performed"] is False
    checks["blueprint_required"]= all(k in bp for k in ["delivery_from_problem","core_contradiction","baseline","method_choice_reason","key_improvements","primary_metrics_and_fairness","key_tables_and_figures","core_conclusion","next_question_interface"])
    checks["blueprint_has_evidence_guard"]= "real results" in bp["evidence_guard"].lower()

    # Quality / figure / checkpoint / invalidation
    checks["claim_evidence_rule"]= len(rules["writing_quality"]["evidence_words"])>=5 and "number" in rules["writing_quality"]["evidence_requirement"].lower()
    checks["formula_must_be_operational"]= "solving" in rules["writing_quality"]["formula_rule"].lower()
    checks["figure_style_floor"]= "dashboard" in rules["figure_table_floor"]["forbidden_styles"] and rules["figure_table_floor"]["traceability_required"] is True
    checks["checkpoint_inherits_00B"]= rules["checkpoint"]["inherits"]=="protocol/00B/checkpoint_protocol.md"
    checks["pause_keeps_step"]= rules["checkpoint"]["on_pause"]["next_allowed"]=="CURRENT_STEP" and rules["checkpoint"]["on_pause"]["pass_token"] is None
    checks["freeze_invalidation"]= "redo STEP13" in rules["freeze_invalidation"]["action"] and len(rules["freeze_invalidation"]["trigger_after_step13"])>=5

    # State machine target after successful 00-C
    req=["CURRENT_ALLOWED_STEP","LAST_PASS_TOKEN","LATEST_WORD","LATEST_FREEZE_PACKAGE","OPEN_P0","NEXT_ALLOWED"]
    checks["state_fields"]=all(k in state for k in req)
    checks["state_postpass_00D"]=state["CURRENT_ALLOWED_STEP"]=="00-D" and state["NEXT_ALLOWED"]=="00-D"
    checks["state_last_token"]=state["LAST_PASS_TOKEN"]==PASS_TOKEN
    checks["state_no_open_p0"]=state["OPEN_P0"]==[]
    checks["state_latest_word_nullable_before_B"]=state["LATEST_WORD"] is None
    checks["state_latest_freeze_nullable_before_A"]=state["LATEST_FREEZE_PACKAGE"] is None
    checks["illegal_jump_stops"]=state["guard_rules"]["illegal_jump_action"]=="STOP"

    # No Word is created by this protocol package.
    checks["no_word_in_00C"]=not any(p.suffix.lower() in {".doc",".docx"} for p in (ROOT/"protocol/00C").rglob("*") if p.is_file())

    # Create run binding according to 00-B.
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    cfg_sha=sha256_file(cfg_path); env_sha=sha256_file(env_path); data_sha=sha256_file(data_path)
    data_id=f"DATA-DRAFT:{data_sha[:16]}"
    binding=binding_object(f"git:{head}",data_id,cfg_sha,SEED,env_sha)
    digest=binding_digest(binding)
    run_id=make_run_id("00-C",digest)
    out=ROOT/"outputs/runs"/run_id
    (out/"logs").mkdir(parents=True,exist_ok=False)
    (out/"manifests").mkdir()
    (out/"checkpoints/batch-001").mkdir(parents=True)

    # Actual checkpoint protocol smoke test.
    cp=out/"checkpoints/batch-001/checkpoint_state.json"
    cp.write_text(json.dumps({"completed_batches":["rules-and-ledgers-validation"],"pending_batches":[],"natural_batch":"00-C consistency validation"},indent=2)+"\n",encoding="utf-8")
    resume=make_resume_token(run_id,"batch-001",sha256_file(cp))
    (out/"checkpoints/batch-001/RESUME_TOKEN.txt").write_text(resume+"\n",encoding="utf-8")
    checks["resume_token_usable"]=resume.startswith("RESUME_")

    validation={"ok":all(checks.values()),"checks":checks,"binding_digest":digest,"resume_token_smoke_test":resume}
    (out/"logs/validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    pip_freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (out/"manifests/pip_freeze.txt").write_text(pip_freeze,encoding="utf-8")
    runtime={
      "captured_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
      "platform":platform.platform(),"system":platform.system(),"machine":platform.machine(),
      "python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),
      "git_commit":head,"gpu":None,"cuda":None,"pip_freeze_sha256":sha256_file(out/"manifests/pip_freeze.txt")
    }
    (out/"manifests/environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    files,tree_sha=output_inventory(out)
    manifest={
      "schema_version":"00C-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"00-C",
      "status":"completed" if validation["ok"] else "failed",
      "created_utc":runtime["captured_utc"],"execution_mode":"chat_plus_git",
      "binding":{**binding,"binding_digest":digest},
      "code":{"repository":"Mhhhh958/mathmodeling","commit":head,"code_manifest":"protocol/00C/code_manifest.json","code_manifest_sha256":sha256_file(code_manifest_path)},
      "data":{"data_manifest":"protocol/00B/data_manifest.draft.json","data_manifest_sha256":data_sha,"raw_data_version_id":None,"provisional_data_version_id":data_id},
      "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":env_sha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
      "config":{"path":"protocol/00C/run_config.json","sha256":cfg_sha},
      "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":tree_sha},
      "checkpoint":{"protocol_smoke_resume_token":resume,"completed_batches":["rules-and-ledgers-validation"],"pending_batches":[]}
    }
    (out/"manifests/run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"ok":validation["ok"],"run_id":run_id,"binding_digest":digest,"pass_token":PASS_TOKEN},ensure_ascii=False))
    return 0 if validation["ok"] else 2

if __name__=="__main__":
    raise SystemExit(main())
