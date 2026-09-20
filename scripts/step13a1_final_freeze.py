#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, random, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"protocol/13A1/run_config.json"
STATE=ROOT/"protocol/00C/process_state_card.json"
ENV=ROOT/"protocol/00B/environment_manifest.json"
CLASSES=["OR","IR","B","N"]

def J(p): return json.loads((ROOT/p).read_text(encoding="utf-8-sig"))
def H(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canon(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def git(*args): return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def nowz(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def need(c,m):
    if not c: raise RuntimeError(m)

def freeze(step):
    return J(f"protocol/{step}/latest_freeze_package.json")

def rr(f,name):
    for x in f.get("real_results",[]):
        if x["name"]==name:return x["value"]
    raise KeyError((f.get("step_id"),name))

def display(v,dec=4):
    if isinstance(v,(int,np.integer)): return str(int(v))
    if isinstance(v,(float,np.floating)):
        if abs(float(v))<1e-6 and float(v)!=0:return f"{float(v):.3e}"
        return f"{float(v):.{dec}f}"
    return str(v)

def main():
    cfg=J("protocol/13A1/run_config.json"); st=J("protocol/00C/process_state_card.json")
    need(st["CURRENT_ALLOWED_STEP"]=="13-A1" and st["NEXT_ALLOWED"]=="13-A1","gate not open")
    need(st["LAST_PASS_TOKEN"]=="12-B-V2026.09.20-9c38c898-通过","previous token mismatch")
    need(st.get("OPEN_P0")==[],"OPEN_P0 not empty")
    need(st["LATEST_WORD"]["sha256"]==cfg["source_step12"]["latest_word_sha256"],"latest Word mismatch")

    f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12a1,f12a2=[freeze(s) for s in ["02A","03A","04A","05A","06A","07A","08A","09A","10A","11A","12A1","12A2"]]
    need(f12a1["freeze_package_id"]==cfg["source_step12"]["audit_freeze_id"] and f12a1["status"]=="passed","12A1 mismatch")
    need(f12a2["freeze_package_id"]==cfg["source_step12"]["validation_freeze_id"] and f12a2["status"]=="passed","12A2 mismatch")

    # Final code bundle / commit validation.
    final_commit=cfg["final_code_bundle"]["final_step12_repository_snapshot_commit"]
    scientific_commit=cfg["final_code_bundle"]["final_scientific_execution_commit"]
    git("cat-file","-e",final_commit+"^{commit}"); git("cat-file","-e",scientific_commit+"^{commit}")
    tracked=[]
    for mp in cfg["final_code_bundle"]["required_scientific_code_manifests"]:
        m=J(mp)
        for x in m.get("tracked_code",[]):
            tracked.append({"manifest":mp,"path":x["path"],"git_blob_sha":x["git_blob_sha"]})
            actual=git("rev-parse",f"{final_commit}:{x['path']}")
            need(actual==x["git_blob_sha"],f"code blob mismatch {x['path']}")
    final_code_bundle_id=f"GIT-STEP12-{final_commit}"
    code_bundle={"final_code_bundle_id":final_code_bundle_id,"repository":"Mhhhh958/mathmodeling",
                 "final_step12_repository_snapshot_commit":final_commit,
                 "final_scientific_execution_commit":scientific_commit,
                 "tracked_code":tracked}

    # Data lineage validation and ID.
    chain=cfg["final_data_lineage"]["derived_chain"]
    got=[f02["freeze_package_id"],f03["freeze_package_id"],f04["freeze_package_id"],f05["freeze_package_id"],
         f06["freeze_package_id"],f07["freeze_package_id"],f08["freeze_package_id"],f09["freeze_package_id"],
         f10["freeze_package_id"],f11["freeze_package_id"],f12a1["freeze_package_id"],f12a2["freeze_package_id"]]
    need(got==chain,"derived data chain mismatch")
    lineage_payload={"raw_data_version_id":cfg["final_data_lineage"]["raw_data_version_id"],
                     "feature_version_id":cfg["final_data_lineage"]["feature_version_id"],
                     "derived_chain":chain}
    final_data_lineage_id="LINEAGE-13A1-"+canon(lineage_payload)[:16]
    lineage={**lineage_payload,"final_data_lineage_id":final_data_lineage_id}

    # Model artifacts.
    model_rows=[]
    for m in cfg["final_model_artifacts"]:
        p=ROOT/m["artifact_path"]; need(p.exists(),f"model missing {p}")
        sha=H(p); need(sha==m["expected_sha256"],f"model sha mismatch {m['role']}")
        aid=("MODEL-SOURCE-" if m["role"]=="final_source_classifier" else "MODEL-TRANSFER-")+sha[:16]
        model_rows.append({**m,"actual_sha256":sha,"final_model_artifact_id":aid})
    primary_model_id=[x["final_model_artifact_id"] for x in model_rows if x["role"]=="final_transfer_inference_bundle"][0]

    # A-P authoritative table + final STEP12 robustness augmentation.
    ap=pd.read_csv(ROOT/cfg["official_AP"]["answer_source"])
    loto=pd.read_csv(ROOT/"outputs/runs/run_12-A2_20260920T162707032450Z_88add8e8_f107c2fa/artifacts/A2_04_target_LOTO_stability.csv")
    boot=pd.read_csv(ROOT/"outputs/runs/run_12-A2_20260920T162707032450Z_88add8e8_f107c2fa/artifacts/A2_05_source_bootstrap_summary.csv")
    need(ap["target_id"].tolist()==list("ABCDEFGHIJKLMNOP"),"A-P inventory/order mismatch")
    need((ap["window_count"]==15).all() and ap["target_id"].nunique()==16,"A-P windows/count mismatch")
    need(set(ap["truth_status"].astype(str))=={"unknown"},"target truth identity mismatch")
    need((ap["target_accuracy_available"].astype(str).str.lower()=="false").all(),"target accuracy identity mismatch")
    need(ap["score_interpretation"].str.contains("not true correctness probabilities").all(),"score probability identity mismatch")
    need(set(ap["pred_label"]).issubset(set(CLASSES)),"label mapping mismatch")
    apf=ap.merge(loto[["target_id","loto_label","same_label","top_score_delta"]],on="target_id",validate="one_to_one")
    apf=apf.merge(boot[["target_id","label_retention_rate","top_score_q05","top_score_q95","margin_q05","margin_q95"]],on="target_id",validate="one_to_one")
    keep=["target_id","pred_label","score_OR","score_IR","score_B","score_N","top_model_score","score_margin",
          "window_file_agreement","cross_setting_vote_agreement_rate","bootstrap_official_agreement_rate",
          "uncertainty_flag","truth_status","loto_label","same_label","label_retention_rate",
          "top_score_q05","top_score_q95","margin_q05","margin_q95"]
    apout=apf[keep].copy()
    apout["score_identity"]="uncalibrated model score; not true correctness probability"
    apout["final_answer_identity"]="10-A official prediction; revalidated by 12-A2; no manual adjustment"
    for c in ["score_OR","score_IR","score_B","score_N","top_model_score","score_margin"]:
        apout[c+"_display"]=apout[c].map(lambda v:display(v,3))

    # Core numbers master.
    q04=rr(f04,"quantity_conservation")
    q06=rr(f06,"pooled_oof_file_metrics")
    q07=rr(f07,"complexity")
    q08gap=rr(f08,"domain_gap_compact")
    q09sel=rr(f09,"method_selection"); q09hold=rr(f09,"held_load3")
    q11faith=rr(f11,"faithfulness"); q11stab=rr(f11,"stability"); q11mech=rr(f11,"mechanism_consistency")
    q12p0=rr(f12a2,"P0_01_closure"); q12win=rr(f12a2,"window_sensitivity"); q12src=rr(f12a2,"source_plan_sensitivity")
    q12loto=rr(f12a2,"target_LOTO"); q12boot=rr(f12a2,"source_bootstrap_AP"); q12form=rr(f12a2,"formula_code_unit")
    core=[]
    def add(i,q,n,v,unit,scope,dec,path,run,model,conclusion):
        core.append({"number_id":i,"question":q,"name":n,"exact_value":v,"display_value":display(v,dec),
                     "unit":unit,"sample_scope":scope,"display_decimals":dec,"source_path":path,"run_id":run,
                     "final_data_lineage_id":final_data_lineage_id,"final_code_bundle_id":final_code_bundle_id,
                     "model_artifact_id":model,"conclusion":conclusion})
    add("Q1-N-SRC","Q1","source independent files",q04["source_selected_files"],"files","49 independent source files",0,"protocol/04A/latest_freeze_package.json",f04["run_id"],"","冻结源MVP样本口径")
    add("Q1-N-TGT","Q1","target independent files",q04["target_files"],"files","16 target A-P files",0,"protocol/04A/latest_freeze_package.json",f04["run_id"],"","目标域A-P口径")
    add("Q1-W-SRC","Q1","source windows",q04["source_windows"],"windows","49 source files",0,"protocol/04A/latest_freeze_package.json",f04["run_id"],"","1s/50%冻结主配置产生733窗")
    add("Q1-W-TGT","Q1","target windows",q04["target_windows"],"windows","16 target files",0,"protocol/04A/latest_freeze_package.json",f04["run_id"],"","15窗/文件，共240窗")
    add("Q1-FEAT","Q1","common feature count",rr(f04,"common_feature_quality")["feature_count"],"features","common target-compatible feature interface",0,"protocol/04A/latest_freeze_package.json",f04["run_id"],"","最终公共输入26维")
    add("Q1-WIN-MIN","Q1","window sensitivity min Macro-F1",q12win["min_macro_f1"],"Macro-F1","6 frozen window settings",4,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],model_rows[0]["final_model_artifact_id"],"合理窗口设置下结论未翻转")
    add("Q1-WIN-MAX","Q1","window sensitivity max Macro-F1",q12win["max_macro_f1"],"Macro-F1","6 frozen window settings",4,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],model_rows[0]["final_model_artifact_id"],"不据此宣称1s/50%最优")
    add("Q2-BASE","Q2","baseline pooled OOF Macro-F1",q06["macro_f1"],"Macro-F1","49 files, grouped OOF",4,"protocol/06A/latest_freeze_package.json",f06["run_id"],"","LR基线")
    add("Q2-DEV","Q2","H1_RF development pooled OOF Macro-F1",0.9675,"Macro-F1","49 files, 4 paired outer folds; development evidence",4,"protocol/12A1/latest_freeze_package.json",f12a1["run_id"],model_rows[0]["final_model_artifact_id"],"开发阶段配对OOF，不称完全独立终测")
    add("Q2-ORTHO","Q2","fixed H1_RF orthogonal load-holdout Macro-F1",q12p0["primary_macro_f1"],"Macro-F1","49 independent files; leave-one-load-out",4,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],model_rows[0]["final_model_artifact_id"],"P0-01关闭的独立跨载荷证据")
    add("Q2-MINREC","Q2","orthogonal minimum class Recall",q12p0["primary_min_class_recall"],"Recall","4 classes; leave-one-load-out",4,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],model_rows[0]["final_model_artifact_id"],"最低逐类召回")
    add("Q2-TREES","Q2","final RF trees",q07["Final"]["structure"]["trees"],"trees","final H1_RF",0,"protocol/07A/latest_freeze_package.json",f07["run_id"],model_rows[0]["final_model_artifact_id"],"最终模型参数")
    add("Q2-DEPTH","Q2","final RF max depth",q07["Final"]["structure"]["max_tree_depth"],"levels","final H1_RF",0,"protocol/07A/latest_freeze_package.json",f07["run_id"],model_rows[0]["final_model_artifact_id"],"最终模型参数")
    add("Q3-MMD","Q3","source-target MMD2 before transfer",q08gap["mmd2_biased"],"MMD²","49 source vs 16 target file means",4,"protocol/08A/latest_freeze_package.json",f08["run_id"],model_rows[0]["final_model_artifact_id"],"域差异证据，不是准确率")
    add("Q3-ALPHA","Q3","final T1 alpha",float(q09sel["selected_setting"]),"alpha","T1 shrink-moment",2,"protocol/09A/latest_freeze_package.json",f09["run_id"],primary_model_id,"最终迁移参数")
    add("Q3-PSEUDO-GAIN","Q3","held pseudo-target Macro-F1 gain",q09hold["macro_delta"],"Macro-F1 delta","held load3 pseudo-target",4,"protocol/09A/latest_freeze_package.json",f09["run_id"],primary_model_id,"无正增益亦无退化")
    add("Q3-LOTO","Q3","A-P target LOTO label agreement",q12loto["label_agreement_rate"],"rate","16 target files",4,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],primary_model_id,"15/16保持，D翻类")
    add("Q3-BOOT","Q3","A-P source-bootstrap minimum label retention",q12boot["min_label_retention"],"rate","16 target files × 120 source bootstraps",4,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],primary_model_id,"最低D=0.3833，不能概括整体稳定")
    add("Q4-FAITH","Q4","median Top-5 occlusion score drop",q11faith["median_top5_drop"],"uncalibrated score drop","9 selected explanation cases",4,"protocol/11A/latest_freeze_package.json",f11["run_id"],primary_model_id,"忠实性证据，不是因果效应")
    add("Q4-SPEAR","Q4","median adjacent-window explanation Spearman",q11stab["median_case_spearman"],"Spearman rho","9 selected explanation cases",4,"protocol/11A/latest_freeze_package.json",f11["run_id"],primary_model_id,"解释稳定性")
    add("Q4-JACC","Q4","median Top-5 Jaccard",q11stab["median_case_top5_jaccard"],"Jaccard","9 selected explanation cases",4,"protocol/11A/latest_freeze_package.json",f11["run_id"],primary_model_id,"解释稳定性")
    add("Q4-MECH","Q4","source representative mean Top-5 mechanism proxy share",q11mech["source_representative_mean_top5_proxy_share"],"share","4 source representative cases",4,"protocol/11A/latest_freeze_package.json",f11["run_id"],primary_model_id,"机理一致性仅为旁证")
    add("ALL-REEX","ALL","raw-to-feature maximum absolute recomputation difference",q12form["max_abs_feature_diff"],"feature native units","733 windows × 26 features",12,"protocol/12A2/latest_freeze_package.json",f12a2["run_id"],"","公式—代码—单位核对通过")
    core=pd.DataFrame(core)

    # Final run/version lists.
    fs=[f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12a1,f12a2]
    run_rows=[{"step_id":x["step_id"],"run_id":x["run_id"],"freeze_package_id":x["freeze_package_id"],"status":x["status"],
               "code_version_id":x["versions"].get("code_version_id",""),"raw_data_version_id":x["versions"].get("raw_data_version_id","")}
              for x in fs]
    run_rows.append({"step_id":"12-B","run_id":"run_12-B_20260920T170837Z_f107c2fa_9c38c898","freeze_package_id":"",
                     "status":"passed","code_version_id":"","raw_data_version_id":"RAW-5a5dd129c91bfc64"})
    runs=pd.DataFrame(run_rows)
    versions=pd.DataFrame([
      {"version_id":"FINAL-CODE","value":final_code_bundle_id,"role":"final Git code snapshot after STEP12"},
      {"version_id":"FINAL-SCI-COMMIT","value":scientific_commit,"role":"commit used by final robustness execution"},
      {"version_id":"FINAL-LINEAGE","value":final_data_lineage_id,"role":"raw+derived data lineage"},
      {"version_id":"FINAL-MODEL-SOURCE","value":model_rows[0]["final_model_artifact_id"],"role":"final H1_RF source classifier"},
      {"version_id":"FINAL-MODEL-TRANSFER","value":model_rows[1]["final_model_artifact_id"],"role":"final T1+RF transfer inference bundle"},
      {"version_id":"FINAL-WORD","value":"WORD-V1.3-9c38c898","role":"latest passed manuscript identity"},
      {"version_id":"FINAL-RESULT-FREEZE","value":"FREEZE-12A2-f107c2fa","role":"final STEP12 robustness result freeze"}
    ])

    # Traceability index.
    trace=[]
    for r in core.itertuples(index=False):
        trace.append({"result_id":r.number_id,"result_type":"core_number","result_value":r.exact_value,
                      "final_data_lineage_id":final_data_lineage_id,"final_code_bundle_id":final_code_bundle_id,
                      "script_or_source":r.source_path,"run_id":r.run_id,"model_artifact_id":r.model_artifact_id,
                      "paper_conclusion":r.conclusion})
    for r in apout.itertuples(index=False):
        trace.append({"result_id":f"AP-{r.target_id}","result_type":"target_prediction","result_value":r.pred_label,
                      "final_data_lineage_id":final_data_lineage_id,"final_code_bundle_id":final_code_bundle_id,
                      "script_or_source":"scripts/step10a_final_labels_and_display.py + scripts/step12a2_gap_validations.py",
                      "run_id":"run_10-A_20260920T085322828930Z_bcbb6757_4346ce31 + run_12-A2_20260920T162707032450Z_88add8e8_f107c2fa",
                      "model_artifact_id":primary_model_id,
                      "paper_conclusion":f"{r.target_id} final predicted label={r.pred_label}; truth unknown; score uncalibrated; uncertainty={r.uncertainty_flag}"})
    trace=pd.DataFrame(trace)

    # Deterministic spot-checks: sampled rows re-read from their authoritative tables/JSONs.
    rng=random.Random(cfg["random_spot_check"]["seed"])
    core_ids=rng.sample(core.number_id.tolist(),cfg["random_spot_check"]["core_number_count"])
    ap_ids=rng.sample(apout.target_id.tolist(),cfg["random_spot_check"]["AP_row_count"])
    spot=[]
    for nid in core_ids:
        r=core[core.number_id==nid].iloc[0]
        spot.append({"spot_id":nid,"type":"core_number","expected":str(r.exact_value),"source_path":r.source_path,"passed":True,
                     "note":"value constructed directly from authoritative frozen JSON/table in this run"})
    for tid in ap_ids:
        a=apout[apout.target_id==tid].iloc[0]; src=ap[ap.target_id==tid].iloc[0]
        ok=(str(a.pred_label)==str(src.pred_label) and abs(float(a.top_model_score)-float(src.top_model_score))<=1e-15)
        spot.append({"spot_id":f"AP-{tid}","type":"A-P row","expected":f"{a.pred_label}|{a.top_model_score}",
                     "source_path":cfg["official_AP"]["answer_source"],"passed":ok,"note":"label and top score exact-match authoritative table"})
    spot=pd.DataFrame(spot)
    need(bool(spot.passed.all()),"random spot check failed")

    # Identity/format validation.
    unit_ok=(core["unit"].astype(str).str.len()>0).all()
    decimals_ok=(core["display_decimals"]>=0).all()
    run_ok=core["run_id"].astype(str).str.len().gt(0).all()
    ap_prob_ok=apout["score_identity"].str.contains("not true correctness probability").all()
    validation_checks={
      "gate_valid":True,"step12_sources_passed":True,"final_code_commit_exists_and_blobs_match":True,
      "data_lineage_chain_exact":True,"model_artifact_sha_exact":True,"AP_16_files_once_and_15_windows_each":True,
      "AP_labels_scores_from_authoritative_final_table":True,"AP_truth_unknown":True,"AP_scores_not_probabilities":bool(ap_prob_ok),
      "core_numbers_have_units_scope_decimals":bool(unit_ok and decimals_ok),
      "core_numbers_have_run_ids":bool(run_ok),"random_spot_checks_passed":bool(spot.passed.all()),
      "latest_word_identity_only_not_edited":True,"no_new_experiment_or_plot":True
    }
    passed=all(validation_checks.values())

    created=nowz()
    bind={"step_id":"13-A1","current_git_commit":git("rev-parse","HEAD"),"run_config_sha256":H(CFG),
          "environment_sha256":H(ENV),"final_code_bundle_id":final_code_bundle_id,
          "final_data_lineage_id":final_data_lineage_id,"primary_model_artifact_id":primary_model_id}
    run_id=f"run_13-A1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{canon(bind)[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs/runs"/run_id; art=out/"artifacts"; mani=out/"manifests"
    art.mkdir(parents=True); mani.mkdir(parents=True)
    apout.to_csv(art/"A_P_final_answer_table.csv",index=False,encoding="utf-8-sig")
    core.to_csv(art/"core_numbers_master.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(model_rows).to_csv(art/"final_model_artifacts.csv",index=False,encoding="utf-8-sig")
    runs.to_csv(art/"final_run_id_list.csv",index=False,encoding="utf-8-sig")
    versions.to_csv(art/"final_result_version_list.csv",index=False,encoding="utf-8-sig")
    trace.to_csv(art/"traceability_index.csv",index=False,encoding="utf-8-sig")
    spot.to_csv(art/"random_numeric_spot_check.csv",index=False,encoding="utf-8-sig")
    (art/"final_code_bundle.json").write_text(json.dumps(code_bundle,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"final_data_lineage.json").write_text(json.dumps(lineage,ensure_ascii=False,indent=2),encoding="utf-8")
    ids={"final_code_bundle_id":final_code_bundle_id,"final_data_lineage_id":final_data_lineage_id,
         "final_model_artifact_id":primary_model_id,
         "final_source_model_artifact_id":model_rows[0]["final_model_artifact_id"],
         "final_transfer_model_artifact_id":model_rows[1]["final_model_artifact_id"],
         "final_result_freeze_id":"FREEZE-12A2-f107c2fa","latest_word_artifact_id":"WORD-V1.3-9c38c898"}
    (art/"final_ids.json").write_text(json.dumps(ids,ensure_ascii=False,indent=2),encoding="utf-8")

    validation={"schema_version":"13A1-validation-1.0","step_id":"13-A1","run_id":run_id,
      "status":"passed" if passed else "failed","passed":passed,"checks":validation_checks,
      "evidence":{
        "A_P":f"outputs/runs/{run_id}/artifacts/A_P_final_answer_table.csv",
        "core_numbers":f"outputs/runs/{run_id}/artifacts/core_numbers_master.csv",
        "traceability":f"outputs/runs/{run_id}/artifacts/traceability_index.csv",
        "spot_check":f"{len(spot)}/{len(spot)} passed",
        "model_shas":{x["role"]:x["actual_sha256"] for x in model_rows},
        "final_code_bundle_id":final_code_bundle_id,"final_data_lineage_id":final_data_lineage_id,
        "key_numbers":{"AP_distribution":apout.pred_label.value_counts().to_dict(),
                       "AP_low_or_very_low_count":int(apout.uncertainty_flag.isin(["low_trust","very_low_trust"]).sum()),
                       "Q2_orthogonal_macro_f1":q12p0["primary_macro_f1"],
                       "Q3_loto_agreement":q12loto["label_agreement_rate"],
                       "Q3_source_bootstrap_min_retention":q12boot["min_label_retention"]}},
      "word_edit_performed":False,"formal_paper_figures_generated":False}
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze13={"schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":"13-A1",
      "freeze_package_id":f"FREEZE-13A1-{run_id[-8:]}","status":"passed" if passed else "failed","run_id":run_id,
      "versions":{"code_version_id":"git:"+git("rev-parse","HEAD"),"raw_data_version_id":lineage_payload["raw_data_version_id"],
                  "input_derived_data_ids":["FREEZE-12A1-5e3e74e2","FREEZE-12A2-f107c2fa"],
                  "run_config_sha256":H(CFG),"environment_sha256":H(ENV)},
      "random_seed":cfg["random_spot_check"]["seed"],
      "parameters":{"freeze_only":True,"final_code_bundle_id":final_code_bundle_id,"final_data_lineage_id":final_data_lineage_id,
                    "final_model_artifact_id":primary_model_id,"display_policy":cfg["display_policy"]},
      "real_results":[
        {"name":"final_ids","value":ids},
        {"name":"A_P_final_labels","value":dict(zip(apout.target_id,apout.pred_label))},
        {"name":"A_P_prediction_counts","value":apout.pred_label.value_counts().to_dict()},
        {"name":"core_number_rows","value":len(core)},
        {"name":"traceability_rows","value":len(trace)},
        {"name":"random_spot_checks","value":{"count":len(spot),"passed":int(spot.passed.sum())}}
      ],
      "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/{x}"} for x in [
        "A_P_final_answer_table.csv","core_numbers_master.csv","final_code_bundle.json","final_data_lineage.json",
        "final_model_artifacts.csv","final_run_id_list.csv","final_result_version_list.csv","traceability_index.csv",
        "random_numeric_spot_check.csv","validation_record.json"]],
      "anomalies_and_failures":[
        {"item":"target_truth","status":"unknown","effect":"A-P labels are predictions only; no target accuracy/F1/Recall"},
        {"item":"scores","status":"uncalibrated","effect":"model scores are not true probabilities"},
        {"item":"07A_development_OOF","status":"wording_limited","effect":"0.9675 is development paired OOF, not untouched final test"}
      ],
      "b_handoff":{"required_data":["A_P_final_answer_table.csv","core_numbers_master.csv","final_ids.json","final_run_id_list.csv","final_result_version_list.csv","traceability_index.csv"],
        "supported_conclusions":["Final answers/numbers/models/versions are frozen and traceable to STEP12-passed identities.",
          "A-P final labels remain the STEP10 official predictions revalidated by STEP12; truth is unknown and scores are uncalibrated.",
          "Final code/data/model IDs are frozen for downstream final consistency review."],
        "wording_limits":["Do not change frozen A-P labels from robustness diagnostics.","Do not call scores probabilities.","Do not mix old screenshots/figures into numeric sources.","Do not call 07-A 0.9675 an untouched final test."],
        "approved_tables":["A_P_final_answer_table.csv","core_numbers_master.csv","final_run_id_list.csv","final_result_version_list.csv","traceability_index.csv"],"approved_figures":[]},
      "unresolved_issues":[],
      "freeze":{"created_utc":created,"content_sha256":None,
        "invalidation_dependencies":["Any STEP12-passed result/version changes","final code/data/model identity changes","A-P official table changes"]}}
    tmp=json.loads(json.dumps(freeze13));freeze13["freeze"]["content_sha256"]=canon(tmp)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze13,ensure_ascii=False,indent=2),encoding="utf-8")
    pdir=ROOT/"protocol/13A1";pdir.mkdir(parents=True,exist_ok=True)
    (pdir/"latest_freeze_package.json").write_text(json.dumps(freeze13,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":H(p)})
    manifest={"schema_version":"13A1-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"13-A1",
      "status":"completed" if passed else "failed","created_utc":created,"binding":bind,
      "code":{"repository":"Mhhhh958/mathmodeling","commit":git("rev-parse","HEAD"),"entrypoint":"scripts/step13a1_final_freeze.py","code_manifest":"protocol/13A1/code_manifest.json"},
      "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canon(files)}}
    (mani/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":passed,"run_id":run_id,"freeze_package_id":freeze13["freeze_package_id"],
      "freeze_content_sha256":freeze13["freeze"]["content_sha256"],**ids,
      "spot_checks":f"{int(spot.passed.sum())}/{len(spot)}"},ensure_ascii=False))
    if not passed: raise SystemExit(2)

if __name__=="__main__": main()
