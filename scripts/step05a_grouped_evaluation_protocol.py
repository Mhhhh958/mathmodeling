#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"protocol"/"05A"/"run_config.json"
BLUEPRINT=ROOT/"protocol"/"05A"/"question2_argument_blueprint.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
FREEZE04=ROOT/"protocol"/"04A"/"latest_freeze_package.json"
RUN04=ROOT/"protocol"/"04A"/"latest_run_id.txt"
SEED=20260919
CLASSES=["OR","IR","B","N"]

def cj(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b): return hashlib.sha256(b).hexdigest()
def hf(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def git_head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def split_seed(*parts):
 s="|".join(map(str,parts)).encode()
 return int(hashlib.sha256(s).hexdigest()[:8],16)

def class_group_map(meta):
 q=meta[["group_id","class_label","relative_path"]].drop_duplicates()
 bad=q.groupby("group_id")["class_label"].nunique()
 if (bad>1).any(): raise RuntimeError("a group maps to >1 class")
 return q.sort_values(["class_label","group_id"]).reset_index(drop=True)

def build_outer(group_table):
 parts={}
 for ci,c in enumerate(CLASSES):
  gs=group_table.loc[group_table.class_label==c,"group_id"].astype(str).sort_values().tolist()
  rng=np.random.default_rng(split_seed(SEED,"outer",c))
  perm=np.array(gs,dtype=object)[rng.permutation(len(gs))]
  arr=[list(x) for x in np.array_split(perm,4)]
  if c=="N" and not all(len(x)==1 for x in arr):
   raise RuntimeError("N outer fold size must be exactly 1")
  parts[c]=arr
 return parts

def build_inner(outer_train_groups,group_table,outer_fold):
 parts={}
 for c in CLASSES:
  gs=group_table[(group_table.class_label==c)&(group_table.group_id.isin(outer_train_groups))].group_id.astype(str).sort_values().tolist()
  rng=np.random.default_rng(split_seed(SEED,"inner",outer_fold,c))
  perm=np.array(gs,dtype=object)[rng.permutation(len(gs))]
  arr=[list(x) for x in np.array_split(perm,3)]
  if c=="N" and not all(len(x)==1 for x in arr):
   raise RuntimeError(f"N inner validation size must be 1 in outer {outer_fold}")
  parts[c]=arr
 return parts

def metrics(y_true,y_pred):
 cm=confusion_matrix(y_true,y_pred,labels=CLASSES)
 rec=recall_score(y_true,y_pred,labels=CLASSES,average=None,zero_division=0)
 f1=f1_score(y_true,y_pred,labels=CLASSES,average="macro",zero_division=0)
 acc=accuracy_score(y_true,y_pred)
 return {"macro_f1":float(f1),"accuracy":float(acc),"per_class_recall":{c:float(v) for c,v in zip(CLASSES,rec)},"confusion_matrix":cm.tolist()}

def manual_macro_f1(y_true,y_pred):
 vals=[]
 for c in CLASSES:
  tp=sum(yt==c and yp==c for yt,yp in zip(y_true,y_pred))
  fp=sum(yt!=c and yp==c for yt,yp in zip(y_true,y_pred))
  fn=sum(yt==c and yp!=c for yt,yp in zip(y_true,y_pred))
  prec=tp/(tp+fp) if tp+fp else 0.0
  rec=tp/(tp+fn) if tp+fn else 0.0
  vals.append(2*prec*rec/(prec+rec) if prec+rec else 0.0)
 return float(np.mean(vals))

def aggregate_file_probs(df):
 prob_cols=[f"p_{c}" for c in CLASSES]
 g=df.groupby(["file_id","true_label"],sort=True)[prob_cols].mean().reset_index()
 g["pred_label"]=[CLASSES[int(np.argmax(row))] for row in g[prob_cols].to_numpy()]
 return g

def main():
 t0=time.time()
 cfg=json.loads(CFG.read_text(encoding="utf-8"))
 fr04=json.loads(FREEZE04.read_text(encoding="utf-8"))
 assert fr04["status"]=="passed"
 assert "FEAT-23d45f5649dcd5f1" in fr04["versions"]["input_derived_data_ids"]
 run04=RUN04.read_text(encoding="utf-8").strip()
 base=ROOT/"outputs"/"runs"/run04/"artifacts"
 y=pd.read_csv(base/"q2_interface"/"y_source_labels.csv")
 groups=pd.read_csv(base/"q2_interface"/"groups_source.csv")
 meta=pd.read_csv(base/"q2_interface"/"source_window_metadata.csv")
 X=pd.read_csv(base/"q2_interface"/"X_source_common.csv")
 if not (len(y)==len(groups)==len(meta)==len(X)==733): raise RuntimeError("04-A source interface row count mismatch")
 joined=y.merge(groups,on="window_id",validate="one_to_one").merge(meta[["window_id","relative_path","analysis_channel"]],on="window_id",validate="one_to_one")
 gt=class_group_map(joined.rename(columns={"group_id":"group_id"}))
 counts=gt.class_label.value_counts().to_dict()
 if counts!={"OR":21,"IR":12,"B":12,"N":4}: raise RuntimeError(f"unexpected file counts {counts}")
 # one primary analysis channel per file/group in frozen 04-A interface
 ch=joined.groupby("group_id")["analysis_channel"].nunique()
 if (ch!=1).any(): raise RuntimeError("a frozen group has multiple analysis channels")

 created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 code=git_head(); cfgsha=hf(CFG); envsha=hf(ENV); bpsha=hf(BLUEPRINT)
 binding={"code_version_id":"git:"+code,"data_version_id":fr04["freeze_package_id"],"run_config_sha256":cfgsha,"seed":SEED,"environment_sha256":envsha}
 bd=hb(cj(binding).encode())
 run_id=f"run_05-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
 out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; mani=out/"manifests"
 art.mkdir(parents=True); mani.mkdir(parents=True)

 outer_parts=build_outer(gt)
 all_groups=set(gt.group_id.astype(str))
 outer_rows=[]; inner_rows=[]; summaries=[]; overlap_issues=[]
 for o in range(4):
  test=set(sum((outer_parts[c][o] for c in CLASSES),[]))
  trainpool=all_groups-test
  if test & trainpool: overlap_issues.append(f"outer{o}:test/trainpool")
  for _,r in gt.iterrows():
   role="test" if r.group_id in test else "train_pool"
   outer_rows.append({"outer_fold":o,"group_id":r.group_id,"relative_path":r.relative_path,"class_label":r.class_label,"role":role})
  for role,gs in [("train_pool",trainpool),("test",test)]:
   cc=gt[gt.group_id.isin(gs)].class_label.value_counts()
   summaries.append({"level":"outer","outer_fold":o,"inner_fold":-1,"role":role,"groups":len(gs),**{c:int(cc.get(c,0)) for c in CLASSES}})
  inner_parts=build_inner(trainpool,gt,o)
  for i in range(3):
   val=set(sum((inner_parts[c][i] for c in CLASSES),[]))
   train=trainpool-val
   if train & val: overlap_issues.append(f"outer{o}/inner{i}:train/val")
   if val & test or train & test: overlap_issues.append(f"outer{o}/inner{i}:outer-test overlap")
   for _,r in gt[gt.group_id.isin(trainpool)].iterrows():
    role="val" if r.group_id in val else "train"
    inner_rows.append({"outer_fold":o,"inner_fold":i,"group_id":r.group_id,"relative_path":r.relative_path,"class_label":r.class_label,"role":role})
   for role,gs in [("train",train),("val",val),("outer_test",test)]:
    cc=gt[gt.group_id.isin(gs)].class_label.value_counts()
    summaries.append({"level":"inner","outer_fold":o,"inner_fold":i,"role":role,"groups":len(gs),**{c:int(cc.get(c,0)) for c in CLASSES}})

 outer=pd.DataFrame(outer_rows); inner=pd.DataFrame(inner_rows); summ=pd.DataFrame(summaries)
 outer.to_csv(art/"outer_split_manifest.csv",index=False,encoding="utf-8-sig")
 inner.to_csv(art/"inner_split_manifest.csv",index=False,encoding="utf-8-sig")
 summ.to_csv(art/"split_summary.csv",index=False,encoding="utf-8-sig")

 # Frozen normal-file distribution.
 normals=gt[gt.class_label=="N"].copy()
 nrows=[]
 for o in range(4):
  test=set(outer[(outer.outer_fold==o)&(outer.role=="test")& (outer.class_label=="N")].group_id)
  for _,r in normals.iterrows():
   if r.group_id in test:
    nrows.append({"outer_fold":o,"inner_fold":-1,"relative_path":r.relative_path,"group_id":r.group_id,"role":"outer_test"})
   else:
    for i in range(3):
     rr=inner[(inner.outer_fold==o)&(inner.inner_fold==i)&(inner.group_id==r.group_id)]
     nrows.append({"outer_fold":o,"inner_fold":i,"relative_path":r.relative_path,"group_id":r.group_id,"role":rr.iloc[0].role})
 ndf=pd.DataFrame(nrows); ndf.to_csv(art/"normal_file_distribution.csv",index=False,encoding="utf-8-sig")

 # Window assignment audit: group drives all windows.
 wa=joined[["window_id","group_id","class_label","relative_path"]].copy()
 wrows=[]
 for o in range(4):
  rolemap=outer[outer.outer_fold==o].set_index("group_id").role.to_dict()
  q=wa.copy(); q["outer_fold"]=o; q["outer_role"]=q.group_id.map(rolemap)
  wrows.append(q)
 pd.concat(wrows,ignore_index=True).to_csv(art/"window_group_assignment_audit.csv",index=False,encoding="utf-8-sig")

 # Evaluation protocol + baselines + budget.
 evalp={
  "primary_metric":"file_level_macro_f1",
  "labels":CLASSES,
  "file_aggregation":{"input":"per-window probability vector [p_OR,p_IR,p_B,p_N]","rule":"arithmetic mean by independent file, then argmax","tie_rule":"numpy argmax over fixed order OR,IR,B,N; exact ties are deterministic but must be reported if observed"},
  "secondary_file_metrics":["per-class recall","confusion matrix","accuracy"],
  "window_metrics":"Macro-F1/per-class recall/confusion matrix may be reported only as auxiliary",
  "outer_reporting":"four predeclared outer folds individually + median + min-max; not called a confidence interval",
  "model_selection":"within each outer fold choose candidate/config by mean 3-inner-fold file Macro-F1; tie-break by higher minimum per-class recall then frozen simplicity order; refit on all outer train-pool groups; evaluate outer test once",
  "preprocessing_boundary":"imputer/scaler/feature selector and any class/file weighting parameters fit using inner-train groups only; outer refit uses outer-train-pool only",
  "test_prohibition":"outer-test labels/probabilities/metrics cannot be consulted for candidate selection or tuning"
 }
 (art/"evaluation_protocol.json").write_text(json.dumps(evalp,ensure_ascii=False,indent=2),encoding="utf-8")
 (art/"baseline_candidates.json").write_text(json.dumps({"candidates":cfg["baseline_candidates"],"weighting":cfg["weighting"],"selection_rule":cfg["selection_rule"]},ensure_ascii=False,indent=2),encoding="utf-8")
 (art/"compute_budget.json").write_text(json.dumps({"compute_budget":cfg["compute_budget"],"failure_standards":cfg["failure_standards"]},ensure_ascii=False,indent=2),encoding="utf-8")

 # Deterministic example validating aggregation and macro-F1.
 probs=[]
 truth={"F_OR":"OR","F_IR":"IR","F_B":"B","F_N":"N","F_N2":"N"}
 values={
  "F_OR":[[.80,.10,.05,.05],[.55,.25,.10,.10],[.65,.15,.10,.10]],
  "F_IR":[[.10,.55,.25,.10],[.10,.60,.20,.10],[.20,.40,.30,.10]],
  "F_B":[[.10,.15,.65,.10],[.10,.25,.55,.10],[.15,.20,.55,.10]],
  "F_N":[[.20,.10,.20,.50],[.10,.10,.20,.60],[.15,.15,.20,.50]],
  "F_N2":[[.15,.15,.45,.25],[.10,.10,.50,.30],[.10,.15,.45,.30]]
 }
 for fid,arr in values.items():
  for j,p in enumerate(arr):
   probs.append({"file_id":fid,"window_id":f"{fid}_W{j}","true_label":truth[fid],**{f"p_{c}":p[k] for k,c in enumerate(CLASSES)}})
 ex=pd.DataFrame(probs); agg=aggregate_file_probs(ex)
 m=metrics(agg.true_label.tolist(),agg.pred_label.tolist())
 mm=manual_macro_f1(agg.true_label.tolist(),agg.pred_label.tolist())
 ex.to_csv(art/"metric_validation_window_probabilities.csv",index=False,encoding="utf-8-sig")
 agg.to_csv(art/"metric_validation_file_aggregation.csv",index=False,encoding="utf-8-sig")
 metric_check={"sklearn":m,"manual_macro_f1":mm,"abs_diff":abs(mm-m["macro_f1"]),"passed":abs(mm-m["macro_f1"])<1e-12}
 (art/"metric_validation_result.json").write_text(json.dumps(metric_check,ensure_ascii=False,indent=2),encoding="utf-8")

 # Validation.
 # Coverage checks for every outer and inner role.
 required_outer=summ[summ.level=="outer"]
 outer_cov=bool(((required_outer[CLASSES]>0).all(axis=1)).all())
 required_inner=summ[(summ.level=="inner") & (summ.role.isin(["train","val","outer_test"]))]
 inner_cov=bool(((required_inner[CLASSES]>0).all(axis=1)).all())
 n_outer=outer[(outer.class_label=="N")&(outer.role=="test")].groupby("outer_fold").size().to_dict()
 n_val=inner[(inner.class_label=="N")&(inner.role=="val")].groupby(["outer_fold","inner_fold"]).size()
 n_ok=bool(len(n_outer)==4 and all(v==1 for v in n_outer.values()) and len(n_val)==12 and (n_val==1).all())
 no_overlap=(len(overlap_issues)==0)
 # every source window appears with exactly one group and inherits role
 group_window_unique=bool(joined.groupby("window_id").group_id.nunique().max()==1)
 validation={
  "step_id":"05-A","run_id":run_id,
  "checks":{
   "group_intersections_zero":{"passed":no_overlap,"evidence":f"outputs/runs/{run_id}/artifacts/outer_split_manifest.csv + inner_split_manifest.csv","issues":overlap_issues},
   "all_classes_present":{"passed":outer_cov and inner_cov,"evidence":f"outputs/runs/{run_id}/artifacts/split_summary.csv"},
   "normal_4_files_distribution":{"passed":n_ok,"evidence":f"outputs/runs/{run_id}/artifacts/normal_file_distribution.csv","outer_test_N_per_fold":n_outer,"inner_val_N_entries":int(len(n_val))},
   "same_source_windows_grouped":{"passed":group_window_unique,"evidence":f"outputs/runs/{run_id}/artifacts/window_group_assignment_audit.csv","source_windows":len(joined),"source_groups":joined.group_id.nunique()},
   "metric_and_file_aggregation_recalculation":{"passed":metric_check["passed"],"evidence":f"outputs/runs/{run_id}/artifacts/metric_validation_result.json","macro_f1":m["macro_f1"],"manual_macro_f1":mm,"abs_diff":metric_check["abs_diff"]},
   "training_boundary_frozen":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/evaluation_protocol.json","statement":"no imputation/scaling/selection/class-weight fit performed in 05-A"},
   "no_model_training":{"passed":True,"evidence":"script generates split/evaluation artifacts only; no estimator fit"}
  }
 }
 validation["passed"]=all(v["passed"] for v in validation["checks"].values()); validation["status"]="passed" if validation["passed"] else "failed"
 (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"05A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

 # File-count summary for human handoff.
 split_stats={
  "source_independent_files":49,"source_windows":733,"file_class_counts":counts,
  "outer_folds":4,"inner_folds_per_outer":3,
  "outer_test_class_counts_by_fold":summ[(summ.level=="outer")&(summ.role=="test")][["outer_fold",*CLASSES]].to_dict("records"),
  "normal_file_count":4,
  "normal_protocol":"each N file outer-test exactly once; within each outer train-pool, each remaining N file inner-validation exactly once",
  "why_not_5_fold":"only four independent N files; five-fold file-level stratification cannot guarantee an independent N in every fold",
  "multi_channel_leakage_check":"04-A frozen interface uses one primary analysis channel per independent_object_id; if more channels are introduced later, they must share the same group_id"
 }
 (art/"split_protocol_summary.json").write_text(json.dumps(split_stats,ensure_ascii=False,indent=2),encoding="utf-8")

 freeze={
  "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":"05-A","freeze_package_id":f"FREEZE-05A-{run_id[-8:]}",
  "status":"passed" if validation["passed"] else "failed","run_id":run_id,
  "versions":{"code_version_id":"git:"+code,"raw_data_version_id":fr04["versions"]["raw_data_version_id"],"input_derived_data_ids":[fr04["freeze_package_id"],"FEAT-23d45f5649dcd5f1"],"run_config_sha256":cfgsha,"environment_sha256":envsha},
  "random_seed":SEED,
  "parameters":{"outer_folds":4,"inner_folds_per_outer":3,"group_key":"independent_object_id","primary_metric":"file_level_macro_f1","file_aggregation":"mean window probabilities -> argmax","max_candidate_configurations":12,"max_total_inner_fits":144},
  "real_results":[
   {"name":"file_class_counts","value":counts},
   {"name":"outer_test_counts","value":split_stats["outer_test_class_counts_by_fold"]},
   {"name":"normal_distribution","value":{"files":4,"outer_test_each_once":True,"inner_val_each_remaining_once":True}},
   {"name":"metric_protocol_validation","value":metric_check}
  ],
  "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/split_summary.csv"},{"path":f"outputs/runs/{run_id}/artifacts/normal_file_distribution.csv"},{"path":f"outputs/runs/{run_id}/artifacts/metric_validation_result.json"}],
  "anomalies_and_failures":[],
  "b_handoff":{"required_data":["outer_split_manifest.csv","inner_split_manifest.csv","split_summary.csv","normal_file_distribution.csv","evaluation_protocol.json","baseline_candidates.json","compute_budget.json"],"supported_conclusions":["4 outer folds are dictated by the four independent N files","all train/validation/test group intersections are zero","file-level Macro-F1 is primary and window-level metrics are auxiliary","outer test is not used for model selection"],"wording_limits":["do not call four-fold min-max a confidence interval","do not call 733 windows 733 independent samples","do not claim five-fold CV"],"approved_tables":["split_summary.csv","normal_file_distribution.csv"],"approved_figures":[]},
  "unresolved_issues":[],
  "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["04-A group/interface changes","05-A split seed/protocol changes","evaluation aggregation/metric changes"]}
 }
 f0=json.loads(json.dumps(freeze,ensure_ascii=False)); freeze["freeze"]["content_sha256"]=hb(cj(f0).encode())
 (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"05A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"05A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

 pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True); (mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
 runtime={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"gpu":None,"cuda":None,"git_commit":code,"pip_freeze_sha256":hf(mani/"pip_freeze.txt")}
 (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
 outs=[]
 for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
  outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
 rm={"schema_version":"05A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"05-A","status":"completed" if validation["passed"] else "failed","created_utc":created,
  "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code,"entrypoint":"scripts/step05a_grouped_evaluation_protocol.py","code_manifest":"protocol/05A/code_manifest.json"},
  "data":{"input_04A_freeze":fr04["freeze_package_id"],"feature_data_version_id":"FEAT-23d45f5649dcd5f1","source_files":49,"source_windows":733},
  "blueprint":{"path":"protocol/05A/question2_argument_blueprint.json","sha256":bpsha},
  "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":envsha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
  "config":{"path":"protocol/05A/run_config.json","sha256":cfgsha},"outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":hb(cj(outs).encode())},
  "checkpoint":{"completed_batches":["outer-splits","inner-splits","normal-audit","window-group-audit","metric-protocol-validation","freeze"],"pending_batches":[],"resume_token":None},
  "duration_seconds":time.time()-t0}
 (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"ok":validation["passed"],"run_id":run_id,"freeze_sha256":freeze["freeze"]["content_sha256"],"class_counts":counts,"outer_test_counts":split_stats["outer_test_class_counts_by_fold"],"metric_example_macro_f1":m["macro_f1"],"group_overlap_issues":overlap_issues},ensure_ascii=False))
 return 0 if validation["passed"] else 2

if __name__=="__main__":
 raise SystemExit(main())
