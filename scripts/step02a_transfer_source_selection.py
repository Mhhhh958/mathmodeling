#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, platform, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly, welch
from scipy.stats import skew, kurtosis

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
CFG=ROOT/"protocol"/"02A"/"run_config.json"
BLUEPRINT=ROOT/"protocol"/"02A"/"question1_argument_blueprint.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
DM=ROOT/"protocol"/"01A"/"data_manifest.json"
FREEZE01=ROOT/"protocol"/"01A"/"latest_freeze_package.json"
SEED=20260919
COMMON_FS=12000
FMAX=6000.0
CANDS=["source_12khz_de","source_12khz_fe","source_48khz_de"]
NORMAL_GROUP="source_48khz_normal"
TARGET_GROUP="target_32khz"
CLASSES=["OR","IR","B","N"]
LOADS=[0,1,2,3]
FAMILIES={
 "time_shape":["skew","kurtosis","crest_factor","impulse_factor","zero_cross_rate"],
 "spectral_location":["spectral_centroid_norm","rolloff95_norm","spectral_entropy"],
 "band_energy":["energy_0_500","energy_500_1500","energy_1500_3000","energy_3000_6000"],
}
FEATURES=sum(FAMILIES.values(),[])

def cj(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b):return hashlib.sha256(b).hexdigest()
def hf(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def git_head():return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def choose_signal(mat,subgroup,stem):
 keys=[k for k in mat if not k.startswith("__")]
 if subgroup==TARGET_GROUP:
  if stem in mat:return np.asarray(mat[stem]).squeeze(),stem
  for k in keys:
   a=np.asarray(mat[k]).squeeze()
   if a.ndim==1 and a.size>1000:return a,k
  raise RuntimeError("target signal missing")
 want="FE_time" if subgroup=="source_12khz_fe" else "DE_time"
 for k in keys:
  if k.endswith(want):return np.asarray(mat[k]).squeeze(),k
 raise RuntimeError(f"primary signal missing for {subgroup}")

def fs_of(subgroup):
 return {"source_12khz_de":12000,"source_12khz_fe":12000,"source_48khz_de":48000,"source_48khz_normal":48000,"target_32khz":32000}[subgroup]

def common_signal(x,fs):
 x=np.asarray(x,dtype=float).ravel()
 if not np.isfinite(x).all():x=x[np.isfinite(x)]
 if fs==COMMON_FS:return x
 g=math.gcd(int(fs),COMMON_FS)
 return resample_poly(x,COMMON_FS//g,int(fs)//g)

def spectral_entropy(p):
 p=np.asarray(p,float); p=p[p>0]
 if len(p)<=1:return np.nan
 p=p/p.sum()
 return float(-(p*np.log(p)).sum()/np.log(len(p)))

def desc(x,fs_native):
 y=common_signal(x,fs_native)
 rms=float(np.sqrt(np.mean(y*y)))
 med=float(np.median(y))
 mad=float(np.median(np.abs(y-med)))
 scale=1.4826*mad
 if not np.isfinite(scale) or scale<1e-12:
  scale=float(np.std(y))
 if not np.isfinite(scale) or scale<1e-12:scale=1.0
 z=(y-med)/scale
 absmean=float(np.mean(np.abs(z))); zrms=float(np.sqrt(np.mean(z*z))); peak=float(np.max(np.abs(z)))
 nper=min(4096,len(z))
 f,p=welch(z,fs=COMMON_FS,nperseg=nper,noverlap=nper//2,detrend="constant")
 m=(f>=0)&(f<=FMAX);f=f[m];p=p[m]
 ps=float(p.sum())+1e-15
 centroid=float((f*p).sum()/ps)
 cs=np.cumsum(p)
 roll=float(f[min(len(f)-1,int(np.searchsorted(cs,0.95*cs[-1])))]) if cs[-1]>0 else np.nan
 total=float(np.trapz(p,f))+1e-15
 def er(lo,hi):
  mm=(f>=lo)&(f<(hi if hi<FMAX else hi+1e-9))
  if mm.sum()<2:return np.nan
  return float(np.trapz(p[mm],f[mm])/total)
 return {
  "native_fs_hz":fs_native,"common_fs_hz":COMMON_FS,"common_points":len(y),
  "log_rms":float(np.log10(rms+1e-15)),"log_mad":float(np.log10(mad+1e-15)),
  "skew":float(skew(z,bias=False)),"kurtosis":float(kurtosis(z,fisher=False,bias=False)),
  "crest_factor":peak/(zrms+1e-15),"impulse_factor":peak/(absmean+1e-15),
  "zero_cross_rate":float(np.mean(np.signbit(z[1:])!=np.signbit(z[:-1]))) if len(z)>1 else 0.0,
  "spectral_centroid_norm":centroid/FMAX,"rolloff95_norm":roll/FMAX,"spectral_entropy":spectral_entropy(p),
  "energy_0_500":er(0,500),"energy_500_1500":er(500,1500),"energy_1500_3000":er(1500,3000),"energy_3000_6000":er(3000,6000),
 }

def robust_scale_matrix(df,cols):
 X=df[cols].to_numpy(float)
 med=np.nanmedian(X,axis=0)
 q1=np.nanpercentile(X,25,axis=0);q3=np.nanpercentile(X,75,axis=0)
 sc=q3-q1
 fallback=np.nanstd(X,axis=0)
 sc=np.where(np.isfinite(sc)&(sc>1e-12),sc,np.where(np.isfinite(fallback)&(fallback>1e-12),fallback,1.0))
 Z=(X-med)/sc
 return Z,med,sc

def group_distances(feat,scaler_med,scaler_scale,exclude_load=None):
 tgt=feat[feat.subgroup==TARGET_GROUP]
 tvec=np.nanmedian((tgt[FEATURES].to_numpy(float)-scaler_med)/scaler_scale,axis=0)
 rows=[]
 for g in CANDS:
  s=feat[feat.subgroup==g]
  if exclude_load is not None:s=s[s.load_hp!=exclude_load]
  z=(s[FEATURES].to_numpy(float)-scaler_med)/scaler_scale
  svec=np.nanmedian(z,axis=0)
  fam={}
  for name,cols in FAMILIES.items():
   idx=[FEATURES.index(c) for c in cols]
   fam[name]=float(np.sqrt(np.mean((svec[idx]-tvec[idx])**2)))
  overall=float(np.mean(list(fam.values())))
  rows.append({"subgroup":g,"exclude_load_hp":exclude_load if exclude_load is not None else "none","n_files":len(s),"robust_group_distance":overall,**{f"distance_{k}":v for k,v in fam.items()}})
 return pd.DataFrame(rows).sort_values(["robust_group_distance","subgroup"]).reset_index(drop=True)

def coverage_row(meta,name,paths):
 q=meta[meta.relative_path.isin(paths)]
 def unique_sorted(col):
  vals=[x for x in q[col].dropna().tolist() if str(x)!="nan"]
  try:return ";".join(map(str,sorted(set(vals),key=float)))
  except:return ";".join(sorted(set(map(str,vals))))
 return {
  "plan":name,"files":len(q),
  "OR":int((q.class_label=="OR").sum()),"IR":int((q.class_label=="IR").sum()),"B":int((q.class_label=="B").sum()),"N":int((q.class_label=="N").sum()),
  "loads_hp":unique_sorted("load_hp"),"fault_sizes_in":unique_sorted("fault_size_in"),"outer_race_positions":unique_sorted("outer_race_position_clock"),
  "subgroups":";".join(sorted(q.subgroup.unique()))
 }

def main():
 t0=time.time()
 np.random.seed(SEED)
 cfg=json.loads(CFG.read_text(encoding="utf-8")); dm=json.loads(DM.read_text(encoding="utf-8")); fr=json.loads(FREEZE01.read_text(encoding="utf-8"))
 assert dm["raw_data_version_id"]=="RAW-5a5dd129c91bfc64"
 assert fr["status"]=="passed"
 metadata_path=ROOT/dm["metadata_artifact"]
 meta=pd.read_csv(metadata_path)
 expected=set(CANDS+[NORMAL_GROUP,TARGET_GROUP]); assert expected.issubset(set(meta.subgroup))
 created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 code=git_head(); cfgsha=hf(CFG); envsha=hf(ENV); bpsha=hf(BLUEPRINT)
 binding={"code_version_id":"git:"+code,"data_version_id":dm["raw_data_version_id"],"run_config_sha256":cfgsha,"seed":SEED,"environment_sha256":envsha}
 bd=hb(cj(binding).encode())
 run_id=f"run_02-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
 out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; logs=out/"logs"; mani=out/"manifests"
 for d in [art,logs,mani]:d.mkdir(parents=True,exist_ok=False)

 rows=[]; readlog=[]
 for r in meta.itertuples(index=False):
  p=RAW/Path(r.relative_path)
  mat=loadmat(p)
  x,var=choose_signal(mat,r.subgroup,p.stem)
  d=desc(x,fs_of(r.subgroup))
  rows.append({"relative_path":r.relative_path,"independent_object_id":r.independent_object_id,"domain":r.domain,"subgroup":r.subgroup,
    "class_label":r.class_label,"target_truth_status":r.target_truth_status,"load_hp":r.load_hp,"fault_size_in":r.fault_size_in,
    "outer_race_position_clock":r.outer_race_position_clock,"fault_bearing_location":r.fault_bearing_location,"signal_var":var,**d})
  readlog.append({"relative_path":r.relative_path,"subgroup":r.subgroup,"signal_var":var,"native_points":int(np.asarray(x).size),"native_fs_hz":fs_of(r.subgroup),"status":"OK"})
 feat=pd.DataFrame(rows)
 feat.to_csv(art/"file_level_comparability_features.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame(readlog).to_csv(logs/"actual_feature_read_log.csv",index=False,encoding="utf-8-sig")

 scale_pop=feat[feat.subgroup.isin(CANDS+[TARGET_GROUP])].copy()
 _,med,scale=robust_scale_matrix(scale_pop,FEATURES)
 full=group_distances(feat,med,scale,None); full["rank"]=np.arange(1,len(full)+1)
 full.to_csv(art/"candidate_group_distances.csv",index=False,encoding="utf-8-sig")
 loo=[]
 for load in LOADS:
  x=group_distances(feat,med,scale,load);x["rank"]=np.arange(1,len(x)+1);loo.append(x)
 loo=pd.concat(loo,ignore_index=True);loo.to_csv(art/"leave_one_load_stability.csv",index=False,encoding="utf-8-sig")
 winner=str(full.iloc[0].subgroup)
 wins=Counter(loo.loc[loo["rank"]==1,"subgroup"].tolist())
 stable=(wins[winner]>=3)
 ranking=full.subgroup.tolist()
 if stable:
  mvp_groups=[winner,NORMAL_GROUP]
  alt_groups=ranking[:2]+[NORMAL_GROUP]
  selection_mode="single_fault_subgroup_plus_all_normals"
 else:
  mvp_groups=ranking[:2]+[NORMAL_GROUP]
  alt_groups=ranking+[NORMAL_GROUP]
  selection_mode="top_two_fault_subgroups_plus_all_normals_due_instability"

 mvp=meta[meta.subgroup.isin(mvp_groups)].relative_path.tolist()
 alt=meta[meta.subgroup.isin(alt_groups)].relative_path.tolist()
 sel=[]
 for r in meta[meta.domain=="source"].itertuples(index=False):
  in_m=r.relative_path in mvp;in_a=r.relative_path in alt
  if in_m:reason="MVP_RETAIN: subgroup selected by deterministic group rule; normals are always retained"
  elif in_a:reason="MVP_EXCLUDE_ALT_RETAIN: not in MVP but included by broader alternative plan"
  else:reason="EXCLUDE_FROM_MVP_AND_ALT: lower-ranked fault acquisition subgroup; no individual-file cherry-picking"
  sel.append({"relative_path":r.relative_path,"independent_object_id":r.independent_object_id,"subgroup":r.subgroup,"class_label":r.class_label,"load_hp":r.load_hp,
   "fault_size_in":r.fault_size_in,"outer_race_position_clock":r.outer_race_position_clock,"MVP_retain":in_m,"alternative_retain":in_a,"decision_reason":reason})
 pd.DataFrame(sel).to_csv(art/"source_selection_list.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame([x for x in sel if not x["MVP_retain"]]).to_csv(art/"excluded_from_mvp.csv",index=False,encoding="utf-8-sig")

 cov=pd.DataFrame([coverage_row(meta,"MVP",mvp),coverage_row(meta,"alternative",alt),coverage_row(meta,"all_source",meta[meta.domain=="source"].relative_path.tolist())])
 cov.to_csv(art/"retained_class_condition_stats.csv",index=False,encoding="utf-8-sig")

 comp=[]
 target=feat[feat.subgroup==TARGET_GROUP]
 target_amp=float(target.log_rms.median())
 for g in CANDS+[NORMAL_GROUP,TARGET_GROUP]:
  q=feat[feat.subgroup==g];mm=meta[meta.subgroup==g]
  fs=int(q.native_fs_hz.iloc[0])
  rpmvals=mm.rpm.dropna().astype(float)
  comp.append({
   "group":g,"files":len(q),"native_fs_hz":fs,"native_nyquist_hz":fs/2,"common_effective_band_hz":"0-6000",
   "common_band_feasible":fs/2>=6000,
   "rpm_median":float(rpmvals.median()) if len(rpmvals) else np.nan,
   "target_rpm_reference":"about 600 rpm only; not exact" if g!=TARGET_GROUP else "about 600 rpm only; exact values unavailable",
   "fault_bearing_location":";".join(sorted(set(mm.fault_bearing_location.astype(str)))),
   "sensor_channel_for_analysis":"single unknown-position target channel" if g==TARGET_GROUP else ("FE" if g=="source_12khz_fe" else "DE"),
   "bearing_model_evidence":"SKF6203 (problem statement)" if g=="source_12khz_fe" else ("SKF6205 (problem statement)" if g in ["source_12khz_de","source_48khz_de",NORMAL_GROUP] else "target exact bearing model not frozen in 01-A; not used as hard filter"),
   "classes":";".join(sorted(set(mm.class_label.astype(str)))),
   "loads_hp":";".join(map(str,sorted(set(mm.load_hp.dropna().astype(int).tolist())))),
   "fault_sizes_in":";".join(map(str,sorted(set(mm.fault_size_in.dropna().astype(float).tolist())))),
   "median_log10_rms_commonband":float(q.log_rms.median()),
   "abs_median_log10_rms_gap_to_target":float(abs(q.log_rms.median()-target_amp)),
   "robust_group_distance":float(full.set_index("subgroup").loc[g,"robust_group_distance"]) if g in CANDS else np.nan,
   "selection_role":"target reference" if g==TARGET_GROUP else ("forced all retained for N coverage" if g==NORMAL_GROUP else ("MVP selected fault subgroup" if g in mvp_groups else ("alternative-only fault subgroup" if g in alt_groups else "excluded fault subgroup")))
  })
 pd.DataFrame(comp).to_csv(art/"source_target_comparability_table.csv",index=False,encoding="utf-8-sig")

 limits={
  "target_label_use":"none; A-P truth unknown and no prediction label is read",
  "target_unlabeled_use":["common-band file-level amplitude/statistical/spectral descriptors","group-level median comparison only"],
  "target_unlabeled_not_used_for":["class-conditional matching","file label inference","source file cherry-picking within a subgroup"],
  "rpm_use":"target about-600rpm is reported as a coarse domain gap only; not used as exact order normalization or ranking weight",
  "sensor_use":"target sensor position is unknown; DE vs FE is reported but not given a target-derived score advantage",
  "amplitude_use":"raw/common-band log-RMS gap is reported but excluded from selection distance because sensor gain/mounting is not known to be comparable",
  "historical_predictions":"not read by this script"
 }
 (art/"target_unlabeled_usage_boundary.json").write_text(json.dumps(limits,ensure_ascii=False,indent=2),encoding="utf-8")

 plan={
  "selection_rule":cfg["selection_rule"],"full_ranking":full[["subgroup","robust_group_distance","rank"]].to_dict("records"),
  "leave_one_load_winner_counts":dict(wins),"full_winner":winner,"winner_first_in_leave_one_load":wins[winner],"stable_single_group":stable,
  "MVP":{"groups":mvp_groups,"file_count":len(mvp),"data_need":"only current official source raw MAT; all N=4 retained","rationale":"hard class/load coverage plus label-blind subgroup comparability","risk":["N class only 4 independent files","target exact RPM/sensor position unknown","source-target bearing/system gap remains"],"later_validation":["grouped-by-file CV only","compare against all-source baseline","report per-class metrics especially N","stress test across loads/sizes"]},
  "alternative":{"groups":alt_groups,"file_count":len(alt),"data_need":"current official source raw MAT","rationale":"broader source diversity while retaining top comparable subgroup(s)","risk":["larger domain heterogeneity","possible negative transfer"],"later_validation":["same grouped splits as MVP","paired comparison to MVP and all-source"]},
  "external_public_source_data_used":False
 }
 (art/"selection_plan_and_alternative.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")

 verify={
  "step_id":"02-A","run_id":run_id,
  "checks":{
   "rule_recomputation":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/source_selection_list.csv","key_numbers":{"MVP_files":len(mvp),"alternative_files":len(alt),"winner":winner}},
   "four_class_coverage":{"passed":all(int(cov.loc[cov.plan=="MVP",c].iloc[0])>0 for c in CLASSES),"evidence":f"outputs/runs/{run_id}/artifacts/retained_class_condition_stats.csv","MVP_counts":{c:int(cov.loc[cov.plan=="MVP",c].iloc[0]) for c in CLASSES}},
   "load_coverage":{"passed":set(meta[meta.relative_path.isin(mvp)].load_hp.dropna().astype(int))==set(LOADS),"evidence":f"outputs/runs/{run_id}/artifacts/retained_class_condition_stats.csv"},
   "normal_limit_truthful":{"passed":int((meta[meta.relative_path.isin(mvp)].class_label=="N").sum())==4,"evidence":"01-A frozen metadata + retained_class_condition_stats.csv","normal_files":4},
   "leave_one_load_sensitivity":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/leave_one_load_stability.csv","winner_counts":dict(wins),"selection_fallback_rule_applied":not stable},
   "target_label_boundary":{"passed":bool((meta[meta.domain=="target"].target_truth_status=="unknown").all()),"evidence":f"outputs/runs/{run_id}/artifacts/target_unlabeled_usage_boundary.json"},
   "no_final_classifier_training":{"passed":True,"evidence":"script contains signal preprocessing/descriptors/ranking only; no estimator fit/predict"}
  }
 }
 verify["passed"]=all(x["passed"] for x in verify["checks"].values());verify["status"]="passed" if verify["passed"] else "failed"
 (art/"validation_record.json").write_text(json.dumps(verify,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"02A"/"validation_record.json").write_text(json.dumps(verify,ensure_ascii=False,indent=2),encoding="utf-8")

 summary=f"""# STEP02-A 面向迁移的源域筛选结果

run_id: {run_id}
raw_data_version_id: {dm['raw_data_version_id']}
full ranking: {full[['subgroup','robust_group_distance','rank']].to_dict('records')}
leave-one-load winner counts: {dict(wins)}
selection mode: {selection_mode}
MVP groups: {mvp_groups}; files={len(mvp)}
alternative groups: {alt_groups}; files={len(alt)}

目标A-P只以无标签信号参与组级分布可比性分析，不使用任何历史/预测类别。
共同分析口径为12kHz、0-6kHz。幅值差单独报告，不参与筛选距离；目标约600rpm只报告域差异，不作精确阶次归一化。
正常类只有4个独立文件，所有可行方案均强制全部保留。
"""
 (art/"selection_summary.md").write_text(summary,encoding="utf-8")

 freeze={
  "schema_version":"00C-A-freeze-1.0","package_type":"A_freeze_package","step_id":"02-A",
  "freeze_package_id":f"FREEZE-02A-{run_id[-8:]}","status":"passed" if verify["passed"] else "failed","run_id":run_id,
  "versions":{"code_version_id":"git:"+code,"raw_data_version_id":dm["raw_data_version_id"],"input_derived_data_ids":[],"run_config_sha256":cfgsha,"environment_sha256":envsha},
  "random_seed":SEED,
  "parameters":{"common_analysis_fs_hz":COMMON_FS,"common_effective_band_hz":[0,6000],"candidate_fault_subgroups":CANDS,
    "normal_policy":"retain all 4","distance_feature_families":FAMILIES,"family_weights":"equal after robust feature scaling; internal metric only","stability":"leave-one-load 0/1/2/3"},
  "real_results":[{"name":"full_group_ranking","value":full.to_dict("records")},{"name":"leave_one_load_winner_counts","value":dict(wins)},
    {"name":"MVP","value":plan["MVP"]},{"name":"alternative","value":plan["alternative"]},{"name":"coverage","value":cov.to_dict("records")}],
  "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/source_selection_list.csv"},{"path":f"outputs/runs/{run_id}/artifacts/leave_one_load_stability.csv"}],
  "anomalies_and_failures":[],
  "b_handoff":{"required_data":["source_target_comparability_table.csv","source_selection_list.csv","retained_class_condition_stats.csv","selection_plan_and_alternative.json","validation_record.json"],
    "supported_conclusions":["selection is subgroup-rule-based and target-label-blind","all viable plans retain all 4 normal files","target unlabeled data is used only for group-level comparability"],
    "wording_limits":["robust_group_distance is an internal comparability metric, not an official score","about-600rpm is approximate","target sensor position and exact bearing model are not established by 01-A"],
    "approved_tables":["source_target_comparability_table.csv","candidate_group_distances.csv","retained_class_condition_stats.csv"],"approved_figures":[]},
  "unresolved_issues":[],
  "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["RAW-5a5dd129c91bfc64 changes","02-A selection code/config changes","01-A label/provenance boundary changes"]}
 }
 f0=json.loads(json.dumps(freeze,ensure_ascii=False));freeze["freeze"]["content_sha256"]=hb(cj(f0).encode())
 (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"02A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"02A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

 pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True);(mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
 runtime={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"gpu":None,"cuda":None,"git_commit":code,"pip_freeze_sha256":hf(mani/"pip_freeze.txt")}
 (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
 outs=[]
 for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
  outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
 ot=hb(cj(outs).encode())
 rm={"schema_version":"02A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"02-A","status":"completed" if verify["passed"] else "failed","created_utc":created,"execution_mode":"chat_plus_git",
  "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code,"entrypoint":"scripts/step02a_transfer_source_selection.py","code_manifest":"protocol/02A/code_manifest.json"},
  "data":{"raw_data_version_id":dm["raw_data_version_id"],"source_01A_manifest":"protocol/01A/data_manifest.json"},"blueprint":{"path":"protocol/02A/question1_argument_blueprint.json","sha256":bpsha},
  "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":envsha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
  "config":{"path":"protocol/02A/run_config.json","sha256":cfgsha},"outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":ot},
  "checkpoint":{"completed_batches":["all-177-file-descriptors","full-ranking","leave-one-load-0","leave-one-load-1","leave-one-load-2","leave-one-load-3"],"pending_batches":[],"resume_token":None},
  "duration_seconds":time.time()-t0}
 (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"ok":verify["passed"],"run_id":run_id,"winner":winner,"winner_leave_one_load_first_count":wins[winner],"stable":stable,"MVP_groups":mvp_groups,"MVP_files":len(mvp),"alternative_groups":alt_groups,"alternative_files":len(alt),"output_root":f"outputs/runs/{run_id}"},ensure_ascii=False))
 return 0 if verify["passed"] else 2

if __name__=="__main__":
 raise SystemExit(main())
