#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, random, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import step04a_window_feature_extraction as fex

STEP="12-A2"
CFG=ROOT/"protocol/12A2/run_config.json"
STATE=ROOT/"protocol/00C/process_state_card.json"
ENV=ROOT/"protocol/00B/environment_manifest.json"
RUN02="run_02-A_20260919T162016957185Z_75708ffe_6f36d0b4"
RUN04="run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0"
RUN07="run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb"
RUN09="run_09-A_20260920T070725126712Z_6ef81c21_26f99878"
RUN10="run_10-A_20260920T085322828930Z_bcbb6757_4346ce31"
RUN11="run_11-A_20260920T151947188398Z_54a3c266_1f812c22"
RUN12A1="run_12-A1_20260920T155834545736Z_ea78ae62_5e3e74e2"
B02=ROOT/"outputs/runs"/RUN02/"artifacts"
B04=ROOT/"outputs/runs"/RUN04/"artifacts"
B07=ROOT/"outputs/runs"/RUN07/"artifacts"
B09=ROOT/"outputs/runs"/RUN09
B10=ROOT/"outputs/runs"/RUN10/"artifacts"
B11=ROOT/"outputs/runs"/RUN11/"artifacts"
B12A1=ROOT/"outputs/runs"/RUN12A1/"artifacts"
RAW=ROOT/"data/raw"

CLASSES=["OR","IR","B","N"]
FEATURES=[
 "mean","std","rms","mean_abs","peak_abs","peak_to_peak","skewness","kurtosis",
 "crest_factor","impulse_factor","shape_factor","clearance_factor","zero_cross_rate",
 "spectral_centroid_hz","spectral_rms_hz","spectral_entropy","spectral_flatness",
 "dominant_frequency_hz","rolloff95_hz","band_ratio_0_500","band_ratio_500_1500",
 "band_ratio_1500_3000","band_ratio_3000_5500","envelope_rms","envelope_kurtosis",
 "envelope_spectral_entropy"
]
EPS=1e-12

def nowz(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def J(p): return json.loads((ROOT/p).read_text(encoding="utf-8-sig"))
def hf(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canon(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def git_head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
def require(x,msg):
    if not x: raise RuntimeError(msg)

def native_fs(subgroup):
    if subgroup in {"source_12khz_fe","source_12khz_de"}: return 12000
    if subgroup in {"source_48khz_de","source_48khz_normal"}: return 48000
    raise RuntimeError(f"unknown subgroup {subgroup}")

def load_preprocessed_signal(rel, subgroup, cache):
    key=(str(rel),str(subgroup))
    if key in cache: return cache[key]
    p=RAW/str(rel)
    mat=loadmat(p)
    x,var=fex.primary_signal(mat,str(subgroup),p.stem,"source")
    fs=native_fs(str(subgroup))
    y,info=fex.preprocess_common(x,fs,"source",str(subgroup))
    cache[key]=(np.asarray(y,float),var,fs,info)
    return cache[key]

def extract_plan_windows(selection, plan, window_s=1.0, overlap=0.5, cache=None):
    cache={} if cache is None else cache
    if plan=="MVP":
        m=selection["MVP_retain"].astype(bool)
    elif plan=="alternative":
        m=selection["alternative_retain"].astype(bool)
    elif plan=="all_source":
        m=np.ones(len(selection),dtype=bool)
    else:
        raise ValueError(plan)
    sel=selection[m].copy().sort_values("relative_path")
    rows=[]
    win=int(round(float(window_s)*12000))
    stride=int(round(win*(1.0-float(overlap))))
    require(stride>0,"invalid stride")
    for r in sel.itertuples(index=False):
        y,var,fs,info=load_preprocessed_signal(r.relative_path,r.subgroup,cache)
        start=0; k=0
        while start+win<=len(y):
            feat=fex.common_features(y[start:start+win])
            rows.append({
              "window_key":f"{r.relative_path}|{start}|{start+win}",
              "group_id":str(r.independent_object_id),"relative_path":str(r.relative_path),
              "subgroup":str(r.subgroup),"class_label":str(r.class_label),"load_hp":float(r.load_hp),
              "window_start_common_idx":int(start),"window_end_common_idx_exclusive":int(start+win),
              **feat
            })
            start+=stride; k+=1
    df=pd.DataFrame(rows)
    require(len(df)>0,f"no windows for {plan}")
    require(df[FEATURES].replace([np.inf,-np.inf],np.nan).notna().all().all(),f"nonfinite features {plan}")
    return df,sel

def train_weights(df):
    gc=df.groupby("group_id").size().to_dict()
    gt=df[["group_id","class_label"]].drop_duplicates()
    cc=gt.groupby("class_label").size().to_dict()
    w=np.array([(1.0/gc[g])*(1.0/cc[c]) for g,c in df[["group_id","class_label"]].itertuples(index=False)],float)
    return w*(len(w)/w.sum())

def fit_fixed_rf(df,seed):
    clf=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=1,max_features="sqrt",
                               random_state=int(seed),n_jobs=1)
    clf.fit(df[FEATURES].to_numpy(float),df["class_label"].astype(str).to_numpy(),sample_weight=train_weights(df))
    return clf

def scores_fixed(clf,X):
    raw=clf.predict_proba(np.asarray(X,float))
    pos={str(c):i for i,c in enumerate(clf.classes_)}
    out=np.zeros((len(raw),4),float)
    for j,c in enumerate(CLASSES):
        require(c in pos,f"missing class {c}")
        out[:,j]=raw[:,pos[c]]
    return out

def aggregate_files(df,scores):
    q=df[["group_id","class_label","load_hp"]].copy()
    for j,c in enumerate(CLASSES): q[f"score_{c}"]=scores[:,j]
    cols=[f"score_{c}" for c in CLASSES]
    g=q.groupby(["group_id","class_label","load_hp"],sort=True,dropna=False)[cols].mean().reset_index()
    A=g[cols].to_numpy(float)
    g["pred_label"]=np.asarray(CLASSES,dtype=object)[np.argmax(A,axis=1)]
    order=np.argsort(A,axis=1)
    g["top_model_score"]=A[np.arange(len(A)),order[:,-1]]
    g["score_margin"]=A[np.arange(len(A)),order[:,-1]]-A[np.arange(len(A)),order[:,-2]]
    g["window_count"]=q.groupby("group_id").size().reindex(g["group_id"]).to_numpy()
    return g

def metric(y,p):
    y=np.asarray(y,dtype=object); p=np.asarray(p,dtype=object)
    rec=recall_score(y,p,labels=CLASSES,average=None,zero_division=0)
    cm=confusion_matrix(y,p,labels=CLASSES)
    return {
      "macro_f1":float(f1_score(y,p,labels=CLASSES,average="macro",zero_division=0)),
      "accuracy":float(accuracy_score(y,p)),
      "min_class_recall":float(np.min(rec)),
      **{f"recall_{c}":float(v) for c,v in zip(CLASSES,rec)},
      "confusion_matrix":json.dumps(cm.tolist(),ensure_ascii=False)
    }

def manual_macro_f1_from_cm(cm):
    cm=np.asarray(cm,float); vals=[]
    for i in range(len(CLASSES)):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        den=2*tp+fp+fn
        vals.append(0.0 if den<=0 else 2*tp/den)
    return float(np.mean(vals))

def eval_load_holdout(df,seeds):
    fold_rows=[]; pooled_rows=[]; file_rows=[]
    for seed in seeds:
        parts=[]
        for load in [0,1,2,3]:
            tr=df[df["load_hp"]!=float(load)].copy()
            te=df[df["load_hp"]==float(load)].copy()
            require(tr["group_id"].nunique()+te["group_id"].nunique()==df["group_id"].nunique(),"load split file count mismatch")
            clf=fit_fixed_rf(tr,seed)
            ff=aggregate_files(te,scores_fixed(clf,te[FEATURES].to_numpy(float)))
            m=metric(ff["class_label"],ff["pred_label"])
            fold_rows.append({"seed":seed,"held_load_hp":load,"train_files":tr.group_id.nunique(),"test_files":te.group_id.nunique(),**m})
            ff["seed"]=seed;ff["held_load_hp"]=load
            parts.append(ff);file_rows.append(ff)
        allf=pd.concat(parts,ignore_index=True)
        m=metric(allf["class_label"],allf["pred_label"])
        cm=json.loads(m["confusion_matrix"])
        manual=manual_macro_f1_from_cm(cm)
        pooled_rows.append({"seed":seed,"files":len(allf),**m,"manual_macro_f1":manual,"macro_f1_abs_diff_manual":abs(m["macro_f1"]-manual)})
    return pd.DataFrame(fold_rows),pd.DataFrame(pooled_rows),pd.concat(file_rows,ignore_index=True)

def file_balanced_mean_sd(X,groups):
    s=pd.Series(np.asarray(groups,dtype=object))
    cnt=s.value_counts().to_dict()
    w=np.array([1.0/cnt[g] for g in s],float);w=w/w.sum()
    X=np.asarray(X,float)
    mu=(X*w[:,None]).sum(axis=0)
    var=(((X-mu)**2)*w[:,None]).sum(axis=0)
    return mu,np.sqrt(np.maximum(var,EPS))

def t1_transform(Xt,gt,mus,sds,mut,sdt,alpha=0.25):
    full=((np.asarray(Xt,float)-mut)/np.where(sdt>EPS,sdt,1.0))*sds+mus
    return (1.0-float(alpha))*np.asarray(Xt,float)+float(alpha)*full

def rbf_mmd2(X,Y):
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    Z=np.vstack([X,Y])
    d2=np.sum((Z[:,None,:]-Z[None,:,:])**2,axis=2)
    vals=d2[np.triu_indices_from(d2,k=1)]
    nz=vals[vals>0]; med=float(np.median(nz)) if len(nz) else 1.0
    gamma=1.0/max(2.0*med,EPS)
    Kxx=np.exp(-gamma*np.sum((X[:,None,:]-X[None,:,:])**2,axis=2))
    Kyy=np.exp(-gamma*np.sum((Y[:,None,:]-Y[None,:,:])**2,axis=2))
    Kxy=np.exp(-gamma*np.sum((X[:,None,:]-Y[None,:,:])**2,axis=2))
    return float(Kxx.mean()+Kyy.mean()-2*Kxy.mean()),float(gamma),float(med)

def target_frame():
    X=pd.read_csv(B04/"target_interface"/"X_target_common.csv")
    M=pd.read_csv(B04/"target_interface"/"target_window_metadata.csv")
    d=X.merge(M[["window_id","group_id","relative_path"]],on="window_id",validate="one_to_one")
    d["target_id"]=d["group_id"].map(lambda x:Path(str(x)).stem)
    return d

def source_frozen_frame():
    X=pd.read_csv(B04/"q2_interface"/"X_source_common.csv")
    y=pd.read_csv(B04/"q2_interface"/"y_source_labels.csv")
    g=pd.read_csv(B04/"q2_interface"/"groups_source.csv")
    m=pd.read_csv(B04/"q2_interface"/"source_window_metadata.csv")
    d=X.merge(y,on="window_id",validate="one_to_one").merge(g,on="window_id",validate="one_to_one")
    d=d.merge(m[["window_id","load_hp","relative_path","subgroup","window_start_common_idx","window_end_common_idx_exclusive"]],on="window_id",validate="one_to_one")
    return d

def formula_code_unit_audit(extracted_primary, frozen):
    # Compare all independently re-extracted MVP primary windows to frozen STEP04 features using physical group/start/end key.
    k=["group_id","window_start_common_idx","window_end_common_idx_exclusive"]
    a=extracted_primary[k+FEATURES].copy()
    b=frozen[k+FEATURES].copy()
    z=a.merge(b,on=k,suffixes=("_new","_frozen"),validate="one_to_one")
    require(len(z)==len(frozen)==len(extracted_primary),"primary re-extraction row mismatch")
    diffs={f:float(np.max(np.abs(z[f"{f}_new"].to_numpy(float)-z[f"{f}_frozen"].to_numpy(float)))) for f in FEATURES}
    maxdiff=max(diffs.values())
    fd=pd.read_csv(B04/"feature_dictionary.csv")
    fo=pd.read_csv(B07/"feature_order.csv").sort_values("feature_order")
    common=fd[(fd["tier"]=="common") & (fd["default_q2_use"].astype(str).str.lower().isin(["true","1"]))]
    order_ok=fo["feature_name"].astype(str).tolist()==FEATURES and common["feature_name"].astype(str).tolist()==FEATURES
    units=common.set_index("feature_name")["unit"].astype(str).to_dict()
    unit_ok=all(units.get(f,"").strip() not in {"","nan","None"} for f in FEATURES)
    hz={"spectral_centroid_hz","spectral_rms_hz","dominant_frequency_hz","rolloff95_hz"}
    hz_ok=all(units[f]=="Hz" for f in hz)
    dimensionless={"skewness","kurtosis","crest_factor","impulse_factor","shape_factor","clearance_factor",
                   "zero_cross_rate","spectral_entropy","spectral_flatness","band_ratio_0_500","band_ratio_500_1500",
                   "band_ratio_1500_3000","band_ratio_3000_5500","envelope_kurtosis","envelope_spectral_entropy"}
    dim_ok=all(units[f]=="1" for f in dimensionless)
    return {"row_count":len(z),"feature_count":len(FEATURES),"max_abs_feature_diff":maxdiff,
            "per_feature_max_abs_diff":diffs,"feature_order_exact":order_ok,"units_present":unit_ok,
            "frequency_features_hz":hz_ok,"dimensionless_features_unit1":dim_ok,
            "passed":bool(maxdiff<=1e-10 and order_ok and unit_ok and hz_ok and dim_ok)}

def domain_gap(plan_df,target):
    sf=plan_df.groupby("group_id",sort=True)[FEATURES].mean()
    tf=target.groupby("group_id",sort=True)[FEATURES].mean()
    mu=sf.mean().to_numpy(float); sd=sf.std(ddof=1).to_numpy(float);sd=np.where(sd>EPS,sd,1.0)
    Zs=(sf.to_numpy(float)-mu)/sd; Zt=(tf.to_numpy(float)-mu)/sd
    mmd,gamma,med=rbf_mmd2(Zs,Zt)
    return {"source_files":len(sf),"target_files":len(tf),"mmd2_biased":mmd,"rbf_gamma":gamma,"median_pairwise_sqdist":med}

def official_target_reproduction(target,bundle,official):
    art=bundle["adaptation_artifact"]
    mus=np.asarray(art["source_mean"],float);sds=np.asarray(art["source_sd"],float)
    mut=np.asarray(art["target_mean"],float);sdt=np.asarray(art["target_sd"],float)
    X=t1_transform(target[FEATURES].to_numpy(float),target["group_id"].to_numpy(),mus,sds,mut,sdt,0.25)
    sc=scores_fixed(bundle["classifier"],X)
    tmp=target[["group_id"]].copy()
    for j,c in enumerate(CLASSES):tmp[f"score_{c}"]=sc[:,j]
    g=tmp.groupby("group_id",sort=True)[[f"score_{c}" for c in CLASSES]].mean().reset_index()
    A=g[[f"score_{c}" for c in CLASSES]].to_numpy(float)
    g["pred_label"]=np.asarray(CLASSES,dtype=object)[np.argmax(A,axis=1)]
    g["target_id"]=g["group_id"].map(lambda x:Path(str(x)).stem)
    zz=g.merge(official[["target_id","pred_label",*[f"score_{c}" for c in CLASSES]]],on="target_id",suffixes=("_new","_official"),validate="one_to_one")
    md=max(float(np.max(np.abs(zz[f"score_{c}_new"]-zz[f"score_{c}_official"]))) for c in CLASSES)
    lm=bool((zz["pred_label_new"].astype(str)==zz["pred_label_official"].astype(str)).all())
    return {"max_score_abs_diff":md,"labels_match_16_of_16":lm,"passed":bool(md<=1e-12 and lm)}

def a2_04_loto_target(target,bundle,official):
    art=bundle["adaptation_artifact"]
    mus=np.asarray(art["source_mean"],float);sds=np.asarray(art["source_sd"],float)
    rows=[]
    for tid in sorted(target["target_id"].unique()):
        held=target[target["target_id"]==tid].copy()
        other=target[target["target_id"]!=tid].copy()
        mut,sdt=file_balanced_mean_sd(other[FEATURES].to_numpy(float),other["group_id"].to_numpy())
        Xh=t1_transform(held[FEATURES].to_numpy(float),held["group_id"].to_numpy(),mus,sds,mut,sdt,0.25)
        sc=scores_fixed(bundle["classifier"],Xh)
        mean=sc.mean(axis=0); order=np.argsort(mean)
        pred=CLASSES[int(np.argmax(mean))]
        off=official[official["target_id"]==tid].iloc[0]
        rows.append({
          "target_id":tid,"official_label":str(off["pred_label"]),"loto_label":pred,
          "same_label":pred==str(off["pred_label"]),
          **{f"loto_score_{c}":float(mean[j]) for j,c in enumerate(CLASSES)},
          "loto_top_score":float(mean[order[-1]]),"loto_margin":float(mean[order[-1]]-mean[order[-2]]),
          "official_top_score":float(off["top_model_score"]),"official_margin":float(off["score_margin"]),
          "top_score_delta":float(mean[order[-1]]-float(off["top_model_score"]))
        })
    d=pd.DataFrame(rows)
    return d,{"files":len(d),"label_agreement_count":int(d.same_label.sum()),"label_agreement_rate":float(d.same_label.mean()),
              "changed_files":d.loc[~d.same_label,"target_id"].tolist(),
              "max_abs_top_score_delta":float(np.max(np.abs(d.top_score_delta)))}

def bootstrap_source_uncertainty(source,target,official,cfg):
    target_X=target[FEATURES].to_numpy(float); target_groups=target["group_id"].to_numpy()
    mut,sdt=file_balanced_mean_sd(target_X,target_groups)
    groups_by_class={}
    base_meta=source[["group_id","class_label"]].drop_duplicates()
    for c in CLASSES: groups_by_class[c]=sorted(base_meta.loc[base_meta.class_label==c,"group_id"].astype(str))
    rec=[]
    rep_global=0
    for seed in cfg["tasks"]["A2-05"]["seeds"]:
        rng=np.random.default_rng(int(seed))
        for rr in range(int(cfg["tasks"]["A2-05"]["replicates_per_seed"])):
            parts=[]
            for c in CLASSES:
                ids=groups_by_class[c]
                picks=rng.choice(ids,size=len(ids),replace=True)
                for j,gid in enumerate(picks):
                    q=source[source["group_id"].astype(str)==str(gid)].copy()
                    q["group_id"]=f"BOOT|{seed}|{rr}|{c}|{j}|{gid}"
                    parts.append(q)
            tr=pd.concat(parts,ignore_index=True)
            clf=fit_fixed_rf(tr,int(cfg["fixed_model"]["official_seed"]))
            mus,sds=file_balanced_mean_sd(tr[FEATURES].to_numpy(float),tr["group_id"].to_numpy())
            Xa=t1_transform(target_X,target_groups,mus,sds,mut,sdt,0.25)
            sc=scores_fixed(clf,Xa)
            tmp=pd.DataFrame({"group_id":target_groups})
            for j,c in enumerate(CLASSES):tmp[f"score_{c}"]=sc[:,j]
            gf=tmp.groupby("group_id",sort=True)[[f"score_{c}" for c in CLASSES]].mean().reset_index()
            A=gf[[f"score_{c}" for c in CLASSES]].to_numpy(float)
            ords=np.argsort(A,axis=1)
            gf["label"]=np.asarray(CLASSES,dtype=object)[np.argmax(A,axis=1)]
            gf["top_score"]=A[np.arange(len(A)),ords[:,-1]]
            gf["margin"]=A[np.arange(len(A)),ords[:,-1]]-A[np.arange(len(A)),ords[:,-2]]
            gf["target_id"]=gf.group_id.map(lambda x:Path(str(x)).stem)
            for row in gf.itertuples(index=False):
                off=official[official.target_id==row.target_id].iloc[0]
                rec.append({"bootstrap_seed":seed,"replicate":rr,"replicate_global":rep_global,"target_id":row.target_id,
                            "official_label":str(off.pred_label),"bootstrap_label":str(row.label),
                            "same_official":str(row.label)==str(off.pred_label),
                            "top_score":float(row.top_score),"margin":float(row.margin)})
            rep_global+=1
    d=pd.DataFrame(rec)
    s=d.groupby("target_id",sort=True).agg(
       bootstrap_runs=("same_official","size"),
       official_label=("official_label","first"),
       label_retention_rate=("same_official","mean"),
       top_score_median=("top_score","median"),
       top_score_q05=("top_score",lambda x:float(np.quantile(x,0.05))),
       top_score_q95=("top_score",lambda x:float(np.quantile(x,0.95))),
       margin_median=("margin","median"),
       margin_q05=("margin",lambda x:float(np.quantile(x,0.05))),
       margin_q95=("margin",lambda x:float(np.quantile(x,0.95)))
    ).reset_index()
    return d,s

def main():
    t0=time.time()
    cfg=J("protocol/12A2/run_config.json")
    state=J("protocol/00C/process_state_card.json")
    require(state["CURRENT_ALLOWED_STEP"]=="12-A2" and state["NEXT_ALLOWED"]=="12-A2","12-A2 gate not open")
    require(state["LAST_PASS_TOKEN"]=="12-A1-V2026.09.20-3257a6ab-通过","previous token mismatch")
    require(any(x.get("id")=="P0-01" for x in state.get("OPEN_P0",[])),"P0-01 not registered")
    f12=J("protocol/12A1/latest_freeze_package.json")
    require(f12["freeze_package_id"]=="FREEZE-12A1-5e3e74e2" and f12["status"]=="passed","12-A1 freeze mismatch")

    selection=pd.read_csv(B02/"source_selection_list.csv")
    require(len(selection)==161,"source selection inventory not 161")
    # Robust bool conversion.
    for c in ["MVP_retain","alternative_retain"]:
        selection[c]=selection[c].astype(str).str.lower().map({"true":True,"false":False})
    require(int(selection["MVP_retain"].sum())==49 and int(selection["alternative_retain"].sum())==109,"source plan counts mismatch")

    frozen_source=source_frozen_frame()
    target=target_frame()
    official=pd.read_csv(B10/"A_P_final_label_table.csv")
    require(len(official)==16 and set(official.target_id)==set(list("ABCDEFGHIJKLMNOP")),"official A-P table mismatch")
    bundle=joblib.load(B09/"models"/"final_transfer_bundle.joblib")
    require(bundle["adaptation_artifact"]["method"]=="T1-SHRINK-MOMENT","bundle is not T1")
    require(abs(float(bundle["adaptation_artifact"]["alpha"])-0.25)<1e-12,"bundle alpha mismatch")
    require(list(bundle["feature_names"])==FEATURES,"bundle feature order mismatch")

    cache={}
    # A2-02 re-extract all six frozen window candidates on MVP. Primary also becomes independent formula/code check.
    window_results=[]; window_fold_parts=[]; primary_extracted=None
    for ws,ov in cfg["tasks"]["A2-02"]["window_candidates"]:
        d,_=extract_plan_windows(selection,"MVP",float(ws),float(ov),cache)
        folds,pool,files=eval_load_holdout(d,[cfg["fixed_model"]["official_seed"]])
        pr=pool.iloc[0].to_dict()
        setting=f"{float(ws):.1f}s_overlap{float(ov):.1f}"
        if abs(float(ws)-1.0)<1e-12 and abs(float(ov)-0.5)<1e-12: primary_extracted=d.copy()
        window_results.append({"setting":setting,"window_seconds":float(ws),"overlap":float(ov),
                               "windows":len(d),"files":d.group_id.nunique(),**{k:v for k,v in pr.items() if k not in {"confusion_matrix"}}})
        ff=folds.copy();ff["setting"]=setting;window_fold_parts.append(ff)
    window_summary=pd.DataFrame(window_results)
    primary_val=float(window_summary.loc[(window_summary.window_seconds==1.0)&(window_summary.overlap==0.5),"macro_f1"].iloc[0])
    window_summary["delta_macro_f1_vs_primary"]=window_summary["macro_f1"]-primary_val

    formula_audit=formula_code_unit_audit(primary_extracted,frozen_source)

    # A2-01 exact fixed final H1_RF on orthogonal physical load holdout, 3 seeds.
    a201_folds,a201_pooled,a201_files=eval_load_holdout(frozen_source,cfg["fixed_model"]["seed_sensitivity"])
    primary=a201_pooled[a201_pooled.seed==cfg["fixed_model"]["official_seed"]].iloc[0]
    th=cfg["tasks"]["A2-01"]["closure_thresholds"]
    p0_checks={
      "primary_seed_pooled_macro_f1":float(primary.macro_f1)>=float(th["primary_seed_pooled_macro_f1_min"]),
      "primary_seed_min_class_recall":float(primary.min_class_recall)>=float(th["primary_seed_min_class_recall_min"]),
      "all_seed_pooled_macro_f1":float(a201_pooled.macro_f1.min())>=float(th["all_seed_pooled_macro_f1_min"]),
      "seed_mean_pooled_macro_f1":float(a201_pooled.macro_f1.mean())>=float(th["seed_mean_pooled_macro_f1_min"]),
      "independent_manual_metric_recalc":float(a201_pooled.macro_f1_abs_diff_manual.max())<=1e-12
    }
    p0_closed=all(p0_checks.values())

    # A2-03 source-plan sensitivity: exact frozen feature formulas, same load-held-out metric.
    plan_rows=[];plan_fold=[];plan_cache={"MVP":primary_extracted}
    target_file=target.groupby("group_id",sort=True)[FEATURES].mean()
    for plan in cfg["tasks"]["A2-03"]["source_plans"]:
        if plan=="MVP": d=primary_extracted
        else:
            d,_=extract_plan_windows(selection,plan,1.0,0.5,cache)
        plan_cache[plan]=d
        folds,pool,files=eval_load_holdout(d,[cfg["fixed_model"]["official_seed"]])
        pp=pool.iloc[0]
        gap=domain_gap(d,target)
        plan_rows.append({"plan":plan,"files":d.group_id.nunique(),"windows":len(d),
                          "macro_f1":float(pp.macro_f1),"accuracy":float(pp.accuracy),
                          "min_class_recall":float(pp.min_class_recall),
                          **{f"recall_{c}":float(pp[f"recall_{c}"]) for c in CLASSES},
                          "mmd2_biased":gap["mmd2_biased"]})
        q=folds.copy();q["plan"]=plan;plan_fold.append(q)
    plan_summary=pd.DataFrame(plan_rows)
    for p,n in cfg["tasks"]["A2-03"]["expected_file_counts"].items():
        require(int(plan_summary.loc[plan_summary.plan==p,"files"].iloc[0])==int(n),f"{p} file count mismatch")

    # A2-04 final target transductive self-influence.
    official_repro=official_target_reproduction(target,bundle,official)
    loto,loto_summary=a2_04_loto_target(target,bundle,official)

    # A2-05 source-file composition uncertainty.
    boot_detail,boot_summary=bootstrap_source_uncertainty(frozen_source,target,official,cfg)

    # A2-06: derive N small-sample evidence from existing A2-01/A2-03 results.
    n_load=a201_folds[a201_folds.seed==cfg["fixed_model"]["official_seed"]][["held_load_hp","test_files","recall_N","macro_f1"]].copy()
    n_plan=plan_summary[["plan","files","recall_N","macro_f1"]].copy()
    n_summary={
      "independent_N_files":4,
      "load_holdout_N_recall_values":[float(x) for x in n_load.recall_N],
      "load_holdout_N_recall_min":float(n_load.recall_N.min()),
      "source_plan_N_recall_values":{r.plan:float(r.recall_N) for r in n_plan.itertuples(index=False)},
      "limitation":"N has only four independent files; these sensitivities do not create a high-precision population estimate."
    }

    # Existing A-P reasonable-setting check from STEP10 (do not reinvent selection).
    sens=pd.read_csv(B10/"A_P_aggregation_alpha_sensitivity.csv")
    bs=pd.read_csv(B10/"A_P_bootstrap_seed_stability.csv")
    require(len(sens)>0 and len(bs)>0,"STEP10 stability tables missing")
    cross=sens.groupby("target_id").agg(votes=("pred_label","size"),distinct=("pred_label","nunique")).reset_index()
    official_map=dict(zip(official.target_id,official.pred_label))
    agree=[]
    for tid,g in sens.groupby("target_id"):
        agree.append({"target_id":tid,"setting_votes":len(g),
                      "official_vote_rate":float(np.mean(g.pred_label.astype(str)==str(official_map[tid]))),
                      "distinct_labels":int(g.pred_label.nunique())})
    agree=pd.DataFrame(agree)
    ap_stab={
      "files":16,"nine_setting_min_official_vote_rate":float(agree.official_vote_rate.min()),
      "nine_setting_full_agreement_files":int((agree.official_vote_rate==1.0).sum()),
      "step10_bootstrap_min_agreement":float(official.bootstrap_official_agreement_rate.min()),
      "A2_04_loto_label_agreement_rate":float(loto_summary["label_agreement_rate"]),
      "A2_05_source_bootstrap_min_label_retention":float(boot_summary.label_retention_rate.min())
    }

    # Final explanation check: no new explanation experiments because 12-A1 selected existing sufficient evidence.
    v11=J("protocol/11A/validation_record.json")
    f11=J("protocol/11A/latest_freeze_package.json")
    explanation_check={
      "final_model_identity":bool(v11["checks"]["final_09_10_model_identity_verified"]),
      "target_10A_reproduction":bool(v11["checks"]["target_10A_predictions_exactly_reproduced"]),
      "faithfulness":bool(v11["checks"]["faithfulness_validation_executed"]),
      "stability":bool(v11["checks"]["stability_validation_executed"]),
      "random_sanity":bool(v11["checks"]["randomized_sanity_check_executed_and_effective"]),
      "contradictions_retained":bool(v11["checks"]["failure_contradiction_cases_retained"]),
      "case_pass_rate_faithfulness":next(x["value"]["case_pass_rate"] for x in f11["real_results"] if x["name"]=="faithfulness"),
      "case_pass_rate_stability":next(x["value"]["case_pass_rate"] for x in f11["real_results"] if x["name"]=="stability")
    }
    explanation_check["passed"]=all([explanation_check[k] for k in ["final_model_identity","target_10A_reproduction","faithfulness","stability","random_sanity","contradictions_retained"]])

    # Version chain exact checks.
    f07=J("protocol/07A/latest_freeze_package.json"); f09=J("protocol/09A/latest_freeze_package.json"); f10=J("protocol/10A/latest_freeze_package.json")
    q3=J(f"outputs/runs/{RUN07}/artifacts/q3_interface.json")
    version_check={
      "07_final_family_H1_RF":q3["final_family"]=="H1_RF",
      "07_features_26":q3["feature_input"].startswith("04-A target-compatible 26"),
      "09_source_model_sha":next(x["value"]["source_model_sha256"] for x in f09["real_results"] if x["name"]=="source_retention")=="4ee408ac96054e3b86e1c62c7a7afedc4baabd792390b1ad8bbb338da404d685",
      "10_transfer_freeze_bound":"FREEZE-09A-26f99878" in f10["versions"]["input_derived_data_ids"],
      "11_transfer_and_prediction_bound":"FREEZE-09A-26f99878" in f11["versions"]["input_derived_data_ids"] and "FREEZE-10A-4346ce31" in f11["versions"]["input_derived_data_ids"],
      "official_target_reproduction":official_repro["passed"]
    }
    version_check["passed"]=all(version_check.values())

    # Decide P1 closure status after quantification; these do not gate STEP pass but determine wording.
    win_range=float(window_summary.macro_f1.max()-window_summary.macro_f1.min())
    window_stable_diag=bool(window_summary.macro_f1.min()>=primary_val-0.15)
    mvp=plan_summary[plan_summary.plan=="MVP"].iloc[0]
    alt=plan_summary[plan_summary.plan=="alternative"].iloc[0]
    alls=plan_summary[plan_summary.plan=="all_source"].iloc[0]
    source_plan_diag={"mvp_macro_f1":float(mvp.macro_f1),"alternative_macro_f1":float(alt.macro_f1),"all_source_macro_f1":float(alls.macro_f1),
                      "mvp_mmd2":float(mvp.mmd2_biased),"alternative_mmd2":float(alt.mmd2_biased),"all_source_mmd2":float(alls.mmd2_biased)}
    p1_status={
      "P1-01":{"status":"closed_quantified","stable_enough_for_strong_wording":window_stable_diag,"macro_f1_range":win_range},
      "P1-02":{"status":"closed_quantified","source_plan_sensitivity":source_plan_diag},
      "P1-03":{"status":"closed_quantified","loto_agreement_rate":float(loto_summary["label_agreement_rate"]),"changed_files":loto_summary["changed_files"]},
      "P1-04":{"status":"closed_quantified","min_source_bootstrap_label_retention":float(boot_summary.label_retention_rate.min()),
               "files_below_0_8":boot_summary.loc[boot_summary.label_retention_rate<0.8,"target_id"].tolist()}
    }

    # Claim disposition, dynamically grounded in the actual validation.
    disp=[]
    def add(cat,claim,rewrite,evidence):
        disp.append({"category":cat,"claim":claim,"required_wording_or_action":rewrite,"evidence":evidence})
    if p0_closed:
        add("ALLOW","固定H1_RF具有跨载荷条件的独立稳健性证据",
            "可表述为“固定模型在未参与调参的leave-one-load-out压力测试下仍保持可接受文件级性能”；不得把07-A 0.9675称为完全独立终测。",
            f"A2-01 primary pooled Macro-F1={float(primary.macro_f1):.4f}, min-class Recall={float(primary.min_class_recall):.4f}")
    else:
        add("MUST_DELETE","H1_RF泛化可靠/稳健","P0未关闭，必须删除强泛化表述并回滚核心结论。","A2-01 closure failed")
    add("DOWNGRADE","07-A pooled OOF Macro-F1=0.9675是完全独立最终测试成绩",
        "改为“按预冻结开发规则得到的四折配对OOF结果”；独立稳健性引用12-A2 load-held-out结果。","P0-01 audit + A2-01")
    add("DOWNGRADE","1 s、50%重叠窗口是最优参数",
        "只能称冻结主配置；敏感性范围单独报告，不得从12-A2反向重选。",f"A2-02 Macro-F1 range={win_range:.4f}")
    add("DOWNGRADE","MVP源数据方案具有普遍最优性",
        "只能称02-A按无标签可比性规则冻结的主方案，并报告alternative/all-source敏感性。",json.dumps(source_plan_diag,ensure_ascii=False))
    add("DOWNGRADE","A-P预测整体稳定可靠",
        "必须逐文件报告跨设置/LOTO/源Bootstrap稳定性，并继续突出低可信文件。",json.dumps(ap_stab,ensure_ascii=False))
    add("MUST_DELETE","T1迁移显著提高真实目标诊断准确率",
        "删除；伪目标held-load增益为0且A-P真值未知。","09-A held load3 delta=0; target truth unknown")
    add("MUST_DELETE","MMD下降证明A-P分类正确或更准确","删除；MMD仅为域分布接近证据。","09-A wording limit")
    add("MUST_DELETE","A-P模型分数是真实正确概率","删除；所有分数均未校准。","10-A score interpretation")
    add("MUST_DELETE","问题4解释证明了目标故障真值或因果关系","删除；解释仅说明最终模型响应，目标真值未知。","11-A no causal claim / target truth unknown")
    dispositions=pd.DataFrame(disp)

    # Build validation matrix.
    vm=pd.DataFrame([
      ["A2-01","P0-01","Q2 fixed H1_RF leave-one-load-out + 3 seeds","PASS" if p0_closed else "FAIL",
       f"primary Macro-F1={float(primary.macro_f1):.6f}; minRecall={float(primary.min_class_recall):.6f}; seed min={float(a201_pooled.macro_f1.min()):.6f}; seed mean={float(a201_pooled.macro_f1.mean()):.6f}"],
      ["A2-02","P1-01","6 frozen window settings","PASS_COMPLETED",f"Macro-F1 min/max={float(window_summary.macro_f1.min()):.6f}/{float(window_summary.macro_f1.max()):.6f}; primary={primary_val:.6f}"],
      ["A2-03","P1-02","MVP/alternative/all_source source-plan sensitivity","PASS_COMPLETED",json.dumps(source_plan_diag,ensure_ascii=False)],
      ["A2-04","P1-03","leave-one-target-file-out T1 moments","PASS_COMPLETED",f"label agreement={loto_summary['label_agreement_count']}/16; changed={','.join(loto_summary['changed_files']) or 'none'}"],
      ["A2-05","P1-04","120 stratified source-file bootstraps","PASS_COMPLETED",f"min A-P official-label retention={float(boot_summary.label_retention_rate.min()):.6f}"],
      ["A2-06","P2-01","N small-sample derived summary","LIMIT_REMAINS",f"N files=4; load-holdout recall min={n_summary['load_holdout_N_recall_min']:.6f}"],
      ["FINAL-FORMULA","critical","formula-code-unit raw re-extraction","PASS" if formula_audit["passed"] else "FAIL",f"733 windows; 26 features; max abs diff={formula_audit['max_abs_feature_diff']:.3e}"],
      ["FINAL-VERSION","critical","07/09/10/11 identity chain","PASS" if version_check["passed"] else "FAIL",json.dumps(version_check,ensure_ascii=False)],
      ["FINAL-EXPLAIN","critical","11-A faithfulness/stability/sanity identity","PASS" if explanation_check["passed"] else "FAIL",json.dumps(explanation_check,ensure_ascii=False)]
    ],columns=["validation_id","gap_or_scope","validation","status","key_result"])

    critical_ok=bool(p0_closed and formula_audit["passed"] and version_check["passed"] and explanation_check["passed"] and official_repro["passed"])
    passed=critical_ok

    created=nowz()
    binding={"step_id":STEP,"git_commit":git_head(),"run_config_sha256":hf(CFG),"environment_sha256":hf(ENV),
             "source_state_version":state["state_version"],"source_audit_freeze":"FREEZE-12A1-5e3e74e2"}
    bd=canon(binding)
    run_id=f"run_12-A2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs/runs"/run_id; art=out/"artifacts"; mani=out/"manifests";logs=out/"logs"
    art.mkdir(parents=True);mani.mkdir(parents=True);logs.mkdir(parents=True)

    # Deliverables.
    vm.to_csv(art/"final_validation_matrix.csv",index=False,encoding="utf-8-sig")
    a201_folds.to_csv(art/"A2_01_load_holdout_by_seed_and_load.csv",index=False,encoding="utf-8-sig")
    a201_pooled.to_csv(art/"A2_01_load_holdout_seed_summary.csv",index=False,encoding="utf-8-sig")
    a201_files.to_csv(art/"A2_01_load_holdout_file_predictions.csv",index=False,encoding="utf-8-sig")
    window_summary.to_csv(art/"A2_02_window_parameter_sensitivity.csv",index=False,encoding="utf-8-sig")
    pd.concat(window_fold_parts,ignore_index=True).to_csv(art/"A2_02_window_parameter_by_load.csv",index=False,encoding="utf-8-sig")
    plan_summary.to_csv(art/"A2_03_source_plan_sensitivity.csv",index=False,encoding="utf-8-sig")
    pd.concat(plan_fold,ignore_index=True).to_csv(art/"A2_03_source_plan_by_load.csv",index=False,encoding="utf-8-sig")
    loto.to_csv(art/"A2_04_target_LOTO_stability.csv",index=False,encoding="utf-8-sig")
    boot_detail.to_csv(art/"A2_05_source_bootstrap_detail.csv",index=False,encoding="utf-8-sig")
    boot_summary.to_csv(art/"A2_05_source_bootstrap_summary.csv",index=False,encoding="utf-8-sig")
    n_load.to_csv(art/"A2_06_N_load_holdout_summary.csv",index=False,encoding="utf-8-sig")
    n_plan.to_csv(art/"A2_06_N_source_plan_summary.csv",index=False,encoding="utf-8-sig")
    agree.to_csv(art/"AP_existing_9setting_reaudit.csv",index=False,encoding="utf-8-sig")
    dispositions.to_csv(art/"claim_disposition_allow_downgrade_delete.csv",index=False,encoding="utf-8-sig")
    (art/"formula_code_unit_audit.json").write_text(json.dumps(formula_audit,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"result_version_identity_audit.json").write_text(json.dumps(version_check,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"AP_stability_final_audit.json").write_text(json.dumps(ap_stab,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"explanation_final_audit.json").write_text(json.dumps(explanation_check,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"P0_closure_record.json").write_text(json.dumps({
      "P0-01":{"status":"closed" if p0_closed else "open","checks":p0_checks,
               "primary_seed_metrics":{k:(float(primary[k]) if k in primary and isinstance(primary[k],(float,np.floating,int,np.integer)) else primary.get(k)) for k in ["macro_f1","accuracy","min_class_recall","recall_OR","recall_IR","recall_B","recall_N"]},
               "seed_macro_f1":{str(int(r.seed)):float(r.macro_f1) for r in a201_pooled.itertuples(index=False)},
               "closure_basis":"orthogonal leave-one-load-out fixed-model stress, no held-load tuning"}
    },ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"P1_closure_and_residual_risk.json").write_text(json.dumps(p1_status,ensure_ascii=False,indent=2),encoding="utf-8")
    failures=[]
    if not p0_closed: failures.append({"item":"P0-01","status":"failed_to_close","evidence":"A2_01_load_holdout_seed_summary.csv"})
    unstable_window=window_summary.loc[window_summary.macro_f1<primary_val-0.15]
    for r in unstable_window.itertuples(index=False):
        failures.append({"item":"A2-02-window","status":"sensitivity_degradation","setting":r.setting,"macro_f1":float(r.macro_f1),"delta":float(r.delta_macro_f1_vs_primary)})
    for tid in loto.loc[~loto.same_label,"target_id"]:
        failures.append({"item":"A2-04-target-LOTO","status":"label_flip","target_id":tid})
    for r in boot_summary.loc[boot_summary.label_retention_rate<0.8].itertuples(index=False):
        failures.append({"item":"A2-05-source-bootstrap","status":"low_label_retention","target_id":r.target_id,"retention":float(r.label_retention_rate)})
    if not failures: failures=[{"item":"none","status":"no_new_failure_beyond_retained_P2_limits"}]
    pd.DataFrame(failures).to_csv(art/"failure_and_no_gain_results.csv",index=False,encoding="utf-8-sig")

    validation={
      "schema_version":"12A2-validation-1.0","step_id":STEP,"run_id":run_id,"status":"passed" if passed else "failed","passed":passed,
      "checks":{
        "gate_bound_to_12A2":True,"only_12A1_listed_tasks_executed":True,"no_model_search_or_reselection":True,
        "independent_object_file_level_primary":True,"P0_01_closed":p0_closed,
        "independent_key_metric_recalculation":bool(a201_pooled.macro_f1_abs_diff_manual.max()<=1e-12),
        "formula_code_unit_check":bool(formula_audit["passed"]),
        "result_version_identity_check":bool(version_check["passed"]),
        "official_AP_reproduction":bool(official_repro["passed"]),
        "AP_stability_checked":True,"explanation_faithfulness_stability_final_checked":bool(explanation_check["passed"]),
        "unsupported_claim_disposition_created":True,"word_not_edited":True,"formal_paper_figure_not_generated":True
      },
      "evidence":{
        "A2_01":f"outputs/runs/{run_id}/artifacts/A2_01_load_holdout_seed_summary.csv",
        "P0_closure":f"outputs/runs/{run_id}/artifacts/P0_closure_record.json",
        "A2_02":f"outputs/runs/{run_id}/artifacts/A2_02_window_parameter_sensitivity.csv",
        "A2_03":f"outputs/runs/{run_id}/artifacts/A2_03_source_plan_sensitivity.csv",
        "A2_04":f"outputs/runs/{run_id}/artifacts/A2_04_target_LOTO_stability.csv",
        "A2_05":f"outputs/runs/{run_id}/artifacts/A2_05_source_bootstrap_summary.csv",
        "formula_code_unit":f"outputs/runs/{run_id}/artifacts/formula_code_unit_audit.json",
        "claim_disposition":f"outputs/runs/{run_id}/artifacts/claim_disposition_allow_downgrade_delete.csv",
        "key_numbers":{
          "A2_01_primary_macro_f1":float(primary.macro_f1),
          "A2_01_primary_min_class_recall":float(primary.min_class_recall),
          "A2_01_seed_min_macro_f1":float(a201_pooled.macro_f1.min()),
          "A2_02_macro_f1_range":[float(window_summary.macro_f1.min()),float(window_summary.macro_f1.max())],
          "A2_04_loto_agreement_rate":float(loto_summary["label_agreement_rate"]),
          "A2_05_min_label_retention":float(boot_summary.label_retention_rate.min()),
          "formula_reextract_max_abs_diff":float(formula_audit["max_abs_feature_diff"])
        }
      },
      "open_P0_after_step":[] if p0_closed else ["P0-01"],
      "residual_P2":["P2-01 N only four independent files","P2-02 A-P truth unknown","P2-03 target geometry/exact RPM unknown","P2-04 occlusion reference/off-manifold limitation"],
      "word_edit_performed":False
    }
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze={
      "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":STEP,
      "freeze_package_id":f"FREEZE-12A2-{run_id[-8:]}","status":"passed" if passed else "failed","run_id":run_id,
      "versions":{"code_version_id":"git:"+git_head(),"raw_data_version_id":"RAW-5a5dd129c91bfc64",
                  "input_derived_data_ids":["FREEZE-12A1-5e3e74e2","FREEZE-04A-3e3b97c0","FREEZE-07A-7e8829fb","FREEZE-09A-26f99878","FREEZE-10A-4346ce31","FREEZE-11A-1f812c22"],
                  "run_config_sha256":hf(CFG),"environment_sha256":hf(ENV)},
      "random_seed":cfg["fixed_model"]["seed_sensitivity"],
      "parameters":{"fixed_model":cfg["fixed_model"],"tasks":cfg["tasks"],"no_model_selection":True},
      "real_results":[
        {"name":"P0_01_closure","value":{"closed":p0_closed,"checks":p0_checks,"primary_macro_f1":float(primary.macro_f1),"primary_min_class_recall":float(primary.min_class_recall),"seed_min_macro_f1":float(a201_pooled.macro_f1.min())}},
        {"name":"window_sensitivity","value":{"primary_macro_f1":primary_val,"min_macro_f1":float(window_summary.macro_f1.min()),"max_macro_f1":float(window_summary.macro_f1.max()),"range":win_range}},
        {"name":"source_plan_sensitivity","value":source_plan_diag},
        {"name":"target_LOTO","value":loto_summary},
        {"name":"source_bootstrap_AP","value":{"replicates":int(cfg["tasks"]["A2-05"]["total_replicates"]),"min_label_retention":float(boot_summary.label_retention_rate.min()),"files_below_0_8":boot_summary.loc[boot_summary.label_retention_rate<0.8,"target_id"].tolist()}},
        {"name":"formula_code_unit","value":{"passed":formula_audit["passed"],"max_abs_feature_diff":formula_audit["max_abs_feature_diff"],"feature_count":26}},
        {"name":"AP_stability_final","value":ap_stab},
        {"name":"explanation_final_check","value":explanation_check}
      ],
      "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/{x}"} for x in [
        "final_validation_matrix.csv","A2_01_load_holdout_seed_summary.csv","A2_02_window_parameter_sensitivity.csv",
        "A2_03_source_plan_sensitivity.csv","A2_04_target_LOTO_stability.csv","A2_05_source_bootstrap_summary.csv",
        "failure_and_no_gain_results.csv","claim_disposition_allow_downgrade_delete.csv","validation_record.json"]],
      "anomalies_and_failures":failures,
      "b_handoff":{
        "required_data":["final_validation_matrix.csv","failure_and_no_gain_results.csv","claim_disposition_allow_downgrade_delete.csv","P0_closure_record.json","P1_closure_and_residual_risk.json"],
        "supported_conclusions":[
          "12-A2 uses only prelisted 12-A1 gaps and fixed model/data boundaries.",
          "P0-01 is closed only if the preregistered load-held-out thresholds pass.",
          "Window/source-plan/target-LOTO/source-bootstrap results are robustness diagnostics and do not retune the frozen model.",
          "A-P truth remains unknown and all scores remain uncalibrated.",
          "Q4 final explanation evidence remains bound to the actual 09/10 final model."
        ],
        "wording_limits":[
          "07-A 0.9675 pooled OOF is development paired evidence, not a completely untouched final test.",
          "do not call 1s/50% globally optimal",
          "do not call MVP universally optimal",
          "do not claim real-target accuracy or significant migration gain",
          "do not treat MMD or explanations as correctness/causal proof"
        ],
        "approved_tables":["final_validation_matrix.csv","A2_01_load_holdout_seed_summary.csv","A2_02_window_parameter_sensitivity.csv","A2_03_source_plan_sensitivity.csv","A2_04_target_LOTO_stability.csv","A2_05_source_bootstrap_summary.csv","claim_disposition_allow_downgrade_delete.csv"],
        "approved_figures":[]
      },
      "unresolved_issues":[] if passed else ["P0-01 remains open; 12-A2 cannot pass."],
      "freeze":{"created_utc":created,"content_sha256":None,
                "invalidation_dependencies":["Any 12-A1 gap/protocol change","04-A feature formula/data change","07-A model change","09/10 transfer/prediction change","11-A explanation identity change","12-A2 code/config change"]}
    }
    tmp=json.loads(json.dumps(freeze));freeze["freeze"]["content_sha256"]=canon(tmp)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    pdir=ROOT/"protocol/12A2";pdir.mkdir(parents=True,exist_ok=True)
    (pdir/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
    manifest={"schema_version":"12A2-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":STEP,
              "status":"completed" if passed else "failed","created_utc":created,
              "binding":{**binding,"binding_digest":bd},
              "code":{"repository":"Mhhhh958/mathmodeling","commit":git_head(),"entrypoint":"scripts/step12a2_gap_validations.py","code_manifest":"protocol/12A2/code_manifest.json"},
              "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canon(files)},
              "duration_seconds":float(time.time()-t0)}
    (mani/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (logs/"execution_summary.json").write_text(json.dumps({
      "p0_closed":p0_closed,"p0_checks":p0_checks,"formula_audit_passed":formula_audit["passed"],
      "version_check_passed":version_check["passed"],"explanation_check_passed":explanation_check["passed"],
      "word_edited":False,"formal_paper_figures_generated":False
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":passed,"run_id":run_id,"freeze_package_id":freeze["freeze_package_id"],
                      "freeze_content_sha256":freeze["freeze"]["content_sha256"],
                      "p0_closed":p0_closed,"primary_load_holdout_macro_f1":float(primary.macro_f1),
                      "loto_agreement":loto_summary["label_agreement_rate"],
                      "bootstrap_min_retention":float(boot_summary.label_retention_rate.min()),
                      "formula_max_diff":formula_audit["max_abs_feature_diff"]},ensure_ascii=False))
    if not passed: raise SystemExit(2)

if __name__=="__main__":
    main()
