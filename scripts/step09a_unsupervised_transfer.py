#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

import step08a_domain_diagnosis as q3a

ROOT=Path(__file__).resolve().parents[1]
STEP="09-A"
PRIMARY_SEED=20260919
CLASSES=["OR","IR","B","N"]
EPS=1e-12
RUN04="run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0"
RUN07="run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb"
RUN08="run_08-A_20260920T060936354760Z_049c8ddb_8cd7401f"
FREEZE07="FREEZE-07A-7e8829fb"
FREEZE08="FREEZE-08A-8cd7401f"

B04=ROOT/"outputs"/"runs"/RUN04/"artifacts"
B07=ROOT/"outputs"/"runs"/RUN07
B08=ROOT/"outputs"/"runs"/RUN08/"artifacts"
CFG=ROOT/"protocol"/"09A"/"run_config.json"
PROTO=ROOT/"protocol"/"09A"/"experiment_protocol.json"
CODE_MANIFEST=ROOT/"protocol"/"09A"/"code_manifest.json"
ENV_MANIFEST=ROOT/"protocol"/"00B"/"environment_manifest.json"

X_SOURCE=B04/"q2_interface"/"X_source_common.csv"
Y_SOURCE=B04/"q2_interface"/"y_source_labels.csv"
G_SOURCE=B04/"q2_interface"/"groups_source.csv"
M_SOURCE=B04/"q2_interface"/"source_window_metadata.csv"
X_TARGET=B04/"target_interface"/"X_target_common.csv"
M_TARGET=B04/"target_interface"/"target_window_metadata.csv"
FINAL_MODEL=B07/"models"/"final_source_model.joblib"
FEATURE_ORDER=B07/"artifacts"/"feature_order.csv"
STEP08_BASE=B08/"A_P_no_transfer_file_predictions.csv"

def nowz():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def canonical_hash(obj)->str:
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def git_head()->str:
    return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def file_balanced_weights(groups):
    s=pd.Series(np.asarray(groups,dtype=object))
    cnt=s.value_counts().to_dict()
    w=np.array([1.0/cnt[g] for g in s],float)
    return w/w.sum()

def train_weights(df):
    gc=df.groupby("group_id").size().to_dict()
    gt=df[["group_id","class_label"]].drop_duplicates()
    cc=gt.groupby("class_label").size().to_dict()
    w=np.array([(1.0/gc[g])*(1.0/cc[c]) for g,c in df[["group_id","class_label"]].itertuples(index=False)],float)
    return w*(len(w)/w.sum())

def fit_rf(df,features,seed,extra=None):
    w=train_weights(df)
    if extra is not None:
        extra=np.asarray(extra,float)
        if len(extra)!=len(w): raise RuntimeError("extra weight length mismatch")
        w=w*extra
        w=w*(len(w)/w.sum())
    clf=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=1,max_features="sqrt",random_state=int(seed),n_jobs=1)
    clf.fit(df[features].to_numpy(float),df["class_label"].astype(str).to_numpy(),sample_weight=w)
    return clf

def scores_fixed(clf,X):
    raw=clf.predict_proba(np.asarray(X,float))
    pos={str(c):i for i,c in enumerate(clf.classes_)}
    out=np.zeros((len(raw),len(CLASSES)),float)
    for j,c in enumerate(CLASSES):
        if c not in pos: raise RuntimeError(f"missing class {c}")
        out[:,j]=raw[:,pos[c]]
    return out

def aggregate(groups,scores):
    return q3a.aggregate_unlabeled(np.asarray(groups,dtype=object),np.asarray(scores,float))

def metrics(truth,files):
    z=files.merge(truth[["group_id","class_label"]],on="group_id",validate="one_to_one")
    y=z["class_label"].astype(str).to_numpy(); p=z["pred_label"].astype(str).to_numpy()
    rec=recall_score(y,p,labels=CLASSES,average=None,zero_division=0)
    return {
        "macro_f1":float(f1_score(y,p,labels=CLASSES,average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(y,p)),
        "min_class_recall":float(np.min(rec)),
        **{f"recall_{c}":float(v) for c,v in zip(CLASSES,rec)},
        "confusion_matrix":json.dumps(confusion_matrix(y,p,labels=CLASSES).tolist(),ensure_ascii=False),
        "file_count":int(len(z))
    }

def hard_distribution(files):
    vc=files["pred_label"].value_counts().reindex(CLASSES,fill_value=0)
    p=vc.to_numpy(float)
    p=p/p.sum()
    nz=p[p>0]
    ent=float(-(nz*np.log(nz)).sum()/np.log(len(CLASSES))) if len(nz) else 0.0
    return {
        **{f"pred_count_{c}":int(vc[c]) for c in CLASSES},
        "predicted_class_count":int((vc>0).sum()),
        "max_class_share":float(p.max()),
        "hard_class_entropy_norm":ent
    }

def weighted_mean_sd(X,groups):
    w=file_balanced_weights(groups)
    X=np.asarray(X,float)
    mu=(X*w[:,None]).sum(axis=0)
    var=(((X-mu)**2)*w[:,None]).sum(axis=0)
    return mu,np.sqrt(np.maximum(var,EPS))

def weighted_cov(X,groups,mu=None):
    X=np.asarray(X,float); w=file_balanced_weights(groups)
    if mu is None: mu=(X*w[:,None]).sum(axis=0)
    Z=X-mu
    return (Z*w[:,None]).T@Z/max(1.0-w@w,EPS)

def mat_power_sym(C,power):
    vals,vecs=np.linalg.eigh((C+C.T)/2.0)
    vals=np.maximum(vals,EPS)
    return (vecs*(vals**power))@vecs.T

def T1_transform(train,pseudo,features,alpha):
    Xs=train[features].to_numpy(float); Xt=pseudo[features].to_numpy(float)
    mus,sds=weighted_mean_sd(Xs,train["group_id"].to_numpy())
    mut,sdt=weighted_mean_sd(Xt,pseudo["group_id"].to_numpy())
    full=((Xt-mut)/np.where(sdt>EPS,sdt,1.0))*sds+mus
    Xa=(1.0-alpha)*Xt+alpha*full
    return Xa,{"method":"T1-SHRINK-MOMENT","alpha":float(alpha),"source_mean":mus,"source_sd":sds,"target_mean":mut,"target_sd":sdt}

def T2_transform(train,pseudo,features,lam):
    Xs=train[features].to_numpy(float); Xt=pseudo[features].to_numpy(float)
    gs=train["group_id"].to_numpy(); gt=pseudo["group_id"].to_numpy()
    mus,sds=weighted_mean_sd(Xs,gs)
    Zs=(Xs-mus)/np.where(sds>EPS,sds,1.0)
    Zt=(Xt-mus)/np.where(sds>EPS,sds,1.0)
    ws=file_balanced_weights(gs); wt=file_balanced_weights(gt)
    mzs=(Zs*ws[:,None]).sum(axis=0); mzt=(Zt*wt[:,None]).sum(axis=0)
    Cs=weighted_cov(Zs,gs,mzs)+float(lam)*np.eye(len(features))
    Ct=weighted_cov(Zt,gt,mzt)+float(lam)*np.eye(len(features))
    A=mat_power_sym(Ct,-0.5)@mat_power_sym(Cs,0.5)
    Za=(Zt-mzt)@A+mzs
    Xa=Za*sds+mus
    return Xa,{"method":"T2-CORAL","ridge_lambda":float(lam),"source_raw_mean":mus,"source_raw_sd":sds,"source_z_mean":mzs,"target_z_mean":mzt,"linear_map":A}

def domain_ratio(train,pseudo,features,clip,seed):
    Xs=train[features].to_numpy(float); Xt=pseudo[features].to_numpy(float)
    gs=train["group_id"].to_numpy(); gt=pseudo["group_id"].to_numpy()
    mus,sds=weighted_mean_sd(Xs,gs)
    Zs=(Xs-mus)/np.where(sds>EPS,sds,1.0); Zt=(Xt-mus)/np.where(sds>EPS,sds,1.0)
    X=np.vstack([Zs,Zt]); y=np.r_[np.zeros(len(Zs),int),np.ones(len(Zt),int)]
    ws=file_balanced_weights(gs); wt=file_balanced_weights(gt)
    sw=np.r_[0.5*ws,0.5*wt]
    lr=LogisticRegression(C=1.0,max_iter=3000,random_state=int(seed))
    lr.fit(X,y,sample_weight=sw)
    pt=np.clip(lr.predict_proba(Zs)[:,1],1e-5,1-1e-5)
    ratio=pt/(1-pt)
    raw=ratio.copy()
    ratio=np.clip(ratio,1.0/float(clip),float(clip))
    ratio=ratio/np.mean(ratio)
    ess=float((ratio.sum()**2)/(ratio@ratio))
    return ratio,{"method":"T3-DOMAIN-WEIGHT","clip":float(clip),"domain_classifier":lr,"source_mean":mus,"source_sd":sds,
                  "raw_ratio_min":float(raw.min()),"raw_ratio_median":float(np.median(raw)),"raw_ratio_max":float(raw.max()),
                  "ratio_min":float(ratio.min()),"ratio_median":float(np.median(ratio)),"ratio_max":float(ratio.max()),
                  "window_ess":ess,"window_ess_ratio":ess/len(ratio)}

def apply_method(method,setting,train,pseudo,features,seed,base_clf=None):
    if base_clf is None: base_clf=fit_rf(train,features,seed)
    if method=="A0":
        X=pseudo[features].to_numpy(float); clf=base_clf; artifact={"method":"A0"}
    elif method=="T1":
        X,artifact=T1_transform(train,pseudo,features,float(setting)); clf=base_clf
    elif method=="T2":
        X,artifact=T2_transform(train,pseudo,features,float(setting)); clf=base_clf
    elif method=="T3":
        ratio,artifact=domain_ratio(train,pseudo,features,float(setting),seed); clf=fit_rf(train,features,seed,extra=ratio); X=pseudo[features].to_numpy(float)
    else: raise ValueError(method)
    scores=scores_fixed(clf,X)
    _,files=aggregate(pseudo["group_id"].to_numpy(),scores)
    return scores,files,artifact,clf,X

def evaluate_one(load,method,setting,source,features,seed):
    train=source[source["load_hp"]!=float(load)].copy()
    pseudo=source[source["load_hp"]==float(load)].copy()
    base=fit_rf(train,features,seed)
    scores,files,artifact,clf,X=apply_method(method,setting,train,pseudo,features,seed,base)
    truth=pseudo[["group_id","class_label"]].drop_duplicates()
    m=metrics(truth,files); dist=hard_distribution(files)
    row={"load":int(load),"seed":int(seed),"method":method,"setting":str(setting),**m,**dist}
    z=files.merge(truth,on="group_id",validate="one_to_one")
    for c in CLASSES:
        z[f"true_is_{c}"]=(z["class_label"]==c).astype(int)
    z["correct"]=(z["pred_label"]==z["class_label"]).astype(int)
    z["load"]=int(load); z["seed"]=int(seed); z["method"]=method; z["setting"]=str(setting)
    return row,z,artifact,clf,X,train,pseudo

def candidate_summary(metrics_df,protocol):
    tuning=metrics_df[metrics_df["load"].isin([0,1,2])].copy()
    base=tuning[tuning["method"]=="A0"][["load","macro_f1","min_class_recall"]].rename(columns={"macro_f1":"base_macro_f1","min_class_recall":"base_min_recall"})
    rows=[]
    configs=[("T1","0.25")]+[("T2",str(x)) for x in [0.01,0.1,1.0]]+[("T3",str(x)) for x in [2.0,5.0]]
    for method,setting in configs:
        gm=tuning[tuning["method"]==method].copy()
        gm["_setting_num"]=pd.to_numeric(gm["setting"],errors="coerce")
        target_setting=float(setting)
        g=gm[np.isclose(gm["_setting_num"].to_numpy(float),target_setting,rtol=0,atol=1e-12)].drop(columns=["_setting_num"]).merge(base,on="load")
        if len(g)!=3: raise RuntimeError(f"missing tuning rows {method}/{setting}")
        deltas=g["macro_f1"]-g["base_macro_f1"]
        new_zero=False
        for _,r in g.iterrows():
            b0=tuning[(tuning["method"]=="A0")&(tuning["load"]==r["load"])].iloc[0]
            for c in CLASSES:
                if float(b0[f"recall_{c}"])>0 and float(r[f"recall_{c}"])<=0: new_zero=True
        severe=bool((g["max_class_share"]>protocol["selection"]["family_eligibility"]["simulated_max_predicted_class_share_ceiling"]).any())
        rows.append({
            "method":method,"setting":setting,
            "mean_tuning_macro_f1":float(g["macro_f1"].mean()),
            "mean_tuning_min_class_recall":float(g["min_class_recall"].mean()),
            "mean_gain_vs_A0":float(deltas.mean()),
            "worst_tuning_load_delta":float(deltas.min()),
            "new_zero_recall":bool(new_zero),
            "severe_simulated_collapse":severe
        })
    tab=pd.DataFrame(rows)
    e=protocol["selection"]["family_eligibility"]
    tab["eligible"]=(tab["mean_gain_vs_A0"]>=e["min_mean_macro_f1_gain_vs_A0"])&(tab["worst_tuning_load_delta"]>=e["worst_tuning_load_macro_f1_delta_floor"])&(~tab["new_zero_recall"])&(~tab["severe_simulated_collapse"])
    # best config per family
    best=[]
    for method,g in tab.groupby("method"):
        gg=g.copy()
        # metric first; conservative tie-break: T1 single; T2 larger lambda; T3 smaller clip
        if method=="T2":
            gg["tie"]=-gg["setting"].astype(float)
        elif method=="T3":
            gg["tie"]=gg["setting"].astype(float)
        else: gg["tie"]=0.0
        gg=gg.sort_values(["mean_tuning_macro_f1","mean_tuning_min_class_recall","tie"],ascending=[False,False,True])
        best.append(gg.iloc[0])
    best=pd.DataFrame(best)
    complexity={"T1":0,"T2":1,"T3":2}
    best["complexity_rank"]=best["method"].map(complexity)
    eligible=best[best["eligible"]].sort_values(["mean_tuning_macro_f1","mean_tuning_min_class_recall","complexity_rank"],ascending=[False,False,True])
    if len(eligible)==0:
        raise RuntimeError("No transfer family meets pre-frozen tuning eligibility")
    win=eligible.iloc[0]
    return tab,best,str(win["method"]),str(win["setting"])

def paired_deltas(base_files,transfer_files,truth,load):
    b=base_files.merge(truth,on="group_id",validate="one_to_one")
    t=transfer_files.merge(truth,on="group_id",validate="one_to_one")
    keep=["group_id","class_label","pred_label",*[f"score_{c}" for c in CLASSES]]
    b=b[keep].rename(columns={"pred_label":"A0_pred",**{f"score_{c}":f"A0_score_{c}" for c in CLASSES}})
    t=t[keep].rename(columns={"pred_label":"transfer_pred",**{f"score_{c}":f"transfer_score_{c}" for c in CLASSES}})
    z=b.merge(t,on=["group_id","class_label"],validate="one_to_one")
    z["A0_correct"]=(z["A0_pred"]==z["class_label"]).astype(int)
    z["transfer_correct"]=(z["transfer_pred"]==z["class_label"]).astype(int)
    z["paired_correct_delta"]=z["transfer_correct"]-z["A0_correct"]
    z["load"]=int(load)
    true_idx={c:i for i,c in enumerate(CLASSES)}
    def true_score(row,prefix):
        return row[f"{prefix}_score_{row['class_label']}"]
    z["A0_true_class_score"]=z.apply(lambda r:true_score(r,"A0"),axis=1)
    z["transfer_true_class_score"]=z.apply(lambda r:true_score(r,"transfer"),axis=1)
    z["true_class_score_delta"]=z["transfer_true_class_score"]-z["A0_true_class_score"]
    return z

def weighted_mmd2(X,Y,wx=None,wy=None):
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    Z=np.vstack([X,Y])
    d2=np.sum((Z[:,None,:]-Z[None,:,:])**2,axis=2)
    vals=d2[np.triu_indices_from(d2,k=1)]; nz=vals[vals>0]
    med=float(np.median(nz)) if len(nz) else 1.0
    gamma=1.0/max(2.0*med,EPS)
    Kxx=np.exp(-gamma*np.sum((X[:,None,:]-X[None,:,:])**2,axis=2))
    Kyy=np.exp(-gamma*np.sum((Y[:,None,:]-Y[None,:,:])**2,axis=2))
    Kxy=np.exp(-gamma*np.sum((X[:,None,:]-Y[None,:,:])**2,axis=2))
    if wx is None: wx=np.ones(len(X))/len(X)
    else: wx=np.asarray(wx,float);wx=wx/wx.sum()
    if wy is None: wy=np.ones(len(Y))/len(Y)
    else: wy=np.asarray(wy,float);wy=wy/wy.sum()
    return float(wx@Kxx@wx+wy@Kyy@wy-2.0*wx@Kxy@wy)

def file_means(X,groups):
    d=pd.DataFrame(np.asarray(X,float))
    d["group_id"]=np.asarray(groups,dtype=object)
    return d.groupby("group_id",sort=True).mean()

def target_table(target,files):
    z=files.copy()
    idmap=target[["group_id","relative_path"]].drop_duplicates()
    idmap["target_id"]=idmap["relative_path"].map(lambda x:Path(str(x)).stem)
    z=z.merge(idmap[["group_id","target_id"]],on="group_id",validate="one_to_one")
    z["truth_status"]="unknown"
    return z[["target_id","group_id","truth_status","window_count",*[f"score_{c}" for c in CLASSES],"pred_label","top_model_score","score_margin","score_entropy_norm","window_file_agreement"]].sort_values("target_id")

def main():
    t0=time.time()
    cfg=json.loads(CFG.read_text(encoding="utf-8"))
    protocol=json.loads(PROTO.read_text(encoding="utf-8"))
    if cfg["inputs"]["freeze07"]!=FREEZE07 or cfg["inputs"]["freeze08"]!=FREEZE08: raise RuntimeError("freeze binding mismatch")
    features=pd.read_csv(FEATURE_ORDER)["feature_name"].astype(str).tolist()
    if len(features)!=26: raise RuntimeError("expected 26 features")
    Xs=pd.read_csv(X_SOURCE); ys=pd.read_csv(Y_SOURCE); gs=pd.read_csv(G_SOURCE); ms=pd.read_csv(M_SOURCE)
    Xt=pd.read_csv(X_TARGET); mt=pd.read_csv(M_TARGET)
    if len(Xs)!=733 or len(Xt)!=240 or ms["group_id"].nunique()!=49 or mt["group_id"].nunique()!=16: raise RuntimeError("inventory changed")
    if set(mt["class_label"].astype(str))!={"UNKNOWN_TRUTH"} or set(mt["label_status"].astype(str))!={"unknown_truth"}: raise RuntimeError("A-P truth boundary violated")
    source=Xs.merge(ys,on="window_id",validate="one_to_one").merge(gs,on="window_id",validate="one_to_one")
    source=source.merge(ms[["window_id","group_id","load_hp","relative_path"]],on=["window_id","group_id"],validate="one_to_one")
    target=Xt.merge(mt[["window_id","group_id","relative_path","class_label","label_status"]],on="window_id",validate="one_to_one")
    frozen=joblib.load(FINAL_MODEL)
    if frozen.get("family")!="RandomForestClassifier" or list(frozen.get("feature_names",[]))!=features or list(frozen.get("class_order",[]))!=CLASSES: raise RuntimeError("07-A model interface changed")
    frozen_clf=frozen["classifier"]

    bind={"step_id":STEP,"code_commit":git_head(),"freeze07":FREEZE07,"freeze08":FREEZE08,"config_sha256":sha256_file(CFG),"protocol_sha256":sha256_file(PROTO),"environment_sha256":sha256_file(ENV_MANIFEST)}
    dig=canonical_hash(bind)
    run_id=f"run_09-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{dig[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; models=out/"models"; mani=out/"manifests"; logs=out/"logs"
    for d in [art,models,mani,logs]: d.mkdir(parents=True,exist_ok=False)
    elog=[]
    def log(event,**kw):
        rec={"utc":nowz(),"event":event,**kw};elog.append(rec)

    # Exact STEP08 A0 reproduction on real A-P.
    base_scores=scores_fixed(frozen_clf,target[features].to_numpy(float))
    _,base_files=aggregate(target["group_id"].to_numpy(),base_scores)
    base_tab=target_table(target,base_files)
    base_tab.to_csv(art/"A_P_A0_reproduced.csv",index=False,encoding="utf-8-sig")
    frozen08=pd.read_csv(STEP08_BASE).sort_values("target_id").reset_index(drop=True)
    chk=base_tab.sort_values("target_id").reset_index(drop=True)
    score_cols=[f"score_{c}" for c in CLASSES]
    max_score_diff=float(np.max(np.abs(chk[score_cols].to_numpy(float)-frozen08[score_cols].to_numpy(float))))
    if list(chk["pred_label"])!=list(frozen08["pred_label"]) or max_score_diff>1e-12: raise RuntimeError("STEP08 no-transfer reproduction mismatch")
    log("step08_baseline_reproduced",max_score_diff=max_score_diff)

    # Primary-seed tuning on loads 0/1/2 only.
    rows=[]; pred_frames=[]; artifacts={}
    methods=[("A0","0"),("T1","0.25")]+[("T2",str(x)) for x in cfg["methods"]["T2_ridge_lambda"]]+[("T3",str(x)) for x in cfg["methods"]["T3_density_ratio_clip"]]
    for load in [0,1,2]:
        for method,setting in methods:
            row,z,a,clf,X,train,pseudo=evaluate_one(load,method,setting,source,features,PRIMARY_SEED)
            rows.append(row); pred_frames.append(z); artifacts[f"tune_load{load}_{method}_{setting}"]=a
    tuning_metrics=pd.DataFrame(rows)
    tuning_metrics.to_csv(art/"tuning_metrics_loads0_1_2.csv",index=False,encoding="utf-8-sig")
    pd.concat(pred_frames,ignore_index=True).to_csv(art/"tuning_file_predictions.csv",index=False,encoding="utf-8-sig")
    detail,best,selected_method,selected_setting=candidate_summary(tuning_metrics,protocol)
    detail.to_csv(art/"candidate_config_summary.csv",index=False,encoding="utf-8-sig")
    best.to_csv(art/"candidate_family_summary.csv",index=False,encoding="utf-8-sig")
    (art/"pre_held_selection.json").write_text(json.dumps({"selected_method":selected_method,"selected_setting":selected_setting,"selection_scope":"loads0/1/2 primary seed only","A_P_prediction_appearance_used":False},ensure_ascii=False,indent=2),encoding="utf-8")
    log("family_frozen_before_load3",method=selected_method,setting=selected_setting)

    # Reveal held load3 only after family/config frozen.
    b3,b3f,_,_,_,_,p3=evaluate_one(3,"A0","0",source,features,PRIMARY_SEED)
    t3,t3f,a3,clf3,X3,train3,pseudo3=evaluate_one(3,selected_method,selected_setting,source,features,PRIMARY_SEED)
    held=pd.DataFrame([b3,t3])
    held.to_csv(art/"held_load3_primary_metrics.csv",index=False,encoding="utf-8-sig")
    truth3=pseudo3[["group_id","class_label"]].drop_duplicates()
    pair3=paired_deltas(b3f,t3f,truth3,3)
    pair3.to_csv(art/"held_load3_paired_file_deltas.csv",index=False,encoding="utf-8-sig")
    f=protocol["selection"]["final_load3_acceptance"]
    new_zero=any(b3[f"recall_{c}"]>0 and t3[f"recall_{c}"]<=0 for c in CLASSES)
    held_accept=(t3["macro_f1"]-b3["macro_f1"]>=f["macro_f1_delta_floor"] and t3["min_class_recall"]-b3["min_class_recall"]>=f["min_class_recall_delta_floor"] and not new_zero and t3["max_class_share"]<=f["simulated_max_predicted_class_share_ceiling"])
    held_decision={"selected_method":selected_method,"selected_setting":selected_setting,"A0_macro_f1":b3["macro_f1"],"transfer_macro_f1":t3["macro_f1"],"macro_delta":t3["macro_f1"]-b3["macro_f1"],"A0_min_recall":b3["min_class_recall"],"transfer_min_recall":t3["min_class_recall"],"min_recall_delta":t3["min_class_recall"]-b3["min_class_recall"],"new_zero_recall":new_zero,"max_class_share":t3["max_class_share"],"passed":bool(held_accept)}
    (art/"held_load3_acceptance.json").write_text(json.dumps(held_decision,ensure_ascii=False,indent=2),encoding="utf-8")
    log("held_load3_revealed",**held_decision)

    # Stability after setting frozen: all 4 loads, three seeds, selected method only, paired to A0.
    stab_rows=[]; paired_all=[]
    pred_by_seed={}
    for seed in cfg["stability_seeds"]:
        for load in [0,1,2,3]:
            br,bf,_,_,_,_,pseudo=evaluate_one(load,"A0","0",source,features,seed)
            tr,tf,_,_,_,_,_=evaluate_one(load,selected_method,selected_setting,source,features,seed)
            stab_rows.append({"seed":seed,"load":load,"A0_macro_f1":br["macro_f1"],"transfer_macro_f1":tr["macro_f1"],"macro_delta":tr["macro_f1"]-br["macro_f1"],"A0_min_recall":br["min_class_recall"],"transfer_min_recall":tr["min_class_recall"],"max_class_share":tr["max_class_share"]})
            truth=pseudo[["group_id","class_label"]].drop_duplicates()
            p=paired_deltas(bf,tf,truth,load);p["seed"]=seed;paired_all.append(p)
            if load==3:
                pred_by_seed[seed]=tf[["group_id","pred_label"]].rename(columns={"pred_label":f"pred_{seed}"})
    stab=pd.DataFrame(stab_rows)
    stab.to_csv(art/"selected_method_seed_stability_metrics.csv",index=False,encoding="utf-8-sig")
    paired=pd.concat(paired_all,ignore_index=True)
    paired.to_csv(art/"selected_method_paired_file_deltas_all_seeds.csv",index=False,encoding="utf-8-sig")
    pm=None
    for seed,t in pred_by_seed.items():
        pm=t if pm is None else pm.merge(t,on="group_id",validate="one_to_one")
    predcols=[f"pred_{s}" for s in cfg["stability_seeds"]]
    pm["all_seed_agree"]=pm[predcols].nunique(axis=1)==1
    label_agree=float(pm["all_seed_agree"].mean())
    held_stab=stab[stab["load"]==3]
    seed_rule=protocol["selection"]["seed_stability_after_freeze"]
    stability_pass=(float(held_stab["macro_delta"].min())>=seed_rule["held_load3_worst_macro_delta_floor"] and label_agree>=seed_rule["all_seed_label_agreement_floor"])
    stability_summary={"seeds":cfg["stability_seeds"],"held_load3_macro_delta_by_seed":{str(int(r.seed)):float(r.macro_delta) for r in held_stab.itertuples()},"held_load3_worst_macro_delta":float(held_stab["macro_delta"].min()),"held_load3_all_seed_file_label_agreement":label_agree,"passed":bool(stability_pass)}
    (art/"stability_summary.json").write_text(json.dumps(stability_summary,ensure_ascii=False,indent=2),encoding="utf-8")

    # Pseudo-label threshold sensitivity is explicitly N/A because no method uses pseudo labels.
    pseudo_sens={"pseudo_labels_used":False,"threshold_sensitivity_applicable":False,"reason":protocol["pseudo_label_policy"]["reason"],"error_accumulation_control":"avoid pseudo-label self-training entirely; selection uses only source truth in simulated targets after prediction"}
    (art/"pseudo_label_threshold_sensitivity.json").write_text(json.dumps(pseudo_sens,ensure_ascii=False,indent=2),encoding="utf-8")

    # Source retention. T1/T2 leave the frozen source classifier untouched. T3 gets an actual 4-fold source OOF safety check using unlabeled A-P domain weights.
    retention={}
    if selected_method in ["T1","T2"]:
        src_scores_before=scores_fixed(frozen_clf,source[features].to_numpy(float))
        src_scores_after=scores_fixed(frozen_clf,source[features].to_numpy(float))
        retention={"mode":"exact_classifier_preservation","source_model_sha256":sha256_file(FINAL_MODEL),"max_source_window_score_delta":float(np.max(np.abs(src_scores_before-src_scores_after))),"macro_f1_delta_by_construction":0.0,"passed":True}
    else:
        outer=pd.read_csv(cfg["inputs"]["outer_splits"])
        rrows=[]
        for fold in sorted(outer["outer_fold"].unique()):
            man=outer[outer["outer_fold"]==fold]
            train_groups=set(man.loc[man["role"]=="train_pool","group_id"].astype(str)); test_groups=set(man.loc[man["role"]=="test","group_id"].astype(str))
            tr=source[source["group_id"].isin(train_groups)].copy(); te=source[source["group_id"].isin(test_groups)].copy()
            bclf=fit_rf(tr,features,PRIMARY_SEED)
            bs=scores_fixed(bclf,te[features].to_numpy(float)); _,bf=aggregate(te["group_id"].to_numpy(),bs)
            bm=metrics(te[["group_id","class_label"]].drop_duplicates(),bf)
            ratio,_=domain_ratio(tr,target,features,float(selected_setting),PRIMARY_SEED)
            aclf=fit_rf(tr,features,PRIMARY_SEED,extra=ratio)
            ass=scores_fixed(aclf,te[features].to_numpy(float)); _,af=aggregate(te["group_id"].to_numpy(),ass)
            am=metrics(te[["group_id","class_label"]].drop_duplicates(),af)
            rr={"fold":int(fold),"A0_macro_f1":bm["macro_f1"],"transfer_macro_f1":am["macro_f1"],"macro_delta":am["macro_f1"]-bm["macro_f1"]}
            for c in CLASSES: rr[f"recall_delta_{c}"]=am[f"recall_{c}"]-bm[f"recall_{c}"]
            rrows.append(rr)
        rdf=pd.DataFrame(rrows);rdf.to_csv(art/"source_retention_outer_folds.csv",index=False,encoding="utf-8-sig")
        srule=protocol["source_retention"]
        worst_recall=float(rdf[[f"recall_delta_{c}" for c in CLASSES]].min().min())
        retention={"mode":"T3_weighted_RF_4fold_source_OOF","mean_macro_f1_delta":float(rdf["macro_delta"].mean()),"worst_fold_macro_f1_delta":float(rdf["macro_delta"].min()),"worst_class_recall_delta":worst_recall,"passed":bool(float(rdf["macro_delta"].mean())>=srule["macro_f1_drop_floor"] and worst_recall>=srule["no_class_recall_drop_below"])}
    (art/"source_retention_check.json").write_text(json.dumps(retention,ensure_ascii=False,indent=2),encoding="utf-8")

    pre_real_pass=bool(held_accept and stability_pass and retention["passed"])

    # Final A-P adaptation only after frozen simulated checks pass.
    final_scores=None; final_files=None; final_artifact=None; final_clf=None; final_X=None
    if pre_real_pass:
        final_scores,final_files,final_artifact,final_clf,final_X=apply_method(selected_method,selected_setting,source,target,features,PRIMARY_SEED,frozen_clf if selected_method in ["T1","T2"] else None)
        final_tab=target_table(target,final_files)
        final_tab.to_csv(art/"A_P_final_transfer_file_predictions.csv",index=False,encoding="utf-8-sig")
        # window output
        w=target[["window_id","group_id","relative_path"]].copy()
        for j,c in enumerate(CLASSES): w[f"score_{c}"]=final_scores[:,j]
        w["pred_label"]=np.array(CLASSES,dtype=object)[np.argmax(final_scores,axis=1)]
        w["truth_status"]="unknown"
        w.to_csv(art/"A_P_final_transfer_window_predictions.csv",index=False,encoding="utf-8-sig")

        # pre/post comparison (descriptive, not used to select).
        cmp=base_tab[["target_id","pred_label","top_model_score","score_margin","score_entropy_norm"]].rename(columns={"pred_label":"A0_pred","top_model_score":"A0_top_score","score_margin":"A0_margin","score_entropy_norm":"A0_score_entropy"}).merge(
            final_tab[["target_id","pred_label","top_model_score","score_margin","score_entropy_norm"]].rename(columns={"pred_label":"transfer_pred","top_model_score":"transfer_top_score","score_margin":"transfer_margin","score_entropy_norm":"transfer_score_entropy"}),on="target_id",validate="one_to_one")
        cmp["label_changed"]=cmp["A0_pred"]!=cmp["transfer_pred"]
        cmp.to_csv(art/"A_P_before_after_comparison.csv",index=False,encoding="utf-8-sig")

        # domain discrepancy on independent-file means in source-standardized raw 26D space.
        mus,sds=weighted_mean_sd(source[features].to_numpy(float),source["group_id"].to_numpy())
        Zs=(source[features].to_numpy(float)-mus)/np.where(sds>EPS,sds,1.0)
        Zt=(target[features].to_numpy(float)-mus)/np.where(sds>EPS,sds,1.0)
        sf=file_means(Zs,source["group_id"].to_numpy())
        tf=file_means(Zt,target["group_id"].to_numpy())
        mmd_before=weighted_mmd2(sf.to_numpy(),tf.to_numpy())
        if selected_method in ["T1","T2"]:
            Za=(final_X-mus)/np.where(sds>EPS,sds,1.0)
            tfa=file_means(Za,target["group_id"].to_numpy())
            mmd_after=weighted_mmd2(sf.to_numpy(),tfa.to_numpy())
            mmd_note="source vs adapted target"
        else:
            # Recompute file-mean source weights from final domain-ratio fit.
            ratio,_=domain_ratio(source,target,features,float(selected_setting),PRIMARY_SEED)
            tmp=pd.DataFrame({"group_id":source["group_id"].astype(str),"ratio":ratio})
            fw=tmp.groupby("group_id")["ratio"].mean().reindex(sf.index).to_numpy(float)
            mmd_after=weighted_mmd2(sf.to_numpy(),tf.to_numpy(),wx=fw)
            mmd_note="domain-weighted source vs target"
        gap={"mmd2_before":mmd_before,"mmd2_after":mmd_after,"decreased":bool(mmd_after<mmd_before),"comparison":mmd_note,"warning":"distribution proximity is not diagnostic accuracy"}
        (art/"A_P_domain_discrepancy_before_after.json").write_text(json.dumps(gap,ensure_ascii=False,indent=2),encoding="utf-8")

        dist=hard_distribution(final_files)
        safety=protocol["selection"]["A_P_safety_only_not_selection"]
        collapse={"A0":hard_distribution(base_files),"transfer":dist,"severe_threshold":safety["severe_collapse_max_class_share"],"severe_collapse":bool(dist["max_class_share"]>=safety["severe_collapse_max_class_share"]),"selection_used_this":False}
        (art/"A_P_class_distribution_collapse_check.json").write_text(json.dumps(collapse,ensure_ascii=False,indent=2),encoding="utf-8")

        # Final bundle.
        bundle={"schema_version":"09A-transfer-bundle-1.0","method":selected_method,"setting":selected_setting,"source_model_sha256":sha256_file(FINAL_MODEL),"feature_names":features,"class_order":CLASSES,"target_truth":"unknown","pseudo_labels_used":False,"selection_scope":"simulated loads0/1/2 only; load3 final validation","classifier":final_clf,"adaptation_artifact":final_artifact}
        joblib.dump(bundle,models/"final_transfer_bundle.joblib")
        final_real_pass=not collapse["severe_collapse"]
    else:
        gap=None; collapse=None; final_real_pass=False

    # Per-load primary-seed paired deltas for selected method, including tuning and held load.
    prim_pairs=[]
    for load in [0,1,2,3]:
        br,bf,_,_,_,_,pseudo=evaluate_one(load,"A0","0",source,features,PRIMARY_SEED)
        tr,tf,_,_,_,_,_=evaluate_one(load,selected_method,selected_setting,source,features,PRIMARY_SEED)
        pp=paired_deltas(bf,tf,pseudo[["group_id","class_label"]].drop_duplicates(),load)
        prim_pairs.append(pp)
    ppall=pd.concat(prim_pairs,ignore_index=True)
    ppall.to_csv(art/"primary_seed_paired_file_negative_transfer.csv",index=False,encoding="utf-8-sig")
    neg_summary=ppall.groupby("load").agg(files=("group_id","count"),improved_files=("paired_correct_delta",lambda s:int((s>0).sum())),degraded_files=("paired_correct_delta",lambda s:int((s<0).sum())),unchanged_files=("paired_correct_delta",lambda s:int((s==0).sum())),mean_correct_delta=("paired_correct_delta","mean"),mean_true_score_delta=("true_class_score_delta","mean")).reset_index()
    neg_summary.to_csv(art/"primary_seed_negative_transfer_summary_by_load.csv",index=False,encoding="utf-8-sig")

    checks={
        "gate_inputs_bound":True,
        "A_P_true_labels_unavailable":True,
        "step08_A0_exactly_reproduced":max_score_diff<=1e-12,
        "T1_T2_T3_tuning_runs_complete":set(tuning_metrics["method"])=={"A0","T1","T2","T3"},
        "selection_uses_only_loads0_1_2":True,
        "held_load3_not_used_until_method_frozen":True,
        "held_load3_acceptance_passed":bool(held_accept),
        "pseudo_labels_not_used":True,
        "pseudo_label_threshold_sensitivity_recorded_as_not_applicable":True,
        "three_seed_stability_checked":True,
        "stability_passed":bool(stability_pass),
        "paired_negative_transfer_quantified_per_simulated_file":len(ppall)==49,
        "source_retention_checked":True,
        "source_retention_passed":bool(retention["passed"]),
        "real_A_P_adaptation_executed":bool(pre_real_pass),
        "real_A_P_accuracy_not_computed":True,
        "class_distribution_entropy_and_collapse_checked":collapse is not None,
        "no_unexplained_real_A_P_severe_collapse":bool(final_real_pass),
        "final_transfer_bundle_saved":(models/"final_transfer_bundle.joblib").exists()
    }
    passed=all(checks.values())
    unresolved=[
        "A-P ground truth remains unknown; no real target accuracy/F1/Recall can be verified.",
        "Pseudo-target load shift cannot reproduce all real sensor/RPM/sampling-path differences.",
        "A lower MMD after adaptation is not proof of better diagnosis."
    ]
    decision={"selected_method":selected_method,"selected_setting":selected_setting,"pre_held_selection_evidence":"loads0/1/2 only","held_load3":held_decision,"stability":stability_summary,"source_retention":retention,"A_P_output_safety":collapse,"A_P_truth":"unknown","pseudo_labels_used":False,"passed":passed}
    (art/"final_transfer_selection.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"unverified_items.json").write_text(json.dumps({"items":unresolved},ensure_ascii=False,indent=2),encoding="utf-8")
    validation={"schema_version":"09A-validation-1.0","step_id":STEP,"run_id":run_id,"status":"passed" if passed else "failed","passed":passed,"checks":checks,"key_evidence":{"selection":f"outputs/runs/{run_id}/artifacts/candidate_family_summary.csv","held":f"outputs/runs/{run_id}/artifacts/held_load3_acceptance.json","paired_negative_transfer":f"outputs/runs/{run_id}/artifacts/primary_seed_paired_file_negative_transfer.csv","stability":f"outputs/runs/{run_id}/artifacts/stability_summary.json","source_retention":f"outputs/runs/{run_id}/artifacts/source_retention_check.json","A_P_final":f"outputs/runs/{run_id}/artifacts/A_P_final_transfer_file_predictions.csv" if pre_real_pass else None},"target_accuracy_reported":False,"word_edit_performed":False}
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (logs/"experiment_log.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in elog)+"\n",encoding="utf-8")

    # Runtime provenance.
    pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True);(mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
    env={"captured_utc":nowz(),"platform":platform.platform(),"python_version":platform.python_version(),"machine":platform.machine(),"cpu_count_logical":os.cpu_count(),"gpu":None,"cuda":None,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__,"git_commit":git_head(),"pip_freeze_sha256":sha256_file(mani/"pip_freeze.txt")}
    (mani/"environment_runtime.json").write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf-8")

    real_results=[
        {"name":"method_selection","value":{"selected_method":selected_method,"selected_setting":selected_setting}},
        {"name":"held_load3","value":held_decision},
        {"name":"seed_stability","value":stability_summary},
        {"name":"source_retention","value":retention},
        {"name":"paired_negative_transfer","value":{"degraded_files_total":int((ppall["paired_correct_delta"]<0).sum()),"improved_files_total":int((ppall["paired_correct_delta"]>0).sum()),"unchanged_files_total":int((ppall["paired_correct_delta"]==0).sum())}},
        {"name":"A_P_output","value":{"truth":"unknown","distribution":collapse["transfer"] if collapse else None,"domain_gap":gap}}
    ]
    freeze={"schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":STEP,"freeze_package_id":f"FREEZE-09A-{run_id[-8:]}","status":"passed" if passed else "failed","run_id":run_id,
            "versions":{"code_version_id":"git:"+git_head(),"raw_data_version_id":"RAW-5a5dd129c91bfc64","input_derived_data_ids":[FREEZE07,FREEZE08,"FEAT-23d45f5649dcd5f1"],"run_config_sha256":sha256_file(CFG),"environment_sha256":sha256_file(ENV_MANIFEST)},
            "random_seed":PRIMARY_SEED,"parameters":{"features":26,"class_order":CLASSES,"T1_alpha":0.25,"T2_lambda":cfg["methods"]["T2_ridge_lambda"],"T3_clip":cfg["methods"]["T3_density_ratio_clip"],"tuning_loads":[0,1,2],"final_validation_load":3,"stability_seeds":cfg["stability_seeds"],"pseudo_labels_used":False},
            "real_results":real_results,
            "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/candidate_config_summary.csv"},{"path":f"outputs/runs/{run_id}/artifacts/held_load3_acceptance.json"},{"path":f"outputs/runs/{run_id}/artifacts/selected_method_paired_file_deltas_all_seeds.csv"},{"path":f"outputs/runs/{run_id}/artifacts/source_retention_check.json"},{"path":f"outputs/runs/{run_id}/artifacts/A_P_class_distribution_collapse_check.json"},{"path":f"outputs/runs/{run_id}/artifacts/pseudo_label_threshold_sensitivity.json"}],
            "anomalies_and_failures":[{"item":"pseudo_labeling","status":"not_used_by_design","reason":protocol["pseudo_label_policy"]["reason"]}]+([{"item":"final_selection","status":"failed_safety_or_validation"}] if not passed else []),
            "b_handoff":{"required_data":["final_transfer_selection.json","candidate_config_summary.csv","candidate_family_summary.csv","held_load3_acceptance.json","primary_seed_negative_transfer_summary_by_load.csv","stability_summary.json","source_retention_check.json","A_P_A0_reproduced.csv","A_P_final_transfer_file_predictions.csv","A_P_before_after_comparison.csv","A_P_domain_discrepancy_before_after.json","A_P_class_distribution_collapse_check.json","unverified_items.json"],
                         "supported_conclusions":["Method/config selection used only simulated target loads0/1/2.","Held load3 was evaluated only after method/config freeze.","A-P labels remained unknown; final A-P outputs are predictions only.","Pseudo-label self-training was not used.","Distribution-gap reduction, if observed, is not claimed as target diagnostic accuracy."],
                         "wording_limits":["do not report A-P accuracy/F1/Recall","do not call uncalibrated scores confidence/probability","do not claim MMD reduction proves accuracy improvement","do not claim pseudo-target results equal real target performance"],
                         "approved_tables":["candidate_family_summary.csv","held_load3_primary_metrics.csv","primary_seed_negative_transfer_summary_by_load.csv","A_P_before_after_comparison.csv"],
                         "approved_figures":[]},
            "unresolved_issues":unresolved,
            "freeze":{"created_utc":nowz(),"content_sha256":None,"invalidation_dependencies":["07-A final source model changes","08-A frozen domain/pseudo-target protocol changes","04-A common feature data changes","09-A config/protocol/code changes"]}}
    f0=json.loads(json.dumps(freeze,ensure_ascii=False));freeze["freeze"]["content_sha256"]=canonical_hash(f0)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"09A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"09A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"09A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    rm={"schema_version":"09A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":STEP,"status":"completed" if passed else "failed","created_utc":nowz(),"binding":{**bind,"binding_digest":dig},
        "code":{"repository":"Mhhhh958/mathmodeling","commit":git_head(),"entrypoint":"scripts/step09a_unsupervised_transfer.py","code_manifest":"protocol/09A/code_manifest.json"},
        "data":{"source_files":49,"source_windows":733,"target_files":16,"target_windows":240,"features":26,"target_truth":"unknown"},
        "config":{"path":"protocol/09A/run_config.json","sha256":sha256_file(CFG)},"protocol":{"path":"protocol/09A/experiment_protocol.json","sha256":sha256_file(PROTO)},
        "environment":{"manifest":"protocol/00B/environment_manifest.json","sha256":sha256_file(ENV_MANIFEST),"runtime":"manifests/environment_runtime.json"},
        "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canonical_hash(files)},"duration_seconds":time.time()-t0}
    (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")

    print(json.dumps({"ok":passed,"run_id":run_id,"freeze_package_id":freeze["freeze_package_id"],"freeze_content_sha256":freeze["freeze"]["content_sha256"],"selected_method":selected_method,"selected_setting":selected_setting,"held_load3_macro_delta":held_decision["macro_delta"],"seed_stability_passed":stability_pass,"source_retention_passed":retention["passed"],"A_P_severe_collapse":collapse["severe_collapse"] if collapse else None,"target_accuracy_reported":False},ensure_ascii=False))
    if not passed: raise SystemExit(2)

if __name__=="__main__":
    main()
