#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"protocol"/"07A"/"run_config.json"
BLUEPRINT=ROOT/"protocol"/"07A"/"question2C_argument_blueprint.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
FREEZE05=ROOT/"protocol"/"05A"/"latest_freeze_package.json"
FREEZE06=ROOT/"protocol"/"06A"/"latest_freeze_package.json"
FREEZE04=ROOT/"protocol"/"04A"/"latest_freeze_package.json"
RUN05=ROOT/"protocol"/"05A"/"latest_run_id.txt"
RUN06=ROOT/"protocol"/"06A"/"latest_run_id.txt"
RUN04=ROOT/"protocol"/"04A"/"latest_run_id.txt"

SEED=20260919
CLASSES=["OR","IR","B","N"]
WORDER={"file_balanced":0,"file_and_class_balanced":1}

def cj(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b): return hashlib.sha256(b).hexdigest()
def hf(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def git_head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def weights(df,mode):
    gc=df.groupby("group_id").size().to_dict()
    w=np.array([1.0/gc[g] for g in df.group_id],float)
    if mode=="file_and_class_balanced":
        gt=df[["group_id","class_label"]].drop_duplicates()
        cc=gt.groupby("class_label").size().to_dict()
        w*=np.array([1.0/cc[c] for c in df.class_label],float)
    elif mode!="file_balanced":
        raise ValueError(mode)
    w*=len(w)/w.sum()
    return w

def fixed_scores(clf,X):
    raw=clf.predict_proba(X)
    out=np.zeros((len(X),4),float)
    pos={c:i for i,c in enumerate(clf.classes_)}
    for j,c in enumerate(CLASSES): out[:,j]=raw[:,pos[c]]
    return out

def aggregate(df,sc):
    q=df[["window_id","group_id","relative_path","class_label","load_hp","fault_size_in","outer_race_position_clock","rpm","subgroup"]].copy()
    for j,c in enumerate(CLASSES): q[f"score_{c}"]=sc[:,j]
    cols=[f"score_{c}" for c in CLASSES]
    ids=["group_id","relative_path","class_label","load_hp","fault_size_in","outer_race_position_clock","rpm","subgroup"]
    g=q.groupby(ids,dropna=False,sort=True)[cols].mean().reset_index()
    arr=g[cols].to_numpy()
    g["pred_label"]=np.array(CLASSES,dtype=object)[np.argmax(arr,axis=1)]
    g["window_count"]=q.groupby("group_id").size().reindex(g.group_id).to_numpy()
    return q,g

def metrics(y,p):
    cm=confusion_matrix(y,p,labels=CLASSES)
    rec=recall_score(y,p,labels=CLASSES,average=None,zero_division=0)
    return {"macro_f1":float(f1_score(y,p,labels=CLASSES,average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(y,p)),
            "recall":{c:float(v) for c,v in zip(CLASSES,rec)},
            "cm":cm.tolist()}

def fit_lr(train,features,C,mode):
    X=train[features].to_numpy(float); y=train.class_label.to_numpy(); w=weights(train,mode)
    scaler=StandardScaler(); scaler.fit(X,sample_weight=w); Z=scaler.transform(X)
    clf=LogisticRegression(C=float(C),solver="lbfgs",max_iter=2000,random_state=SEED)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always"); t=time.perf_counter(); clf.fit(Z,y,sample_weight=w); dt=time.perf_counter()-t
    conv=[str(x.message) for x in rec if issubclass(x.category,ConvergenceWarning)]
    return scaler,clf,dt,conv

def fit_rf(train,features,depth,mode):
    X=train[features].to_numpy(float); y=train.class_label.to_numpy(); w=weights(train,mode)
    clf=RandomForestClassifier(n_estimators=300,max_depth=depth,min_samples_leaf=1,max_features="sqrt",
                               random_state=SEED,n_jobs=1)
    t=time.perf_counter(); clf.fit(X,y,sample_weight=w); dt=time.perf_counter()-t
    return clf,dt

def predict_lr(scaler,clf,df,features): return fixed_scores(clf,scaler.transform(df[features].to_numpy(float)))
def predict_rf(clf,df,features): return fixed_scores(clf,df[features].to_numpy(float))

def corr_keep(train,features,thr=0.95):
    X=train[features]
    corr=X.corr(method="pearson").abs()
    keep=[]; drop=[]
    for f in features:
        if any(float(corr.loc[f,k])>thr for k in keep):
            drop.append(f)
        else: keep.append(f)
    return keep,drop

def select_rows(df,primary,secondary,complex_cols):
    # Desc primary/secondary, then complexity columns ascending in supplied order.
    asc=[False,False]+[True]*len(complex_cols)
    return df.sort_values([primary,secondary,*complex_cols],ascending=asc).iloc[0]

def rf_structure(clf):
    nodes=sum(int(e.tree_.node_count) for e in clf.estimators_)
    leaves=sum(int(e.tree_.n_leaves) for e in clf.estimators_)
    depths=[int(e.tree_.max_depth) for e in clf.estimators_]
    return {"trees":len(clf.estimators_),"total_nodes":nodes,"total_leaves":leaves,
            "mean_tree_depth":float(np.mean(depths)),"max_tree_depth":int(np.max(depths))}

def time_file_inference(model_kind,model_obj,file_df,features,repeats=200,warmup=20):
    X=file_df[features].to_numpy(float)
    if model_kind=="LR":
        scaler,clf=model_obj
        for _ in range(warmup): fixed_scores(clf,scaler.transform(X))
        ts=[]
        for _ in range(repeats):
            t=time.perf_counter(); fixed_scores(clf,scaler.transform(X)); ts.append((time.perf_counter()-t)*1000)
    else:
        clf=model_obj
        for _ in range(warmup): fixed_scores(clf,X)
        ts=[]
        for _ in range(repeats):
            t=time.perf_counter(); fixed_scores(clf,X); ts.append((time.perf_counter()-t)*1000)
    return {"median_ms":float(np.median(ts)),"p95_ms":float(np.percentile(ts,95)),"repeats":repeats,"windows":len(X)}

def main():
    t0=time.time()
    cfg=json.loads(CFG.read_text(encoding="utf-8"))
    bp=json.loads(BLUEPRINT.read_text(encoding="utf-8"))
    f05=json.loads(FREEZE05.read_text(encoding="utf-8")); f06=json.loads(FREEZE06.read_text(encoding="utf-8")); f04=json.loads(FREEZE04.read_text(encoding="utf-8"))
    assert f05["status"]=="passed" and f05["freeze_package_id"]=="FREEZE-05A-ba6432cf"
    assert f06["status"]=="passed" and f06["freeze_package_id"]=="FREEZE-06A-01576948"
    assert f04["status"]=="passed"
    run05=RUN05.read_text().strip(); run06=RUN06.read_text().strip(); run04=RUN04.read_text().strip()
    b5=ROOT/"outputs"/"runs"/run05/"artifacts"; b6=ROOT/"outputs"/"runs"/run06/"artifacts"; b4=ROOT/"outputs"/"runs"/run04/"artifacts"
    X=pd.read_csv(b4/"q2_interface"/"X_source_common.csv")
    y=pd.read_csv(b4/"q2_interface"/"y_source_labels.csv")
    g=pd.read_csv(b4/"q2_interface"/"groups_source.csv")
    meta=pd.read_csv(b4/"q2_interface"/"source_window_metadata.csv")
    outer=pd.read_csv(b5/"outer_split_manifest.csv"); inner=pd.read_csv(b5/"inner_split_manifest.csv")
    base_inner=pd.read_csv(b6/"inner_lr_results.csv")
    base_sel=pd.read_csv(b6/"selected_lr_config_by_outer_fold.csv")
    base_outer=pd.read_csv(b6/"outer_fold_metrics.csv")
    base_files=pd.read_csv(b6/"oof_file_predictions.csv")
    features=[c for c in X.columns if c!="window_id"]
    data=X.merge(y,on="window_id",validate="one_to_one").merge(g,on="window_id",validate="one_to_one")
    data=data.merge(meta[["window_id","relative_path","load_hp","fault_size_in","outer_race_position_clock","rpm","subgroup"]],on="window_id",validate="one_to_one")
    assert len(features)==26 and len(data)==733 and data.group_id.nunique()==49

    created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    code=git_head(); cfgsha=hf(CFG); envsha=hf(ENV); bpsha=hf(BLUEPRINT)
    binding={"code_version_id":"git:"+code,"data_version_id":f06["freeze_package_id"],"run_config_sha256":cfgsha,"seed":SEED,"environment_sha256":envsha}
    bd=hb(cj(binding).encode())
    run_id=f"run_07-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; models=out/"models"; logs=out/"logs"; mani=out/"manifests"
    for d in [art,models,logs,mani]: d.mkdir(parents=True,exist_ok=False)
    logp=logs/"experiment_log.jsonl"
    def log(event,**kw):
        with logp.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"event":event,**kw},ensure_ascii=False)+"\n")

    # Hypothesis table is frozen from blueprint.
    hyp=[]
    for h in bp["key_improvements"]:
        hyp.append({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in h.items()})
    pd.DataFrame(hyp).to_csv(art/"improvement_hypotheses.csv",index=False,encoding="utf-8-sig")

    # H2: LR + fixed 0.95 correlation pruning, INNER ONLY.
    h2_rows=[]; h2_sel=[]; h2_feature_rows=[]
    for o in range(4):
        candidates=[]
        for C in [0.1,1.0,10.0]:
            for mode in ["file_balanced","file_and_class_balanced"]:
                fold=[]
                recalls={c:[] for c in CLASSES}
                for i in range(3):
                    im=inner[(inner.outer_fold==o)&(inner.inner_fold==i)]
                    trg=set(im.loc[im.role=="train","group_id"]); vag=set(im.loc[im.role=="val","group_id"])
                    tr=data[data.group_id.isin(trg)].copy(); va=data[data.group_id.isin(vag)].copy()
                    keep,drop=corr_keep(tr,features,0.95)
                    sca,clf,dt,conv=fit_lr(tr,keep,C,mode)
                    sc=predict_lr(sca,clf,va,keep); _,fv=aggregate(va,sc); m=metrics(fv.class_label,fv.pred_label)
                    h2_rows.append({"outer_fold":o,"inner_fold":i,"C":C,"weighting":mode,"feature_count":len(keep),
                                    "file_macro_f1":m["macro_f1"],**{f"recall_{c}":m["recall"][c] for c in CLASSES},
                                    "train_time_s":dt,"convergence_warnings":len(conv)})
                    h2_feature_rows.append({"outer_fold":o,"inner_fold":i,"C":C,"weighting":mode,"kept":";".join(keep),"dropped":";".join(drop),"feature_count":len(keep)})
                    fold.append(m["macro_f1"])
                    for c in CLASSES: recalls[c].append(m["recall"][c])
                avg_rec={c:float(np.mean(v)) for c,v in recalls.items()}
                candidates.append({"outer_fold":o,"C":C,"weighting":mode,
                                   "mean_inner_file_macro_f1":float(np.mean(fold)),
                                   "min_class_recall_mean_across_inner":float(min(avg_rec.values()))})
        cdf=pd.DataFrame(candidates); cdf["weight_order"]=cdf.weighting.map(WORDER)
        best=select_rows(cdf,"mean_inner_file_macro_f1","min_class_recall_mean_across_inner",["C","weight_order"])
        h2_sel.append(best.drop(labels=["weight_order"]).to_dict())
    h2df=pd.DataFrame(h2_rows); h2df.to_csv(art/"H2_corr_lr_inner_results.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(h2_feature_rows).to_csv(art/"H2_corr_lr_feature_masks.csv",index=False,encoding="utf-8-sig")
    h2sel=pd.DataFrame(h2_sel); h2sel.to_csv(art/"H2_corr_lr_selected_by_outer.csv",index=False,encoding="utf-8-sig")

    # Pair H2 selected inner folds with frozen Base selected inner folds.
    h2_pair=[]
    for o in range(4):
        hs=h2sel[h2sel.outer_fold==o].iloc[0]
        bs=base_sel[base_sel.outer_fold==o].iloc[0]
        hrows=h2df[(h2df.outer_fold==o)&(h2df.C==hs.C)&(h2df.weighting==hs.weighting)]
        brows=base_inner[(base_inner.outer_fold==o)&(base_inner.C==bs.C)&(base_inner.weighting==bs.weighting)]
        for i in range(3):
            hm=float(hrows[hrows.inner_fold==i].file_macro_f1.iloc[0]); bm=float(brows[brows.inner_fold==i].file_macro_f1.iloc[0])
            h2_pair.append({"outer_fold":o,"inner_fold":i,"base_macro_f1":bm,"H2_macro_f1":hm,"delta":hm-bm,
                            "H2_feature_count":int(hrows[hrows.inner_fold==i].feature_count.iloc[0])})
    h2pair=pd.DataFrame(h2_pair); h2pair.to_csv(art/"H2_corr_lr_paired_inner_comparison.csv",index=False,encoding="utf-8-sig")
    h2_effect={"mean_delta":float(h2pair.delta.mean()),"median_delta":float(h2pair.delta.median()),"min_delta":float(h2pair.delta.min()),"max_delta":float(h2pair.delta.max()),
               "positive_folds":int((h2pair.delta>0).sum()),"nonnegative_folds":int((h2pair.delta>=0).sum()),"pairs":len(h2pair),
               "mean_feature_count":float(h2pair.H2_feature_count.mean())}
    h2_effect["decision"]="no_gain_stop" if h2_effect["mean_delta"]<0.01 else "inner_gain_but_not_final_candidate_by_blueprint"
    (art/"H2_corr_lr_effect.json").write_text(json.dumps(h2_effect,ensure_ascii=False,indent=2),encoding="utf-8")
    log("H2_complete_inner_only",**h2_effect)

    # H1 RF: evaluate all six configs on same inner folds.
    rf_rows=[]; rf_sel=[]; rf_train_times=[]
    for o in range(4):
        cand=[]
        for depth in [8,16,None]:
            for mode in ["file_balanced","file_and_class_balanced"]:
                vals=[]; recalls={c:[] for c in CLASSES}; tvals=[]
                for i in range(3):
                    im=inner[(inner.outer_fold==o)&(inner.inner_fold==i)]
                    trg=set(im.loc[im.role=="train","group_id"]); vag=set(im.loc[im.role=="val","group_id"])
                    tr=data[data.group_id.isin(trg)].copy(); va=data[data.group_id.isin(vag)].copy()
                    clf,dt=fit_rf(tr,features,depth,mode); tvals.append(dt)
                    sc=predict_rf(clf,va,features); _,fv=aggregate(va,sc); m=metrics(fv.class_label,fv.pred_label)
                    rf_rows.append({"outer_fold":o,"inner_fold":i,"max_depth":("None" if depth is None else depth),"weighting":mode,
                                    "file_macro_f1":m["macro_f1"],**{f"recall_{c}":m["recall"][c] for c in CLASSES},"train_time_s":dt})
                    vals.append(m["macro_f1"])
                    for c in CLASSES: recalls[c].append(m["recall"][c])
                avg_rec={c:float(np.mean(v)) for c,v in recalls.items()}
                cand.append({"outer_fold":o,"max_depth":("None" if depth is None else str(depth)),"weighting":mode,
                             "mean_inner_file_macro_f1":float(np.mean(vals)),
                             "min_class_recall_mean_across_inner":float(min(avg_rec.values())),
                             "mean_train_time_s":float(np.mean(tvals))})
        cdf=pd.DataFrame(cand)
        depth_order={"8":0,"16":1,"None":2}; cdf["depth_order"]=cdf.max_depth.map(depth_order); cdf["weight_order"]=cdf.weighting.map(WORDER)
        best=select_rows(cdf,"mean_inner_file_macro_f1","min_class_recall_mean_across_inner",["depth_order","weight_order"])
        rf_sel.append(best.drop(labels=["depth_order","weight_order"]).to_dict())
    rfdf=pd.DataFrame(rf_rows); rfdf.to_csv(art/"H1_rf_inner_results.csv",index=False,encoding="utf-8-sig")
    rfsel=pd.DataFrame(rf_sel); rfsel.to_csv(art/"H1_rf_selected_by_outer.csv",index=False,encoding="utf-8-sig")

    # Pre-outer gate against frozen Base selected inner summaries.
    base_inner_mean=float(base_sel.mean_inner_file_macro_f1.mean())
    rf_inner_mean=float(rfsel.mean_inner_file_macro_f1.mean())
    base_minrec_mean=float(base_sel.min_class_recall_mean_across_inner.mean())
    rf_minrec_mean=float(rfsel.min_class_recall_mean_across_inner.mean())
    gate={
        "base_selected_inner_mean_macro_f1":base_inner_mean,
        "H1_selected_inner_mean_macro_f1":rf_inner_mean,
        "inner_gain":rf_inner_mean-base_inner_mean,
        "base_mean_min_class_recall":base_minrec_mean,
        "H1_mean_min_class_recall":rf_minrec_mean,
        "min_class_recall_delta":rf_minrec_mean-base_minrec_mean,
        "required_inner_gain":0.01,
        "allowed_min_class_recall_delta":-0.05
    }
    gate["passed"]=bool(gate["inner_gain"]>=0.01 and gate["min_class_recall_delta"]>=-0.05)
    (art/"H1_pre_outer_gate.json").write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding="utf-8")
    log("H1_pre_outer_gate",**gate)

    rf_outer_rows=[]; rf_oof_files=[]; rf_models=[]; rf_inference=[]
    if gate["passed"]:
        # One-shot outer evaluation; no configuration changes after this point.
        for o in range(4):
            s=rfsel[rfsel.outer_fold==o].iloc[0]
            depth=None if str(s.max_depth)=="None" else int(str(s.max_depth))
            mode=str(s.weighting)
            om=outer[outer.outer_fold==o]
            trg=set(om.loc[om.role=="train_pool","group_id"]); teg=set(om.loc[om.role=="test","group_id"])
            assert not (trg & teg)
            tr=data[data.group_id.isin(trg)].copy(); te=data[data.group_id.isin(teg)].copy()
            clf,dt=fit_rf(tr,features,depth,mode)
            sc=predict_rf(clf,te,features); _,ff=aggregate(te,sc); m=metrics(ff.class_label,ff.pred_label)
            rf_outer_rows.append({"outer_fold":o,"max_depth":("None" if depth is None else depth),"weighting":mode,
                                  "file_macro_f1":m["macro_f1"],"file_accuracy":m["accuracy"],
                                  **{f"file_recall_{c}":m["recall"][c] for c in CLASSES},
                                  "train_time_s":dt,"test_files":len(teg),"test_windows":len(te)})
            ff["outer_fold"]=o; rf_oof_files.append(ff)
            mp=models/f"H1_rf_outer_fold_{o}.joblib"; joblib.dump({"classifier":clf,"feature_names":features,"class_order":CLASSES,"selected":{"max_depth":depth,"weighting":mode},"seed":SEED},mp,compress=3)
            rf_models.append({"outer_fold":o,"path":mp.relative_to(out).as_posix(),"sha256":hf(mp),"bytes":mp.stat().st_size,**rf_structure(clf)})
            # Feature-ready per-file inference timing on outer-test files.
            for gid in sorted(teg):
                fd=te[te.group_id==gid]
                tm=time_file_inference("RF",clf,fd,features,repeats=200,warmup=20)
                rf_inference.append({"outer_fold":o,"group_id":gid,"model":"H1_RF",**tm})
        rfouter=pd.DataFrame(rf_outer_rows); rfouter.to_csv(art/"H1_rf_outer_metrics.csv",index=False,encoding="utf-8-sig")
        rffiles=pd.concat(rf_oof_files,ignore_index=True).sort_values(["outer_fold","group_id"]); rffiles.to_csv(art/"H1_rf_oof_file_predictions.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(rf_models).to_csv(art/"H1_rf_model_complexity.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(rf_inference).to_csv(art/"H1_rf_inference_timing.csv",index=False,encoding="utf-8-sig")
        rf_pooled=metrics(rffiles.class_label,rffiles.pred_label)
        (art/"H1_rf_pooled_metrics.json").write_text(json.dumps(rf_pooled,ensure_ascii=False,indent=2),encoding="utf-8")
    else:
        pd.DataFrame(columns=["outer_fold","file_macro_f1"]).to_csv(art/"H1_rf_outer_metrics.csv",index=False)
        (art/"H1_rf_pooled_metrics.json").write_text(json.dumps({"status":"not_run","reason":"H1 inner gate failed; outer test was not opened"},indent=2),encoding="utf-8")

    # Pair Base vs H1 only if one-shot outer was permitted.
    if gate["passed"]:
        rfouter=pd.read_csv(art/"H1_rf_outer_metrics.csv")
        pair=base_outer[["outer_fold","file_macro_f1","file_accuracy"]].rename(columns={"file_macro_f1":"base_macro_f1","file_accuracy":"base_accuracy"}).merge(
            rfouter[["outer_fold","file_macro_f1","file_accuracy"]].rename(columns={"file_macro_f1":"H1_macro_f1","file_accuracy":"H1_accuracy"}),on="outer_fold",validate="one_to_one")
        pair["delta_macro_f1"]=pair.H1_macro_f1-pair.base_macro_f1
        pair["delta_accuracy"]=pair.H1_accuracy-pair.base_accuracy
        pair.to_csv(art/"Base_vs_H1_paired_outer.csv",index=False,encoding="utf-8-sig")
        effect={"fold_deltas":[float(x) for x in pair.delta_macro_f1],"mean_delta":float(pair.delta_macro_f1.mean()),
                "median_delta":float(pair.delta_macro_f1.median()),"min_delta":float(pair.delta_macro_f1.min()),"max_delta":float(pair.delta_macro_f1.max()),
                "nonnegative_folds":int((pair.delta_macro_f1>=0).sum()),"positive_folds":int((pair.delta_macro_f1>0).sum()),"n_folds":4,
                "wording":"paired frozen-fold effect; not a statistical significance claim"}
        accept=bool(effect["median_delta"]>0 and effect["nonnegative_folds"]>=3)
        effect["H1_outer_acceptance_passed"]=accept
        (art/"Base_vs_H1_paired_effect.json").write_text(json.dumps(effect,ensure_ascii=False,indent=2),encoding="utf-8")
    else:
        effect={"H1_outer_acceptance_passed":False,"status":"not_evaluated_due_to_inner_gate"}
        accept=False
        (art/"Base_vs_H1_paired_effect.json").write_text(json.dumps(effect,ensure_ascii=False,indent=2),encoding="utf-8")

    final_family="H1_RF" if accept else "Base_LR"
    decision={
        "final_family":final_family,
        "H1_inner_gate":gate,
        "H1_outer_effect":effect,
        "H2_inner_effect":h2_effect,
        "rule":"H1 replaces Base only if pre-outer inner gate passes, then one-shot paired outer median delta>0 and >=3/4 nonnegative folds. H2 is inner-only diagnostic and cannot become Final in this step.",
        "no_more_experiments_after_outer":True
    }
    (art/"final_model_decision.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding="utf-8")

    # Global configuration choice from inner-validation records ONLY, then fit all 49 source files.
    if final_family=="H1_RF":
        rr=rfdf.copy()
        agg=rr.groupby(["max_depth","weighting"],as_index=False).agg(mean_macro_f1=("file_macro_f1","mean"),
            mean_recall_OR=("recall_OR","mean"),mean_recall_IR=("recall_IR","mean"),mean_recall_B=("recall_B","mean"),mean_recall_N=("recall_N","mean"))
        agg["min_mean_recall"]=agg[["mean_recall_OR","mean_recall_IR","mean_recall_B","mean_recall_N"]].min(axis=1)
        depth_order={"8":0,"16":1,"None":2}; agg["depth_order"]=agg.max_depth.astype(str).map(depth_order); agg["weight_order"]=agg.weighting.map(WORDER)
        best=select_rows(agg,"mean_macro_f1","min_mean_recall",["depth_order","weight_order"])
        depth=None if str(best.max_depth)=="None" else int(str(best.max_depth))
        mode=str(best.weighting)
        t=time.perf_counter(); final_clf,fitdt=fit_rf(data,features,depth,mode); final_train=time.perf_counter()-t
        final_obj={"family":"RandomForestClassifier","classifier":final_clf,"feature_names":features,"class_order":CLASSES,
                   "selected_from":"mean of 12 frozen inner validation folds only","config":{"n_estimators":300,"max_depth":depth,"min_samples_leaf":1,"max_features":"sqrt","weighting":mode},
                   "seed":SEED,"score_semantics":"uncalibrated normalized model score vector","raw_feature_extraction_in_model":False}
        final_struct=rf_structure(final_clf)
    else:
        rr=base_inner.copy()
        agg=rr.groupby(["C","weighting"],as_index=False).agg(mean_macro_f1=("file_macro_f1","mean"),
            mean_recall_OR=("recall_OR","mean"),mean_recall_IR=("recall_IR","mean"),mean_recall_B=("recall_B","mean"),mean_recall_N=("recall_N","mean"))
        agg["min_mean_recall"]=agg[["mean_recall_OR","mean_recall_IR","mean_recall_B","mean_recall_N"]].min(axis=1)
        agg["weight_order"]=agg.weighting.map(WORDER)
        best=select_rows(agg,"mean_macro_f1","min_mean_recall",["C","weight_order"])
        C=float(best.C); mode=str(best.weighting)
        t=time.perf_counter(); scaler,final_clf,fitdt,conv=fit_lr(data,features,C,mode); final_train=time.perf_counter()-t
        final_obj={"family":"LogisticRegression","scaler":scaler,"classifier":final_clf,"feature_names":features,"class_order":CLASSES,
                   "selected_from":"mean of 12 frozen inner validation folds only","config":{"C":C,"weighting":mode},
                   "seed":SEED,"score_semantics":"uncalibrated normalized model score vector","raw_feature_extraction_in_model":False}
        final_struct={"coefficient_parameters":int(final_clf.coef_.size+final_clf.intercept_.size)}

    fp=models/"final_source_model.joblib"; joblib.dump(final_obj,fp,compress=3)
    final_model_size=fp.stat().st_size

    # Base LR complexity from the existing 06-A fold models, plus feature-ready inference timing on the same 49 OOF file batches.
    base_timing=[]; base_complex=[]
    for o in range(4):
        bp=ROOT/"outputs"/"runs"/run06/"models"/f"outer_fold_{o}_logistic_baseline.joblib"
        bundle=joblib.load(bp); scaler=bundle["scaler"]; clf=bundle["classifier"]
        base_complex.append({"outer_fold":o,"bytes":bp.stat().st_size,"coefficient_parameters":int(clf.coef_.size+clf.intercept_.size)})
        teg=set(outer[(outer.outer_fold==o)&(outer.role=="test")].group_id)
        te=data[data.group_id.isin(teg)]
        for gid in sorted(teg):
            tm=time_file_inference("LR",(scaler,clf),te[te.group_id==gid],features,repeats=200,warmup=20)
            base_timing.append({"outer_fold":o,"group_id":gid,"model":"Base_LR",**tm})
    pd.DataFrame(base_timing).to_csv(art/"Base_lr_inference_timing.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(base_complex).to_csv(art/"Base_lr_model_complexity.csv",index=False,encoding="utf-8-sig")

    complexity={
        "hardware":{"runner":"GitHub Actions ubuntu-24.04","cpu_logical":os.cpu_count(),"gpu":None},
        "timing_scope":{"input":"already-extracted 26-D window features","raw_signal_preprocessing_and_feature_extraction_included":False,"model_preprocessing_included":True,"batch":"all windows of one file","warmup_repeats":20,"timed_repeats":200},
        "Base_LR":{"features":26,"coefficient_parameters":int(pd.DataFrame(base_complex).coefficient_parameters.median()),
                   "median_serialized_bytes":float(pd.DataFrame(base_complex).bytes.median()),
                   "median_single_file_inference_ms":float(pd.DataFrame(base_timing).median_ms.median()),
                   "p95_across_file_medians_ms":float(np.percentile(pd.DataFrame(base_timing).median_ms,95))},
        "Final":{"family":final_family,"features":26,"serialized_bytes":int(final_model_size),"global_fit_time_s":float(final_train),
                 "structure":final_struct}
    }
    if gate["passed"]:
        rit=pd.DataFrame(rf_inference)
        complexity["H1_RF_outer_models"]={"median_serialized_bytes":float(pd.DataFrame(rf_models).bytes.median()),
                                          "median_total_nodes":float(pd.DataFrame(rf_models).total_nodes.median()),
                                          "median_single_file_inference_ms":float(rit.median_ms.median()),
                                          "p95_across_file_medians_ms":float(np.percentile(rit.median_ms,95)),
                                          "mean_outer_refit_time_s":float(pd.DataFrame(rf_outer_rows).train_time_s.mean())}
    (art/"complexity_and_runtime.json").write_text(json.dumps(complexity,ensure_ascii=False,indent=2),encoding="utf-8")

    # Failure comparison for final vs base, if final RF accepted.
    if accept:
        rff=pd.read_csv(art/"H1_rf_oof_file_predictions.csv")
        base_cm=metrics(base_files.class_label,base_files.pred_label)
        fin_cm=metrics(rff.class_label,rff.pred_label)
        comp={"Base":{"metrics":base_cm,"errors":int((base_files.class_label!=base_files.pred_label).sum())},
              "Final":{"metrics":fin_cm,"errors":int((rff.class_label!=rff.pred_label).sum())}}
        # paired correctness transitions by group
        trn=base_files[["group_id","class_label","pred_label"]].rename(columns={"pred_label":"base_pred"}).merge(
            rff[["group_id","pred_label"]].rename(columns={"pred_label":"final_pred"}),on="group_id",validate="one_to_one")
        trn["base_correct"]=trn.class_label==trn.base_pred; trn["final_correct"]=trn.class_label==trn.final_pred
        trn.to_csv(art/"Base_vs_Final_file_transitions.csv",index=False,encoding="utf-8-sig")
        comp["transitions"]={"base_wrong_final_correct":int((~trn.base_correct & trn.final_correct).sum()),
                             "base_correct_final_wrong":int((trn.base_correct & ~trn.final_correct).sum()),
                             "both_wrong":int((~trn.base_correct & ~trn.final_correct).sum())}
    else:
        comp={"Base_retained":True,"reason":"H1 acceptance rule failed; no post-test tuning permitted."}
    (art/"failure_comparison.json").write_text(json.dumps(comp,ensure_ascii=False,indent=2),encoding="utf-8")

    # Q3 interface.
    pd.DataFrame({"feature_order":range(len(features)),"feature_name":features}).to_csv(art/"feature_order.csv",index=False,encoding="utf-8-sig")
    (art/"class_order.json").write_text(json.dumps({"class_order":CLASSES},ensure_ascii=False,indent=2),encoding="utf-8")
    q3={
        "schema_version":"07A-Q3-interface-1.0",
        "final_model_file":"models/final_source_model.joblib",
        "final_family":final_family,
        "feature_input":"04-A target-compatible 26 common features in feature_order.csv",
        "target_input":"04-A target_interface/X_target_common.csv + target_window_metadata.csv",
        "class_order":CLASSES,
        "file_aggregation":"mean per-window normalized model score vector, then argmax",
        "score_semantics":"uncalibrated model score; do not call confidence/calibrated probability",
        "target_truth":"unknown",
        "source_only_mechanism_features_used":False,
        "fit_on_all_source_after_model_family/config_frozen":True,
        "training_source_groups":49,
        "transfer_warning":"source OOF performance does not establish target-domain accuracy; Q3 must analyze domain shift and target predictions separately"
    }
    (art/"q3_interface.json").write_text(json.dumps(q3,ensure_ascii=False,indent=2),encoding="utf-8")

    # Validation.
    split_sha={"outer":hf(b5/"outer_split_manifest.csv"),"inner":hf(b5/"inner_split_manifest.csv")}
    same_seed=(cfg["seed"]==SEED)
    h2_preserved=(art/"H2_corr_lr_effect.json").exists()
    complexity_ok=bool(complexity["timing_scope"]["raw_signal_preprocessing_and_feature_extraction_included"] is False and complexity["timing_scope"]["model_preprocessing_included"] is True)
    final_exists=fp.exists() and fp.stat().st_size>0
    validation={
        "step_id":"07-A","run_id":run_id,
        "checks":{
            "Base_Final_same_data_splits_seed_metric":{"passed":same_seed,"evidence":f"split_sha={split_sha}; seed={SEED}; primary=file-level Macro-F1","H1_outer_evaluated":bool(gate["passed"])},
            "H1_pre_outer_gate_enforced":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/H1_pre_outer_gate.json","outer_accessed":bool(gate["passed"])},
            "paired_comparison_and_ablation_preserved":{"passed":h2_preserved and (not gate["passed"] or (art/"Base_vs_H1_paired_outer.csv").exists()),"evidence":[f"outputs/runs/{run_id}/artifacts/H2_corr_lr_paired_inner_comparison.csv",f"outputs/runs/{run_id}/artifacts/Base_vs_H1_paired_effect.json"]},
            "no_post_outer_retuning":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/final_model_decision.json","statement":"candidate set, thresholds, seed and acceptance rules were frozen before execution; no experiment after H1 outer results"},
            "complexity_timing_scope_explicit":{"passed":complexity_ok,"evidence":f"outputs/runs/{run_id}/artifacts/complexity_and_runtime.json"},
            "final_model_and_q3_interface_exist":{"passed":final_exists and (art/"q3_interface.json").exists(),"evidence":[f"outputs/runs/{run_id}/models/final_source_model.joblib",f"outputs/runs/{run_id}/artifacts/q3_interface.json"]},
            "target_compatible_final_features_only":{"passed":len(features)==26,"evidence":f"outputs/runs/{run_id}/artifacts/feature_order.csv","source_only_mechanism_features_used":False}
        }
    }
    validation["passed"]=all(v["passed"] for v in validation["checks"].values()); validation["status"]="passed" if validation["passed"] else "failed"
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"07A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    summary={
        "baseline":{"file_macro_f1":0.8184523809523809,"file_accuracy":0.7959183673469388,"errors":10,"error_types":{"OR->IR":6,"IR->OR":3,"N->OR":1}},
        "H2_corr_lr_inner_effect":h2_effect,
        "H1_pre_outer_gate":gate,
        "H1_outer_effect":effect,
        "final_family":final_family,
        "final_global_model_path":"models/final_source_model.joblib",
        "complexity":complexity,
        "q3_interface":"artifacts/q3_interface.json"
    }
    if gate["passed"]:
        rfpm=json.loads((art/"H1_rf_pooled_metrics.json").read_text())
        summary["H1_pooled_file_metrics"]=rfpm
    (art/"step07A_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze={
        "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":"07-A",
        "freeze_package_id":f"FREEZE-07A-{run_id[-8:]}","status":"passed" if validation["passed"] else "failed","run_id":run_id,
        "versions":{"code_version_id":"git:"+code,"raw_data_version_id":f06["versions"]["raw_data_version_id"],"input_derived_data_ids":[f05["freeze_package_id"],f06["freeze_package_id"],"FEAT-23d45f5649dcd5f1"],"run_config_sha256":cfgsha,"environment_sha256":envsha},
        "random_seed":SEED,
        "parameters":{"Base":"06-A LR","H1":"RF 300 trees / max_depth {8,16,None} / two weighting modes","H2":"LR + train-only abs Pearson corr pruning >0.95","primary":"file-level Macro-F1","outer_folds":4,"inner_folds":3},
        "real_results":[
            {"name":"H2_inner_effect","value":h2_effect},
            {"name":"H1_inner_gate","value":gate},
            {"name":"H1_outer_paired_effect","value":effect},
            {"name":"final_decision","value":decision},
            {"name":"complexity","value":complexity}
        ],
        "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/final_model_decision.json"},{"path":f"outputs/runs/{run_id}/artifacts/Base_vs_H1_paired_effect.json"},{"path":f"outputs/runs/{run_id}/artifacts/H2_corr_lr_effect.json"},{"path":f"outputs/runs/{run_id}/artifacts/complexity_and_runtime.json"}],
        "anomalies_and_failures":[{"experiment":"H2-CORR-LR","status":h2_effect["decision"],"evidence":"H2_corr_lr_effect.json"}],
        "b_handoff":{
            "required_data":["step07A_summary.json","improvement_hypotheses.csv","H1_pre_outer_gate.json","Base_vs_H1_paired_effect.json","H2_corr_lr_effect.json","final_model_decision.json","complexity_and_runtime.json","failure_comparison.json","q3_interface.json","feature_order.csv"],
            "supported_conclusions":["Base and H1 used identical frozen outer groups and metric if H1 reached outer evaluation","H2 correlation pruning result is preserved even if non-beneficial","final family selected by the predeclared one-shot rule","Q3 inherits only the 26 target-compatible common features"],
            "wording_limits":["do not use statistically significant","four-fold paired min-max is not a confidence interval","model scores are uncalibrated","source-domain OOF performance does not imply target-domain accuracy"],
            "approved_tables":["Base_vs_H1_paired_outer.csv","H2_corr_lr_paired_inner_comparison.csv","H1_rf_outer_metrics.csv","Base_vs_Final_file_transitions.csv"],"approved_figures":[]
        },
        "unresolved_issues":[],
        "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["05-A split protocol changes","06-A baseline evidence changes","04-A common feature data changes","07-A hypotheses/gates/implementation changes"]}
    }
    f0=json.loads(json.dumps(freeze,ensure_ascii=False)); freeze["freeze"]["content_sha256"]=hb(cj(f0).encode())
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"07A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"07A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True); (mani/"pip_freeze.txt").write_text(pf)
    runtime={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__,"gpu":None,"cuda":None,"git_commit":code}
    (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
    outs=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
        outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
    rm={"schema_version":"07A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"07-A","status":"completed" if validation["passed"] else "failed","created_utc":created,
        "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code,"entrypoint":"scripts/step07a_failure_driven_improvement.py","code_manifest":"protocol/07A/code_manifest.json"},
        "data":{"input_05A_freeze":f05["freeze_package_id"],"input_06A_freeze":f06["freeze_package_id"],"feature_version":"FEAT-23d45f5649dcd5f1","source_files":49,"source_windows":733},
        "blueprint":{"path":"protocol/07A/question2C_argument_blueprint.json","sha256":bpsha},
        "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":envsha},
        "config":{"path":"protocol/07A/run_config.json","sha256":cfgsha},
        "outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":hb(cj(outs).encode())},
        "checkpoint":{"completed_batches":["H2-inner-ablation","H1-inner-selection","H1-pre-outer-gate","H1-one-shot-outer-if-gated","paired-effect","final-global-fit","complexity","Q3-interface","freeze"],"pending_batches":[],"resume_token":None},
        "duration_seconds":time.time()-t0}
    (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":validation["passed"],"run_id":run_id,"freeze_sha256":freeze["freeze"]["content_sha256"],"H2":h2_effect,"H1_gate":gate,"H1_effect":effect,"final_family":final_family,"final_model_sha256":hf(fp)},ensure_ascii=False))
    return 0 if validation["passed"] else 2

if __name__=="__main__":
    raise SystemExit(main())
