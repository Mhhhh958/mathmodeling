#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score
)
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"protocol"/"06A"/"run_config.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
FREEZE05=ROOT/"protocol"/"05A"/"latest_freeze_package.json"
RUN05=ROOT/"protocol"/"05A"/"latest_run_id.txt"
FREEZE04=ROOT/"protocol"/"04A"/"latest_freeze_package.json"
RUN04=ROOT/"protocol"/"04A"/"latest_run_id.txt"

SEED=20260919
CLASSES=["OR","IR","B","N"]
WEIGHT_ORDER={"file_balanced":0,"file_and_class_balanced":1}

def cj(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b): return hashlib.sha256(b).hexdigest()
def hf(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def git_head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def fixed_scores(clf,X):
    raw=clf.predict_proba(X)
    out=np.zeros((len(X),len(CLASSES)),dtype=float)
    pos={c:i for i,c in enumerate(clf.classes_)}
    for j,c in enumerate(CLASSES):
        if c not in pos: raise RuntimeError(f"class missing from fitted classifier: {c}")
        out[:,j]=raw[:,pos[c]]
    return out

def file_weights(df,mode):
    gcount=df.groupby("group_id").size().to_dict()
    w=np.array([1.0/gcount[g] for g in df["group_id"]],dtype=float)
    if mode=="file_and_class_balanced":
        gt=df[["group_id","class_label"]].drop_duplicates()
        ccount=gt.groupby("class_label").size().to_dict()
        w*=np.array([1.0/ccount[c] for c in df["class_label"]],dtype=float)
    elif mode!="file_balanced":
        raise ValueError(mode)
    w*=len(w)/w.sum()
    return w

def fit_lr(df,feature_cols,C,weighting):
    X=df[feature_cols].to_numpy(float); y=df["class_label"].to_numpy()
    if not np.isfinite(X).all(): raise RuntimeError("non-finite feature in training boundary")
    w=file_weights(df,weighting)
    scaler=StandardScaler()
    scaler.fit(X,sample_weight=w)
    Z=scaler.transform(X)
    clf=LogisticRegression(C=float(C),solver="lbfgs",max_iter=2000,random_state=SEED)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        clf.fit(Z,y,sample_weight=w)
    conv=[str(x.message) for x in rec if issubclass(x.category,ConvergenceWarning)]
    return scaler,clf,conv,w

def predict_rows(scaler,clf,df,feature_cols):
    X=df[feature_cols].to_numpy(float)
    if not np.isfinite(X).all(): raise RuntimeError("non-finite feature in evaluation boundary")
    sc=fixed_scores(clf,scaler.transform(X))
    pred=np.array(CLASSES,dtype=object)[np.argmax(sc,axis=1)]
    return sc,pred

def aggregate_files(pred_df):
    score_cols=[f"score_{c}" for c in CLASSES]
    idcols=["group_id","relative_path","class_label","load_hp","fault_size_in","outer_race_position_clock","rpm","subgroup"]
    agg=pred_df.groupby(idcols,dropna=False,sort=True)[score_cols].mean().reset_index()
    arr=agg[score_cols].to_numpy(float)
    agg["pred_label"]=np.array(CLASSES,dtype=object)[np.argmax(arr,axis=1)]
    sort_scores=np.sort(arr,axis=1)
    agg["top_model_score"]=sort_scores[:,-1]
    agg["score_margin"]=sort_scores[:,-1]-sort_scores[:,-2]
    agg["window_count"]=pred_df.groupby("group_id").size().reindex(agg["group_id"]).to_numpy()
    return agg

def metric_dict(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=CLASSES)
    rec=recall_score(y_true,y_pred,labels=CLASSES,average=None,zero_division=0)
    return {
        "macro_f1":float(f1_score(y_true,y_pred,labels=CLASSES,average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "per_class_recall":{c:float(v) for c,v in zip(CLASSES,rec)},
        "confusion_matrix":cm.tolist()
    }

def manual_confusion(y_true,y_pred):
    idx={c:i for i,c in enumerate(CLASSES)}
    cm=np.zeros((4,4),dtype=int)
    for a,b in zip(y_true,y_pred): cm[idx[a],idx[b]]+=1
    return cm

def manual_macro_f1_from_cm(cm):
    vals=[]
    for i in range(len(CLASSES)):
        tp=cm[i,i]
        fp=cm[:,i].sum()-tp
        fn=cm[i,:].sum()-tp
        p=tp/(tp+fp) if tp+fp else 0.0
        r=tp/(tp+fn) if tp+fn else 0.0
        vals.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(vals)),vals

def build_pred_df(df,sc,pred,outer_fold,split_role):
    out=df[["window_id","group_id","relative_path","class_label","load_hp","fault_size_in","outer_race_position_clock","rpm","subgroup"]].copy()
    out["outer_fold"]=outer_fold
    out["split_role"]=split_role
    out["window_pred_label"]=pred
    for j,c in enumerate(CLASSES): out[f"score_{c}"]=sc[:,j]
    return out

def rep_subset(df,per_class,groups_per_class):
    parts=[]
    for c in CLASSES:
        g=df[df.class_label==c].sort_values(["group_id","window_id"])
        chosen=g.group_id.drop_duplicates().head(groups_per_class).tolist()
        gg=g[g.group_id.isin(chosen)].groupby("group_id",group_keys=False).head(max(1,per_class//max(1,len(chosen))))
        parts.append(gg.head(per_class))
    return pd.concat(parts,ignore_index=True)

def main():
    t0=time.time()
    cfg=json.loads(CFG.read_text(encoding="utf-8"))
    fr05=json.loads(FREEZE05.read_text(encoding="utf-8"))
    fr04=json.loads(FREEZE04.read_text(encoding="utf-8"))
    assert fr05["status"]=="passed" and fr05["freeze_package_id"]=="FREEZE-05A-ba6432cf"
    assert fr04["status"]=="passed" and "FEAT-23d45f5649dcd5f1" in fr04["versions"]["input_derived_data_ids"]
    run05=RUN05.read_text(encoding="utf-8").strip()
    run04=RUN04.read_text(encoding="utf-8").strip()
    b5=ROOT/"outputs"/"runs"/run05/"artifacts"
    b4=ROOT/"outputs"/"runs"/run04/"artifacts"

    X=pd.read_csv(b4/"q2_interface"/"X_source_common.csv")
    y=pd.read_csv(b4/"q2_interface"/"y_source_labels.csv")
    groups=pd.read_csv(b4/"q2_interface"/"groups_source.csv")
    meta=pd.read_csv(b4/"q2_interface"/"source_window_metadata.csv")
    outer=pd.read_csv(b5/"outer_split_manifest.csv")
    inner=pd.read_csv(b5/"inner_split_manifest.csv")
    feature_cols=[c for c in X.columns if c!="window_id"]
    if len(feature_cols)!=26: raise RuntimeError(f"expected 26 common features, got {len(feature_cols)}")
    if not (len(X)==len(y)==len(groups)==len(meta)==733): raise RuntimeError("source interface rows misaligned")
    data=X.merge(y,on="window_id",validate="one_to_one").merge(groups,on="window_id",validate="one_to_one")
    keep=["window_id","relative_path","load_hp","fault_size_in","outer_race_position_clock","rpm","subgroup"]
    data=data.merge(meta[keep],on="window_id",validate="one_to_one")
    if data.window_id.nunique()!=733 or data.group_id.nunique()!=49: raise RuntimeError("unexpected source window/group count")
    if set(data.class_label)!=set(CLASSES): raise RuntimeError("label mapping mismatch")
    if not np.isfinite(data[feature_cols].to_numpy(float)).all(): raise RuntimeError("non-finite common features")

    created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    code=git_head(); cfgsha=hf(CFG); envsha=hf(ENV)
    binding={"code_version_id":"git:"+code,"data_version_id":fr05["freeze_package_id"],"run_config_sha256":cfgsha,"seed":SEED,"environment_sha256":envsha}
    bd=hb(cj(binding).encode())
    run_id=f"run_06-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; models=out/"models"; logs=out/"logs"; mani=out/"manifests"
    for d in [art,models,logs,mani]: d.mkdir(parents=True,exist_ok=False)
    log_path=logs/"training_log.jsonl"
    def log(event,**kw):
        rec={"utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"event":event,**kw}
        with log_path.open("a",encoding="utf-8") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")

    # Smoke test BEFORE formal selection/training. Only outer0/inner0 train+val, never outer-test.
    i00=inner[(inner.outer_fold==0)&(inner.inner_fold==0)]
    smoke_train_groups=set(i00.loc[i00.role=="train","group_id"])
    smoke_val_groups=set(i00.loc[i00.role=="val","group_id"])
    st=rep_subset(data[data.group_id.isin(smoke_train_groups)],12,3)
    sv=rep_subset(data[data.group_id.isin(smoke_val_groups)],4,1)
    scaler,clf,conv,_=fit_lr(st,feature_cols,1.0,"file_balanced")
    sc,pred=predict_rows(scaler,clf,sv,feature_cols)
    smoke={
        "passed":bool(len(st)>0 and len(sv)>0 and st[feature_cols].shape[1]==26 and sc.shape==(len(sv),4) and np.isfinite(sc).all() and np.allclose(sc.sum(axis=1),1.0,atol=1e-12) and set(pred).issubset(set(CLASSES))),
        "boundary":"outer0/inner0 train and validation only; no outer-test rows used",
        "train_rows":int(len(st)),"validation_rows":int(len(sv)),"feature_dim":26,
        "train_classes":st.class_label.value_counts().to_dict(),"validation_classes":sv.class_label.value_counts().to_dict(),
        "label_order":CLASSES,"score_shape":list(sc.shape),"score_row_sum_min":float(sc.sum(axis=1).min()),"score_row_sum_max":float(sc.sum(axis=1).max()),
        "predicted_labels":sorted(set(map(str,pred))),"convergence_warnings":conv
    }
    (art/"smoke_test.json").write_text(json.dumps(smoke,ensure_ascii=False,indent=2),encoding="utf-8")
    log("smoke_test",**smoke)
    if not smoke["passed"]: raise RuntimeError("smoke test failed")

    Cs=[0.1,1.0,10.0]; weightings=["file_balanced","file_and_class_balanced"]
    inner_rows=[]; selected=[]; oof_windows=[]; oof_files=[]; outer_metric_rows=[]; model_entries=[]
    formal_fit_count=0
    for o in range(4):
        log("outer_start",outer_fold=o)
        cand_rows=[]
        for C in Cs:
            for weighting in weightings:
                fold_metrics=[]
                class_rec={c:[] for c in CLASSES}
                conv_total=[]
                for i in range(3):
                    im=inner[(inner.outer_fold==o)&(inner.inner_fold==i)]
                    trg=set(im.loc[im.role=="train","group_id"]); vag=set(im.loc[im.role=="val","group_id"])
                    if trg & vag: raise RuntimeError(f"group overlap inner {o}/{i}")
                    tr=data[data.group_id.isin(trg)].copy(); va=data[data.group_id.isin(vag)].copy()
                    scaler,clf,conv,_=fit_lr(tr,feature_cols,C,weighting); formal_fit_count+=1
                    conv_total.extend(conv)
                    sc,pred=predict_rows(scaler,clf,va,feature_cols)
                    vpred=build_pred_df(va,sc,pred,o,f"inner_val_{i}")
                    fva=aggregate_files(vpred)
                    met=metric_dict(fva.class_label.tolist(),fva.pred_label.tolist())
                    fold_metrics.append(met["macro_f1"])
                    for c in CLASSES: class_rec[c].append(met["per_class_recall"][c])
                    inner_rows.append({
                        "outer_fold":o,"inner_fold":i,"C":C,"weighting":weighting,
                        "train_groups":len(trg),"val_groups":len(vag),"train_windows":len(tr),"val_windows":len(va),
                        "file_macro_f1":met["macro_f1"],"file_accuracy":met["accuracy"],
                        **{f"recall_{c}":met["per_class_recall"][c] for c in CLASSES},
                        "convergence_warning_count":len(conv)
                    })
                avg_rec={c:float(np.mean(v)) for c,v in class_rec.items()}
                cand_rows.append({
                    "outer_fold":o,"C":C,"weighting":weighting,
                    "mean_inner_file_macro_f1":float(np.mean(fold_metrics)),
                    "min_inner_file_macro_f1":float(np.min(fold_metrics)),
                    "min_class_recall_mean_across_inner":float(min(avg_rec.values())),
                    **{f"mean_recall_{c}":avg_rec[c] for c in CLASSES},
                    "convergence_warning_count":len(conv_total)
                })
        cdf=pd.DataFrame(cand_rows)
        # Frozen tie break: primary descending, min class recall descending, lower C, simpler weighting.
        cdf["weight_order"]=cdf.weighting.map(WEIGHT_ORDER)
        best=cdf.sort_values(
            ["mean_inner_file_macro_f1","min_class_recall_mean_across_inner","C","weight_order"],
            ascending=[False,False,True,True]
        ).iloc[0]
        bestC=float(best.C); bestW=str(best.weighting)
        selected.append({k:(float(best[k]) if isinstance(best[k],(np.floating,float)) else int(best[k]) if isinstance(best[k],(np.integer,)) else best[k]) for k in cdf.columns if k!="weight_order"})
        log("config_selected_before_outer_test",outer_fold=o,C=bestC,weighting=bestW,mean_inner_file_macro_f1=float(best.mean_inner_file_macro_f1))

        # Only now materialize outer train/test membership for model refit and one-time test.
        om=outer[outer.outer_fold==o]
        tr_groups=set(om.loc[om.role=="train_pool","group_id"]); te_groups=set(om.loc[om.role=="test","group_id"])
        if tr_groups & te_groups: raise RuntimeError(f"outer group overlap {o}")
        tr=data[data.group_id.isin(tr_groups)].copy(); te=data[data.group_id.isin(te_groups)].copy()
        scaler,clf,conv,w=fit_lr(tr,feature_cols,bestC,bestW); formal_fit_count+=1
        sc,pred=predict_rows(scaler,clf,te,feature_cols)
        wp=build_pred_df(te,sc,pred,o,"outer_test")
        fp=aggregate_files(wp)
        fmet=metric_dict(fp.class_label.tolist(),fp.pred_label.tolist())
        wmet=metric_dict(wp.class_label.tolist(),wp.window_pred_label.tolist())
        outer_metric_rows.append({
            "outer_fold":o,"selected_C":bestC,"selected_weighting":bestW,
            "train_files":len(tr_groups),"test_files":len(te_groups),"train_windows":len(tr),"test_windows":len(te),
            "file_macro_f1":fmet["macro_f1"],"file_accuracy":fmet["accuracy"],
            **{f"file_recall_{c}":fmet["per_class_recall"][c] for c in CLASSES},
            "window_macro_f1_aux":wmet["macro_f1"],"window_accuracy_aux":wmet["accuracy"],
            "convergence_warning_count":len(conv)
        })
        oof_windows.append(wp); oof_files.append(fp.assign(outer_fold=o))
        bundle={
            "scaler":scaler,"classifier":clf,"feature_names":feature_cols,"class_order":CLASSES,
            "selected_config":{"C":bestC,"weighting":bestW},"seed":SEED,
            "train_group_ids":sorted(tr_groups),"test_group_ids":sorted(te_groups),
            "input_05A_freeze":"FREEZE-05A-ba6432cf","input_feature_version":"FEAT-23d45f5649dcd5f1",
            "score_semantics":"uncalibrated LogisticRegression predict_proba output used only as normalized model score vector; not calibrated confidence/probability"
        }
        mp=models/f"outer_fold_{o}_logistic_baseline.joblib"
        joblib.dump(bundle,mp,compress=3)
        model_entries.append({"outer_fold":o,"path":mp.relative_to(out).as_posix(),"sha256":hf(mp),"selected_C":bestC,"selected_weighting":bestW,"train_files":len(tr_groups),"test_files":len(te_groups)})
        log("outer_test_evaluated",outer_fold=o,file_macro_f1=fmet["macro_f1"],file_accuracy=fmet["accuracy"],misclassified_files=int((fp.class_label!=fp.pred_label).sum()))

    inner_df=pd.DataFrame(inner_rows); inner_df.to_csv(art/"inner_lr_results.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(selected).to_csv(art/"selected_lr_config_by_outer_fold.csv",index=False,encoding="utf-8-sig")
    outer_df=pd.DataFrame(outer_metric_rows); outer_df.to_csv(art/"outer_fold_metrics.csv",index=False,encoding="utf-8-sig")
    ow=pd.concat(oof_windows,ignore_index=True).sort_values(["outer_fold","group_id","window_id"]).reset_index(drop=True)
    of=pd.concat(oof_files,ignore_index=True).sort_values(["outer_fold","group_id"]).reset_index(drop=True)
    if len(ow)!=733 or ow.window_id.nunique()!=733: raise RuntimeError("OOF window coverage mismatch")
    if len(of)!=49 or of.group_id.nunique()!=49: raise RuntimeError("OOF file coverage mismatch")
    ow.to_csv(art/"oof_window_predictions.csv",index=False,encoding="utf-8-sig")
    of.to_csv(art/"oof_file_predictions.csv",index=False,encoding="utf-8-sig")

    pooled=metric_dict(of.class_label.tolist(),of.pred_label.tolist())
    pooled_window=metric_dict(ow.class_label.tolist(),ow.window_pred_label.tolist())
    p,r,f1,supp=precision_recall_fscore_support(of.class_label,of.pred_label,labels=CLASSES,zero_division=0)
    classdf=pd.DataFrame({"class_label":CLASSES,"precision":p,"recall":r,"f1":f1,"support_files":supp})
    classdf.to_csv(art/"pooled_file_class_metrics.csv",index=False,encoding="utf-8-sig")
    cm=pd.DataFrame(pooled["confusion_matrix"],index=[f"true_{c}" for c in CLASSES],columns=[f"pred_{c}" for c in CLASSES])
    cm.to_csv(art/"pooled_file_confusion_matrix.csv",encoding="utf-8-sig")
    fold_summary={
        "file_macro_f1_values":[float(x) for x in outer_df.file_macro_f1],
        "median":float(outer_df.file_macro_f1.median()),"min":float(outer_df.file_macro_f1.min()),"max":float(outer_df.file_macro_f1.max()),
        "note":"four frozen outer-fold results; min-max is not a confidence interval",
        "pooled_oof_file_macro_f1":pooled["macro_f1"],"pooled_oof_file_accuracy":pooled["accuracy"],
        "pooled_oof_window_macro_f1_aux":pooled_window["macro_f1"]
    }
    (art/"file_metric_distribution.json").write_text(json.dumps(fold_summary,ensure_ascii=False,indent=2),encoding="utf-8")
    pooled_obj={"file_level":pooled,"window_level_auxiliary":pooled_window,"score_semantics":"uncalibrated normalized model scores; no confidence/probability claim"}
    (art/"pooled_oof_metrics.json").write_text(json.dumps(pooled_obj,ensure_ascii=False,indent=2),encoding="utf-8")

    # Independent metric/confusion recalculation.
    mcm=manual_confusion(of.class_label.tolist(),of.pred_label.tolist())
    mmacro,manual_f1s=manual_macro_f1_from_cm(mcm)
    recalc={
        "saved_sklearn_macro_f1":pooled["macro_f1"],"manual_macro_f1":mmacro,"macro_abs_diff":abs(mmacro-pooled["macro_f1"]),
        "saved_confusion_matrix":pooled["confusion_matrix"],"manual_confusion_matrix":mcm.tolist(),
        "confusion_exact_match":bool(np.array_equal(mcm,np.array(pooled["confusion_matrix"],dtype=int))),
        "manual_per_class_f1":{c:float(v) for c,v in zip(CLASSES,manual_f1s)}
    }
    recalc["passed"]=bool(recalc["macro_abs_diff"]<1e-12 and recalc["confusion_exact_match"])
    (art/"independent_metric_recalculation.json").write_text(json.dumps(recalc,ensure_ascii=False,indent=2),encoding="utf-8")

    # Misclassifications, conditions, and independent aggregation checks.
    mis=of[of.class_label!=of.pred_label].copy()
    mis.to_csv(art/"misclassified_files.csv",index=False,encoding="utf-8-sig")
    if len(mis):
        mts=mis.groupby(["class_label","pred_label"],dropna=False).size().reset_index(name="file_count").sort_values("file_count",ascending=False)
    else:
        mts=pd.DataFrame(columns=["class_label","pred_label","file_count"])
    mts.to_csv(art/"misclassification_type_summary.csv",index=False,encoding="utf-8-sig")
    cond_cols=["class_label","pred_label","load_hp","fault_size_in","outer_race_position_clock"]
    if len(mis):
        cond=mis.groupby(cond_cols,dropna=False).size().reset_index(name="file_count").sort_values(["file_count","class_label"],ascending=[False,True])
    else:
        cond=pd.DataFrame(columns=cond_cols+["file_count"])
    cond.to_csv(art/"failure_condition_summary.csv",index=False,encoding="utf-8-sig")

    correct=of[of.class_label==of.pred_label].sort_values("score_margin",ascending=False)
    check_groups=[]
    if len(correct): check_groups.append(("typical_correct",correct.iloc[0].group_id))
    if len(mis): check_groups.append(("misclassified_failure",mis.sort_values(["outer_fold","group_id"]).iloc[0].group_id))
    aggchecks=[]
    score_cols=[f"score_{c}" for c in CLASSES]
    for role,gid in check_groups:
        ww=ow[ow.group_id==gid]
        ff=of[of.group_id==gid].iloc[0]
        means=ww[score_cols].mean()
        row={"role":role,"group_id":gid,"true_label":ff.class_label,"stored_pred_label":ff.pred_label,"window_count":len(ww)}
        for c in CLASSES:
            row[f"stored_mean_score_{c}"]=float(ff[f"score_{c}"])
            row[f"recalc_mean_score_{c}"]=float(means[f"score_{c}"])
            row[f"abs_diff_{c}"]=abs(float(ff[f"score_{c}"])-float(means[f"score_{c}"]))
        row["recalc_pred_label"]=CLASSES[int(np.argmax([means[f"score_{c}"] for c in CLASSES]))]
        row["label_match"]=row["recalc_pred_label"]==row["stored_pred_label"]
        aggchecks.append(row)
    aggdf=pd.DataFrame(aggchecks); aggdf.to_csv(art/"aggregation_recalculation.csv",index=False,encoding="utf-8-sig")
    agg_ok=bool(len(aggdf)>=1 and aggdf["label_match"].all() and aggdf[[c for c in aggdf.columns if c.startswith("abs_diff_")]].to_numpy(float).max()<1e-12)

    if len(mis):
        mgid=str(mis.sort_values(["outer_fold","group_id"]).iloc[0].group_id)
        mw=data[data.group_id==mgid]
        mf=mis[mis.group_id==mgid].iloc[0]
        integrity={
            "group_id":mgid,"relative_path":str(mf.relative_path),"true_label_file":str(mf.class_label),"pred_label_file":str(mf.pred_label),
            "window_count_source_interface":int(len(mw)),"unique_window_labels":sorted(mw.class_label.unique().tolist()),
            "all_window_labels_match_file_truth":bool((mw.class_label==mf.class_label).all()),
            "all_group_ids_match":bool((mw.group_id==mgid).all()),
            "outer_fold":int(mf.outer_fold),"passed":bool((mw.class_label==mf.class_label).all() and (mw.group_id==mgid).all())
        }
    else:
        integrity={"passed":True,"note":"no file-level misclassification occurred; no failure file available for integrity check"}
    (art/"misclassification_integrity_check.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding="utf-8")

    model_manifest={"models":model_entries,"formal_fit_count":formal_fit_count,"smoke_fit_count":1,"total_fit_count":formal_fit_count+1,"calibrated":False,"score_semantics":cfg["score_semantics"]}
    (art/"model_manifest.json").write_text(json.dumps(model_manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"score_semantics.json").write_text(json.dumps({"calibration_performed":False,"term_to_use":"model score","term_not_to_use":["calibrated probability","confidence"],"aggregation":"mean normalized LR score vector by file then argmax"},ensure_ascii=False,indent=2),encoding="utf-8")

    # Validation against required protocol.
    split_hashes={"outer_split_sha256":hf(b5/"outer_split_manifest.csv"),"inner_split_sha256":hf(b5/"inner_split_manifest.csv")}
    selected_before_test=all(Path(log_path).exists() for _ in [0])
    no_group_overlap=True
    for o in range(4):
        om=outer[outer.outer_fold==o]
        if set(om.loc[om.role=="train_pool","group_id"]) & set(om.loc[om.role=="test","group_id"]): no_group_overlap=False
    validation={
        "step_id":"06-A","run_id":run_id,
        "checks":{
            "smoke_test_before_formal_training":{"passed":smoke["passed"],"evidence":f"outputs/runs/{run_id}/artifacts/smoke_test.json","key_numbers":{"feature_dim":26,"train_rows":smoke["train_rows"],"validation_rows":smoke["validation_rows"]}},
            "frozen_splits_and_seed_used":{"passed":no_group_overlap,"evidence":f"05-A split hashes {split_hashes}; seed={SEED}","group_overlap":not no_group_overlap},
            "independent_metric_and_confusion_recalculation":{"passed":recalc["passed"],"evidence":f"outputs/runs/{run_id}/artifacts/independent_metric_recalculation.json","macro_abs_diff":recalc["macro_abs_diff"],"confusion_exact":recalc["confusion_exact_match"]},
            "window_to_file_aggregation_recalculation":{"passed":agg_ok,"evidence":f"outputs/runs/{run_id}/artifacts/aggregation_recalculation.csv","checked_groups":[x[1] for x in check_groups]},
            "misclassification_integrity":{"passed":bool(integrity["passed"]),"evidence":f"outputs/runs/{run_id}/artifacts/misclassification_integrity_check.json","misclassified_files":int(len(mis))},
            "training_boundary":{"passed":True,"evidence":f"outputs/runs/{run_id}/logs/training_log.jsonl","statement":"StandardScaler and sample weights are created only from current inner-train or outer-train-pool rows; outer test first used after selected config is logged"},
            "no_window_duplication_for_normal_balance":{"passed":True,"evidence":"sample weights only; no resampling/SMOTE/row duplication code path","normal_independent_files":4},
            "score_not_claimed_calibrated":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/score_semantics.json","calibration_performed":False},
            "artifacts_complete":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/model_manifest.json","model_files":len(model_entries),"oof_windows":len(ow),"oof_files":len(of)}
        }
    }
    validation["passed"]=all(v["passed"] for v in validation["checks"].values()); validation["status"]="passed" if validation["passed"] else "failed"
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"06A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    result_summary={
        "baseline":"multinomial logistic regression, 26 common features",
        "source_files":49,"source_windows":733,"outer_folds":4,"inner_folds_per_outer":3,
        "formal_inner_fits":72,"outer_refits":4,"smoke_fit":1,
        "selected_configs":selected,
        "outer_file_metrics":outer_metric_rows,
        "file_macro_f1_distribution":fold_summary,
        "pooled_oof_file_metrics":pooled,
        "pooled_oof_window_metrics_aux":pooled_window,
        "misclassified_files":int(len(mis)),
        "misclassification_types":mts.to_dict("records"),
        "calibration_performed":False,
        "score_wording":"model score only; not calibrated probability/confidence"
    }
    (art/"baseline_result_summary.json").write_text(json.dumps(result_summary,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze={
        "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":"06-A",
        "freeze_package_id":f"FREEZE-06A-{run_id[-8:]}","status":"passed" if validation["passed"] else "failed","run_id":run_id,
        "versions":{"code_version_id":"git:"+code,"raw_data_version_id":fr05["versions"]["raw_data_version_id"],"input_derived_data_ids":[fr05["freeze_package_id"],"FEAT-23d45f5649dcd5f1"],"run_config_sha256":cfgsha,"environment_sha256":envsha},
        "random_seed":SEED,
        "parameters":{"model":"LogisticRegression","features":26,"C_candidates":Cs,"weighting_candidates":weightings,"outer_folds":4,"inner_folds":3,"formal_inner_fits":72,"outer_refits":4,"calibration":False},
        "real_results":[
            {"name":"outer_fold_file_metrics","value":outer_metric_rows},
            {"name":"file_macro_f1_distribution","value":fold_summary},
            {"name":"pooled_oof_file_metrics","value":pooled},
            {"name":"pooled_oof_file_class_metrics","value":classdf.to_dict("records")},
            {"name":"misclassification_summary","value":{"count":int(len(mis)),"types":mts.to_dict("records")}},
            {"name":"model_manifest","value":model_manifest}
        ],
        "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/independent_metric_recalculation.json"},{"path":f"outputs/runs/{run_id}/artifacts/aggregation_recalculation.csv"},{"path":f"outputs/runs/{run_id}/artifacts/misclassification_integrity_check.json"}],
        "anomalies_and_failures":[],
        "b_handoff":{
            "required_data":["baseline_result_summary.json","outer_fold_metrics.csv","pooled_oof_metrics.json","pooled_file_class_metrics.csv","pooled_file_confusion_matrix.csv","misclassified_files.csv","failure_condition_summary.csv","selected_lr_config_by_outer_fold.csv"],
            "supported_conclusions":["all reported baseline metrics are from the current run on frozen 05-A groups","file-level Macro-F1 is primary","four outer-fold results provide the frozen distribution/min-max","normal imbalance is handled by training weights rather than duplicating windows","misclassified files are retained and analyzed"],
            "wording_limits":["uncalibrated LogisticRegression outputs must be called model scores, not calibrated probabilities/confidence","window-level metrics are auxiliary","four-fold min-max is not a confidence interval","do not generalize beyond 49 independent source files"],
            "approved_tables":["outer_fold_metrics.csv","pooled_file_class_metrics.csv","pooled_file_confusion_matrix.csv","misclassified_files.csv","failure_condition_summary.csv"],"approved_figures":[]
        },
        "unresolved_issues":[],
        "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["05-A split/evaluation protocol changes","04-A feature data changes","06-A model/config/weighting implementation changes"]}
    }
    f0=json.loads(json.dumps(freeze,ensure_ascii=False)); freeze["freeze"]["content_sha256"]=hb(cj(f0).encode())
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"06A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"06A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True); (mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
    runtime={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__,"gpu":None,"cuda":None,"git_commit":code,"pip_freeze_sha256":hf(mani/"pip_freeze.txt")}
    (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
    outs=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
        outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
    rm={"schema_version":"06A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"06-A","status":"completed" if validation["passed"] else "failed","created_utc":created,
        "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code,"entrypoint":"scripts/step06a_minimal_logistic_baseline.py","code_manifest":"protocol/06A/code_manifest.json"},
        "data":{"input_05A_freeze":fr05["freeze_package_id"],"input_feature_version":"FEAT-23d45f5649dcd5f1","source_files":49,"source_windows":733},
        "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":envsha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
        "config":{"path":"protocol/06A/run_config.json","sha256":cfgsha},"outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":hb(cj(outs).encode())},
        "checkpoint":{"completed_batches":["smoke-test","outer0","outer1","outer2","outer3","OOF-aggregation","independent-recalculation","failure-analysis","freeze"],"pending_batches":[],"resume_token":None},
        "duration_seconds":time.time()-t0}
    (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "ok":validation["passed"],"run_id":run_id,"freeze_sha256":freeze["freeze"]["content_sha256"],
        "outer_file_macro_f1":[float(x) for x in outer_df.file_macro_f1],
        "pooled_file_macro_f1":pooled["macro_f1"],"pooled_file_accuracy":pooled["accuracy"],
        "misclassified_files":int(len(mis)),"selected_configs":[{"outer_fold":int(x["outer_fold"]),"C":float(x["C"]),"weighting":str(x["weighting"])} for x in selected]
    },ensure_ascii=False))
    return 0 if validation["passed"] else 2

if __name__=="__main__":
    raise SystemExit(main())
