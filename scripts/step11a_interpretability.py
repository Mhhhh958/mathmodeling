#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn

ROOT=Path(__file__).resolve().parents[1]
STEP="11-A"
CLASSES=["OR","IR","B","N"]
SEED=20260919
TOPK=5
RANDOM_NULL_N=200
EPS=1e-12

RUN09="run_09-A_20260920T070725126712Z_6ef81c21_26f99878"
RUN10="run_10-A_20260920T085322828930Z_bcbb6757_4346ce31"
RUN07="run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb"
RUN04="run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0"

STATE=ROOT/"protocol/00C/process_state_card.json"
BP=ROOT/"protocol/11A/question4_argument_blueprint.json"
CFG=ROOT/"protocol/11A/run_config.json"
ENV=ROOT/"protocol/00B/environment_manifest.json"
F09=ROOT/"protocol/09A/latest_freeze_package.json"
F10=ROOT/"protocol/10A/latest_freeze_package.json"

MODEL=ROOT/"outputs/runs"/RUN09/"models/final_transfer_bundle.joblib"
T10=ROOT/"outputs/runs"/RUN10/"artifacts/A_P_final_label_table.csv"
W10=ROOT/"outputs/runs"/RUN10/"artifacts/A_P_recomputed_window_predictions.csv"
OOF=ROOT/"outputs/runs"/RUN07/"artifacts/H1_rf_oof_file_predictions.csv"
XS=ROOT/"outputs/runs"/RUN04/"artifacts/q2_interface/X_source_common.csv"
YS=ROOT/"outputs/runs"/RUN04/"artifacts/q2_interface/y_source_labels.csv"
GS=ROOT/"outputs/runs"/RUN04/"artifacts/q2_interface/groups_source.csv"
MS=ROOT/"outputs/runs"/RUN04/"artifacts/q2_interface/source_window_metadata.csv"
XT=ROOT/"outputs/runs"/RUN04/"artifacts/target_interface/X_target_common.csv"
MT=ROOT/"outputs/runs"/RUN04/"artifacts/target_interface/target_window_metadata.csv"

IMPULSE={"kurtosis","crest_factor","impulse_factor","clearance_factor","envelope_kurtosis"}
SPECTRAL={"spectral_centroid_hz","spectral_rms_hz","spectral_entropy","spectral_flatness",
          "dominant_frequency_hz","rolloff95_hz","band_ratio_0_500","band_ratio_500_1500",
          "band_ratio_1500_3000","band_ratio_3000_5500","envelope_spectral_entropy"}
MECH=IMPULSE|SPECTRAL

def nowz():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def canonical_hash(obj)->str:
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def git_head()->str:
    return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def scores_fixed(clf,X):
    raw=clf.predict_proba(np.asarray(X,float))
    pos={str(c):i for i,c in enumerate(clf.classes_)}
    return np.column_stack([raw[:,pos[c]] for c in CLASSES])

def t1_transform(bundle,Xt):
    art=bundle["adaptation_artifact"]
    mus=np.asarray(art["source_mean"],float); sds=np.asarray(art["source_sd"],float)
    mut=np.asarray(art["target_mean"],float); sdt=np.asarray(art["target_sd"],float)
    full=((Xt-mut)/np.where(sdt>EPS,sdt,1.0))*sds+mus
    alpha=float(bundle["setting"])
    return (1-alpha)*Xt+alpha*full

def file_score(clf,X,class_idx):
    return float(scores_fixed(clf,X)[:,class_idx].mean())

def joint_drop(clf,X,class_idx,features_idx,ref):
    base=file_score(clf,X,class_idx)
    Z=np.asarray(X,float).copy()
    for j in features_idx: Z[:,j]=ref[j]
    return base-file_score(clf,Z,class_idx)

def individual_contrib(clf,X,class_idx,ref):
    base=file_score(clf,X,class_idx)
    out=np.zeros(X.shape[1],float)
    for j in range(X.shape[1]):
        Z=np.asarray(X,float).copy(); Z[:,j]=ref[j]
        out[j]=base-file_score(clf,Z,class_idx)
    return base,out

def window_contrib(clf,x,class_idx,ref):
    x=np.asarray(x,float).reshape(1,-1)
    base=float(scores_fixed(clf,x)[0,class_idx])
    out=np.zeros(x.shape[1],float)
    for j in range(x.shape[1]):
        Z=x.copy(); Z[0,j]=ref[j]
        out[j]=base-float(scores_fixed(clf,Z)[0,class_idx])
    return base,out

def top_set(v,k=TOPK):
    return set(np.argsort(np.asarray(v))[-k:].tolist())

def jacc(a,b):
    u=a|b
    return float(len(a&b)/len(u)) if u else 1.0

def safe_spearman(a,b):
    r=spearmanr(a,b).statistic
    return 0.0 if not np.isfinite(r) else float(r)

def label_from_scores(clf,X):
    m=scores_fixed(clf,X).mean(axis=0)
    return CLASSES[int(np.argmax(m))],m

def case_source_selection(oof):
    rows=[]
    for c in CLASSES:
        q=oof[(oof["class_label"].astype(str)==c)&(oof["pred_label"].astype(str)==c)].copy()
        if len(q)==0: raise RuntimeError(f"no correct OOF source file for {c}")
        q["true_score"]=q[f"score_{c}"].astype(float)
        r=q.sort_values(["true_score","group_id"],ascending=[False,True]).iloc[0]
        rows.append({"case_id":f"SRC_{c}_REP","domain":"source","group_id":str(r.group_id),
                     "selection_role":f"known-label representative {c}","true_label":c,
                     "selection_evidence":"highest correct OOF true-class score"})
    mis=oof[oof["pred_label"].astype(str)!=oof["class_label"].astype(str)].copy()
    if len(mis):
        arr=mis[[f"score_{c}" for c in CLASSES]].to_numpy(float)
        sr=np.sort(arr,axis=1)
        mis["oof_margin"]=sr[:,-1]-sr[:,-2]
        r=mis.sort_values(["oof_margin","group_id"],ascending=[True,True]).iloc[0]
        rows.append({"case_id":"SRC_OOF_DIFFICULT","domain":"source","group_id":str(r.group_id),
                     "selection_role":"known-label historical OOF difficult case","true_label":str(r.class_label),
                     "selection_evidence":"smallest OOF top-two score margin among OOF errors"})
    return rows

def case_target_selection(t):
    rows=[]
    selected=set()
    stable=(np.isclose(t["window_file_agreement"],1.0)&np.isclose(t["cross_setting_vote_agreement_rate"],1.0)&np.isclose(t["bootstrap_official_agreement_rate"],1.0))
    for c in ["OR","B"]:
        q=t[stable&(t["pred_label"].astype(str)==c)].copy()
        if len(q)==0: raise RuntimeError(f"no fully stable target {c} case")
        r=q.sort_values(["top_model_score","target_id"],ascending=[False,True]).iloc[0]
        tid=str(r.target_id); selected.add(tid)
        rows.append({"case_id":f"TGT_STABLE_{c}","domain":"target","group_id":str(r.group_id),
                     "selection_role":f"stable predicted {c} representative","true_label":"UNKNOWN_TRUTH",
                     "selection_evidence":"window/vote/bootstrap all 1; highest top_model_score"})
    q=t[~t["target_id"].astype(str).isin(selected)].copy()
    r=q.sort_values(["bootstrap_official_agreement_rate","target_id"],ascending=[True,True]).iloc[0]
    tid=str(r.target_id); selected.add(tid)
    rows.append({"case_id":"TGT_LOW_BOOT","domain":"target","group_id":str(r.group_id),
                 "selection_role":"lowest bootstrap-stability target case","true_label":"UNKNOWN_TRUTH",
                 "selection_evidence":"minimum frozen bootstrap official-label agreement"})
    q=t[~t["target_id"].astype(str).isin(selected)].copy()
    r=q.sort_values(["cross_setting_vote_agreement_rate","target_id"],ascending=[True,True]).iloc[0]
    rows.append({"case_id":"TGT_LOW_SETTING","domain":"target","group_id":str(r.group_id),
                 "selection_role":"lowest cross-setting-vote remaining target case","true_label":"UNKNOWN_TRUTH",
                 "selection_evidence":"minimum frozen 9-setting vote agreement after prior selections"})
    return rows

def mechanism_direction(case_mean,ref,proto,feat_idx):
    out={}
    for j in feat_idx:
        a=float(case_mean[j]-ref[j]); b=float(proto[j]-ref[j])
        if abs(b)<1e-12: out[j]=None
        else: out[j]=bool(a*b>0)
    return out

def main():
    t0=time.time()
    state=json.loads(STATE.read_text(encoding="utf-8"))
    bp=json.loads(BP.read_text(encoding="utf-8"))
    cfg=json.loads(CFG.read_text(encoding="utf-8"))
    f09=json.loads(F09.read_text(encoding="utf-8"))
    f10=json.loads(F10.read_text(encoding="utf-8"))
    if state["CURRENT_ALLOWED_STEP"]!="11-A" or state["NEXT_ALLOWED"]!="11-A" or state["LAST_PASS_TOKEN"]!="10-B-V2026.09.20-49e30c62-通过" or state.get("OPEN_P0")!=[]:
        raise RuntimeError("11-A gate not open")
    if f09["freeze_package_id"]!="FREEZE-09A-26f99878" or f09["status"]!="passed": raise RuntimeError("09-A freeze mismatch")
    if f10["freeze_package_id"]!="FREEZE-10A-4346ce31" or f10["status"]!="passed": raise RuntimeError("10-A freeze mismatch")
    if sha256_file(MODEL)!=cfg["final_transfer_bundle_sha256"]: raise RuntimeError("final transfer bundle SHA mismatch")

    bundle=joblib.load(MODEL)
    if bundle.get("method")!="T1" or abs(float(bundle.get("setting"))-0.25)>1e-12: raise RuntimeError("final transfer method mismatch")
    features=list(bundle["feature_names"])
    if len(features)!=26 or list(bundle["class_order"])!=CLASSES: raise RuntimeError("feature/class interface mismatch")
    clf=bundle["classifier"]

    xs=pd.read_csv(XS); ys=pd.read_csv(YS); gs=pd.read_csv(GS); ms=pd.read_csv(MS)
    xt=pd.read_csv(XT); mt=pd.read_csv(MT); t10=pd.read_csv(T10); oof=pd.read_csv(OOF)
    if list(xs.columns[1:])!=features or list(xt.columns[1:])!=features: raise RuntimeError("feature order mismatch")
    source=xs.merge(ys,on="window_id",validate="one_to_one").merge(gs,on="window_id",validate="one_to_one")
    source=source.merge(ms[["window_id","window_start_s","relative_path"]],on="window_id",how="left",validate="one_to_one")
    target=xt.merge(mt[["window_id","group_id","relative_path","window_start_s","class_label","label_status"]],on="window_id",validate="one_to_one")
    if set(target["class_label"].astype(str))!={"UNKNOWN_TRUTH"} or set(target["label_status"].astype(str))!={"unknown_truth"}:
        raise RuntimeError("target truth boundary violated")

    Xt_raw=target[features].to_numpy(float)
    Xt_ad=t1_transform(bundle,Xt_raw)
    target_ad=target[["window_id","group_id","relative_path","window_start_s","class_label","label_status"]].copy()
    for j,f in enumerate(features): target_ad[f]=Xt_ad[:,j]

    # Reproduce 10-A final labels/scores.
    repro=[]
    for gid,g in target_ad.groupby("group_id",sort=True):
        pred,m=label_from_scores(clf,g[features].to_numpy(float))
        tid=Path(str(gid)).stem
        refrow=t10[t10["target_id"].astype(str)==tid].iloc[0]
        repro.append({"target_id":tid,"pred_label_recomputed":pred,"pred_label_10A":str(refrow.pred_label),
                      "max_score_abs_diff":float(np.max(np.abs(m-refrow[[f"score_{c}" for c in CLASSES]].to_numpy(float))))})
    repro=pd.DataFrame(repro)
    if not (repro["pred_label_recomputed"]==repro["pred_label_10A"]).all() or repro["max_score_abs_diff"].max()>1e-12:
        raise RuntimeError("cannot exactly reproduce 10-A target outputs")

    # Source independent-file reference values/prototypes.
    sf=source.groupby("group_id",sort=True)[features].mean()
    file_label=source[["group_id","class_label"]].drop_duplicates().set_index("group_id").loc[sf.index]["class_label"].astype(str)
    ref=sf.median(axis=0).to_numpy(float)
    proto={c:sf[file_label==c].mean(axis=0).to_numpy(float) for c in CLASSES}
    src_mu=sf.mean(axis=0).to_numpy(float)
    src_sd=sf.std(axis=0,ddof=1).replace(0,np.nan).fillna(1.0).to_numpy(float)

    case_defs=case_source_selection(oof)+case_target_selection(t10)
    case_inv=[]
    contrib_rows=[]; win_rows=[]; faith_rows=[]; null_rows=[]; stab_rows=[]; mech_rows=[]; migration_case_rows=[]
    rng=np.random.default_rng(SEED)
    for cd in case_defs:
        gid=cd["group_id"]; dom=cd["domain"]
        if dom=="source":
            g=source[source["group_id"].astype(str)==gid].sort_values("window_start_s").copy()
            X=g[features].to_numpy(float)
            true_label=str(g["class_label"].iloc[0])
            explained_label,m=label_from_scores(clf,X)
            truth_status="known_source"
            target_id=""
            tenA_label=""
            input_space="source common features -> frozen RF"
        else:
            g=target_ad[target_ad["group_id"].astype(str)==gid].sort_values("window_start_s").copy()
            X=g[features].to_numpy(float)
            target_id=Path(gid).stem
            r=t10[t10["target_id"].astype(str)==target_id].iloc[0]
            tenA_label=str(r.pred_label)
            explained_label,m=label_from_scores(clf,X)
            if explained_label!=tenA_label: raise RuntimeError(f"{target_id} label drift")
            true_label="UNKNOWN_TRUTH"; truth_status="unknown_target"
            input_space="T1(alpha=0.25)-adapted common features -> frozen RF"

        cidx=CLASSES.index(explained_label)
        base,vec=individual_contrib(clf,X,cidx,ref)
        order=np.argsort(vec)[::-1]
        top=order[:TOPK].tolist(); low=order[-TOPK:][::-1].tolist()
        pool=np.array([j for j in range(len(features)) if j not in top],int)
        fixed_random=rng.choice(pool,size=TOPK,replace=False).tolist()
        top_drop=joint_drop(clf,X,cidx,top,ref)
        low_drop=joint_drop(clf,X,cidx,low,ref)
        random_drop=joint_drop(clf,X,cidx,fixed_random,ref)
        random_drops=[]
        for rr in range(RANDOM_NULL_N):
            idx=rng.choice(np.arange(len(features)),size=TOPK,replace=False).tolist()
            d=joint_drop(clf,X,cidx,idx,ref); random_drops.append(d)
            null_rows.append({"case_id":cd["case_id"],"replicate":rr,"drop":d,"features":";".join(features[j] for j in idx)})
        null_med=float(np.median(random_drops))
        pct=float((np.sum(np.asarray(random_drops)<top_drop)+0.5*np.sum(np.asarray(random_drops)==top_drop))/len(random_drops))
        faith_pass=bool(top_drop>0 and top_drop>low_drop and top_drop>null_med)
        faith_rows.append({"case_id":cd["case_id"],"domain":dom,"group_id":gid,"explained_label":explained_label,
                           "original_score":base,"top5_drop":top_drop,"low5_drop":low_drop,"fixed_random5_drop":random_drop,
                           "random_null_median":null_med,"top5_null_percentile":pct,"faithfulness_case_pass":faith_pass,
                           "top5_features":";".join(features[j] for j in top),
                           "low5_features":";".join(features[j] for j in low),
                           "fixed_random5_features":";".join(features[j] for j in fixed_random)})
        cm=X.mean(axis=0)
        dirmap=mechanism_direction(cm,ref,proto[explained_label],top)
        mech_top=[j for j in top if features[j] in MECH]
        aligned=[dirmap[j] for j in mech_top if dirmap[j] is not None]
        proxy_share=float(len(mech_top)/TOPK)
        align_rate=float(np.mean(aligned)) if aligned else np.nan
        mech_rows.append({"case_id":cd["case_id"],"domain":dom,"group_id":gid,"true_label":true_label,
                          "explained_label":explained_label,"top5_mechanism_proxy_share":proxy_share,
                          "prototype_direction_alignment_rate":align_rate,
                          "top5_mechanism_features":";".join(features[j] for j in mech_top),
                          "mechanism_note":"directional consistency with source true-class prototype; not causality and not target truth"})
        for rank,j in enumerate(order,1):
            fam="impulse" if features[j] in IMPULSE else ("spectral_structure" if features[j] in SPECTRAL else "other")
            contrib_rows.append({"case_id":cd["case_id"],"domain":dom,"group_id":gid,"target_id":target_id,
                                 "true_label":true_label,"truth_status":truth_status,"explained_label":explained_label,
                                 "feature":features[j],"rank":rank,"file_score_drop_when_occluded":float(vec[j]),
                                 "feature_family":fam,"is_mechanism_proxy":bool(features[j] in MECH),
                                 "case_input_mean":float(cm[j]),"source_reference_median_file_mean":float(ref[j]),
                                 "predicted_class_source_prototype":float(proto[explained_label][j]),
                                 "prototype_direction_aligned":dirmap.get(j,None)})
        # Window-level contributions and adjacent stability.
        wvecs=[]
        for _,wr in g.iterrows():
            xb=wr[features].to_numpy(float)
            ws,wv=window_contrib(clf,xb,cidx,ref); wvecs.append(wv)
            for j,f in enumerate(features):
                win_rows.append({"case_id":cd["case_id"],"window_id":str(wr.window_id),"window_start_s":float(wr.window_start_s),
                                 "domain":dom,"explained_label":explained_label,"feature":f,
                                 "window_score":ws,"score_drop_when_occluded":float(wv[j])})
        rhos=[]; jacs=[]
        for i in range(len(wvecs)-1):
            rhos.append(safe_spearman(wvecs[i],wvecs[i+1]))
            jacs.append(jacc(top_set(wvecs[i]),top_set(wvecs[i+1])))
        medrho=float(np.median(rhos)) if rhos else np.nan
        medjac=float(np.median(jacs)) if jacs else np.nan
        stab_pass=bool(medrho>=0.30 and medjac>=0.20)
        stab_rows.append({"case_id":cd["case_id"],"domain":dom,"group_id":gid,
                          "adjacent_pair_count":len(rhos),"median_adjacent_spearman":medrho,
                          "median_adjacent_top5_jaccard":medjac,"stability_case_pass":stab_pass,
                          "spearman_threshold":0.30,"jaccard_threshold":0.20})
        if dom=="target":
            raw=target[target["group_id"].astype(str)==gid].sort_values("window_start_s")[features].to_numpy(float)
            rmean=raw.mean(axis=0); amean=X.mean(axis=0)
            move=np.abs((amean-rmean)/src_sd)
            for j in np.argsort(move)[::-1][:10]:
                migration_case_rows.append({"case_id":cd["case_id"],"target_id":target_id,"feature":features[j],
                                            "abs_T1_input_shift_in_source_sd":float(move[j]),
                                            "raw_file_mean":float(rmean[j]),"adapted_file_mean":float(amean[j])})
        case_inv.append({"case_id":cd["case_id"],"domain":dom,"group_id":gid,"target_id":target_id,
                         "selection_role":cd["selection_role"],"selection_evidence":cd["selection_evidence"],
                         "true_label":true_label,"truth_status":truth_status,"explained_label":explained_label,
                         "tenA_final_label":tenA_label,"final_model_label_matches_true":(explained_label==true_label if dom=="source" else ""),
                         "original_explained_class_score":base,"input_space":input_space})

    case_inv=pd.DataFrame(case_inv); contrib=pd.DataFrame(contrib_rows); wincon=pd.DataFrame(win_rows)
    faith=pd.DataFrame(faith_rows); null=pd.DataFrame(null_rows); stab=pd.DataFrame(stab_rows); mech=pd.DataFrame(mech_rows)
    migcase=pd.DataFrame(migration_case_rows)

    # Global T1 process explanation across all target files.
    tf0=target.groupby("group_id",sort=True)[features].mean()
    tf1=target_ad.groupby("group_id",sort=True)[features].mean()
    before=np.abs((tf0.mean(axis=0).to_numpy(float)-src_mu)/src_sd)
    after=np.abs((tf1.mean(axis=0).to_numpy(float)-src_mu)/src_sd)
    mig=pd.DataFrame({"feature":features,"source_standardized_mean_gap_before":before,
                      "source_standardized_mean_gap_after":after,
                      "gap_change":after-before,
                      "relative_reduction":np.where(before>EPS,(before-after)/before,np.nan)})
    mig["rank_before"]=mig["source_standardized_mean_gap_before"].rank(ascending=False,method="min").astype(int)
    mig=mig.sort_values("source_standardized_mean_gap_before",ascending=False).reset_index(drop=True)

    faith_rate=float(faith["faithfulness_case_pass"].mean())
    stability_rate=float(stab["stability_case_pass"].mean())
    source_reps=mech[mech["case_id"].str.match(r"SRC_(OR|IR|B|N)_REP")]
    mech_proxy_mean=float(source_reps["top5_mechanism_proxy_share"].mean())
    mech_align_median=float(source_reps["prototype_direction_alignment_rate"].median())
    mechanism_valid=bool(mech_proxy_mean>=0.60 and mech_align_median>=0.50)
    faith_valid=bool(faith_rate>=2/3)
    stability_valid=bool(stability_rate>=2/3)
    validation_type_count=int(faith_valid)+int(stability_valid)+int(mechanism_valid)
    sanity_valid=bool((faith["top5_null_percentile"]>=0.75).mean()>=2/3 and float(faith["top5_null_percentile"].median())>=0.80)

    # Contradictions/failures must be retained.
    contradictions=[]
    merged=case_inv.merge(faith[["case_id","faithfulness_case_pass","top5_drop","low5_drop","top5_null_percentile"]],on="case_id")
    merged=merged.merge(stab[["case_id","stability_case_pass","median_adjacent_spearman","median_adjacent_top5_jaccard"]],on="case_id")
    merged=merged.merge(mech[["case_id","top5_mechanism_proxy_share","prototype_direction_alignment_rate"]],on="case_id")
    for _,r in merged.iterrows():
        reasons=[]
        if not bool(r.faithfulness_case_pass): reasons.append("faithfulness control failed")
        if not bool(r.stability_case_pass): reasons.append("adjacent-window explanation stability below frozen thresholds")
        if np.isfinite(r.prototype_direction_alignment_rate) and r.prototype_direction_alignment_rate<0.5:
            reasons.append("top mechanism-feature directions weakly aligned with predicted-class source prototype")
        if r.domain=="source" and r.final_model_label_matches_true is False: reasons.append("final RF output differs from known source label")
        if r.case_id=="SRC_OOF_DIFFICULT": reasons.append("historical OOF misclassification/difficult case retained by design")
        if reasons:
            contradictions.append({"case_id":r.case_id,"domain":r.domain,"group_id":r.group_id,
                                   "true_label":r.true_label,"explained_label":r.explained_label,
                                   "reasons":"; ".join(reasons),
                                   "interpretation_limit":"explanation describes model response; mismatch is not hidden or converted into causal evidence"})
    contradictions=pd.DataFrame(contradictions)

    created=nowz()
    binding={"step_id":STEP,"git_commit":git_head(),"freeze09":"FREEZE-09A-26f99878",
             "freeze10":"FREEZE-10A-4346ce31","model_sha256":sha256_file(MODEL),
             "blueprint_sha256":sha256_file(BP),"run_config_sha256":sha256_file(CFG),
             "environment_sha256":sha256_file(ENV)}
    bd=canonical_hash(binding)
    run_id=f"run_11-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs/runs"/run_id; art=out/"artifacts"; figs=out/"figures"; mani=out/"manifests"; logs=out/"logs"
    for d in [art,figs,mani,logs]: d.mkdir(parents=True,exist_ok=False)

    case_inv.to_csv(art/"case_inventory.csv",index=False,encoding="utf-8-sig")
    contrib.to_csv(art/"file_feature_contributions.csv",index=False,encoding="utf-8-sig")
    wincon.to_csv(art/"window_feature_contributions.csv",index=False,encoding="utf-8-sig")
    faith.to_csv(art/"faithfulness_ablation.csv",index=False,encoding="utf-8-sig")
    null.to_csv(art/"sanity_random5_null.csv",index=False,encoding="utf-8-sig")
    stab.to_csv(art/"explanation_stability.csv",index=False,encoding="utf-8-sig")
    mech.to_csv(art/"mechanism_consistency.csv",index=False,encoding="utf-8-sig")
    mig.to_csv(art/"migration_feature_shift.csv",index=False,encoding="utf-8-sig")
    migcase.to_csv(art/"target_case_T1_input_shift.csv",index=False,encoding="utf-8-sig")
    contradictions.to_csv(art/"failure_and_contradiction_cases.csv",index=False,encoding="utf-8-sig")
    repro.to_csv(art/"10A_reproduction_check.csv",index=False,encoding="utf-8-sig")

    # Compact case table.
    ctab=merged[["case_id","domain","group_id","target_id","selection_role","true_label","truth_status","explained_label",
                 "original_explained_class_score","faithfulness_case_pass","top5_drop","low5_drop","top5_null_percentile",
                 "stability_case_pass","median_adjacent_spearman","median_adjacent_top5_jaccard",
                 "top5_mechanism_proxy_share","prototype_direction_alignment_rate"]].copy()
    topfeat=(contrib[contrib["rank"]<=5].sort_values(["case_id","rank"]).groupby("case_id")["feature"].apply(lambda x:";".join(x))).rename("top5_features")
    ctab=ctab.merge(topfeat,on="case_id",how="left")
    ctab.to_csv(art/"explanation_case_table.csv",index=False,encoding="utf-8-sig")

    # Figure 1: T1 process shift, top 10 by before gap.
    m10=mig.head(10).iloc[::-1]
    y=np.arange(len(m10))
    fig,ax=plt.subplots(figsize=(8.7,5.6))
    ax.scatter(m10["source_standardized_mean_gap_before"],y,marker="o",label="Before T1")
    ax.scatter(m10["source_standardized_mean_gap_after"],y,marker="x",label="After T1")
    for yi,a,b in zip(y,m10["source_standardized_mean_gap_before"],m10["source_standardized_mean_gap_after"]):
        ax.plot([a,b],[yi,yi],linewidth=1)
    ax.set_yticks(y,m10["feature"])
    ax.set_xlabel("Absolute target-source mean gap (source-file SD units)")
    ax.set_title("T1 adaptation: largest marginal feature shifts")
    ax.legend()
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); fig.savefig(figs/"q4_T1_feature_shift.png",dpi=180); plt.close(fig)

    # Figure 2: Top-5 contributions per selected case.
    c5=contrib[contrib["rank"]<=5].copy()
    # horizontal ranked points, one row per case-feature label
    c5["label"]=c5["case_id"]+" | "+c5["feature"]
    c5=c5.sort_values(["case_id","rank"],ascending=[True,False]).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(10.5,max(6,0.25*len(c5)+1.5)))
    yy=np.arange(len(c5))
    ax.scatter(c5["file_score_drop_when_occluded"],yy)
    ax.axvline(0,linewidth=1)
    ax.set_yticks(yy,c5["label"],fontsize=7)
    ax.set_xlabel("Drop in explained-class file score after single-feature occlusion")
    ax.set_title("Final-model case explanations (positive = supporting model output)")
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); fig.savefig(figs/"q4_case_top5_contributions.png",dpi=180); plt.close(fig)

    # Figure 3: Faithfulness controls.
    fplot=faith.copy()
    x=np.arange(len(fplot))
    fig,ax=plt.subplots(figsize=(9.4,5.2))
    ax.scatter(x-0.16,fplot["top5_drop"],label="Top-5")
    ax.scatter(x,fplot["low5_drop"],label="Low-5")
    ax.scatter(x+0.16,fplot["fixed_random5_drop"],label="Fixed random-5")
    ax.axhline(0,linewidth=1)
    ax.set_xticks(x,fplot["case_id"],rotation=35,ha="right")
    ax.set_ylabel("Drop in explained-class file score")
    ax.set_title("Faithfulness control: high-contribution vs low/random features")
    ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(figs/"q4_faithfulness_controls.png",dpi=180); plt.close(fig)

    analysis={
      "final_model_identity":{"transfer":"T1","alpha":0.25,"classifier":"STEP07 frozen H1_RF","features":26,
                              "aggregation":"mean window class scores -> argmax","model_sha256":sha256_file(MODEL)},
      "selected_cases":case_inv.to_dict(orient="records"),
      "migration_process":{
        "median_relative_mean_gap_reduction":float(np.nanmedian(mig["relative_reduction"])),
        "mean_before_gap":float(mig["source_standardized_mean_gap_before"].mean()),
        "mean_after_gap":float(mig["source_standardized_mean_gap_after"].mean()),
        "warning":"marginal gap reduction describes T1 input alignment only; it is not target diagnostic accuracy"
      },
      "faithfulness":{"valid":faith_valid,"case_pass_rate":faith_rate,
                      "median_top5_drop":float(faith["top5_drop"].median()),
                      "median_low5_drop":float(faith["low5_drop"].median()),
                      "median_random_null_drop":float(faith["random_null_median"].median())},
      "stability":{"valid":stability_valid,"case_pass_rate":stability_rate,
                   "median_case_spearman":float(stab["median_adjacent_spearman"].median()),
                   "median_case_top5_jaccard":float(stab["median_adjacent_top5_jaccard"].median())},
      "mechanism_consistency":{"valid":mechanism_valid,
                               "source_representative_mean_top5_proxy_share":mech_proxy_mean,
                               "source_representative_median_prototype_direction_alignment":mech_align_median,
                               "warning":"feature-direction consistency is not causal proof; target exact fault frequency is not asserted"},
      "sanity_check":{"valid":sanity_valid,
                      "median_top5_null_percentile":float(faith["top5_null_percentile"].median()),
                      "fraction_cases_percentile_ge_0_75":float((faith["top5_null_percentile"]>=0.75).mean())},
      "validation_type_count":validation_type_count,
      "contradiction_case_count":int(len(contradictions)),
      "target_truth":"unknown",
      "target_accuracy_reported":False,
      "causal_claim_made":False
    }
    (art/"problem4_analysis.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding="utf-8")

    checks={
      "gate_bound_to_11A":True,
      "final_09_10_model_identity_verified":True,
      "model_sha_matches_frozen":sha256_file(MODEL)==cfg["final_transfer_bundle_sha256"],
      "target_10A_predictions_exactly_reproduced":bool((repro["pred_label_recomputed"]==repro["pred_label_10A"]).all() and repro["max_score_abs_diff"].max()<=1e-12),
      "target_truth_remains_unknown":True,
      "source_known_label_cases_present":bool((case_inv["domain"]=="source").sum()>=4),
      "target_prediction_cases_present":bool((case_inv["domain"]=="target").sum()>=4),
      "faithfulness_validation_executed":len(faith)==len(case_inv),
      "stability_validation_executed":len(stab)==len(case_inv),
      "mechanism_consistency_executed":len(mech)==len(case_inv),
      "at_least_two_validation_types_effective":validation_type_count>=2,
      "randomized_sanity_check_executed_and_effective":sanity_valid,
      "failure_contradiction_cases_retained":len(contradictions)>0,
      "no_causal_claim":True,
      "no_target_accuracy":True,
      "word_not_edited":True
    }
    passed=all(checks.values())
    validation={
      "schema_version":"11A-validation-1.0","step_id":STEP,"run_id":run_id,
      "status":"passed" if passed else "failed","passed":passed,"checks":checks,
      "evidence":{
        "model":f"{MODEL.relative_to(ROOT)} sha256={sha256_file(MODEL)}",
        "10A_reproduction":f"max score diff={repro['max_score_abs_diff'].max():.3e}; 16/16 labels match",
        "case_inventory":f"outputs/runs/{run_id}/artifacts/case_inventory.csv",
        "faithfulness":f"pass_rate={faith_rate:.3f}; median top5 drop={faith['top5_drop'].median():.6f}; median low5 drop={faith['low5_drop'].median():.6f}",
        "stability":f"pass_rate={stability_rate:.3f}; median Spearman={stab['median_adjacent_spearman'].median():.3f}; median Jaccard={stab['median_adjacent_top5_jaccard'].median():.3f}",
        "mechanism":f"source representative proxy share={mech_proxy_mean:.3f}; direction alignment median={mech_align_median:.3f}",
        "sanity":f"median top5 random-null percentile={faith['top5_null_percentile'].median():.3f}",
        "contradictions":f"{len(contradictions)} retained in failure_and_contradiction_cases.csv"
      },
      "valid_validation_types":{"faithfulness":faith_valid,"stability":stability_valid,"mechanism_consistency":mechanism_valid},
      "target_accuracy_reported":False,"word_edit_performed":False
    }
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    pip_freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (mani/"pip_freeze.txt").write_text(pip_freeze,encoding="utf-8")
    envrt={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),
           "machine":platform.machine(),"cpu_count_logical":os.cpu_count(),"gpu":None,
           "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
           "sklearn":sklearn.__version__,"joblib":joblib.__version__,
           "git_commit":git_head(),"pip_freeze_sha256":sha256_file(mani/"pip_freeze.txt")}
    (mani/"environment_runtime.json").write_text(json.dumps(envrt,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze={
      "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":STEP,
      "freeze_package_id":f"FREEZE-11A-{run_id[-8:]}","status":"passed" if passed else "failed","run_id":run_id,
      "versions":{"code_version_id":"git:"+git_head(),"raw_data_version_id":"RAW-5a5dd129c91bfc64",
                  "input_derived_data_ids":["FREEZE-09A-26f99878","FREEZE-10A-4346ce31","FEAT-23d45f5649dcd5f1"],
                  "run_config_sha256":sha256_file(CFG),"environment_sha256":sha256_file(ENV)},
      "random_seed":SEED,
      "parameters":{"method":"file/window feature occlusion on actual final RF inputs","top_k":TOPK,
                    "random_null_subsets":RANDOM_NULL_N,"stability_spearman_threshold":0.30,
                    "stability_top5_jaccard_threshold":0.20,"faithfulness_case_pass_rate_floor":2/3,
                    "stability_case_pass_rate_floor":2/3},
      "real_results":[
        {"name":"selected_cases","value":case_inv["case_id"].tolist()},
        {"name":"faithfulness","value":analysis["faithfulness"]},
        {"name":"stability","value":analysis["stability"]},
        {"name":"mechanism_consistency","value":analysis["mechanism_consistency"]},
        {"name":"sanity_check","value":analysis["sanity_check"]},
        {"name":"migration_process","value":analysis["migration_process"]},
        {"name":"contradiction_case_count","value":int(len(contradictions))}
      ],
      "validation_evidence":[
        {"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},
        {"path":f"outputs/runs/{run_id}/artifacts/explanation_case_table.csv"},
        {"path":f"outputs/runs/{run_id}/artifacts/faithfulness_ablation.csv"},
        {"path":f"outputs/runs/{run_id}/artifacts/explanation_stability.csv"},
        {"path":f"outputs/runs/{run_id}/artifacts/mechanism_consistency.csv"},
        {"path":f"outputs/runs/{run_id}/artifacts/failure_and_contradiction_cases.csv"},
        {"path":f"outputs/runs/{run_id}/figures/q4_case_top5_contributions.png"},
        {"path":f"outputs/runs/{run_id}/figures/q4_faithfulness_controls.png"},
        {"path":f"outputs/runs/{run_id}/figures/q4_T1_feature_shift.png"}
      ],
      "anomalies_and_failures":[
        {"item":"target_truth","status":"unknown","effect":"target explanations cannot verify diagnosis correctness"},
        {"item":"exact_target_fault_frequency","status":"not_asserted","reason":"target lacks frozen equivalent bearing geometry/exact RPM interface"},
        {"item":"contradiction_cases","status":"retained","count":int(len(contradictions))}
      ],
      "b_handoff":{
        "required_data":["explanation_case_table.csv","file_feature_contributions.csv","migration_feature_shift.csv",
                         "faithfulness_ablation.csv","explanation_stability.csv","mechanism_consistency.csv",
                         "failure_and_contradiction_cases.csv","problem4_analysis.json"],
        "supported_conclusions":[
          "Explanations are computed on the actual frozen T1(alpha=0.25)->RF pipeline input and file aggregation.",
          "High-contribution feature occlusion is compared with low/random controls; random-subset sanity null is retained.",
          "Adjacent overlapping-window explanation stability is quantified and failures are retained.",
          "Mechanism proxy/prototype-direction agreement is descriptive, not causal; target truth remains unknown."
        ],
        "wording_limits":[
          "do not call feature contribution causal",
          "do not report A-P target accuracy/F1/Recall",
          "do not claim exact target BPFO/BPFI/BSF/FTF match without frozen geometry/RPM",
          "do not hide unstable or mechanism-inconsistent explanations"
        ],
        "approved_tables":["explanation_case_table.csv","faithfulness_ablation.csv","explanation_stability.csv","failure_and_contradiction_cases.csv"],
        "approved_figures":["q4_T1_feature_shift.png","q4_case_top5_contributions.png","q4_faithfulness_controls.png"]
      },
      "unresolved_issues":[
        "A-P ground truth remains unknown.",
        "Occlusion uses a source file-balanced median reference and therefore explains sensitivity relative to that reference, not a unique causal decomposition.",
        "Engineered-feature explanations do not localize raw waveform time samples unless a separate raw-signal mapping is introduced."
      ],
      "freeze":{"created_utc":created,"content_sha256":None,
                "invalidation_dependencies":["09-A final transfer bundle changes","10-A final A-P labels/aggregation changes","04-A 26-feature data changes","11-A blueprint/code changes"]}
    }
    tmp=json.loads(json.dumps(freeze)); freeze["freeze"]["content_sha256"]=canonical_hash(tmp)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol/11A/latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol/11A/validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol/11A/latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    manifest={"schema_version":"11A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":STEP,
              "status":"completed" if passed else "failed","created_utc":created,
              "binding":{**binding,"binding_digest":bd},
              "code":{"repository":"Mhhhh958/mathmodeling","commit":git_head(),"entrypoint":"scripts/step11a_interpretability.py","code_manifest":"protocol/11A/code_manifest.json"},
              "data":{"source_files":int(source["group_id"].nunique()),"source_windows":int(len(source)),
                      "target_files":int(target["group_id"].nunique()),"target_windows":int(len(target)),
                      "features":len(features),"target_truth":"unknown"},
              "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canonical_hash(files)},
              "duration_seconds":float(time.time()-t0)}
    (mani/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":passed,"run_id":run_id,"freeze_package_id":freeze["freeze_package_id"],
                      "freeze_content_sha256":freeze["freeze"]["content_sha256"],
                      "validation_types":validation["valid_validation_types"],
                      "faithfulness_pass_rate":faith_rate,"stability_pass_rate":stability_rate,
                      "sanity_valid":sanity_valid,"contradiction_count":len(contradictions)},ensure_ascii=False))
    if not passed: raise SystemExit(2)

if __name__=="__main__":
    main()
