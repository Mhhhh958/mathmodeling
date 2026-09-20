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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
STEP = "10-A"
EXPECTED_PASS = "09-B-V2026.09.20-cbbd4e50-通过"
FREEZE09 = "FREEZE-09A-26f99878"
RUN09 = "run_09-A_20260920T070725126712Z_6ef81c21_26f99878"
RUN07 = "run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb"
RUN04 = "run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0"
CLASSES = ["OR", "IR", "B", "N"]
EXPECTED_IDS = list("ABCDEFGHIJKLMNOP")
EPS = 1e-12

STATE = ROOT / "protocol" / "00C" / "process_state_card.json"
FREEZE09_PATH = ROOT / "protocol" / "09A" / "latest_freeze_package.json"
CFG = ROOT / "protocol" / "10A" / "run_config.json"
PROTO = ROOT / "protocol" / "10A" / "inference_protocol.json"
CODE_MANIFEST = ROOT / "protocol" / "10A" / "code_manifest.json"
ENV_BASE = ROOT / "protocol" / "00B" / "environment_manifest.json"

B09 = ROOT / "outputs" / "runs" / RUN09
B07 = ROOT / "outputs" / "runs" / RUN07 / "artifacts"
B04 = ROOT / "outputs" / "runs" / RUN04 / "artifacts"

MODEL = B09 / "models" / "final_transfer_bundle.joblib"
STEP09_FILE = B09 / "artifacts" / "A_P_final_transfer_file_predictions.csv"
STEP09_WIN = B09 / "artifacts" / "A_P_final_transfer_window_predictions.csv"
A0_FILE = B09 / "artifacts" / "A_P_A0_reproduced.csv"
STEP09_STAB = B09 / "artifacts" / "stability_summary.json"

XT_PATH = B04 / "target_interface" / "X_target_common.csv"
MT_PATH = B04 / "target_interface" / "target_window_metadata.csv"
XS_PATH = B04 / "q2_interface" / "X_source_common.csv"
YS_PATH = B04 / "q2_interface" / "y_source_labels.csv"
GS_PATH = B04 / "q2_interface" / "groups_source.csv"
SOURCE_OOF_PATH = B07 / "H1_rf_oof_file_predictions.csv"

MECH_FEATURES = [
    "zero_cross_rate","spectral_centroid_hz","spectral_rms_hz","spectral_entropy",
    "spectral_flatness","dominant_frequency_hz","rolloff95_hz",
    "band_ratio_0_500","band_ratio_500_1500","band_ratio_1500_3000","band_ratio_3000_5500",
    "envelope_kurtosis","envelope_spectral_entropy"
]

def nowz():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")).hexdigest()

def git_head() -> str:
    return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"], text=True).strip()

def entropy_rows(scores: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(scores,float), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])

def scores_fixed(clf, X: np.ndarray) -> np.ndarray:
    raw = clf.predict_proba(np.asarray(X,float))
    pos = {str(c):i for i,c in enumerate(clf.classes_)}
    out = np.zeros((len(raw), len(CLASSES)), float)
    for j,c in enumerate(CLASSES):
        if c not in pos:
            raise RuntimeError(f"frozen classifier missing class {c}")
        out[:,j] = raw[:,pos[c]]
    return out

def t1_transform_from_bundle(bundle, Xt: np.ndarray, alpha: float) -> np.ndarray:
    art = bundle["adaptation_artifact"]
    if art.get("method") != "T1-SHRINK-MOMENT":
        raise RuntimeError("STEP09 final bundle is not T1-SHRINK-MOMENT")
    mus = np.asarray(art["source_mean"], float)
    sds = np.asarray(art["source_sd"], float)
    mut = np.asarray(art["target_mean"], float)
    sdt = np.asarray(art["target_sd"], float)
    full = ((Xt-mut) / np.where(sdt > EPS, sdt, 1.0)) * sds + mus
    return (1.0-float(alpha))*Xt + float(alpha)*full

def aggregate_mean(groups, scores):
    df = pd.DataFrame({"group_id":np.asarray(groups,dtype=object)})
    for j,c in enumerate(CLASSES):
        df[f"score_{c}"] = scores[:,j]
    g = df.groupby("group_id", sort=True)[[f"score_{c}" for c in CLASSES]].mean().reset_index()
    arr = g[[f"score_{c}" for c in CLASSES]].to_numpy(float)
    g["pred_label"] = np.array(CLASSES,dtype=object)[np.argmax(arr,axis=1)]
    order = np.argsort(arr,axis=1)
    g["top_model_score"] = arr[np.arange(len(arr)),order[:,-1]]
    g["score_margin"] = arr[np.arange(len(arr)),order[:,-1]] - arr[np.arange(len(arr)),order[:,-2]]
    g["score_entropy_norm"] = entropy_rows(arr)
    counts = df.groupby("group_id").size().rename("window_count").reset_index()
    g = g.merge(counts,on="group_id")
    wp = np.array(CLASSES,dtype=object)[np.argmax(scores,axis=1)]
    df["win_pred"] = wp
    df = df.merge(g[["group_id","pred_label"]],on="group_id")
    agr = (df["win_pred"] == df["pred_label"]).groupby(df["group_id"]).mean().rename("window_file_agreement").reset_index()
    return g.merge(agr,on="group_id"), df

def aggregate_median(groups, scores):
    df = pd.DataFrame({"group_id":np.asarray(groups,dtype=object)})
    for j,c in enumerate(CLASSES):
        df[f"score_{c}"] = scores[:,j]
    g = df.groupby("group_id", sort=True)[[f"score_{c}" for c in CLASSES]].median().reset_index()
    arr = g[[f"score_{c}" for c in CLASSES]].to_numpy(float)
    g["pred_label"] = np.array(CLASSES,dtype=object)[np.argmax(arr,axis=1)]
    return g[["group_id","pred_label"]]

def aggregate_majority(groups, scores):
    hard = np.argmax(scores,axis=1)
    rows=[]
    groups=np.asarray(groups,dtype=object)
    for gid in sorted(pd.unique(groups)):
        idx=np.where(groups==gid)[0]
        counts=np.bincount(hard[idx],minlength=len(CLASSES))
        mx=counts.max()
        tied=np.where(counts==mx)[0]
        if len(tied)==1:
            win=int(tied[0])
        else:
            means=scores[idx].mean(axis=0)
            win=int(tied[np.argmax(means[tied])])
        rows.append({"group_id":gid,"pred_label":CLASSES[win]})
    return pd.DataFrame(rows)

def target_ids_from_groups(groups):
    return [Path(str(g)).stem for g in groups]

def source_uncertainty_reference(oof: pd.DataFrame):
    score_cols=[f"score_{c}" for c in CLASSES]
    arr=oof[score_cols].to_numpy(float)
    order=np.argsort(arr,axis=1)
    o=oof.copy()
    o["top_model_score"]=arr[np.arange(len(arr)),order[:,-1]]
    o["score_margin"]=arr[np.arange(len(arr)),order[:,-1]]-arr[np.arange(len(arr)),order[:,-2]]
    o["score_entropy_norm"]=entropy_rows(arr)
    correct=o[o["pred_label"].astype(str)==o["class_label"].astype(str)].copy()
    if len(correct)<20:
        raise RuntimeError("insufficient correct source OOF files")
    return {
        "reference_population":"correct STEP07 source OOF independent files",
        "n_files":int(len(correct)),
        "top_score_q10":float(correct["top_model_score"].quantile(0.10)),
        "margin_q10":float(correct["score_margin"].quantile(0.10)),
        "entropy_q90":float(correct["score_entropy_norm"].quantile(0.90)),
        "interpretation":"source-derived descriptive thresholds only; not target correctness calibration"
    }

def file_means(df: pd.DataFrame, features):
    return df.groupby("group_id",sort=True)[features].mean()

def mechanism_evidence(source: pd.DataFrame, target_raw: pd.DataFrame, target_adapted: pd.DataFrame, pred_map, features):
    sf=file_means(source,features)
    tf0=file_means(target_raw,features)
    tf1=file_means(target_adapted,features)
    labels=source[["group_id","class_label"]].drop_duplicates().set_index("group_id").loc[sf.index]["class_label"].astype(str).to_numpy()

    mech=MECH_FEATURES
    idx=[features.index(f) for f in mech]
    S=sf.to_numpy(float)
    mu=S.mean(axis=0)
    sd=S.std(axis=0,ddof=1)
    sd=np.where(sd>EPS,sd,1.0)
    Zs=(S-mu)/sd
    Z0=(tf0.to_numpy(float)-mu)/sd
    Z1=(tf1.to_numpy(float)-mu)/sd

    cents={}; radii={}
    for c in CLASSES:
        z=Zs[labels==c][:,idx]
        cen=z.mean(axis=0)
        cents[c]=cen
        d=np.linalg.norm(z-cen,axis=1)/math.sqrt(len(idx))
        radii[c]=float(np.percentile(d,95))

    rows=[]
    for i,gid in enumerate(tf1.index.astype(str)):
        pred=pred_map[gid]
        d0={c:float(np.linalg.norm(Z0[i,idx]-cents[c])/math.sqrt(len(idx))) for c in CLASSES}
        d1={c:float(np.linalg.norm(Z1[i,idx]-cents[c])/math.sqrt(len(idx))) for c in CLASSES}
        nearest0=min(d0,key=d0.get); nearest1=min(d1,key=d1.get)
        diffs=np.abs(Z1[i,idx]-cents[pred])
        best=np.argsort(diffs)[:3]
        worst=np.argsort(diffs)[-3:][::-1]
        rows.append({
            "target_id":Path(gid).stem,
            "group_id":gid,
            "pred_label":pred,
            "mechanism_nearest_before":nearest0,
            "mechanism_nearest_after":nearest1,
            "mechanism_distance_to_pred_before":d0[pred],
            "mechanism_distance_to_pred_after":d1[pred],
            "mechanism_pred_distance_over_source_p95":float(d1[pred]/max(radii[pred],EPS)),
            "mechanism_similarity_index":float(1.0/(1.0+d1[pred])),
            "mechanism_prediction_agree":bool(nearest1==pred),
            "three_most_consistent_features":";".join(f"{mech[j]}(|z-diff|={diffs[j]:.3f})" for j in best),
            "three_largest_deviations":";".join(f"{mech[j]}(|z-diff|={diffs[j]:.3f})" for j in worst),
            "evidence_role":"explanatory source-prototype evidence only; not target truth"
        })
    return pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True), sf, tf0, tf1, Zs, Z0, Z1, labels, radii

def make_transfer_display(fig_path: Path, sf, tf0, tf1, labels, pred_map):
    S=sf.to_numpy(float)
    mu=S.mean(axis=0); sd=S.std(axis=0,ddof=1); sd=np.where(sd>EPS,sd,1.0)
    Zs=(S-mu)/sd; Z0=(tf0.to_numpy(float)-mu)/sd; Z1=(tf1.to_numpy(float)-mu)/sd
    pca=PCA(n_components=2,random_state=0).fit(Zs)
    Cs=pca.transform(Zs); C0=pca.transform(Z0); C1=pca.transform(Z1)
    src_idx=np.array([CLASSES.index(x) for x in labels],int)
    tids=[Path(str(x)).stem for x in tf0.index.astype(str)]
    tgt_idx=np.array([CLASSES.index(pred_map[str(g)]) for g in tf0.index.astype(str)],int)

    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2),sharex=True,sharey=True)
    scat=None
    for ax,C,title in zip(axes,[C0,C1],["Before T1 adaptation","After T1 adaptation (alpha=0.25)"]):
        ax.scatter(Cs[:,0],Cs[:,1],c=src_idx,marker="o",s=30,alpha=0.55,vmin=0,vmax=3)
        scat=ax.scatter(C[:,0],C[:,1],c=tgt_idx,marker="x",s=60,vmin=0,vmax=3)
        for j,tid in enumerate(tids):
            ax.annotate(tid,(C[j,0],C[j,1]),xytext=(3,2),textcoords="offset points",fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("PC1 (same source-fitted PCA)")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    cbar=fig.colorbar(scat,ax=axes.ravel().tolist(),ticks=[0,1,2,3],fraction=0.025,pad=0.02)
    cbar.ax.set_yticklabels(CLASSES)
    fig.suptitle("Source circles = true labels; target x = final predicted labels; clustering is descriptive only")
    fig.subplots_adjust(left=0.07,right=0.90,bottom=0.12,top=0.86,wspace=0.12)
    fig.savefig(fig_path,dpi=180,bbox_inches="tight")
    plt.close(fig)
    return [float(x) for x in pca.explained_variance_ratio_]

def make_mechanism_heatmap(fig_path: Path, mech_df: pd.DataFrame):
    vals=[]
    for _,r in mech_df.iterrows():
        vals.append([
            r["mechanism_distance_to_pred_after"] if r["pred_label"]==c else np.nan
            for c in CLASSES
        ])
    # More useful matrix is recomputed from saved per-class distances not available here; plot pred-distance + agreement strip instead.
    fig,ax=plt.subplots(figsize=(9.2,5.8))
    x=np.arange(len(mech_df))
    y=mech_df["mechanism_distance_to_pred_after"].to_numpy(float)
    ax.scatter(x,y,marker="o")
    ax.axhline(1.0,linestyle="--",linewidth=1)
    ax.set_xticks(x,mech_df["target_id"].tolist())
    ax.set_ylabel("Distance to predicted-class mechanism prototype")
    ax.set_xlabel("Target file")
    ax.set_title("Mechanism-side evidence (source-standardized prototype distance; explanatory only)")
    for i,r in mech_df.iterrows():
        ax.annotate(str(r["mechanism_nearest_after"]),(i,y[i]),xytext=(0,5),textcoords="offset points",ha="center",fontsize=7)
    ax.grid(axis="y",alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_path,dpi=180)
    plt.close(fig)

def main():
    t0=time.time()
    state=json.loads(STATE.read_text(encoding="utf-8"))
    freeze09=json.loads(FREEZE09_PATH.read_text(encoding="utf-8"))
    cfg=json.loads(CFG.read_text(encoding="utf-8"))
    proto=json.loads(PROTO.read_text(encoding="utf-8"))

    if state["CURRENT_ALLOWED_STEP"]!="10-A" or state["NEXT_ALLOWED"]!="10-A" or state["LAST_PASS_TOKEN"]!=EXPECTED_PASS or state.get("OPEN_P0")!=[]:
        raise RuntimeError("10-A gate is not currently open")
    if freeze09["freeze_package_id"]!=FREEZE09 or freeze09["status"]!="passed":
        raise RuntimeError("STEP09 freeze mismatch")
    if sha256_file(MODEL)!=proto["source_freeze"]["model_sha256"]:
        raise RuntimeError("STEP09 frozen transfer bundle SHA mismatch")

    bundle=joblib.load(MODEL)
    if bundle.get("method")!="T1" or str(bundle.get("setting"))!="0.25":
        raise RuntimeError(f"expected frozen T1 alpha=0.25, got {bundle.get('method')} {bundle.get('setting')}")
    if bundle.get("target_truth")!="unknown" or bundle.get("pseudo_labels_used") is not False:
        raise RuntimeError("target-truth or pseudo-label boundary changed")
    features=list(bundle["feature_names"])
    if len(features)!=26 or list(bundle["class_order"])!=CLASSES:
        raise RuntimeError("frozen feature/class interface changed")

    Xt=pd.read_csv(XT_PATH)
    Mt=pd.read_csv(MT_PATH)
    Xs=pd.read_csv(XS_PATH)
    Ys=pd.read_csv(YS_PATH)
    Gs=pd.read_csv(GS_PATH)
    source_oof=pd.read_csv(SOURCE_OOF_PATH)
    saved09=pd.read_csv(STEP09_FILE)
    saved09w=pd.read_csv(STEP09_WIN)
    a0=pd.read_csv(A0_FILE)

    if list(Xt.columns[1:])!=features or list(Xs.columns[1:])!=features:
        raise RuntimeError("feature order mismatch")
    if set(Mt["class_label"].astype(str).unique())!={"UNKNOWN_TRUTH"} or set(Mt["label_status"].astype(str).unique())!={"unknown_truth"}:
        raise RuntimeError("A-P truth boundary violated")
    if len(Xt)!=240 or Mt["group_id"].nunique()!=16:
        raise RuntimeError("target inventory mismatch")
    counts=Mt.groupby("group_id").size()
    if not (counts==15).all():
        raise RuntimeError("expected exactly 15 frozen windows per target file")
    target_ids=sorted([Path(str(x)).stem for x in counts.index])
    if target_ids!=EXPECTED_IDS:
        raise RuntimeError(f"A-P inventory mismatch: {target_ids}")

    target=Xt.merge(Mt[["window_id","group_id","relative_path"]],on="window_id",validate="one_to_one")
    source=Xs.merge(Ys,on="window_id",validate="one_to_one").merge(Gs,on="window_id",validate="one_to_one")

    Xt_arr=target[features].to_numpy(float)
    Xoff=t1_transform_from_bundle(bundle,Xt_arr,0.25)
    scores=scores_fixed(bundle["classifier"],Xoff)
    official,win_df=aggregate_mean(target["group_id"].to_numpy(),scores)
    official["target_id"]=official["group_id"].map(lambda x:Path(str(x)).stem)
    official=official.sort_values("target_id").reset_index(drop=True)

    # Reproduce STEP09 exactly.
    chk=official.merge(saved09,on="target_id",suffixes=("_10","_09"),validate="one_to_one")
    max_score_diff=0.0
    for c in CLASSES:
        max_score_diff=max(max_score_diff,float(np.max(np.abs(chk[f"score_{c}_10"]-chk[f"score_{c}_09"]))))
    labels_match=bool((chk["pred_label_10"].astype(str)==chk["pred_label_09"].astype(str)).all())
    if max_score_diff>1e-12 or not labels_match:
        raise RuntimeError(f"STEP10 official inference does not reproduce STEP09: maxdiff={max_score_diff} labels={labels_match}")

    # Save recomputed window and file outputs.
    wout=target[["window_id","group_id","relative_path"]].copy()
    for j,c in enumerate(CLASSES):
        wout[f"score_{c}"]=scores[:,j]
    wout["window_pred_label"]=np.array(CLASSES,dtype=object)[np.argmax(scores,axis=1)]
    wout["truth_status"]="unknown"

    # Cross-setting sensitivity (same frozen classifier only).
    variant_rows=[]
    variant_labels={}
    alpha_scores={}
    for alpha in proto["stability_checks"]["hyperparameter_sensitivity"]["alpha_values"]:
        Xa=t1_transform_from_bundle(bundle,Xt_arr,float(alpha))
        Sa=scores_fixed(bundle["classifier"],Xa)
        alpha_scores[float(alpha)]=Sa
        mean,_=aggregate_mean(target["group_id"].to_numpy(),Sa)
        med=aggregate_median(target["group_id"].to_numpy(),Sa)
        maj=aggregate_majority(target["group_id"].to_numpy(),Sa)
        for agg_name,tab in [("mean",mean),("median",med),("majority",maj)]:
            tab=tab.copy()
            tab["target_id"]=tab["group_id"].map(lambda x:Path(str(x)).stem)
            for _,r in tab.iterrows():
                variant_rows.append({"target_id":r["target_id"],"alpha":float(alpha),"aggregation":agg_name,"pred_label":str(r["pred_label"])})
                variant_labels[(r["target_id"],float(alpha),agg_name)]=str(r["pred_label"])
    variants=pd.DataFrame(variant_rows).sort_values(["target_id","alpha","aggregation"])
    official_map=dict(zip(official["target_id"],official["pred_label"]))
    vote_summary=[]
    for tid in EXPECTED_IDS:
        labs=variants.loc[variants["target_id"]==tid,"pred_label"].astype(str)
        agr=float((labs==official_map[tid]).mean())
        vc=labs.value_counts()
        vote_summary.append({"target_id":tid,"cross_setting_vote_agreement_rate":agr,"cross_setting_vote_label":str(vc.index[0]),"cross_setting_distinct_labels":int(vc.size)})
    vote_summary=pd.DataFrame(vote_summary)

    # Bootstrap window-composition stability using official frozen scores.
    bootstrap_rows=[]; boot_summary=[]
    groups=target["group_id"].to_numpy(dtype=object)
    for tid in EXPECTED_IDS:
        gid=f"target_domain/train_bearing_a_to_p/{tid}.mat"
        idx=np.where(groups==gid)[0]
        if len(idx)!=15:
            raise RuntimeError(f"{tid}: expected 15 windows")
        labs_all=[]
        seed_agrs=[]
        for seed in proto["stability_checks"]["random_resampling"]["seeds"]:
            rng=np.random.default_rng(int(seed))
            labs=[]
            for rep in range(int(proto["stability_checks"]["random_resampling"]["replicates_per_seed"])):
                pick=rng.choice(idx,size=len(idx),replace=True)
                m=scores[pick].mean(axis=0)
                lab=CLASSES[int(np.argmax(m))]
                labs.append(lab); labs_all.append(lab)
            agr=float(np.mean(np.array(labs,dtype=object)==official_map[tid]))
            mode=str(pd.Series(labs).value_counts().index[0])
            seed_agrs.append(agr)
            bootstrap_rows.append({"target_id":tid,"seed":int(seed),"replicates":len(labs),"seed_mode_label":mode,"official_label_agreement_rate":agr})
        boot_summary.append({
            "target_id":tid,
            "bootstrap_official_agreement_rate":float(np.mean(np.array(labs_all,dtype=object)==official_map[tid])),
            "bootstrap_min_seed_agreement_rate":float(min(seed_agrs)),
            "bootstrap_seed_modes_agree":bool(len(set(r["seed_mode_label"] for r in bootstrap_rows if r["target_id"]==tid))==1)
        })
    bootstrap=pd.DataFrame(bootstrap_rows)
    boot_summary=pd.DataFrame(boot_summary)

    # Source-derived score reference.
    ref=source_uncertainty_reference(source_oof)

    # Mechanism side evidence before/after adaptation.
    target_raw=target[["group_id",*features]].copy()
    target_ad=target[["group_id"]].copy()
    for j,f in enumerate(features):
        target_ad[f]=Xoff[:,j]
    pred_map_group=dict(zip(official["group_id"].astype(str),official["pred_label"].astype(str)))
    mech,sf,tf0,tf1,Zs,Z0,Z1,source_file_labels,radii=mechanism_evidence(source,target_raw,target_ad,pred_map_group,features)

    # No-transfer vs transfer.
    comp=a0[["target_id","pred_label","top_model_score","score_margin","score_entropy_norm","window_file_agreement"]].rename(columns={
        "pred_label":"A0_pred_label","top_model_score":"A0_top_model_score","score_margin":"A0_score_margin",
        "score_entropy_norm":"A0_score_entropy_norm","window_file_agreement":"A0_window_file_agreement"
    }).merge(
        official[["target_id","pred_label","top_model_score","score_margin","score_entropy_norm","window_file_agreement"]],
        on="target_id",validate="one_to_one")
    comp["label_changed_A0_to_T1"]=comp["A0_pred_label"].astype(str)!=comp["pred_label"].astype(str)

    # Assemble final answer table and uncertainty.
    table=official[["target_id","group_id","window_count",*[f"score_{c}" for c in CLASSES],"pred_label","top_model_score","score_margin","score_entropy_norm","window_file_agreement"]].copy()
    table=table.merge(comp[["target_id","A0_pred_label","A0_top_model_score","A0_score_margin","label_changed_A0_to_T1"]],on="target_id",validate="one_to_one")
    table=table.merge(vote_summary,on="target_id",validate="one_to_one")
    table=table.merge(boot_summary,on="target_id",validate="one_to_one")
    table=table.merge(mech,on=["target_id","group_id","pred_label"],validate="one_to_one")

    table["score_evidence_flag"]=(table["top_model_score"]<ref["top_score_q10"]) | (table["score_margin"]<ref["margin_q10"]) | (table["score_entropy_norm"]>ref["entropy_q90"])
    table["window_stability_evidence_flag"]=(table["window_file_agreement"]<0.80) | (table["cross_setting_vote_agreement_rate"]<0.80) | (table["bootstrap_official_agreement_rate"]<0.90)
    table["mechanism_evidence_flag"]=(~table["mechanism_prediction_agree"].astype(bool)) | (table["mechanism_pred_distance_over_source_p95"]>1.0)
    table["uncertainty_evidence_category_count"]=table[["score_evidence_flag","window_stability_evidence_flag","mechanism_evidence_flag"]].sum(axis=1)
    table["uncertainty_flag"]=np.select(
        [table["uncertainty_evidence_category_count"]>=3,table["uncertainty_evidence_category_count"]>=2,table["uncertainty_evidence_category_count"]==1],
        ["very_low_trust","low_trust","watch"],
        default="no_flag"
    )
    def why(r):
        parts=[]
        if r["score_evidence_flag"]:
            parts.append(f"score: top={r['top_model_score']:.3f}, margin={r['score_margin']:.3f}, entropy={r['score_entropy_norm']:.3f}; source refs q10/top={ref['top_score_q10']:.3f}, q10/margin={ref['margin_q10']:.3f}, q90/entropy={ref['entropy_q90']:.3f}")
        if r["window_stability_evidence_flag"]:
            parts.append(f"window/stability: window={r['window_file_agreement']:.3f}, setting-vote={r['cross_setting_vote_agreement_rate']:.3f}, bootstrap={r['bootstrap_official_agreement_rate']:.3f}")
        if r["mechanism_evidence_flag"]:
            parts.append(f"mechanism: nearest={r['mechanism_nearest_after']}, predicted={r['pred_label']}, distance/p95={r['mechanism_pred_distance_over_source_p95']:.3f}")
        return " | ".join(parts) if parts else "no frozen warning criterion triggered"
    table["uncertainty_reasons"]=table.apply(why,axis=1)
    table["score_interpretation"]="uncalibrated model scores; not true correctness probabilities"
    table["truth_status"]="unknown"
    table["target_accuracy_available"]=False
    table=table.sort_values("target_id").reset_index(drop=True)

    low=table[table["uncertainty_flag"].isin(["low_trust","very_low_trust"])].copy()

    # Run identity after inputs/protocol are bound.
    created=nowz()
    binding={
        "step_id":STEP,
        "code_commit":git_head(),
        "freeze09":FREEZE09,
        "model_sha256":sha256_file(MODEL),
        "run_config_sha256":sha256_file(CFG),
        "inference_protocol_sha256":sha256_file(PROTO),
        "environment_sha256":sha256_file(ENV_BASE)
    }
    bd=canonical_hash(binding)
    run_id=f"run_10-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs"/"runs"/run_id
    art=out/"artifacts"; figs=out/"figures"; mani=out/"manifests"; logs=out/"logs"
    for d in [art,figs,mani,logs]: d.mkdir(parents=True,exist_ok=False)

    # Persist outputs.
    wout.to_csv(art/"A_P_recomputed_window_predictions.csv",index=False,encoding="utf-8-sig")
    table.to_csv(art/"A_P_final_label_table.csv",index=False,encoding="utf-8-sig")
    table[["target_id","pred_label"]].to_csv(art/"A_P_final_labels_for_answer.csv",index=False,encoding="utf-8-sig")
    comp.to_csv(art/"A_P_no_transfer_vs_transfer.csv",index=False,encoding="utf-8-sig")
    variants.to_csv(art/"A_P_aggregation_alpha_sensitivity.csv",index=False,encoding="utf-8-sig")
    bootstrap.to_csv(art/"A_P_bootstrap_seed_stability.csv",index=False,encoding="utf-8-sig")
    low.to_csv(art/"low_confidence_cases.csv",index=False,encoding="utf-8-sig")
    mech.to_csv(art/"mechanism_side_evidence.csv",index=False,encoding="utf-8-sig")
    (art/"source_uncertainty_reference.json").write_text(json.dumps(ref,ensure_ascii=False,indent=2),encoding="utf-8")

    pca_var=make_transfer_display(figs/"transfer_display_before_after.png",sf,tf0,tf1,source_file_labels,pred_map_group)
    make_mechanism_heatmap(figs/"mechanism_prototype_side_evidence.png",mech)

    labels_dict={r.target_id:r.pred_label for r in table[["target_id","pred_label"]].itertuples(index=False)}
    analysis={
        "final_labels":labels_dict,
        "prediction_counts":{c:int((table["pred_label"]==c).sum()) for c in CLASSES},
        "changed_vs_no_transfer_files":table.loc[table["label_changed_A0_to_T1"],"target_id"].tolist(),
        "mean_window_file_agreement":float(table["window_file_agreement"].mean()),
        "min_cross_setting_vote_agreement":float(table["cross_setting_vote_agreement_rate"].min()),
        "min_bootstrap_official_agreement":float(table["bootstrap_official_agreement_rate"].min()),
        "low_confidence_files":low["target_id"].tolist(),
        "very_low_confidence_files":table.loc[table["uncertainty_flag"]=="very_low_trust","target_id"].tolist(),
        "mechanism_prediction_agreement_count":int(table["mechanism_prediction_agree"].sum()),
        "pca_explained_variance_ratio":pca_var,
        "target_truth":"unknown",
        "target_accuracy_reported":False,
        "score_warning":"uncalibrated scores are not correctness probabilities",
        "visualization_warning":"PCA clustering is descriptive only and does not prove target correctness",
        "problem3_result_analysis":"Final answer is the fixed mean-score aggregation of STEP09 T1(alpha=0.25) frozen outputs. Stability uses only fixed-model sensitivity/resampling; no retraining or target-label tuning occurred."
    }
    (art/"problem3_result_analysis.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding="utf-8")

    validation_checks={
        "gate_bound_to_10A":True,
        "freeze09_passed_and_bound":True,
        "step09_frozen_bundle_sha_matches":sha256_file(MODEL)==proto["source_freeze"]["model_sha256"],
        "bundle_method_T1_alpha025":bundle.get("method")=="T1" and str(bundle.get("setting"))=="0.25",
        "A_P_truth_unavailable":True,
        "exact_16_files_A_to_P_once_each":table["target_id"].tolist()==EXPECTED_IDS and table["target_id"].nunique()==16,
        "exact_15_windows_each":bool((table["window_count"]==15).all()),
        "label_mapping_consistent":set(table["pred_label"]).issubset(set(CLASSES)),
        "official_aggregation_fixed_before_run":proto["official_aggregation"]["rule_id"]=="MEAN_SCORE_ARGMAX_V1",
        "step09_file_results_exactly_reproduced":labels_match and max_score_diff<=1e-12,
        "no_manual_label_adjustment":True,
        "random_seed_window_bootstrap_checked":len(bootstrap)==16*3,
        "reasonable_aggregation_and_alpha_sensitivity_checked":len(variants)==16*9,
        "mechanism_side_evidence_for_all_16":len(mech)==16 and mech["target_id"].tolist()==EXPECTED_IDS,
        "low_confidence_requires_two_evidence_categories":bool((low["uncertainty_evidence_category_count"]>=2).all()),
        "uncalibrated_scores_not_called_correctness_probability":True,
        "same_PCA_projector_before_after":True,
        "source_true_vs_target_predicted_identity_explicit":True,
        "target_accuracy_not_computed":True,
        "word_edit_performed":False
    }
    passed=all(validation_checks.values())
    validation={
        "schema_version":"10A-validation-1.0","step_id":STEP,"run_id":run_id,
        "status":"passed" if passed else "failed","passed":passed,
        "checks":validation_checks,
        "evidence":{
            "final_table":f"outputs/runs/{run_id}/artifacts/A_P_final_label_table.csv",
            "window_recompute":f"outputs/runs/{run_id}/artifacts/A_P_recomputed_window_predictions.csv",
            "step09_reproduction":f"max class-score difference={max_score_diff:.3e}; labels_match={labels_match}",
            "stability":f"min 9-setting agreement={analysis['min_cross_setting_vote_agreement']:.3f}; min bootstrap agreement={analysis['min_bootstrap_official_agreement']:.3f}",
            "low_confidence":f"{len(low)} files: {','.join(low['target_id'].tolist()) if len(low) else 'none'}",
            "mechanism":f"mechanism prototype agrees with model for {analysis['mechanism_prediction_agreement_count']}/16 files",
            "visualization":f"outputs/runs/{run_id}/figures/transfer_display_before_after.png",
            "model_sha256":sha256_file(MODEL),
            "input_freeze":FREEZE09
        },
        "target_accuracy_reported":False,
        "word_edit_performed":False
    }
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    # Environment/manifests.
    pip_freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (mani/"pip_freeze.txt").write_text(pip_freeze,encoding="utf-8")
    env={
        "captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),
        "machine":platform.machine(),"cpu_count_logical":os.cpu_count(),"gpu":None,
        "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
        "sklearn":sklearn.__version__,"joblib":joblib.__version__,
        "git_commit":git_head(),"pip_freeze_sha256":sha256_file(mani/"pip_freeze.txt")
    }
    (mani/"environment_runtime.json").write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze={
        "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":STEP,
        "freeze_package_id":f"FREEZE-10A-{run_id[-8:]}","status":"passed" if passed else "failed","run_id":run_id,
        "versions":{
            "code_version_id":"git:"+git_head(),"raw_data_version_id":"RAW-5a5dd129c91bfc64",
            "input_derived_data_ids":[FREEZE09,"FEAT-23d45f5649dcd5f1"],
            "run_config_sha256":sha256_file(CFG),"environment_sha256":sha256_file(ENV_BASE)
        },
        "random_seed":[20260919,20260920,20260921],
        "parameters":{
            "frozen_transfer_method":"T1","alpha":0.25,"features":26,"class_order":CLASSES,
            "official_aggregation":"mean of 15 window class scores then argmax",
            "sensitivity_alpha":[0.20,0.25,0.30],
            "sensitivity_aggregations":["mean","median","majority"],
            "bootstrap_replicates_per_seed":200
        },
        "real_results":[
            {"name":"A_P_final_labels","value":labels_dict},
            {"name":"prediction_counts","value":analysis["prediction_counts"]},
            {"name":"changed_vs_no_transfer","value":analysis["changed_vs_no_transfer_files"]},
            {"name":"stability","value":{"min_cross_setting_vote_agreement":analysis["min_cross_setting_vote_agreement"],"min_bootstrap_agreement":analysis["min_bootstrap_official_agreement"]}},
            {"name":"low_confidence_files","value":analysis["low_confidence_files"]},
            {"name":"mechanism_prototype_agreement_count","value":analysis["mechanism_prediction_agreement_count"]}
        ],
        "validation_evidence":[
            {"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},
            {"path":f"outputs/runs/{run_id}/artifacts/A_P_final_label_table.csv"},
            {"path":f"outputs/runs/{run_id}/artifacts/A_P_aggregation_alpha_sensitivity.csv"},
            {"path":f"outputs/runs/{run_id}/artifacts/A_P_bootstrap_seed_stability.csv"},
            {"path":f"outputs/runs/{run_id}/artifacts/mechanism_side_evidence.csv"},
            {"path":f"outputs/runs/{run_id}/figures/transfer_display_before_after.png"}
        ],
        "anomalies_and_failures":[
            {"item":"target_truth","status":"unknown","effect":"no target accuracy/F1/Recall can be computed"},
            {"item":"legacy_q3c_script","status":"not_used","reason":"legacy script bound to old 28-feature/A0 interface; STEP10 uses new frozen-chain script"}
        ],
        "b_handoff":{
            "required_data":[
                "A_P_final_label_table.csv","A_P_final_labels_for_answer.csv","A_P_no_transfer_vs_transfer.csv",
                "A_P_aggregation_alpha_sensitivity.csv","A_P_bootstrap_seed_stability.csv",
                "mechanism_side_evidence.csv","low_confidence_cases.csv","problem3_result_analysis.json",
                "transfer_display_before_after.png","mechanism_prototype_side_evidence.png"
            ],
            "supported_conclusions":[
                "A-P final labels are predictions from the frozen STEP09 T1(alpha=0.25) model and fixed mean-score aggregation.",
                "16 files A-P are present exactly once and each uses 15 frozen windows.",
                "Uncertainty flags use pre-frozen score/window/mechanism evidence rules.",
                "Mechanism prototype evidence is explanatory only and target truth remains unknown."
            ],
            "wording_limits":[
                "do not report target accuracy/F1/Recall",
                "do not call model scores correctness probabilities",
                "do not use PCA clustering/prototype similarity as proof of correctness",
                "do not hide low-confidence files"
            ],
            "approved_tables":["A_P_final_label_table.csv","low_confidence_cases.csv","mechanism_side_evidence.csv"],
            "approved_figures":["transfer_display_before_after.png","mechanism_prototype_side_evidence.png"]
        },
        "unresolved_issues":[
            "A-P ground truth remains unknown.",
            "Prototype similarity and PCA geometry cannot verify diagnostic correctness."
        ],
        "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["09-A final transfer bundle changes","04-A target/source common feature data changes","10-A inference protocol or code changes"]}
    }
    tmp=json.loads(json.dumps(freeze,ensure_ascii=False))
    freeze["freeze"]["content_sha256"]=canonical_hash(tmp)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"10A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"10A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"10A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    manifest={
        "schema_version":"10A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":STEP,
        "status":"completed" if passed else "failed","created_utc":created,
        "binding":{**binding,"binding_digest":bd},
        "code":{"repository":"Mhhhh958/mathmodeling","commit":git_head(),"entrypoint":"scripts/step10a_final_labels_and_display.py","code_manifest":"protocol/10A/code_manifest.json"},
        "data":{"source_files":int(source["group_id"].nunique()),"source_windows":int(len(source)),"target_files":16,"target_windows":240,"features":26,"target_truth":"unknown"},
        "protocol":{"path":"protocol/10A/inference_protocol.json","sha256":sha256_file(PROTO)},
        "config":{"path":"protocol/10A/run_config.json","sha256":sha256_file(CFG)},
        "environment":{"manifest":"protocol/00B/environment_manifest.json","sha256":sha256_file(ENV_BASE),"runtime":"manifests/environment_runtime.json"},
        "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canonical_hash(files)},
        "duration_seconds":float(time.time()-t0)
    }
    (mani/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    print(json.dumps({
        "ok":passed,"run_id":run_id,"freeze_package_id":freeze["freeze_package_id"],
        "freeze_content_sha256":freeze["freeze"]["content_sha256"],
        "labels":labels_dict,"low_confidence_files":analysis["low_confidence_files"],
        "min_cross_setting_vote_agreement":analysis["min_cross_setting_vote_agreement"],
        "min_bootstrap_agreement":analysis["min_bootstrap_official_agreement"],
        "target_accuracy_reported":False
    },ensure_ascii=False))
    if not passed:
        raise SystemExit(2)

if __name__=="__main__":
    main()
