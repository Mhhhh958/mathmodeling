from __future__ import annotations

from pathlib import Path
import hashlib, json, math, os, platform, subprocess, sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "outputs" / "q1c_feature_extraction"
SRC_CSV = IN_DIR / "q2_source_raw.csv"
INTERFACE_JSON = IN_DIR / "q2_interface.json"
OUT = ROOT / "outputs" / "q2b_minimal_baseline"
MODELS = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 20260916
LABELS = ["OR", "IR", "B", "N"]
C_GRID = [0.1, 1.0, 10.0]
EPS = 1e-12


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics(y_true, y_pred):
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    d = {
        "n_files": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "min_class_recall": float(np.min(rc)),
    }
    for lab, p, r, ff, s in zip(LABELS, pr, rc, f1, sup):
        d[f"precision_{lab}"] = float(p)
        d[f"recall_{lab}"] = float(r)
        d[f"f1_{lab}"] = float(ff)
        d[f"support_{lab}"] = int(s)
    return d


def file_table(df):
    cols = ["independent_object_id", "class_label", "load_hp"]
    g = df[cols].drop_duplicates().copy()
    if g["independent_object_id"].duplicated().any():
        raise RuntimeError("One independent_object_id maps to multiple metadata rows")
    return g.sort_values("independent_object_id").reset_index(drop=True)


def make_weights(train_df):
    file_counts = train_df.groupby("independent_object_id").size().to_dict()
    meta = train_df[["independent_object_id", "class_label"]].drop_duplicates()
    if meta["independent_object_id"].duplicated().any():
        raise RuntimeError("Group has multiple labels")
    class_file_counts = meta.groupby("class_label")["independent_object_id"].nunique().to_dict()
    n_files = meta["independent_object_id"].nunique()
    k = len(LABELS)
    w = []
    for _, r in train_df.iterrows():
        ni = file_counts[r["independent_object_id"]]
        nc = class_file_counts[r["class_label"]]
        w.append((1.0 / ni) * (n_files / (k * nc)))
    return np.asarray(w, float)


def build_pipe(C):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=float(C), penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED)),
    ])


def fit_pipe(train_df, feats, C):
    pipe = build_pipe(C)
    X = train_df[feats].to_numpy(float)
    y = train_df["class_label"].astype(str).to_numpy()
    w = make_weights(train_df)
    pipe.fit(X, y, clf__sample_weight=w)
    return pipe


def predict_windows(pipe, d, feats, outer_test_load, selected_C):
    X = d[feats].to_numpy(float)
    p = pipe.predict_proba(X)
    cls = list(pipe.named_steps["clf"].classes_)
    out = d[["window_id", "independent_object_id", "class_label", "load_hp"]].copy().reset_index(drop=True)
    out["outer_test_load"] = outer_test_load
    out["selected_C"] = selected_C
    for lab in LABELS:
        out[f"p_{lab}"] = p[:, cls.index(lab)]
    out["pred_label_window"] = out[[f"p_{x}" for x in LABELS]].idxmax(axis=1).str.replace("p_", "", regex=False)
    return out


def aggregate_files(win_df):
    pcols = [f"p_{x}" for x in LABELS]
    agg = win_df.groupby(["outer_test_load", "selected_C", "independent_object_id", "class_label", "load_hp"], as_index=False)[pcols].mean()
    cnt = win_df.groupby("independent_object_id").size().rename("n_windows").reset_index()
    agg = agg.merge(cnt, on="independent_object_id", how="left")
    agg["pred_label_file"] = agg[pcols].idxmax(axis=1).str.replace("p_", "", regex=False)
    arr = np.sort(agg[pcols].to_numpy(float), axis=1)
    agg["confidence_top1"] = np.max(agg[pcols].to_numpy(float), axis=1)
    agg["margin_top1_top2"] = arr[:, -1] - arr[:, -2]
    return agg


def smoke_test(df, feats):
    if len(feats) != 28:
        raise RuntimeError(f"Expected 28 common features, got {len(feats)}")
    if set(df["class_label"].unique()) != set(LABELS):
        raise RuntimeError(f"Bad labels: {sorted(df['class_label'].unique())}")
    reps = []
    for lab in LABELS:
        rid = df.loc[df["class_label"] == lab, "independent_object_id"].drop_duplicates().iloc[0]
        reps.append(rid)
    mini = df[df["independent_object_id"].isin(reps)].copy()
    if not np.isfinite(mini[feats].to_numpy(float)).all():
        raise RuntimeError("Smoke input has non-finite common feature")
    pipe = fit_pipe(mini, feats, 1.0)
    p = pipe.predict_proba(mini[feats].to_numpy(float))
    ok = p.shape == (len(mini), 4) and np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    result = {
        "status": "PASS" if ok else "FAIL",
        "representative_groups": reps,
        "rows": int(len(mini)),
        "feature_count": len(feats),
        "labels": LABELS,
        "probability_shape": list(p.shape),
        "probability_row_sum_max_abs_error": float(np.max(np.abs(p.sum(axis=1) - 1.0))),
    }
    (OUT / "smoke_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ok:
        raise RuntimeError("Smoke test failed")
    return result


def inner_select(df, feats, outer_test_load):
    dev_loads = [x for x in [0,1,2,3] if x != outer_test_load]
    rows = []
    for C in C_GRID:
        for val_load in dev_loads:
            tr = df[df["load_hp"].isin([x for x in dev_loads if x != val_load])].copy()
            va = df[df["load_hp"] == val_load].copy()
            tr_groups = set(tr["independent_object_id"])
            va_groups = set(va["independent_object_id"])
            if tr_groups & va_groups:
                raise RuntimeError("Inner group leakage")
            pipe = fit_pipe(tr, feats, C)
            wf = predict_windows(pipe, va, feats, outer_test_load, C)
            ff = aggregate_files(wf)
            m = metrics(ff["class_label"], ff["pred_label_file"])
            rows.append({"outer_test_load":outer_test_load,"C":C,"inner_val_load":val_load,**m})
    detail = pd.DataFrame(rows)
    s = detail.groupby(["outer_test_load","C"], as_index=False).agg(
        mean_macro_f1=("macro_f1","mean"),
        mean_balanced_accuracy=("balanced_accuracy","mean"),
        mean_min_class_recall=("min_class_recall","mean"),
    )
    # frozen tie order: macro F1 -> balanced accuracy -> min recall -> simpler (smaller C)
    s = s.sort_values(["mean_macro_f1","mean_balanced_accuracy","mean_min_class_recall","C"], ascending=[False,False,False,True]).reset_index(drop=True)
    return float(s.iloc[0]["C"]), detail, s


def manual_macro_f1(y_true, y_pred):
    vals = []
    for lab in LABELS:
        tp = sum((a == lab and b == lab) for a,b in zip(y_true,y_pred))
        fp = sum((a != lab and b == lab) for a,b in zip(y_true,y_pred))
        fn = sum((a == lab and b != lab) for a,b in zip(y_true,y_pred))
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn) if (tp+fn) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        vals.append(f)
    return float(sum(vals)/len(vals))


def main():
    interface = json.loads(INTERFACE_JSON.read_text(encoding="utf-8"))
    feats = list(interface["feature_columns"])
    forbidden = set(interface["forbidden_as_model_features"])
    if forbidden.intersection(feats):
        raise RuntimeError(f"Leakage feature present: {forbidden.intersection(feats)}")
    df = pd.read_csv(SRC_CSV)
    if len(df) != 400:
        raise RuntimeError(f"Expected 400 source windows, got {len(df)}")
    ft = file_table(df)
    counts = ft["class_label"].value_counts().to_dict()
    if len(ft) != 56 or counts != {"OR":28,"B":12,"IR":12,"N":4}:
        raise RuntimeError(f"Unexpected file inventory: n={len(ft)}, counts={counts}")
    if sorted(ft["load_hp"].unique().tolist()) != [0.0,1.0,2.0,3.0]:
        raise RuntimeError("Unexpected load set")
    load_class = ft.groupby(["load_hp","class_label"]).size().unstack(fill_value=0)
    for load in [0,1,2,3]:
        if not all(int(load_class.loc[load, x]) == v for x,v in {"OR":7,"IR":3,"B":3,"N":1}.items()):
            raise RuntimeError(f"Load {load} class coverage not frozen pattern")

    smoke = smoke_test(df, feats)

    # exact split manifests
    outer_rows, inner_rows = [], []
    for test_load in [0,1,2,3]:
        for _, r in ft.iterrows():
            outer_rows.append({"outer_test_load":test_load,"independent_object_id":r.independent_object_id,"class_label":r.class_label,"load_hp":r.load_hp,"role":"test" if r.load_hp==test_load else "development"})
        dev_loads = [x for x in [0,1,2,3] if x != test_load]
        for val_load in dev_loads:
            for _, r in ft[ft["load_hp"].isin(dev_loads)].iterrows():
                inner_rows.append({"outer_test_load":test_load,"inner_val_load":val_load,"independent_object_id":r.independent_object_id,"class_label":r.class_label,"load_hp":r.load_hp,"role":"validation" if r.load_hp==val_load else "train"})
    pd.DataFrame(outer_rows).to_csv(OUT/"outer_split_manifest.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(inner_rows).to_csv(OUT/"inner_split_manifest.csv",index=False,encoding="utf-8-sig")

    all_inner_detail=[]; all_inner_summary=[]; all_w=[]; all_f=[]; fold_metrics=[]; b0_rows=[]
    for test_load in [0,1,2,3]:
        dev = df[df["load_hp"] != test_load].copy()
        test = df[df["load_hp"] == test_load].copy()
        if set(dev["independent_object_id"]) & set(test["independent_object_id"]):
            raise RuntimeError("Outer group leakage")
        C, detail, summary = inner_select(df, feats, test_load)
        all_inner_detail.append(detail); all_inner_summary.append(summary)
        pipe = fit_pipe(dev, feats, C)
        joblib.dump({"pipeline":pipe,"feature_columns":feats,"labels":LABELS,"selected_C":C,"outer_test_load":test_load,"random_seed":SEED}, MODELS/f"logreg_outer_T{test_load}_C{C:g}.joblib")
        wf = predict_windows(pipe, test, feats, test_load, C)
        ff = aggregate_files(wf)
        all_w.append(wf); all_f.append(ff)
        m = metrics(ff["class_label"], ff["pred_label_file"])
        fold_metrics.append({"outer_test_load":test_load,"selected_C":C,**m})
        # frozen B0 comparator: development-file majority class
        dev_file = dev[["independent_object_id","class_label"]].drop_duplicates()
        majority = dev_file["class_label"].value_counts().idxmax()
        b0_pred = np.repeat(majority, len(ff))
        bm = metrics(ff["class_label"], b0_pred)
        b0_rows.append({"outer_test_load":test_load,"majority_label":majority,**bm})

    inner_detail=pd.concat(all_inner_detail,ignore_index=True); inner_summary=pd.concat(all_inner_summary,ignore_index=True)
    win=pd.concat(all_w,ignore_index=True); files=pd.concat(all_f,ignore_index=True)
    foldm=pd.DataFrame(fold_metrics); b0=pd.DataFrame(b0_rows)
    inner_detail.to_csv(OUT/"inner_cv_detail.csv",index=False,encoding="utf-8-sig")
    inner_summary.to_csv(OUT/"inner_cv_summary.csv",index=False,encoding="utf-8-sig")
    win.to_csv(OUT/"oof_window_predictions.csv",index=False,encoding="utf-8-sig")
    files.to_csv(OUT/"oof_file_predictions.csv",index=False,encoding="utf-8-sig")
    foldm.to_csv(OUT/"outer_fold_file_metrics.csv",index=False,encoding="utf-8-sig")
    b0.to_csv(OUT/"b0_majority_metrics.csv",index=False,encoding="utf-8-sig")

    pooled = metrics(files["class_label"], files["pred_label_file"])
    pooled_win = {
        "n_windows":int(len(win)),
        "macro_f1":float(f1_score(win["class_label"],win["pred_label_window"],labels=LABELS,average="macro",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(win["class_label"],win["pred_label_window"])),
        "accuracy":float(accuracy_score(win["class_label"],win["pred_label_window"])),
    }
    cm = pd.DataFrame(confusion_matrix(files["class_label"],files["pred_label_file"],labels=LABELS), index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
    cm.to_csv(OUT/"confusion_matrix_file.csv",encoding="utf-8-sig")
    cmw = pd.DataFrame(confusion_matrix(win["class_label"],win["pred_label_window"],labels=LABELS), index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
    cmw.to_csv(OUT/"confusion_matrix_window_aux.csv",encoding="utf-8-sig")

    # per-fold distribution summary
    dist = {k:float(v) for k,v in {
        "macro_f1_mean":foldm.macro_f1.mean(),"macro_f1_std":foldm.macro_f1.std(ddof=1),"macro_f1_min":foldm.macro_f1.min(),"macro_f1_max":foldm.macro_f1.max(),
        "balanced_accuracy_mean":foldm.balanced_accuracy.mean(),"balanced_accuracy_std":foldm.balanced_accuracy.std(ddof=1),"accuracy_mean":foldm.accuracy.mean(),"accuracy_std":foldm.accuracy.std(ddof=1)
    }.items()}
    summary = {"file_level_pooled_oof":pooled,"outer_fold_distribution":dist,"window_level_auxiliary":pooled_win}
    (OUT/"metrics_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    # misclassifications + failed loads
    mis = files[files["class_label"] != files["pred_label_file"]].copy().sort_values(["outer_test_load","class_label","independent_object_id"])
    mis.to_csv(OUT/"misclassified_files.csv",index=False,encoding="utf-8-sig")
    fail_load = foldm.sort_values(["macro_f1","balanced_accuracy"]).copy()
    fail_load.to_csv(OUT/"failure_cases_by_load.csv",index=False,encoding="utf-8-sig")

    # independent recalc from written CSV: manual pooled Macro-F1 + one file probability aggregation
    reread_f = pd.read_csv(OUT/"oof_file_predictions.csv")
    manual_f1 = manual_macro_f1(reread_f["class_label"].tolist(),reread_f["pred_label_file"].tolist())
    ref_f1 = float(pooled["macro_f1"])
    sample_id = "data/raw/source_domain/cwru_48khz_de/B007_0.mat"
    reread_w = pd.read_csv(OUT/"oof_window_predictions.csv")
    sw = reread_w[reread_w["independent_object_id"] == sample_id].copy()
    sf = reread_f[reread_f["independent_object_id"] == sample_id].iloc[0]
    rec_probs = {lab:float(sw[f"p_{lab}"].mean()) for lab in LABELS}
    rec_pred = max(rec_probs, key=rec_probs.get)
    stored_probs = {lab:float(sf[f"p_{lab}"]) for lab in LABELS}
    max_prob_diff = max(abs(rec_probs[k]-stored_probs[k]) for k in LABELS)
    recalc = {
        "manual_macro_f1":manual_f1,"sklearn_macro_f1":ref_f1,"macro_f1_abs_diff":abs(manual_f1-ref_f1),
        "sample_independent_object_id":sample_id,"sample_window_count":int(len(sw)),"recalc_mean_probabilities":rec_probs,"stored_file_probabilities":stored_probs,
        "max_probability_abs_diff":max_prob_diff,"recalc_pred_label":rec_pred,"stored_pred_label":str(sf["pred_label_file"]),
        "pass":bool(abs(manual_f1-ref_f1)<1e-12 and max_prob_diff<1e-12 and rec_pred==sf["pred_label_file"]),
    }
    (OUT/"recalculation_evidence.json").write_text(json.dumps(recalc,ensure_ascii=False,indent=2),encoding="utf-8")

    # acceptance checks
    selected = foldm[["outer_test_load","selected_C"]].to_dict("records")
    selected_Cs = [float(x["selected_C"]) for x in selected]
    b0_pooled_pred=[]; b0_true=[]
    for test_load in [0,1,2,3]:
        ff=files[files["outer_test_load"]==test_load]
        maj=b0.loc[b0["outer_test_load"]==test_load,"majority_label"].iloc[0]
        b0_true.extend(ff["class_label"].tolist()); b0_pooled_pred.extend([maj]*len(ff))
    b0_pooled_macro=float(f1_score(b0_true,b0_pooled_pred,labels=LABELS,average="macro",zero_division=0))
    failure_flags={
        "any_file_group_leakage":False,
        "pooled_macro_f1_not_above_b0":bool(pooled["macro_f1"] <= b0_pooled_macro),
        "any_pooled_class_recall_zero":bool(any(pooled[f"recall_{x}"]==0 for x in LABELS)),
    }
    validation = {
        "status":"PASS",
        "checks":{
            "smoke_test_pass":smoke["status"]=="PASS",
            "source_56_files_400_windows":len(ft)==56 and len(df)==400,
            "outer_groups_disjoint":True,
            "all_outer_tests_have_four_classes":all(int(r["n_files"])==14 for _,r in foldm.iterrows()),
            "training_boundary_pipeline":True,
            "no_feature_name_leakage":not bool(forbidden.intersection(feats)),
            "oof_files_exactly_56_once":len(files)==56 and files["independent_object_id"].nunique()==56,
            "oof_windows_exactly_400_once":len(win)==400 and win["window_id"].nunique()==400,
            "independent_recalc_pass":recalc["pass"],
            "model_files_saved":len(list(MODELS.glob("*.joblib")))==4,
        },
        "performance_failure_flags_from_step05":failure_flags,
        "selected_C_by_outer_fold":selected,
        "pooled_file_macro_f1":float(pooled["macro_f1"]),
        "b0_pooled_macro_f1":b0_pooled_macro,
        "note":"PASS means computational/protocol validation passed. Step05 performance failure flags are reported separately and are not hidden."
    }
    if not all(validation["checks"].values()):
        validation["status"]="FAIL"
    (OUT/"validation_report.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    run_manifest={
        "step":"Q2B_STEP06","generated_utc":datetime.now(timezone.utc).isoformat(),"git_sha_at_run_start":git_sha(),"python":sys.version,"platform":platform.platform(),"random_seed":SEED,
        "input_source_csv":str(SRC_CSV.relative_to(ROOT)),"input_source_sha256":sha256(SRC_CSV),"input_interface_json":str(INTERFACE_JSON.relative_to(ROOT)),"input_interface_sha256":sha256(INTERFACE_JSON),
        "feature_count":len(feats),"labels":LABELS,"outer_test_loads":[0,1,2,3],"C_grid":C_GRID,"model":"L2 LogisticRegression","imputer":"median fit on training only","scaler":"StandardScaler fit on training only","feature_selection":"none","class_imbalance":"file-normalized x class-file-balanced sample weight","aggregation":"mean window class probabilities per file","validation_status":validation["status"]
    }
    (OUT/"run_manifest.json").write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    lines=[
        "Q2B / STEP06 minimal viable baseline",
        f"status={validation['status']}",
        f"smoke={smoke['status']}",
        f"selected_Cs={selected_Cs}",
        f"file_pooled_macro_f1={pooled['macro_f1']:.6f}",
        f"file_pooled_balanced_accuracy={pooled['balanced_accuracy']:.6f}",
        f"file_pooled_accuracy={pooled['accuracy']:.6f}",
        "class_recalls="+",".join(f"{x}:{pooled[f'recall_{x}']:.6f}" for x in LABELS),
        f"window_aux_macro_f1={pooled_win['macro_f1']:.6f}",
        f"misclassified_files={len(mis)}",
        f"manual_recalc_pass={recalc['pass']}",
        f"performance_failure_flags={failure_flags}",
    ]
    (OUT/"result_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
    if validation["status"] != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
