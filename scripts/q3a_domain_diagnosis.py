from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
Q1C = ROOT / "outputs" / "q1c_feature_extraction"
Q2C = ROOT / "outputs" / "q2c_failure_driven_improvement"
RAW_AUDIT = ROOT / "outputs" / "raw_audit" / "raw_file_audit.csv"
SRC_CSV = Q1C / "q2_source_raw.csv"
TGT_CSV = Q1C / "q2_target_raw.csv"
Q3_INTERFACE = Q2C / "q3_interface.json"
MODEL_FILE = Q2C / "models" / "final_source_model.joblib"
SOURCE_OOF = Q2C / "final_method_oof_file_predictions.csv"
OUT = ROOT / "outputs" / "q3a_domain_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 20260916
LABELS = ["OR", "IR", "B", "N"]
LOADS = [0, 1, 2, 3]
EPS = 1e-12
FINAL_RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


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


def entropy_rows(P):
    P = np.clip(np.asarray(P, float), EPS, 1.0)
    return -np.sum(P * np.log(P), axis=1) / np.log(P.shape[1])


def aggregate_probabilities(df_meta, P, class_order, group_col="independent_object_id"):
    tmp = df_meta[["window_id", group_col, "class_label"]].copy().reset_index(drop=True)
    for lab in LABELS:
        tmp[f"p_{lab}"] = P[:, class_order.index(lab)]
    pcols = [f"p_{x}" for x in LABELS]
    grouped = tmp.groupby([group_col, "class_label"], as_index=False)[pcols].mean()
    cnt = tmp.groupby(group_col).size().rename("n_windows").reset_index()
    grouped = grouped.merge(cnt, on=group_col, how="left")
    arr = grouped[pcols].to_numpy(float)
    grouped["pred_label"] = [LABELS[i] for i in np.argmax(arr, axis=1)]
    sortedp = np.sort(arr, axis=1)
    grouped["confidence"] = sortedp[:, -1]
    grouped["margin"] = sortedp[:, -1] - sortedp[:, -2]
    grouped["entropy_norm"] = entropy_rows(arr)
    win_pred = np.array([LABELS[i] for i in np.argmax(tmp[pcols].to_numpy(float), axis=1)])
    tmp["pred_label_window"] = win_pred
    consistency = tmp.merge(grouped[[group_col, "pred_label"]], on=group_col, how="left")
    consistency["agree"] = consistency["pred_label_window"] == consistency["pred_label"]
    ctab = consistency.groupby(group_col)["agree"].mean().rename("window_consistency").reset_index()
    grouped = grouped.merge(ctab, on=group_col, how="left")
    return tmp, grouped


def metric_dict(y_true, y_pred):
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    d = {
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


def file_equal_weights(df):
    n = df.groupby("independent_object_id").size().to_dict()
    return np.array([1.0 / n[x] for x in df["independent_object_id"]], float)


def weighted_mean_std(X, w):
    w = np.asarray(w, float)
    w = w / np.sum(w)
    mu = np.sum(X * w[:, None], axis=0)
    var = np.sum(((X - mu) ** 2) * w[:, None], axis=0)
    return mu, np.sqrt(np.maximum(var, EPS))


def training_weights(df):
    file_n = df.groupby("independent_object_id").size().to_dict()
    meta = df[["independent_object_id", "class_label"]].drop_duplicates()
    class_files = meta.groupby("class_label")["independent_object_id"].nunique().to_dict()
    n_files = meta["independent_object_id"].nunique()
    return np.array([
        (1.0 / file_n[r.independent_object_id]) * (n_files / (len(LABELS) * class_files[r.class_label]))
        for r in df[["independent_object_id", "class_label"]].itertuples(index=False)
    ], float)


def build_frozen_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            **FINAL_RF_CONFIG,
            random_state=SEED,
            n_jobs=1,
        )),
    ])


def fit_frozen_rf(df, feats):
    pipe = build_frozen_rf()
    X = df[feats].to_numpy(float)
    y = df["class_label"].astype(str).to_numpy()
    pipe.fit(X, y, clf__sample_weight=training_weights(df))
    return pipe


def rbf_mmd2(X, Y):
    Z = np.vstack([X, Y])
    # Median heuristic on pairwise squared distances excluding diagonal.
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    vals = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
    gamma = 1.0 / max(2.0 * med, EPS)
    Kxx = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    Kyy = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    Kxy = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    # Biased empirical MMD^2: descriptive evidence only.
    mmd2 = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return mmd2, gamma, med


def context_summary(src, tgt, audit):
    src_m1 = audit[
        audit["relative_path"].astype(str).str.startswith("source_domain/cwru_48khz_de/")
        | audit["relative_path"].astype(str).str.startswith("source_domain/cwru_48khz_normal/")
    ].copy()
    tgt_raw = audit[audit["relative_path"].astype(str).str.startswith("target_domain/train_bearing_a_to_p/")].copy()
    amp_cols = ["td_rms", "td_peak_abs", "td_ptp", "env_rms"]
    sf = src.groupby("independent_object_id")[amp_cols].mean()
    tf = tgt.groupby("independent_object_id")[amp_cols].mean()
    return {
        "source": {
            "independent_files": int(src["independent_object_id"].nunique()),
            "windows": int(len(src)),
            "sampling_hz": sorted(map(int, src["fs_hz"].dropna().unique())),
            "rpm_min": float(src["rpm_value"].min()),
            "rpm_max": float(src["rpm_value"].max()),
            "channel_values": sorted(map(str, src["channel"].dropna().unique())),
            "sensor_position_values": sorted(map(str, src["sensor_position"].dropna().unique())),
            "duration_min_s_raw_audit": float(src_m1["duration_min_s"].min()),
            "duration_max_s_raw_audit": float(src_m1["duration_max_s"].max()),
            "amplitude_file_medians": {c: float(sf[c].median()) for c in amp_cols},
        },
        "target": {
            "independent_files": int(tgt["independent_object_id"].nunique()),
            "windows": int(len(tgt)),
            "sampling_hz": sorted(map(int, tgt["fs_hz"].dropna().unique())),
            "rpm_value_recorded": sorted(map(float, tgt["rpm_value"].dropna().unique())),
            "rpm_source": sorted(map(str, tgt["rpm_source"].dropna().unique())),
            "channel_values": sorted(map(str, tgt["channel"].dropna().unique())),
            "sensor_position_values": sorted(map(str, tgt["sensor_position"].dropna().unique())),
            "duration_min_s_raw_audit": float(tgt_raw["duration_min_s"].min()),
            "duration_max_s_raw_audit": float(tgt_raw["duration_max_s"].max()),
            "amplitude_file_medians": {c: float(tf[c].median()) for c in amp_cols},
        },
        "interpretation_boundaries": {
            "common_fault_mechanism": "Both domains concern local rolling-bearing states OR/IR/B/N; impacts, modulation, and envelope/statistical structure are transferable concepts.",
            "not_assumed_common": "Bearing geometry, exact sensor mounting/transfer path, exact target RPM per file, absolute amplitude scale, and source SKF6205 characteristic frequencies are not assumed equal in target.",
            "source_geometry_specific_features_on_target": "prohibited",
        },
    }


def feature_shift_tables(src, tgt, feats):
    # File means are the independent comparison unit.
    sf = src.groupby("independent_object_id")[feats].mean()
    tf = tgt.groupby("independent_object_id")[feats].mean()
    mu = sf.mean().to_numpy(float)
    sd = sf.std(ddof=1).to_numpy(float)
    sd_safe = np.where(sd > EPS, sd, 1.0)
    Zs = (sf.to_numpy(float) - mu) / sd_safe
    Zt = (tf.to_numpy(float) - mu) / sd_safe
    rows = []
    for j, f in enumerate(feats):
        s = sf[f].to_numpy(float); t = tf[f].to_numpy(float)
        ks = ks_2samp(s, t, alternative="two-sided", method="auto")
        rows.append({
            "feature": f,
            "source_file_mean": float(np.mean(s)),
            "source_file_std": float(np.std(s, ddof=1)),
            "target_file_mean": float(np.mean(t)),
            "target_file_std": float(np.std(t, ddof=1)),
            "standardized_mean_shift_source_sd": float((np.mean(t) - np.mean(s)) / max(np.std(s, ddof=1), EPS)),
            "abs_standardized_mean_shift": float(abs((np.mean(t) - np.mean(s)) / max(np.std(s, ddof=1), EPS))),
            "wasserstein_over_source_sd": float(wasserstein_distance(s, t) / max(np.std(s, ddof=1), EPS)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue_descriptive": float(ks.pvalue),
        })
    tab = pd.DataFrame(rows).sort_values("abs_standardized_mean_shift", ascending=False).reset_index(drop=True)
    mmd2, gamma, med = rbf_mmd2(Zs, Zt)
    return tab, sf, tf, Zs, Zt, {"mmd2_biased": mmd2, "rbf_gamma": gamma, "median_pairwise_sq_distance": med}


def pca_and_domain_classifier(sf, tf, feats):
    Xs = sf[feats].to_numpy(float); Xt = tf[feats].to_numpy(float)
    scaler = StandardScaler().fit(Xs)
    X = np.vstack([scaler.transform(Xs), scaler.transform(Xt)])
    domain = np.array([0] * len(Xs) + [1] * len(Xt))
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    C = pca.transform(X)
    ids = list(sf.index.astype(str)) + list(tf.index.astype(str))
    dom_names = ["source"] * len(Xs) + ["target"] * len(Xt)
    ptab = pd.DataFrame({"independent_object_id": ids, "domain": dom_names, "pc1": C[:, 0], "pc2": C[:, 1]})
    fig, ax = plt.subplots(figsize=(7, 5))
    for d in ["source", "target"]:
        q = ptab[ptab["domain"] == d]
        ax.scatter(q["pc1"], q["pc2"], label=d, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Source vs target file-level 28-feature PCA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "source_target_pca.png", dpi=180)
    plt.close(fig)

    # File-level domain separability; descriptive, not a fault metric.
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=SEED))])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    aucs = cross_val_score(pipe, np.vstack([Xs, Xt]), domain, scoring="roc_auc", cv=cv)
    return ptab, {
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "domain_classifier_file_level_auc_mean": float(np.mean(aucs)),
        "domain_classifier_file_level_auc_std": float(np.std(aucs, ddof=1)),
        "domain_classifier_note": "Domain labels only; no fault labels from target are used.",
    }


def no_transfer_target(bundle, tgt, feats):
    pipe = bundle["pipeline"]
    X = tgt[feats].to_numpy(float)
    P = pipe.predict_proba(X)
    cls = list(pipe.named_steps["clf"].classes_)
    win, files = aggregate_probabilities(tgt, P, cls)
    files["target_id"] = files["independent_object_id"].map(lambda x: Path(str(x)).stem)
    files["truth_status"] = "unknown"
    cols = ["target_id", "independent_object_id", "truth_status", "n_windows"] + [f"p_{x}" for x in LABELS] + ["pred_label", "confidence", "margin", "entropy_norm", "window_consistency"]
    return win, files[cols].sort_values("target_id").reset_index(drop=True)


def classifier_output_shift(source_oof, target_files):
    pcols = [f"p_{x}" for x in LABELS]
    so = source_oof.copy()
    if "confidence_top1" in so.columns:
        sconf = so["confidence_top1"].to_numpy(float)
    else:
        sconf = so[pcols].max(axis=1).to_numpy(float)
    sent = entropy_rows(so[pcols].to_numpy(float))
    return {
        "source_oof_file_count": int(len(so)),
        "target_file_count": int(len(target_files)),
        "source_oof_confidence_median": float(np.median(sconf)),
        "target_no_transfer_confidence_median": float(target_files["confidence"].median()),
        "source_oof_entropy_median": float(np.median(sent)),
        "target_no_transfer_entropy_median": float(target_files["entropy_norm"].median()),
        "target_prediction_counts": {k: int(v) for k, v in target_files["pred_label"].value_counts().to_dict().items()},
        "target_mean_window_consistency": float(target_files["window_consistency"].mean()),
        "warning": "Target predictions are model outputs only; target truth is unavailable and no target accuracy is computed.",
    }


def class_structure(src, tgt, feats, target_pred_files):
    sf = src.groupby(["independent_object_id", "class_label"], as_index=False)[feats].mean()
    tf = tgt.groupby("independent_object_id", as_index=False)[feats].mean()
    mu = sf[feats].mean().to_numpy(float)
    sd = sf[feats].std(ddof=1).to_numpy(float)
    sd = np.where(sd > EPS, sd, 1.0)
    Zs = (sf[feats].to_numpy(float) - mu) / sd
    Zt = (tf[feats].to_numpy(float) - mu) / sd
    centroids = {}
    radii = {}
    for lab in LABELS:
        z = Zs[sf["class_label"].to_numpy() == lab]
        c = z.mean(axis=0)
        centroids[lab] = c
        dist = np.linalg.norm(z - c, axis=1) / math.sqrt(len(feats))
        radii[lab] = {"median": float(np.median(dist)), "p95": float(np.percentile(dist, 95))}
    rows = []
    pred_map = dict(zip(target_pred_files["independent_object_id"], target_pred_files["pred_label"]))
    for i, oid in enumerate(tf["independent_object_id"].astype(str)):
        ds = {lab: float(np.linalg.norm(Zt[i] - centroids[lab]) / math.sqrt(len(feats))) for lab in LABELS}
        order = sorted(ds, key=ds.get)
        nearest = order[0]; second = order[1]
        rows.append({
            "independent_object_id": oid,
            "target_id": Path(oid).stem,
            **{f"distance_to_{lab}": ds[lab] for lab in LABELS},
            "nearest_source_class_centroid": nearest,
            "second_nearest_source_class_centroid": second,
            "centroid_distance_margin_second_minus_first": float(ds[second] - ds[nearest]),
            "nearest_distance_over_source_class_p95_radius": float(ds[nearest] / max(radii[nearest]["p95"], EPS)),
            "no_transfer_pred_label": pred_map[oid],
            "model_centroid_agree": bool(pred_map[oid] == nearest),
            "truth_status": "unknown",
        })
    return pd.DataFrame(rows).sort_values("target_id"), radii


def moment_align_target_to_source(train_df, target_df, feats):
    # Transductive, unsupervised. No class labels are read from target_df here.
    Xs = train_df[feats].to_numpy(float)
    Xt = target_df[feats].to_numpy(float)
    mus, sds = weighted_mean_std(Xs, file_equal_weights(train_df))
    mut, sdt = weighted_mean_std(Xt, file_equal_weights(target_df))
    Xa = ((Xt - mut) / np.where(sdt > EPS, sdt, 1.0)) * sds + mus
    return Xa


def simulated_target_experiment(src, feats):
    rows = []
    details = []
    for held in LOADS:
        train = src[src["load_hp"] != held].copy()
        sim = src[src["load_hp"] == held].copy()
        pipe = fit_frozen_rf(train, feats)
        cls = list(pipe.named_steps["clf"].classes_)

        # A0 no transfer: labels are not used until after predictions are aggregated.
        P0 = pipe.predict_proba(sim[feats].to_numpy(float))
        _, F0 = aggregate_probabilities(sim, P0, cls)
        m0 = metric_dict(F0["class_label"], F0["pred_label"])
        rows.append({"simulated_target_load": held, "method": "A0_no_transfer", "adaptation_uses_target_labels": False, **m0})

        # A1 simple moment alignment: target labels are deliberately not passed to adaptation function.
        Xa = moment_align_target_to_source(train, sim.drop(columns=["class_label"]).assign(class_label="HIDDEN"), feats)
        P1 = pipe.predict_proba(Xa)
        # aggregation needs metadata and hidden truth only for post-hoc evaluation; prediction does not consume it.
        _, F1 = aggregate_probabilities(sim, P1, cls)
        m1 = metric_dict(F1["class_label"], F1["pred_label"])
        rows.append({"simulated_target_load": held, "method": "A1_unlabeled_moment_alignment", "adaptation_uses_target_labels": False, **m1})

        d0 = F0[["independent_object_id", "class_label", "pred_label", "confidence", "margin"]].rename(columns={"pred_label": "pred_no_transfer", "confidence": "confidence_no_transfer", "margin": "margin_no_transfer"})
        d1 = F1[["independent_object_id", "pred_label", "confidence", "margin"]].rename(columns={"pred_label": "pred_moment_align", "confidence": "confidence_moment_align", "margin": "margin_moment_align"})
        q = d0.merge(d1, on="independent_object_id", how="inner")
        q["simulated_target_load"] = held
        details.append(q)
    tab = pd.DataFrame(rows)
    det = pd.concat(details, ignore_index=True)
    summary = tab.groupby("method", as_index=False).agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        min_macro_f1=("macro_f1", "min"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_min_class_recall=("min_class_recall", "mean"),
    )
    base = summary[summary["method"] == "A0_no_transfer"].iloc[0]
    summary["mean_macro_f1_gain_vs_no_transfer"] = summary["mean_macro_f1"] - float(base["mean_macro_f1"])
    return tab, summary, det


def main():
    interface = json.loads(Q3_INTERFACE.read_text(encoding="utf-8"))
    feats = list(interface["feature_columns"])
    if len(feats) != 28:
        raise RuntimeError(f"Expected frozen 28 features, got {len(feats)}")
    src = pd.read_csv(SRC_CSV)
    tgt = pd.read_csv(TGT_CSV)
    audit = pd.read_csv(RAW_AUDIT)
    source_oof = pd.read_csv(SOURCE_OOF)
    bundle = joblib.load(MODEL_FILE)
    if list(bundle["feature_columns"]) != feats:
        raise RuntimeError("STEP07 model feature interface changed")
    if len(src) != 400 or src["independent_object_id"].nunique() != 56:
        raise RuntimeError("Source STEP04 inventory changed")
    if len(tgt) != 240 or tgt["independent_object_id"].nunique() != 16:
        raise RuntimeError("Target STEP04 inventory changed")
    if set(tgt["class_label"].astype(str).unique()) != {"UNKNOWN"} or set(tgt["truth_status"].astype(str).unique()) != {"unknown"}:
        raise RuntimeError("Target truth boundary changed")

    # 1) Operating/raw context difference.
    context = context_summary(src, tgt, audit)
    (OUT / "operating_context_comparison.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) Feature-distribution difference at independent-file level.
    shift, sf, tf, Zs, Zt, mmd = feature_shift_tables(src, tgt, feats)
    shift.to_csv(OUT / "feature_shift_file_level.csv", index=False, encoding="utf-8-sig")
    mmd["comparison_unit"] = "independent-file mean feature vector"
    mmd["note"] = "Descriptive evidence only; MMD is not a target-accuracy estimate."
    (OUT / "mmd_summary.json").write_text(json.dumps(mmd, ensure_ascii=False, indent=2), encoding="utf-8")

    ptab, pca_domain = pca_and_domain_classifier(sf, tf, feats)
    ptab.to_csv(OUT / "pca_file_coordinates.csv", index=False, encoding="utf-8-sig")
    (OUT / "pca_domain_separation.json").write_text(json.dumps(pca_domain, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Frozen source model -> target A-P, no adaptation.
    tgt_win, tgt_files = no_transfer_target(bundle, tgt, feats)
    tgt_win.to_csv(OUT / "target_A_P_no_transfer_window_predictions.csv", index=False, encoding="utf-8-sig")
    tgt_files.to_csv(OUT / "target_A_P_no_transfer_file_predictions.csv", index=False, encoding="utf-8-sig")

    output_shift = classifier_output_shift(source_oof, tgt_files)
    (OUT / "classifier_output_shift.json").write_text(json.dumps(output_shift, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Class-conditional structure without target labels.
    cstruct, source_radii = class_structure(src, tgt, feats, tgt_files)
    cstruct.to_csv(OUT / "target_class_structure_diagnostic.csv", index=False, encoding="utf-8-sig")
    (OUT / "source_class_radius_reference.json").write_text(json.dumps(source_radii, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Explicit unlabeled-target use ledger.
    ledger = {
        "target_files": [Path(x).stem for x in sorted(tgt["independent_object_id"].unique())],
        "target_windows": int(len(tgt)),
        "truth_status": "unknown",
        "used_now_for": [
            "frozen-source-model no-transfer prediction",
            "global/file-level feature distribution diagnostics",
            "domain-separation PCA/MMD/domain-classifier diagnostics",
            "source-class-centroid distance diagnostics",
            "classifier confidence/entropy/consistency diagnostics",
        ],
        "allowed_later_for_unsupervised_adaptation": [
            "target marginal moments/covariance (e.g. moment alignment or CORAL)",
            "domain-classifier probabilities for source instance reweighting",
            "unlabeled target model probabilities for stability diagnostics only",
        ],
        "forbidden": [
            "target accuracy/F1/recall calculation",
            "supervised model selection on A-P",
            "treating no-transfer predictions as truth",
            "injecting historical target predictions as labels",
            "using SKF6205 source geometry as target bearing geometry",
        ],
    }
    (OUT / "target_unlabeled_usage_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) Evaluable simulated-target scheme: hold each load out, hide labels during adaptation.
    sim_detail, sim_summary, sim_files = simulated_target_experiment(src, feats)
    sim_detail.to_csv(OUT / "simulated_target_by_load_metrics.csv", index=False, encoding="utf-8-sig")
    sim_summary.to_csv(OUT / "simulated_target_method_summary.csv", index=False, encoding="utf-8-sig")
    sim_files.to_csv(OUT / "simulated_target_file_predictions.csv", index=False, encoding="utf-8-sig")

    # 7) Candidate plan is frozen before a complex transfer model is trained on real target.
    promotion = {
        "primary_metric": "mean file-level Macro-F1 across four simulated held-out loads",
        "minimum_mean_macro_f1_gain_for_promotion": 0.02,
        "maximum_allowed_single_load_macro_f1_drop": 0.05,
        "class_recall_rule": "A candidate must not introduce a zero class recall on a simulated load where no-transfer recall for that class was nonzero.",
        "selection_data": "simulated source-held-out target only; never A-P truth",
        "stop_rule": "At most two additional candidate families after A1. Stop if neither satisfies the predeclared rule; retain no-transfer source model for target reporting.",
    }
    candidates = [
        {
            "candidate": "A1_unlabeled_moment_alignment",
            "role": "simple transfer baseline",
            "problem_targeted": "large marginal location/scale shift, especially absolute amplitude and spectral/envelope statistics",
            "target_input_used": "all unlabeled target windows only for file-balanced feature means and standard deviations",
            "extra_assumption": "dominant shift is approximately feature-wise affine and not strongly class-conditional",
            "status_in_step08": "actually evaluated on simulated held-out loads; not promoted to A-P as a final method in STEP08",
        },
        {
            "candidate": "A2_CORAL",
            "role": "candidate 1",
            "problem_targeted": "cross-feature covariance shift not handled by independent moment matching",
            "target_input_used": "unlabeled target covariance only",
            "extra_assumption": "a global second-order linear alignment preserves class structure sufficiently",
            "status_in_step08": "design only; defer real target adaptation",
        },
        {
            "candidate": "A3_domain_classifier_instance_weighting",
            "role": "candidate 2",
            "problem_targeted": "only a subset of source files/windows may resemble the target distribution",
            "target_input_used": "unlabeled target domain membership to fit a domain classifier; target fault labels not used",
            "extra_assumption": "covariate-shift style reweighting is meaningful and source support overlaps target support",
            "status_in_step08": "design only; defer real target adaptation",
        },
    ]
    (OUT / "transfer_candidate_plan.json").write_text(json.dumps({"promotion_rule": promotion, "candidates": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary evidence.
    top_shift = shift.head(8)[["feature", "abs_standardized_mean_shift", "wasserstein_over_source_sd", "ks_statistic"]].to_dict(orient="records")
    sim_map = {r["method"]: float(r["mean_macro_f1"]) for r in sim_summary.to_dict(orient="records")}
    summary = {
        "status": "PASS",
        "source_files": 56,
        "target_files": 16,
        "target_truth_known": False,
        "no_transfer_target_prediction_count": int(len(tgt_files)),
        "no_transfer_target_prediction_distribution": output_shift["target_prediction_counts"],
        "top_feature_shifts": top_shift,
        "mmd2_biased_file_level": float(mmd["mmd2_biased"]),
        "domain_classifier_auc_mean_file_level": float(pca_domain["domain_classifier_file_level_auc_mean"]),
        "source_oof_confidence_median": float(output_shift["source_oof_confidence_median"]),
        "target_no_transfer_confidence_median": float(output_shift["target_no_transfer_confidence_median"]),
        "simulated_target_no_transfer_mean_macro_f1": sim_map.get("A0_no_transfer"),
        "simulated_target_moment_alignment_mean_macro_f1": sim_map.get("A1_unlabeled_moment_alignment"),
        "target_accuracy_reported": False,
        "complex_transfer_trained_on_real_target": False,
    }
    (OUT / "domain_diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "frozen_step07_model_loaded": MODEL_FILE.exists() and list(bundle["feature_columns"]) == feats,
        "source_56_files_400_windows": len(src) == 400 and src["independent_object_id"].nunique() == 56,
        "target_16_files_240_windows": len(tgt) == 240 and tgt["independent_object_id"].nunique() == 16,
        "target_truth_remains_unknown": set(tgt["class_label"].astype(str).unique()) == {"UNKNOWN"},
        "no_transfer_A_P_predictions_saved": len(tgt_files) == 16 and tgt_files["target_id"].nunique() == 16,
        "no_target_accuracy_computed": summary["target_accuracy_reported"] is False,
        "domain_difference_three_levels": (OUT / "feature_shift_file_level.csv").exists() and (OUT / "classifier_output_shift.json").exists() and (OUT / "target_class_structure_diagnostic.csv").exists(),
        "unlabeled_target_use_recorded": (OUT / "target_unlabeled_usage_ledger.json").exists(),
        "simulated_target_evaluable": len(sim_detail) == 8 and set(sim_detail["simulated_target_load"]) == set(LOADS),
        "simulated_adaptation_no_label_access": not bool(sim_detail["adaptation_uses_target_labels"].any()),
        "transfer_candidates_and_rule_frozen": (OUT / "transfer_candidate_plan.json").exists(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {"status": status, "checks": checks, "summary": summary}
    (OUT / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "step": "Q3A_STEP08",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start": git_sha(),
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "random_seed": SEED,
        "source_feature_sha256": sha256(SRC_CSV),
        "target_feature_sha256": sha256(TGT_CSV),
        "q3_interface_sha256": sha256(Q3_INTERFACE),
        "step07_model_sha256": sha256(MODEL_FILE),
        "feature_count": len(feats),
        "real_target_adaptation_trained": False,
        "simulated_target_adaptation": "A1 feature-wise moment alignment using unlabeled held-out-load target moments",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Q3A / STEP08 domain diagnosis and no-transfer baseline",
        f"status={status}",
        "target_truth_known=False",
        f"source_files={src['independent_object_id'].nunique()} target_files={tgt['independent_object_id'].nunique()}",
        f"mmd2_file_level={mmd['mmd2_biased']:.6f}",
        f"domain_classifier_auc={pca_domain['domain_classifier_file_level_auc_mean']:.6f}",
        f"source_oof_confidence_median={output_shift['source_oof_confidence_median']:.6f}",
        f"target_no_transfer_confidence_median={output_shift['target_no_transfer_confidence_median']:.6f}",
        f"target_prediction_counts={json.dumps(output_shift['target_prediction_counts'], ensure_ascii=False, sort_keys=True)}",
        f"sim_no_transfer_mean_macro_f1={sim_map.get('A0_no_transfer', float('nan')):.6f}",
        f"sim_moment_align_mean_macro_f1={sim_map.get('A1_unlabeled_moment_alignment', float('nan')):.6f}",
        "target_accuracy_reported=False",
        "complex_real_target_transfer_trained=False",
    ]
    (OUT / "result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
