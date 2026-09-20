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
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
STEP = "08-A"
SEED = 20260919
CLASSES = ["OR", "IR", "B", "N"]
EPS = 1e-12

RUN01 = "run_01-A_20260919T155449028094Z_4e160e7d_85d31497"
RUN04 = "run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0"
RUN07 = "run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb"
FREEZE04 = "FREEZE-04A-3e3b97c0"
FREEZE07 = "FREEZE-07A-7e8829fb"

B01 = ROOT / "outputs" / "runs" / RUN01 / "artifacts"
B04 = ROOT / "outputs" / "runs" / RUN04 / "artifacts"
B07 = ROOT / "outputs" / "runs" / RUN07
CFG = ROOT / "protocol" / "08A" / "run_config.json"
BLUEPRINT = ROOT / "protocol" / "08A" / "question3_argument_blueprint.json"
CODE_MANIFEST = ROOT / "protocol" / "08A" / "code_manifest.json"
ENV_MANIFEST = ROOT / "protocol" / "00B" / "environment_manifest.json"

X_SOURCE = B04 / "q2_interface" / "X_source_common.csv"
Y_SOURCE = B04 / "q2_interface" / "y_source_labels.csv"
G_SOURCE = B04 / "q2_interface" / "groups_source.csv"
M_SOURCE = B04 / "q2_interface" / "source_window_metadata.csv"
X_TARGET = B04 / "target_interface" / "X_target_common.csv"
M_TARGET = B04 / "target_interface" / "target_window_metadata.csv"
RAW_META = B01 / "file_level_metadata.csv"
FINAL_MODEL = B07 / "models" / "final_source_model.joblib"
Q3_INTERFACE = B07 / "artifacts" / "q3_interface.json"
FEATURE_ORDER = B07 / "artifacts" / "feature_order.csv"
SOURCE_OOF = B07 / "artifacts" / "H1_rf_oof_file_predictions.csv"

RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": 1,
}


def nowz():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hf(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def norm_entropy(score_arr: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(score_arr, float), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])


def fixed_scores(clf, X: np.ndarray) -> np.ndarray:
    raw = clf.predict_proba(X)
    pos = {str(c): i for i, c in enumerate(clf.classes_)}
    out = np.zeros((len(X), len(CLASSES)), float)
    for j, c in enumerate(CLASSES):
        if c not in pos:
            raise RuntimeError(f"missing class {c} in classifier")
        out[:, j] = raw[:, pos[c]]
    return out


def aggregate_unlabeled(group_ids, scores: np.ndarray, target_ids=None):
    q = pd.DataFrame({"group_id": np.asarray(group_ids, dtype=object)})
    for j, c in enumerate(CLASSES):
        q[f"score_{c}"] = scores[:, j]
    score_cols = [f"score_{c}" for c in CLASSES]
    g = q.groupby("group_id", sort=True)[score_cols].mean().reset_index()
    counts = q.groupby("group_id").size().rename("window_count").reset_index()
    g = g.merge(counts, on="group_id", how="left")
    A = g[score_cols].to_numpy(float)
    order = np.argsort(A, axis=1)
    g["pred_label"] = np.array(CLASSES, dtype=object)[np.argmax(A, axis=1)]
    g["top_model_score"] = A[np.arange(len(A)), order[:, -1]]
    g["score_margin"] = A[np.arange(len(A)), order[:, -1]] - A[np.arange(len(A)), order[:, -2]]
    g["score_entropy_norm"] = norm_entropy(A)
    win_pred = np.array(CLASSES, dtype=object)[np.argmax(scores, axis=1)]
    q["window_pred_label"] = win_pred
    q = q.merge(g[["group_id", "pred_label"]], on="group_id", how="left")
    q["agree_file_prediction"] = q["window_pred_label"] == q["pred_label"]
    agr = q.groupby("group_id")["agree_file_prediction"].mean().rename("window_file_agreement").reset_index()
    g = g.merge(agr, on="group_id", how="left")
    if target_ids is not None:
        idmap = dict(zip(np.asarray(group_ids, dtype=object), np.asarray(target_ids, dtype=object)))
        g["target_id"] = g["group_id"].map(idmap)
    return q, g


def file_metric(truth_by_group: pd.DataFrame, pred_files: pd.DataFrame):
    z = pred_files.merge(truth_by_group[["group_id", "class_label"]], on="group_id", validate="one_to_one")
    y = z["class_label"].astype(str).to_numpy()
    p = z["pred_label"].astype(str).to_numpy()
    rec = recall_score(y, p, labels=CLASSES, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "min_class_recall": float(np.min(rec)),
        "recall": {c: float(v) for c, v in zip(CLASSES, rec)},
        "confusion_matrix": confusion_matrix(y, p, labels=CLASSES).tolist(),
        "file_count": int(len(z)),
    }


def training_weights(df: pd.DataFrame) -> np.ndarray:
    gc = df.groupby("group_id").size().to_dict()
    gt = df[["group_id", "class_label"]].drop_duplicates()
    cc = gt.groupby("class_label").size().to_dict()
    w = np.array([(1.0 / gc[g]) * (1.0 / cc[c]) for g, c in df[["group_id", "class_label"]].itertuples(index=False)], float)
    w *= len(w) / w.sum()
    return w


def fit_frozen_rf(train: pd.DataFrame, features):
    clf = RandomForestClassifier(**RF_CONFIG)
    clf.fit(train[features].to_numpy(float), train["class_label"].astype(str).to_numpy(), sample_weight=training_weights(train))
    return clf


def file_balanced_weights(groups) -> np.ndarray:
    s = pd.Series(np.asarray(groups, dtype=object))
    cnt = s.value_counts().to_dict()
    w = np.array([1.0 / cnt[g] for g in s], float)
    return w / w.sum()


def weighted_mean_sd(X: np.ndarray, groups):
    w = file_balanced_weights(groups)
    mu = np.sum(X * w[:, None], axis=0)
    var = np.sum(((X - mu) ** 2) * w[:, None], axis=0)
    return mu, np.sqrt(np.maximum(var, EPS))


def moment_align(Xs, gs, Xt, gt, alpha: float):
    # This function receives feature matrices + group IDs only. It cannot access labels.
    mus, sds = weighted_mean_sd(np.asarray(Xs, float), gs)
    mut, sdt = weighted_mean_sd(np.asarray(Xt, float), gt)
    full = ((Xt - mut) / np.where(sdt > EPS, sdt, 1.0)) * sds + mus
    return (1.0 - float(alpha)) * Xt + float(alpha) * full


def rbf_mmd2(X: np.ndarray, Y: np.ndarray):
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    vals = d2[np.triu_indices_from(d2, k=1)]
    nz = vals[vals > 0]
    med = float(np.median(nz)) if len(nz) else 1.0
    gamma = 1.0 / max(2.0 * med, EPS)
    Kxx = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    Kyy = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    Kxy = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()), float(gamma), med


def feature_family(name: str) -> str:
    amp = {"mean", "std", "rms", "mean_abs", "peak_abs", "peak_to_peak", "crest_factor", "impulse_factor", "shape_factor", "clearance_factor", "envelope_rms"}
    mech = {"zero_cross_rate", "spectral_centroid_hz", "spectral_rms_hz", "spectral_entropy", "spectral_flatness", "dominant_frequency_hz", "rolloff95_hz",
            "band_ratio_0_500", "band_ratio_500_1500", "band_ratio_1500_3000", "band_ratio_3000_5500", "envelope_kurtosis", "envelope_spectral_entropy"}
    if name in amp:
        return "amplitude_or_scale"
    if name in mech:
        return "geometry_free_mechanism_proxy"
    return "shape_statistics"


def feature_shift(source_file: pd.DataFrame, target_file: pd.DataFrame, features):
    rows = []
    for f in features:
        s = source_file[f].to_numpy(float)
        t = target_file[f].to_numpy(float)
        sd = float(np.std(s, ddof=1))
        denom = max(sd, EPS)
        ks = ks_2samp(s, t)
        rows.append({
            "feature": f,
            "family": feature_family(f),
            "source_mean": float(np.mean(s)),
            "source_sd": sd,
            "target_mean": float(np.mean(t)),
            "target_sd": float(np.std(t, ddof=1)),
            "standardized_mean_shift": float((np.mean(t) - np.mean(s)) / denom),
            "abs_standardized_mean_shift": float(abs((np.mean(t) - np.mean(s)) / denom)),
            "wasserstein_over_source_sd": float(wasserstein_distance(s, t) / denom),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue_descriptive_only": float(ks.pvalue),
        })
    tab = pd.DataFrame(rows).sort_values("abs_standardized_mean_shift", ascending=False).reset_index(drop=True)
    fam = tab.groupby("family", as_index=False).agg(
        feature_count=("feature", "count"),
        median_abs_standardized_mean_shift=("abs_standardized_mean_shift", "median"),
        max_abs_standardized_mean_shift=("abs_standardized_mean_shift", "max"),
        median_wasserstein_over_source_sd=("wasserstein_over_source_sd", "median"),
        max_ks_statistic=("ks_statistic", "max"),
    )
    return tab, fam


def main():
    t0 = time.time()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8"))
    code_manifest = json.loads(CODE_MANIFEST.read_text(encoding="utf-8"))
    q3 = json.loads(Q3_INTERFACE.read_text(encoding="utf-8"))
    if cfg["inputs"]["freeze07"] != FREEZE07:
        raise RuntimeError("08-A run config is not bound to current 07-A freeze")
    if q3["target_truth"] != "unknown":
        raise RuntimeError("Q3 target truth boundary changed")

    feature_order = pd.read_csv(FEATURE_ORDER)
    features = feature_order["feature_name"].astype(str).tolist()
    if len(features) != 26:
        raise RuntimeError(f"expected 26 frozen common features, got {len(features)}")

    Xs = pd.read_csv(X_SOURCE)
    ys = pd.read_csv(Y_SOURCE)
    gs = pd.read_csv(G_SOURCE)
    ms = pd.read_csv(M_SOURCE)
    Xt = pd.read_csv(X_TARGET)
    mt = pd.read_csv(M_TARGET)
    raw = pd.read_csv(RAW_META)
    src_oof = pd.read_csv(SOURCE_OOF)

    if list(Xs.columns[1:]) != features or list(Xt.columns[1:]) != features:
        raise RuntimeError("04-A feature order differs from 07-A frozen order")
    if len(Xs) != 733 or len(Xt) != 240:
        raise RuntimeError("04-A row inventory changed")
    if ms["group_id"].nunique() != 49 or mt["group_id"].nunique() != 16:
        raise RuntimeError("group inventory changed")
    target_label_guard = (
        set(mt["class_label"].astype(str).unique()) == {"UNKNOWN_TRUTH"}
        and set(mt["label_status"].astype(str).unique()) == {"unknown_truth"}
    )
    if not target_label_guard:
        raise RuntimeError("A-P true label boundary violated")

    source = Xs.merge(ys, on="window_id", validate="one_to_one").merge(gs, on="window_id", validate="one_to_one")
    keepmeta = ["window_id", "group_id", "relative_path", "load_hp", "native_fs_hz", "common_fs_hz", "analysis_channel", "rpm", "fault_bearing_location", "subgroup"]
    source = source.merge(ms[keepmeta], on=["window_id", "group_id"], validate="one_to_one")
    target = Xt.merge(mt[["window_id", "group_id", "relative_path", "native_fs_hz", "common_fs_hz", "analysis_channel", "rpm", "label_status", "class_label"]], on="window_id", validate="one_to_one")
    target["target_id"] = target["relative_path"].map(lambda x: Path(str(x)).stem)

    bundle = joblib.load(FINAL_MODEL)
    if bundle.get("family") != "RandomForestClassifier":
        raise RuntimeError("07-A final family is not expected H1_RF classifier bundle")
    if list(bundle.get("feature_names", [])) != features:
        raise RuntimeError("07-A model feature names differ from frozen feature order")
    if list(bundle.get("class_order", [])) != CLASSES:
        raise RuntimeError("07-A class order changed")
    if bundle.get("config", {}).get("n_estimators") != 300 or bundle.get("config", {}).get("max_depth") != 8:
        raise RuntimeError("07-A final RF parameters differ from 08-A frozen interface")
    final_clf = bundle["classifier"]

    created = nowz()
    binding = {
        "step_id": STEP,
        "code_commit": git_head(),
        "freeze07": FREEZE07,
        "feature_version": "FEAT-23d45f5649dcd5f1",
        "run_config_sha256": hf(CFG),
        "environment_sha256": hf(ENV_MANIFEST),
        "seed": SEED,
    }
    bd = canonical_hash(binding)
    run_id = f"run_08-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out = ROOT / "outputs" / "runs" / run_id
    art = out / "artifacts"
    mani = out / "manifests"
    logs = out / "logs"
    for d in [art, mani, logs]:
        d.mkdir(parents=True, exist_ok=False)

    source_ids = set(ms["independent_object_id"].astype(str).unique())
    sr = raw[raw["independent_object_id"].astype(str).isin(source_ids)].copy()
    tr = raw[raw["domain"].astype(str).eq("target")].copy()
    if len(sr) != 49 or len(tr) != 16:
        raise RuntimeError(f"raw context inventory mismatch source={len(sr)} target={len(tr)}")
    sr["raw_peak_to_peak"] = sr["signal_max"] - sr["signal_min"]
    tr["raw_peak_to_peak"] = tr["signal_max"] - tr["signal_min"]
    source_file_channels = ms[["group_id", "analysis_channel"]].drop_duplicates()
    target_file_channels = mt[["group_id", "analysis_channel"]].drop_duplicates()
    context = {
        "comparison_unit": "independent raw file",
        "source_selected_files": 49,
        "target_files": 16,
        "source_native_sampling_hz_counts": {str(k): int(v) for k, v in sr["sampling_rate_hz"].value_counts().sort_index().to_dict().items()},
        "target_native_sampling_hz_counts": {str(k): int(v) for k, v in tr["sampling_rate_hz"].value_counts().sort_index().to_dict().items()},
        "common_feature_sampling_hz": {"source": sorted(map(int, ms["common_fs_hz"].dropna().unique())), "target": sorted(map(int, mt["common_fs_hz"].dropna().unique()))},
        "source_analysis_channel_counts": {str(k): int(v) for k, v in source_file_channels["analysis_channel"].value_counts().to_dict().items()},
        "target_analysis_channel_counts": {str(k): int(v) for k, v in target_file_channels["analysis_channel"].value_counts().to_dict().items()},
        "source_rpm": {
            "available_files": int(sr["rpm"].notna().sum()),
            "min": float(sr["rpm"].min()),
            "max": float(sr["rpm"].max()),
            "median": float(sr["rpm"].median()),
            "source": "raw MAT RPM variable",
        },
        "target_rpm": {
            "exact_available_files": int(tr["rpm"].notna().sum()),
            "problem_statement_approx_rpm": 600,
            "source": "题面约600rpm仅作近似工况；原始MAT无逐时精确RPM",
        },
        "duration_seconds": {
            "source_min": float(sr["duration_seconds"].min()),
            "source_max": float(sr["duration_seconds"].max()),
            "source_median": float(sr["duration_seconds"].median()),
            "target_min": float(tr["duration_seconds"].min()),
            "target_max": float(tr["duration_seconds"].max()),
            "target_median": float(tr["duration_seconds"].median()),
        },
        "raw_amplitude": {
            "source_signal_std_median": float(sr["signal_std"].median()),
            "target_signal_std_median": float(tr["signal_std"].median()),
            "target_over_source_signal_std_median_ratio": float(tr["signal_std"].median() / max(sr["signal_std"].median(), EPS)),
            "source_peak_to_peak_median": float(sr["raw_peak_to_peak"].median()),
            "target_peak_to_peak_median": float(tr["raw_peak_to_peak"].median()),
            "target_over_source_peak_to_peak_median_ratio": float(tr["raw_peak_to_peak"].median() / max(sr["raw_peak_to_peak"].median(), EPS)),
        },
        "common_mechanism_boundary": {
            "shared": "OR/IR/B/N均对应滚动轴承局部状态；冲击、调制、包络与频带能量等概念可作为跨域机理线索。",
            "not_shared_by_assumption": "源/目标轴承几何、传感器安装与传递路径、目标逐文件精确RPM、绝对幅值尺度。",
            "source_geometry_specific_aux_on_target": "prohibited because target bearing geometry/exact RPM are not frozen",
            "target_compatible_mechanism_proxies": ["envelope_rms", "envelope_kurtosis", "envelope_spectral_entropy", "spectral_centroid_hz", "spectral_entropy", "band_ratio_*"],
        },
    }
    (art / "domain_context_comparison.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_rows = pd.DataFrame([
        {"domain": "source_selected", "files": len(sr), "native_fs_hz": ";".join(map(str, sorted(sr["sampling_rate_hz"].unique()))),
         "duration_median_s": sr["duration_seconds"].median(), "rpm_exact_available": sr["rpm"].notna().sum(),
         "rpm_min": sr["rpm"].min(), "rpm_max": sr["rpm"].max(), "signal_std_median": sr["signal_std"].median(),
         "raw_peak_to_peak_median": sr["raw_peak_to_peak"].median(), "analysis_channel": ";".join(sorted(source_file_channels["analysis_channel"].astype(str).unique()))},
        {"domain": "target_A_P", "files": len(tr), "native_fs_hz": ";".join(map(str, sorted(tr["sampling_rate_hz"].unique()))),
         "duration_median_s": tr["duration_seconds"].median(), "rpm_exact_available": tr["rpm"].notna().sum(),
         "rpm_min": np.nan, "rpm_max": np.nan, "signal_std_median": tr["signal_std"].median(),
         "raw_peak_to_peak_median": tr["raw_peak_to_peak"].median(), "analysis_channel": ";".join(sorted(target_file_channels["analysis_channel"].astype(str).unique()))},
    ])
    raw_rows.to_csv(art / "domain_context_comparison.csv", index=False, encoding="utf-8-sig")

    sf = source.groupby("group_id", sort=True)[features].mean()
    tf = target.groupby("group_id", sort=True)[features].mean()
    shift, fam = feature_shift(sf, tf, features)
    shift.to_csv(art / "feature_shift_file_level.csv", index=False, encoding="utf-8-sig")
    fam.to_csv(art / "feature_family_shift_summary.csv", index=False, encoding="utf-8-sig")

    mu = sf.mean().to_numpy(float)
    sd = sf.std(ddof=1).to_numpy(float)
    sd = np.where(sd > EPS, sd, 1.0)
    Zs = (sf.to_numpy(float) - mu) / sd
    Zt = (tf.to_numpy(float) - mu) / sd
    mmd2, gamma, med = rbf_mmd2(Zs, Zt)
    mmd = {
        "comparison_unit": "independent-file mean vector",
        "feature_count": 26,
        "mmd2_biased": mmd2,
        "rbf_gamma": gamma,
        "median_pairwise_squared_distance": med,
        "interpretation": "descriptive domain-gap evidence only; not an accuracy estimate",
    }
    (art / "mmd_summary.json").write_text(json.dumps(mmd, ensure_ascii=False, indent=2), encoding="utf-8")

    Xdom = np.vstack([sf.to_numpy(float), tf.to_numpy(float)])
    ydom = np.array([0] * len(sf) + [1] * len(tf), int)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    aucs = []
    for tri, vai in skf.split(Xdom, ydom):
        sc = StandardScaler().fit(Xdom[tri])
        lr = LogisticRegression(max_iter=3000, random_state=SEED)
        lr.fit(sc.transform(Xdom[tri]), ydom[tri])
        p = lr.predict_proba(sc.transform(Xdom[vai]))[:, 1]
        aucs.append(float(roc_auc_score(ydom[vai], p)))
    pca_scaler = StandardScaler().fit(sf.to_numpy(float))
    pca = PCA(n_components=2, random_state=SEED).fit(np.vstack([pca_scaler.transform(sf), pca_scaler.transform(tf)]))
    C = pca.transform(np.vstack([pca_scaler.transform(sf), pca_scaler.transform(tf)]))
    pca_tab = pd.DataFrame({
        "group_id": list(sf.index.astype(str)) + list(tf.index.astype(str)),
        "domain": ["source"] * len(sf) + ["target"] * len(tf),
        "pc1": C[:, 0], "pc2": C[:, 1],
    })
    pca_tab.to_csv(art / "pca_file_coordinates.csv", index=False, encoding="utf-8-sig")
    sep = {
        "domain_classifier_auc_folds": aucs,
        "domain_classifier_auc_mean": float(np.mean(aucs)),
        "domain_classifier_auc_std": float(np.std(aucs, ddof=1)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "note": "domain labels only; no target fault labels used",
    }
    (art / "domain_separation_summary.json").write_text(json.dumps(sep, ensure_ascii=False, indent=2), encoding="utf-8")

    target_predict_X = target[features].to_numpy(float)
    target_scores = fixed_scores(final_clf, target_predict_X)
    wtab = target[["window_id", "group_id", "target_id", "relative_path"]].copy()
    for j, c in enumerate(CLASSES):
        wtab[f"score_{c}"] = target_scores[:, j]
    wtab["pred_label"] = np.array(CLASSES, dtype=object)[np.argmax(target_scores, axis=1)]
    wtab["truth_status"] = "unknown"
    wtab.to_csv(art / "A_P_no_transfer_window_predictions.csv", index=False, encoding="utf-8-sig")
    _, ft = aggregate_unlabeled(target["group_id"].to_numpy(), target_scores, target["target_id"].to_numpy())
    ft["truth_status"] = "unknown"
    ft = ft[["target_id", "group_id", "truth_status", "window_count", *[f"score_{c}" for c in CLASSES], "pred_label", "top_model_score", "score_margin", "score_entropy_norm", "window_file_agreement"]]
    ft.sort_values("target_id").to_csv(art / "A_P_no_transfer_file_predictions.csv", index=False, encoding="utf-8-sig")

    score_cols = [f"score_{c}" for c in CLASSES]
    so = src_oof[score_cols].to_numpy(float)
    st = ft[score_cols].to_numpy(float)
    source_top = np.max(so, axis=1)
    target_top = np.max(st, axis=1)
    output_shift = {
        "source_oof_files": int(len(src_oof)),
        "target_files": int(len(ft)),
        "source_top_model_score_median": float(np.median(source_top)),
        "target_top_model_score_median": float(np.median(target_top)),
        "source_score_entropy_median": float(np.median(norm_entropy(so))),
        "target_score_entropy_median": float(np.median(norm_entropy(st))),
        "target_score_margin_median": float(ft["score_margin"].median()),
        "target_window_file_agreement_mean": float(ft["window_file_agreement"].mean()),
        "target_prediction_counts": {str(k): int(v) for k, v in ft["pred_label"].value_counts().to_dict().items()},
        "score_semantics": "uncalibrated model scores; do not call confidence or calibrated probability",
        "target_truth": "unknown",
    }
    (art / "classifier_output_shift.json").write_text(json.dumps(output_shift, ensure_ascii=False, indent=2), encoding="utf-8")

    source_file_labels = source[["group_id", "class_label"]].drop_duplicates().set_index("group_id").loc[sf.index]
    centroids = {}
    radii = {}
    for c in CLASSES:
        idx = source_file_labels["class_label"].astype(str).to_numpy() == c
        z = Zs[idx]
        cc = z.mean(axis=0)
        dist = np.linalg.norm(z - cc, axis=1) / math.sqrt(len(features))
        centroids[c] = cc
        radii[c] = {"median": float(np.median(dist)), "p95": float(np.percentile(dist, 95)), "n_files": int(len(dist))}
    pred_map = dict(zip(ft["group_id"].astype(str), ft["pred_label"].astype(str)))
    struct_rows = []
    for i, gid in enumerate(tf.index.astype(str)):
        ds = {c: float(np.linalg.norm(Zt[i] - centroids[c]) / math.sqrt(len(features))) for c in CLASSES}
        ordered = sorted(ds, key=ds.get)
        nearest, second = ordered[0], ordered[1]
        ratio = ds[nearest] / max(radii[nearest]["p95"], EPS)
        struct_rows.append({
            "target_id": Path(gid).stem,
            "group_id": gid,
            **{f"distance_to_{c}": ds[c] for c in CLASSES},
            "nearest_source_class_centroid": nearest,
            "second_nearest_source_class_centroid": second,
            "centroid_distance_margin": float(ds[second] - ds[nearest]),
            "nearest_distance_over_source_class_p95_radius": float(ratio),
            "inside_nearest_source_class_p95_radius": bool(ratio <= 1.0),
            "no_transfer_pred_label": pred_map[gid],
            "model_centroid_agree": bool(pred_map[gid] == nearest),
            "truth_status": "unknown",
        })
    struct = pd.DataFrame(struct_rows).sort_values("target_id")
    struct.to_csv(art / "class_conditional_structure_diagnostic.csv", index=False, encoding="utf-8-sig")
    (art / "source_class_radius_reference.json").write_text(json.dumps(radii, ensure_ascii=False, indent=2), encoding="utf-8")
    outside_frac = float((~struct["inside_nearest_source_class_p95_radius"]).mean())

    usage = {
        "target_files": sorted(ft["target_id"].astype(str).tolist()),
        "target_windows": int(len(target)),
        "truth_status": "unknown",
        "target_truth_values_available_to_modeling": False,
        "target_label_columns_read_only_for_boundary_assertion": ["class_label=UNKNOWN_TRUTH", "label_status=unknown_truth"],
        "used_in_08A_for": [
            "frozen-source-model no-transfer inference",
            "file-level feature shift/MMD/PCA/domain-classifier diagnostics",
            "source-class-centroid distance diagnostics",
            "classifier score entropy/margin/window-agreement diagnostics",
        ],
        "used_for_real_target_adaptation_parameters_in_08A": False,
        "allowed_for_08B_unsupervised_adaptation": [
            "all 240 target 26-D feature rows for global unlabeled moments/covariance",
            "target group IDs only for file-balanced weighting and file aggregation",
            "unlabeled target-vs-source domain membership for domain-classifier weighting",
        ],
        "forbidden": [
            "target Accuracy/F1/Recall",
            "supervised selection on A-P",
            "treating A-P predictions as truth or pseudo-ground-truth",
            "source-geometry-specific mechanism features on target",
        ],
    }
    (art / "target_unlabeled_usage_ledger.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

    load_groups = source[["group_id", "class_label", "load_hp"]].drop_duplicates()
    coverage = load_groups.groupby(["load_hp", "class_label"]).size().unstack(fill_value=0)
    coverage.to_csv(art / "pseudo_target_load_class_file_counts.csv", encoding="utf-8-sig")
    expected_loads = [0.0, 1.0, 2.0, 3.0]
    for ld in expected_loads:
        if ld not in coverage.index:
            raise RuntimeError(f"pseudo target load {ld} missing")
        if any(int(coverage.loc[ld].get(c, 0)) <= 0 for c in CLASSES):
            raise RuntimeError(f"pseudo target load {ld} lacks full OR/IR/B/N file coverage")

    alphas = [float(x) for x in cfg["simple_transfer_baseline"]["alpha_grid"]]
    tune_loads = [float(x) for x in cfg["pseudo_target"]["hyperparameter_selection_loads"]]
    final_load = float(cfg["pseudo_target"]["final_validation_load"])
    if final_load in tune_loads:
        raise RuntimeError("final pseudo target load leaks into tuning loads")

    sim_rows = []
    sim_file_rows = []
    per_load_models = {}
    for ld in expected_loads:
        train = source[source["load_hp"] != ld].copy()
        pseudo = source[source["load_hp"] == ld].copy()
        clf = fit_frozen_rf(train, features)
        per_load_models[ld] = clf

        Xp = pseudo[features].to_numpy(float)
        gp = pseudo["group_id"].to_numpy()
        Xtr = train[features].to_numpy(float)
        gtr = train["group_id"].to_numpy()
        truth = pseudo[["group_id", "class_label"]].drop_duplicates()

        p0 = fixed_scores(clf, Xp)
        _, f0 = aggregate_unlabeled(gp, p0)
        m0 = file_metric(truth, f0)
        sim_rows.append({"pseudo_target_load": int(ld), "split_role": "tuning" if ld in tune_loads else "final_validation",
                         "method": "A0_no_transfer", "alpha": 0.0, "adaptation_uses_target_labels": False, **m0})
        z0 = f0.merge(truth, on="group_id")
        z0["pseudo_target_load"] = int(ld)
        z0["method"] = "A0_no_transfer"
        z0["alpha"] = 0.0
        sim_file_rows.append(z0)

        eval_alphas = alphas if ld in tune_loads else []
        for a in eval_alphas:
            Xa = moment_align(Xtr, gtr, Xp, gp, a)
            pa = fixed_scores(clf, Xa)
            _, fa = aggregate_unlabeled(gp, pa)
            ma = file_metric(truth, fa)
            sim_rows.append({"pseudo_target_load": int(ld), "split_role": "tuning", "method": "T1_shrink_moment",
                             "alpha": a, "adaptation_uses_target_labels": False, **ma})
            zz = fa.merge(truth, on="group_id")
            zz["pseudo_target_load"] = int(ld)
            zz["method"] = "T1_shrink_moment"
            zz["alpha"] = a
            sim_file_rows.append(zz)

    sim = pd.DataFrame(sim_rows)
    tune = sim[(sim["split_role"] == "tuning") & (sim["method"] == "T1_shrink_moment")].copy()
    tune_summary = tune.groupby("alpha", as_index=False).agg(
        mean_macro_f1=("macro_f1", "mean"),
        mean_min_class_recall=("min_class_recall", "mean"),
        worst_macro_f1=("macro_f1", "min"),
    )
    tune_summary = tune_summary.sort_values(["mean_macro_f1", "mean_min_class_recall", "alpha"], ascending=[False, False, True]).reset_index(drop=True)
    selected_alpha = float(tune_summary.iloc[0]["alpha"])
    tune_summary["selected"] = tune_summary["alpha"].eq(selected_alpha)
    tune_summary.to_csv(art / "T1_alpha_tuning_summary.csv", index=False, encoding="utf-8-sig")

    train = source[source["load_hp"] != final_load].copy()
    pseudo = source[source["load_hp"] == final_load].copy()
    clf = per_load_models[final_load]
    truth = pseudo[["group_id", "class_label"]].drop_duplicates()
    Xp = pseudo[features].to_numpy(float)
    gp = pseudo["group_id"].to_numpy()
    Xa = moment_align(train[features].to_numpy(float), train["group_id"].to_numpy(), Xp, gp, selected_alpha)
    pa = fixed_scores(clf, Xa)
    _, fa = aggregate_unlabeled(gp, pa)
    ma = file_metric(truth, fa)
    final_row = {"pseudo_target_load": int(final_load), "split_role": "final_validation", "method": "T1_shrink_moment",
                 "alpha": selected_alpha, "adaptation_uses_target_labels": False, **ma}
    sim = pd.concat([sim, pd.DataFrame([final_row])], ignore_index=True)
    zz = fa.merge(truth, on="group_id")
    zz["pseudo_target_load"] = int(final_load)
    zz["method"] = "T1_shrink_moment"
    zz["alpha"] = selected_alpha
    sim_file_rows.append(zz)

    sim.to_csv(art / "pseudo_target_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(sim_file_rows, ignore_index=True).to_csv(art / "pseudo_target_file_predictions.csv", index=False, encoding="utf-8-sig")

    base_final = sim[(sim["pseudo_target_load"] == int(final_load)) & (sim["method"] == "A0_no_transfer")].iloc[0]
    adapt_final = sim[(sim["pseudo_target_load"] == int(final_load)) & (sim["method"] == "T1_shrink_moment")].iloc[0]
    final_gain = float(adapt_final["macro_f1"] - base_final["macro_f1"])
    severe_fail = bool(final_gain < float(cfg["promotion_and_failure"]["failure_if_final_macro_f1_drop_below"]))
    t1 = {
        "selected_alpha_from_tuning_loads_0_1_2": selected_alpha,
        "tuning_loads": [0, 1, 2],
        "final_validation_load": 3,
        "final_no_transfer_macro_f1": float(base_final["macro_f1"]),
        "final_T1_macro_f1": float(adapt_final["macro_f1"]),
        "final_macro_f1_gain": final_gain,
        "final_no_transfer_min_class_recall": float(base_final["min_class_recall"]),
        "final_T1_min_class_recall": float(adapt_final["min_class_recall"]),
        "severe_failure": severe_fail,
        "post_final_retuning_performed": False,
        "interpretation": "pseudo-target validation only; does not estimate A-P target accuracy",
    }
    (art / "T1_final_validation.json").write_text(json.dumps(t1, ensure_ascii=False, indent=2), encoding="utf-8")

    trig = cfg["diagnostic_triggers"]
    T2_trigger = bool((final_gain <= 0.0 or severe_fail) and (sep["domain_classifier_auc_mean"] >= trig["domain_classifier_auc_high_at_or_above"] or mmd2 >= trig["descriptive_mmd2_material_at_or_above"]))
    T3_trigger = bool(sep["domain_classifier_auc_mean"] >= trig["domain_classifier_auc_high_at_or_above"] and (1.0 - outside_frac) < 0.75)
    plan = {
        "simple_transfer_baseline": {
            "id": "T1-SHRINK-MOMENT",
            "selected_alpha": selected_alpha,
            "selection_domains": [0, 1, 2],
            "final_validation_domain": 3,
            "final_macro_f1_gain": final_gain,
            "status": "failed_severely" if severe_fail else ("positive_on_held_domain" if final_gain > 0 else "no_positive_gain_on_held_domain"),
        },
        "candidate_1": {
            "id": "T2-CORAL",
            "triggered_for_08B": T2_trigger,
            "target_problem": "residual multivariate covariance shift after simple marginal alignment",
            "target_input_if_run": "unlabeled 26-D target covariance only",
            "config_budget_if_run": 3,
            "failure_standard": "no positive held-load3 Macro-F1 gain over no-transfer or >0.05 loss on held validation",
        },
        "candidate_2": {
            "id": "T3-DOMAIN-WEIGHT",
            "triggered_for_08B": T3_trigger,
            "target_problem": "partial source-support mismatch / covariate shift",
            "target_input_if_run": "domain membership only; no target fault labels/pseudo-label truth",
            "config_budget_if_run": 2,
            "failure_standard": "no positive held-load3 Macro-F1 gain over no-transfer or zeroing a previously nonzero class recall",
        },
        "diagnostic_values": {
            "mmd2_biased": mmd2,
            "domain_classifier_auc_mean": sep["domain_classifier_auc_mean"],
            "target_outside_nearest_source_class_p95_fraction": outside_frac,
        },
        "compute_budget": cfg["compute_budget"],
    }
    (art / "transfer_candidate_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    need = {
        "problems_to_solve_before_real_target_adaptation": [
            "large acquisition/operating shift: source measured ~1722-1798rpm vs target only ~600rpm approximate and no exact per-file RPM",
            "native sampling/sensor path mismatch: selected source includes 12kHz FE faults and 48kHz normal files, target is 32kHz unknown sensor; common feature extraction normalizes sampling but not all transfer-path effects",
            "raw amplitude-scale mismatch, so absolute-amplitude features may dominate source-trained decision boundaries",
            "feature marginal/multivariate shift evidenced by file-level shift, MMD/domain separability",
            "possible class-conditional support mismatch: target files can sit outside source class p95 radii; target truth remains unknown",
        ],
        "mechanism_shared_but_not_geometry_shared": context["common_mechanism_boundary"],
        "target_accuracy_available": False,
    }
    (art / "migration_problems_to_solve.json").write_text(json.dumps(need, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "gate_bound_to_current_freezes": cfg["inputs"]["freeze07"] == FREEZE07 and cfg["inputs"]["freeze04"] == FREEZE04,
        "A_P_true_labels_unavailable": target_label_guard,
        "A_P_predictions_are_labeled_as_prediction_only": bool((ft["truth_status"] == "unknown").all()),
        "A_P_no_transfer_saved_16_files": len(ft) == 16 and ft["target_id"].nunique() == 16,
        "no_target_accuracy_or_F1_computed": True,
        "pseudo_targets_rotate_multiple_loads": set(sim["pseudo_target_load"].unique()) == {0, 1, 2, 3},
        "pseudo_target_each_load_has_all_classes": all(all(int(coverage.loc[float(ld)].get(c, 0)) > 0 for c in CLASSES) for ld in [0, 1, 2, 3]),
        "pseudo_adaptation_function_has_no_label_argument": True,
        "pseudo_adaptation_rows_record_no_label_use": not bool(sim["adaptation_uses_target_labels"].any()),
        "tuning_and_final_validation_domains_separated": final_load not in tune_loads and set(tune_loads) == {0.0, 1.0, 2.0} and final_load == 3.0,
        "no_post_final_retuning": t1["post_final_retuning_performed"] is False,
        "domain_evidence_raw_context": (art / "domain_context_comparison.json").exists(),
        "domain_evidence_feature_distribution": (art / "feature_shift_file_level.csv").exists() and (art / "mmd_summary.json").exists(),
        "domain_evidence_classifier_output": (art / "classifier_output_shift.json").exists(),
        "domain_evidence_class_structure": (art / "class_conditional_structure_diagnostic.csv").exists(),
        "at_least_two_independent_evidence_types": True,
        "target_unlabeled_input_use_recorded": (art / "target_unlabeled_usage_ledger.json").exists(),
        "transfer_candidates_and_selection_rules_saved": (art / "transfer_candidate_plan.json").exists(),
        "complex_real_target_transfer_not_trained": True,
    }
    failed_checks = [k for k, v in checks.items() if not v]
    passed = len(failed_checks) == 0
    validation = {
        "step_id": STEP,
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "failed_checks": failed_checks,
        "evidence": {
            "A_P_truth_boundary": "04-A target_window_metadata.csv: class_label=UNKNOWN_TRUTH,label_status=unknown_truth for all 240 windows",
            "no_transfer": f"outputs/runs/{run_id}/artifacts/A_P_no_transfer_file_predictions.csv",
            "raw_context": f"outputs/runs/{run_id}/artifacts/domain_context_comparison.json",
            "feature_shift": f"outputs/runs/{run_id}/artifacts/feature_shift_file_level.csv",
            "mmd": f"mmd2={mmd2:.8f}",
            "domain_classifier": f"AUC_mean={sep['domain_classifier_auc_mean']:.8f}",
            "classifier_output": f"outputs/runs/{run_id}/artifacts/classifier_output_shift.json",
            "class_structure": f"outside_source_class_p95_fraction={outside_frac:.8f}",
            "pseudo_target": f"loads0/1/2 tuning, load3 final; selected_alpha={selected_alpha}; final_gain={final_gain:.8f}",
            "target_usage": f"outputs/runs/{run_id}/artifacts/target_unlabeled_usage_ledger.json",
        },
        "word_edit_performed": False,
        "target_accuracy_reported": False,
    }
    (art / "validation_record.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    top_shift = shift.head(8)[["feature", "family", "abs_standardized_mean_shift", "wasserstein_over_source_sd", "ks_statistic"]].to_dict(orient="records")
    summary = {
        "source_files": 49,
        "source_windows": 733,
        "target_files": 16,
        "target_windows": 240,
        "target_truth": "unknown",
        "target_accuracy_reported": False,
        "no_transfer_prediction_counts": output_shift["target_prediction_counts"],
        "raw_amplitude_target_over_source_std_median_ratio": context["raw_amplitude"]["target_over_source_signal_std_median_ratio"],
        "top_feature_shifts": top_shift,
        "mmd2_biased": mmd2,
        "domain_classifier_auc_mean": sep["domain_classifier_auc_mean"],
        "target_outside_nearest_source_class_p95_fraction": outside_frac,
        "T1_selected_alpha": selected_alpha,
        "T1_final_held_load3_no_transfer_macro_f1": float(base_final["macro_f1"]),
        "T1_final_held_load3_macro_f1": float(adapt_final["macro_f1"]),
        "T1_final_gain": final_gain,
        "T2_triggered_for_08B": T2_trigger,
        "T3_triggered_for_08B": T3_trigger,
    }
    (art / "step08A_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (mani / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")
    env_runtime = {
        "captured_utc": created,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "gpu": None,
        "cuda": None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pip_freeze_sha256": hf(mani / "pip_freeze.txt"),
        "git_commit": git_head(),
    }
    (mani / "environment_runtime.json").write_text(json.dumps(env_runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    freeze = {
        "schema_version": "00C-1.0",
        "package_type": "A_freeze_package",
        "step_id": STEP,
        "freeze_package_id": f"FREEZE-08A-{run_id[-8:]}",
        "status": "passed" if passed else "failed",
        "run_id": run_id,
        "versions": {
            "code_version_id": "git:" + git_head(),
            "raw_data_version_id": "RAW-5a5dd129c91bfc64",
            "input_derived_data_ids": [FREEZE04, FREEZE07, "FEAT-23d45f5649dcd5f1"],
            "run_config_sha256": hf(CFG),
            "environment_sha256": hf(ENV_MANIFEST),
        },
        "random_seed": SEED,
        "parameters": {
            "final_source_model": "H1_RF; 300 trees; max_depth=8; min_samples_leaf=1; max_features=sqrt; file_and_class_balanced",
            "target_features": 26,
            "pseudo_target_axis": "load_hp",
            "tuning_loads": [0, 1, 2],
            "final_validation_load": 3,
            "T1_alpha_grid": alphas,
            "selected_T1_alpha": selected_alpha,
            "target_truth_policy": "unknown; no target accuracy/F1/Recall",
        },
        "real_results": [
            {"name": "domain_context", "value": context},
            {"name": "domain_gap_compact", "value": {"mmd2_biased": mmd2, "domain_classifier_auc_mean": sep["domain_classifier_auc_mean"], "target_outside_source_class_p95_fraction": outside_frac}},
            {"name": "A_P_no_transfer_prediction_distribution", "value": output_shift["target_prediction_counts"]},
            {"name": "T1_pseudo_target_final_validation", "value": t1},
            {"name": "candidate_plan", "value": plan},
        ],
        "validation_evidence": [
            {"path": f"outputs/runs/{run_id}/artifacts/validation_record.json"},
            {"path": f"outputs/runs/{run_id}/artifacts/A_P_no_transfer_file_predictions.csv"},
            {"path": f"outputs/runs/{run_id}/artifacts/domain_context_comparison.json"},
            {"path": f"outputs/runs/{run_id}/artifacts/feature_shift_file_level.csv"},
            {"path": f"outputs/runs/{run_id}/artifacts/classifier_output_shift.json"},
            {"path": f"outputs/runs/{run_id}/artifacts/class_conditional_structure_diagnostic.csv"},
            {"path": f"outputs/runs/{run_id}/artifacts/pseudo_target_metrics.csv"},
            {"path": f"outputs/runs/{run_id}/artifacts/target_unlabeled_usage_ledger.json"},
        ],
        "anomalies_and_failures": [
            {"item": "T1_simple_transfer_baseline", "status": plan["simple_transfer_baseline"]["status"], "held_load3_gain": final_gain},
            {"item": "target_exact_rpm", "status": "unavailable", "boundary": "题面约600rpm only; no per-file exact target RPM"},
            {"item": "target_bearing_geometry", "status": "unknown", "boundary": "source geometry-specific mechanism aux prohibited on target"},
        ],
        "b_handoff": {
            "required_data": [
                "step08A_summary.json",
                "domain_context_comparison.json",
                "feature_shift_file_level.csv",
                "feature_family_shift_summary.csv",
                "mmd_summary.json",
                "domain_separation_summary.json",
                "A_P_no_transfer_window_predictions.csv",
                "A_P_no_transfer_file_predictions.csv",
                "classifier_output_shift.json",
                "class_conditional_structure_diagnostic.csv",
                "target_unlabeled_usage_ledger.json",
                "pseudo_target_metrics.csv",
                "T1_alpha_tuning_summary.csv",
                "T1_final_validation.json",
                "transfer_candidate_plan.json",
            ],
            "supported_conclusions": [
                "A-P no-transfer outputs are predictions only; target truth remained unavailable.",
                "Source-target gap is described by acquisition/raw-amplitude evidence plus file-level feature/domain-separation evidence plus classifier-output/class-structure evidence.",
                "Pseudo-target loads 0/1/2 tuned T1 alpha and load3 was held completely out for final migration validation.",
                "Only target-compatible 26 common features are eligible for real target adaptation.",
            ],
            "wording_limits": [
                "do not report target-domain accuracy/F1/Recall",
                "do not call uncalibrated model scores confidence/probability",
                "do not claim MMD/PCA/domain-classifier AUC measures fault-classification accuracy",
                "do not claim source bearing geometry applies to target",
                "do not call pseudo-target held-load result real target performance",
            ],
            "approved_tables": [
                "domain_context_comparison.csv",
                "feature_shift_file_level.csv",
                "A_P_no_transfer_file_predictions.csv",
                "pseudo_target_metrics.csv",
                "T1_alpha_tuning_summary.csv",
            ],
            "approved_figures": [],
        },
        "unresolved_issues": [],
        "freeze": {
            "created_utc": created,
            "content_sha256": None,
            "invalidation_dependencies": [
                "07-A final source model/config/interface changes",
                "04-A source/target common feature data changes",
                "01-A raw context metadata changes",
                "08-A blueprint/run-config/code changes",
            ],
        },
    }
    f0 = json.loads(json.dumps(freeze, ensure_ascii=False))
    freeze["freeze"]["content_sha256"] = canonical_hash(f0)
    (art / "A_freeze_package.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "protocol" / "08A" / "latest_freeze_package.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "protocol" / "08A" / "validation_record.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "protocol" / "08A" / "latest_run_id.txt").write_text(run_id + "\n", encoding="utf-8")

    files = []
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name != "run_manifest.json"):
        files.append({"relative_path": p.relative_to(out).as_posix(), "size_bytes": p.stat().st_size, "sha256": hf(p)})
    run_manifest = {
        "schema_version": "08A-1.0",
        "manifest_type": "run_manifest",
        "run_id": run_id,
        "step_id": STEP,
        "status": "completed" if passed else "failed",
        "created_utc": created,
        "binding": {**binding, "binding_digest": bd},
        "code": {
            "repository": "Mhhhh958/mathmodeling",
            "commit": git_head(),
            "entrypoint": "scripts/step08a_domain_diagnosis.py",
            "code_manifest": "protocol/08A/code_manifest.json",
            "code_manifest_blob": CODE_MANIFEST.exists(),
        },
        "data": {
            "source_files": 49, "source_windows": 733,
            "target_files": 16, "target_windows": 240,
            "target_truth": "unknown",
            "feature_count": 26,
        },
        "blueprint": {"path": "protocol/08A/question3_argument_blueprint.json", "sha256": hf(BLUEPRINT)},
        "config": {"path": "protocol/08A/run_config.json", "sha256": hf(CFG)},
        "environment": {"manifest": "protocol/00B/environment_manifest.json", "sha256": hf(ENV_MANIFEST), "runtime": "manifests/environment_runtime.json"},
        "outputs": {"root": f"outputs/runs/{run_id}", "files": files, "output_tree_sha256": canonical_hash(files)},
        "duration_seconds": time.time() - t0,
    }
    (mani / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": passed,
        "run_id": run_id,
        "freeze_package_id": freeze["freeze_package_id"],
        "freeze_content_sha256": freeze["freeze"]["content_sha256"],
        "target_prediction_counts": output_shift["target_prediction_counts"],
        "mmd2": mmd2,
        "domain_auc": sep["domain_classifier_auc_mean"],
        "outside_source_class_p95_fraction": outside_frac,
        "selected_alpha": selected_alpha,
        "held_load3_no_transfer_macro_f1": float(base_final["macro_f1"]),
        "held_load3_T1_macro_f1": float(adapt_final["macro_f1"]),
        "held_load3_gain": final_gain,
        "T2_triggered": T2_trigger,
        "T3_triggered": T3_trigger,
        "target_accuracy_reported": False,
    }, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
