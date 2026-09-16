from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse STEP08 metric/aggregation conventions exactly.
import q3a_domain_diagnosis as q3a

ROOT = Path(__file__).resolve().parents[1]
Q1C = ROOT / "outputs" / "q1c_feature_extraction"
Q2C = ROOT / "outputs" / "q2c_failure_driven_improvement"
Q3A = ROOT / "outputs" / "q3a_domain_diagnosis"
OUT = ROOT / "outputs" / "q3b_unsupervised_transfer"
MODELS = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SRC_CSV = Q1C / "q2_source_raw.csv"
TGT_CSV = Q1C / "q2_target_raw.csv"
Q3_INTERFACE = Q2C / "q3_interface.json"
SOURCE_MODEL = Q2C / "models" / "final_source_model.joblib"
STEP08_A1 = Q3A / "simulated_target_method_summary.csv"
STEP08_PLAN = Q3A / "transfer_candidate_plan.json"
STEP08_NO_TRANSFER = Q3A / "target_A_P_no_transfer_file_predictions.csv"

SEED = 20260916
SEEDS = [20260916, 20260917, 20260918]
LABELS = ["OR", "IR", "B", "N"]
LOADS = [0, 1, 2, 3]
CORAL_REGS = [1e-3, 1e-2, 1e-1]
IW_CLIPS = [2.0, 5.0, 10.0]
EPS = 1e-9
PROMOTION = {
    "min_mean_macro_f1_gain": 0.02,
    "max_single_load_macro_f1_drop": 0.05,
    "no_new_zero_recall": True,
    "selection_source": "four simulated held-out source loads only; never A-P truth or appearance",
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


def build_rf(seed: int):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )),
    ])


def fit_rf(df: pd.DataFrame, feats: list[str], seed: int, extra_weights=None):
    pipe = build_rf(seed)
    base_w = q3a.training_weights(df)
    if extra_weights is not None:
        ew = np.asarray(extra_weights, float)
        if len(ew) != len(df):
            raise RuntimeError("extra weight length mismatch")
        base_w = base_w * ew
    base_w = base_w / max(np.mean(base_w), EPS)
    pipe.fit(df[feats].to_numpy(float), df["class_label"].astype(str).to_numpy(), clf__sample_weight=base_w)
    return pipe


def transform_z(pipe, df: pd.DataFrame, feats: list[str]):
    X = df[feats].to_numpy(float)
    Xi = pipe.named_steps["imputer"].transform(X)
    return pipe.named_steps["scaler"].transform(Xi)


def predict_direct(pipe, df: pd.DataFrame, feats: list[str]):
    P = pipe.predict_proba(df[feats].to_numpy(float))
    cls = list(pipe.named_steps["clf"].classes_)
    win, files = q3a.aggregate_probabilities(df, P, cls)
    return win, files


def predict_from_z(pipe, Z: np.ndarray, df: pd.DataFrame):
    clf = pipe.named_steps["clf"]
    P = clf.predict_proba(Z)
    cls = list(clf.classes_)
    win, files = q3a.aggregate_probabilities(df, P, cls)
    return win, files


def mat_power_sym(C: np.ndarray, power: float):
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    vals = np.maximum(vals, EPS)
    return (vecs * (vals ** power)) @ vecs.T


def coral_target_to_source(pipe, src: pd.DataFrame, tgt: pd.DataFrame, feats: list[str], reg: float):
    Zs = transform_z(pipe, src, feats)
    Zt = transform_z(pipe, tgt, feats)
    ms = Zs.mean(axis=0)
    mt = Zt.mean(axis=0)
    Cs = np.cov(Zs, rowvar=False) + reg * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + reg * np.eye(Zt.shape[1])
    A = mat_power_sym(Ct, -0.5) @ mat_power_sym(Cs, 0.5)
    Zta = (Zt - mt) @ A + ms
    bundle = {
        "method": "A2_CORAL_target_to_source",
        "regularization": float(reg),
        "target_mean_in_source_standardized_space": mt,
        "source_mean_in_source_standardized_space": ms,
        "linear_map": A,
        "definition": "Zt_aligned=(Zt-mu_t) Ct_reg^{-1/2} Cs_reg^{1/2}+mu_s; frozen source RF is unchanged",
        "pseudo_labels_used": False,
    }
    return Zta, bundle


def file_equal_domain_weights(df: pd.DataFrame):
    w = q3a.file_equal_weights(df)
    return w / max(w.sum(), EPS)


def fit_domain_ratio(pipe, src: pd.DataFrame, tgt: pd.DataFrame, feats: list[str], seed: int, clip: float):
    Zs = transform_z(pipe, src, feats)
    Zt = transform_z(pipe, tgt, feats)
    X = np.vstack([Zs, Zt])
    y = np.r_[np.zeros(len(Zs), dtype=int), np.ones(len(Zt), dtype=int)]
    ws = file_equal_domain_weights(src)
    wt = file_equal_domain_weights(tgt)
    # Equal total prior mass for source and target, so posterior odds approximate density ratio.
    sw = np.r_[ws, wt]
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    clf.fit(X, y, sample_weight=sw)
    pt = np.clip(clf.predict_proba(Zs)[:, 1], 1e-5, 1 - 1e-5)
    ratio = pt / (1.0 - pt)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    ratio = ratio / max(np.mean(ratio), EPS)
    ess = float((ratio.sum() ** 2) / np.sum(ratio ** 2))
    ftmp = pd.DataFrame({"independent_object_id": src["independent_object_id"].astype(str), "ratio": ratio})
    file_ratio = ftmp.groupby("independent_object_id")["ratio"].mean()
    fr = file_ratio.to_numpy(float)
    file_ess = float((fr.sum() ** 2) / np.sum(fr ** 2))
    diagnostics = {
        "clip": float(clip),
        "window_weight_min": float(np.min(ratio)),
        "window_weight_median": float(np.median(ratio)),
        "window_weight_max": float(np.max(ratio)),
        "window_ess": ess,
        "window_ess_ratio": ess / len(ratio),
        "file_ess": file_ess,
        "file_ess_ratio": file_ess / len(fr),
        "fraction_at_upper_clip": float(np.mean(ratio >= clip / max(np.mean(np.clip(pt / (1 - pt), 1.0 / clip, clip)), EPS) - 1e-9)),
        "pseudo_labels_used": False,
        "domain_classifier": "LogisticRegression on source-standardized 28 features; domain labels only",
    }
    return ratio, clf, diagnostics


def aggregate_metric(files: pd.DataFrame):
    return q3a.metric_dict(files["class_label"].astype(str), files["pred_label"].astype(str))


def evaluate_simulated(src_all: pd.DataFrame, feats: list[str], seed: int):
    rows = []
    pred_rows = []
    weight_rows = []
    for load in LOADS:
        src = src_all[src_all["load_hp"] != load].copy().reset_index(drop=True)
        tgt = src_all[src_all["load_hp"] == load].copy().reset_index(drop=True)
        base = fit_rf(src, feats, seed)

        _, ff0 = predict_direct(base, tgt, feats)
        m0 = aggregate_metric(ff0)
        rows.append({"method": "A0_no_transfer", "setting": "frozen_rf", "seed": seed, "simulated_target_load": load, **m0})
        pred_rows.append(ff0.assign(method="A0_no_transfer", setting="frozen_rf", seed=seed, simulated_target_load=load))

        for reg in CORAL_REGS:
            Zta, _ = coral_target_to_source(base, src, tgt, feats, reg)
            _, ff = predict_from_z(base, Zta, tgt)
            m = aggregate_metric(ff)
            rows.append({"method": "A2_CORAL", "setting": f"reg={reg:g}", "seed": seed, "simulated_target_load": load, **m})
            pred_rows.append(ff.assign(method="A2_CORAL", setting=f"reg={reg:g}", seed=seed, simulated_target_load=load))

        for clip in IW_CLIPS:
            ratio, domclf, d = fit_domain_ratio(base, src, tgt, feats, seed, clip)
            adapted = fit_rf(src, feats, seed, extra_weights=ratio)
            _, ff = predict_direct(adapted, tgt, feats)
            m = aggregate_metric(ff)
            rows.append({"method": "A3_instance_weighting", "setting": f"clip={clip:g}", "seed": seed, "simulated_target_load": load, **m})
            pred_rows.append(ff.assign(method="A3_instance_weighting", setting=f"clip={clip:g}", seed=seed, simulated_target_load=load))
            weight_rows.append({"seed": seed, "simulated_target_load": load, **d})
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True), pd.DataFrame(weight_rows)


def select_setting_and_method(metrics_seed0: pd.DataFrame):
    base = metrics_seed0[metrics_seed0["method"] == "A0_no_transfer"].sort_values("simulated_target_load").reset_index(drop=True)
    base_mean = float(base["macro_f1"].mean())
    detail = []
    for method in ["A2_CORAL", "A3_instance_weighting"]:
        m = metrics_seed0[metrics_seed0["method"] == method]
        for setting, g in m.groupby("setting"):
            g = g.sort_values("simulated_target_load").reset_index(drop=True)
            if len(g) != 4:
                raise RuntimeError("candidate setting missing simulated load")
            drops = g["macro_f1"].to_numpy() - base["macro_f1"].to_numpy()
            no_new_zero = True
            for i in range(4):
                for lab in LABELS:
                    if float(base.loc[i, f"recall_{lab}"]) > 0 and float(g.loc[i, f"recall_{lab}"]) <= 0:
                        no_new_zero = False
            mean_macro = float(g["macro_f1"].mean())
            mean_bal = float(g["balanced_accuracy"].mean())
            gain = mean_macro - base_mean
            worst_drop = float(np.min(drops))
            eligible = (
                gain >= PROMOTION["min_mean_macro_f1_gain"]
                and worst_drop >= -PROMOTION["max_single_load_macro_f1_drop"]
                and no_new_zero
            )
            detail.append({
                "method": method,
                "setting": setting,
                "mean_macro_f1": mean_macro,
                "mean_balanced_accuracy": mean_bal,
                "gain_vs_A0": gain,
                "worst_single_load_macro_delta": worst_drop,
                "no_new_zero_recall": bool(no_new_zero),
                "eligible": bool(eligible),
            })
    tab = pd.DataFrame(detail)
    # Best setting per family is chosen by simulated-target performance only.
    best = tab.sort_values(["method", "mean_macro_f1", "mean_balanced_accuracy"], ascending=[True, False, False]).groupby("method", as_index=False).first()
    eligible = best[best["eligible"]].sort_values(["mean_macro_f1", "mean_balanced_accuracy"], ascending=[False, False])
    if len(eligible):
        r = eligible.iloc[0]
        selected_method = str(r["method"])
        selected_setting = str(r["setting"])
        reason = "Promoted solely by the frozen simulated-target rule; A-P predictions were not consulted."
    else:
        selected_method = "A0_no_transfer"
        selected_setting = "frozen_rf"
        reason = "Neither A2 nor A3 satisfied the frozen promotion rule; retain the STEP08 no-transfer model."
    summary = pd.concat([
        pd.DataFrame([{
            "method": "A0_no_transfer", "setting": "frozen_rf", "mean_macro_f1": base_mean,
            "mean_balanced_accuracy": float(base["balanced_accuracy"].mean()), "gain_vs_A0": 0.0,
            "worst_single_load_macro_delta": 0.0, "no_new_zero_recall": True, "eligible": True,
        }]),
        best,
    ], ignore_index=True)
    return selected_method, selected_setting, reason, tab, summary


def file_level_z(pipe, df: pd.DataFrame, feats: list[str]):
    Z = transform_z(pipe, df, feats)
    zdf = pd.DataFrame(Z, columns=feats)
    zdf["independent_object_id"] = df["independent_object_id"].astype(str).to_numpy()
    return zdf.groupby("independent_object_id")[feats].mean()


def weighted_mmd2(X, Y, wx=None):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    vals = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
    gamma = 1.0 / max(2.0 * med, EPS)
    Kxx = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    Kyy = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    Kxy = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    if wx is None:
        wx = np.ones(len(X)) / len(X)
    else:
        wx = np.asarray(wx, float); wx = wx / wx.sum()
    wy = np.ones(len(Y)) / len(Y)
    return float(wx @ Kxx @ wx + wy @ Kyy @ wy - 2.0 * wx @ Kxy @ wy)


def apply_real_method(src, tgt, feats, source_bundle, method, setting, seed):
    # Official seed uses the exact frozen STEP07 source model for A0/A2.
    if seed == SEED:
        base = source_bundle["pipeline"]
    else:
        base = fit_rf(src, feats, seed)

    if method == "A0_no_transfer":
        win, files = predict_direct(base, tgt, feats)
        artifact = {"method": method, "pseudo_labels_used": False}
        return win, files, artifact, base, None

    if method == "A2_CORAL":
        reg = float(setting.split("=")[1])
        Zta, coral = coral_target_to_source(base, src, tgt, feats, reg)
        win, files = predict_from_z(base, Zta, tgt)
        return win, files, coral, base, Zta

    if method == "A3_instance_weighting":
        clip = float(setting.split("=")[1])
        ratio, domclf, diag = fit_domain_ratio(base, src, tgt, feats, seed, clip)
        adapted = fit_rf(src, feats, seed, extra_weights=ratio)
        win, files = predict_direct(adapted, tgt, feats)
        artifact = {"method": method, "clip": clip, "diagnostics": diag, "pseudo_labels_used": False}
        artifact["source_importance_weights"] = ratio
        artifact["domain_classifier"] = domclf
        return win, files, artifact, adapted, None
    raise ValueError(method)


def target_table(files: pd.DataFrame):
    f = files.copy()
    f["target_id"] = f["independent_object_id"].map(lambda x: Path(str(x)).stem)
    f["truth_status"] = "unknown"
    cols = ["target_id", "independent_object_id", "truth_status", "n_windows"] + [f"p_{x}" for x in LABELS] + ["pred_label", "confidence", "margin", "entropy_norm", "window_consistency"]
    return f[cols].sort_values("target_id").reset_index(drop=True)


def target_stability(src, tgt, feats, source_bundle, method, setting):
    file_tabs = []
    for seed in SEEDS:
        _, ff, _, _, _ = apply_real_method(src, tgt, feats, source_bundle, method, setting, seed)
        t = target_table(ff)[["target_id", "pred_label", "confidence"]].rename(columns={"pred_label": f"pred_seed_{seed}", "confidence": f"confidence_seed_{seed}"})
        file_tabs.append(t)
    merged = file_tabs[0]
    for t in file_tabs[1:]:
        merged = merged.merge(t, on="target_id", how="inner")
    pred_cols = [f"pred_seed_{s}" for s in SEEDS]
    merged["all_seed_labels_agree"] = merged[pred_cols].nunique(axis=1) == 1
    return merged, {
        "seeds": SEEDS,
        "file_label_agreement_rate_all_three": float(merged["all_seed_labels_agree"].mean()),
        "files_with_seed_disagreement": merged.loc[~merged["all_seed_labels_agree"], "target_id"].tolist(),
    }


def main():
    t0 = time.perf_counter()
    interface = json.loads(Q3_INTERFACE.read_text(encoding="utf-8"))
    feats = list(interface["feature_columns"])
    src = pd.read_csv(SRC_CSV)
    tgt = pd.read_csv(TGT_CSV)
    source_bundle = joblib.load(SOURCE_MODEL)
    if len(src) != 400 or src["independent_object_id"].nunique() != 56 or len(tgt) != 240 or tgt["independent_object_id"].nunique() != 16 or len(feats) != 28:
        raise RuntimeError("Frozen STEP07/08 inventory changed")
    if not (tgt["truth_status"].astype(str) == "unknown").all():
        raise RuntimeError("Target truth unexpectedly present")

    objective = {
        "A2_CORAL": {
            "retained_source_parts": "Frozen 28-feature definition, source-fitted imputer/scaler, STEP07 RF classifier and file aggregation.",
            "targets": "global covariance/location mismatch after source standardization",
            "objective": "non-iterative linear target-to-source covariance alignment; no label loss and no pseudo-labels",
            "optimization": "closed-form eigendecomposition of regularized covariance matrices",
            "weights": "regularization chosen only by simulated held-out loads",
        },
        "A3_instance_weighting": {
            "retained_source_parts": "Frozen 28-feature definition, RF architecture/hyperparameters and file aggregation; classifier is refit on source labels only.",
            "targets": "covariate shift where only a subset of source support resembles target",
            "objective": "source supervised RF loss with source sample weight multiplied by clipped domain-density-ratio estimate",
            "optimization": "domain LogisticRegression on unlabeled domain identity, then weighted RF fitting",
            "weights": "posterior odds p(target|x)/p(source|x) under equal domain priors, clipped; clip chosen only on simulated held-out loads",
        },
        "pseudo_label_policy": {
            "used": False,
            "reason": "STEP08 target predictions collapsed to OR and target truth is unknown; self-training would risk reinforcing source-model bias.",
        },
    }
    (OUT / "method_objectives.json").write_text(json.dumps(objective, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "frozen_promotion_rule.json").write_text(json.dumps(PROMOTION, ensure_ascii=False, indent=2), encoding="utf-8")

    # Main simulated-target tuning/evaluation on the single frozen seed.
    sim, sim_pred, iw_diag = evaluate_simulated(src, feats, SEED)
    sim.to_csv(OUT / "simulated_target_all_methods_metrics.csv", index=False, encoding="utf-8-sig")
    sim_pred.to_csv(OUT / "simulated_target_file_predictions.csv", index=False, encoding="utf-8-sig")
    iw_diag.to_csv(OUT / "instance_weighting_diagnostics_by_load.csv", index=False, encoding="utf-8-sig")
    selected_method, selected_setting, selection_reason, setting_detail, method_summary = select_setting_and_method(sim)
    setting_detail.to_csv(OUT / "candidate_setting_selection.csv", index=False, encoding="utf-8-sig")
    method_summary.to_csv(OUT / "method_selection_summary.csv", index=False, encoding="utf-8-sig")

    # Bring forward the simple A1 failure from STEP08 as a fixed comparator, not retuned here.
    if STEP08_A1.exists():
        pd.read_csv(STEP08_A1).to_csv(OUT / "step08_A1_simple_baseline_reference.csv", index=False, encoding="utf-8-sig")

    # Stability of the selected method on simulated targets across three fixed seeds.
    stab_rows = []
    for seed in SEEDS:
        sm, _, _ = evaluate_simulated(src, feats, seed)
        g = sm[(sm["method"] == selected_method) & (sm["setting"] == selected_setting)]
        for _, r in g.iterrows():
            stab_rows.append(r.to_dict())
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT / "selected_method_simulated_seed_stability.csv", index=False, encoding="utf-8-sig")
    stab_summary = {
        "method": selected_method,
        "setting": selected_setting,
        "seeds": SEEDS,
        "macro_f1_mean_over_12_seed_load_cases": float(stab["macro_f1"].mean()),
        "macro_f1_std_over_12_seed_load_cases": float(stab["macro_f1"].std(ddof=1)),
        "minimum_macro_f1_over_12_seed_load_cases": float(stab["macro_f1"].min()),
    }

    # Real A-P: run A0 and only the simulated-rule-selected final method. No target metric is computed.
    base_pipe = source_bundle["pipeline"]
    bwin, bfile = predict_direct(base_pipe, tgt, feats)
    btab = target_table(bfile)
    btab.to_csv(OUT / "target_A_P_A0_no_transfer.csv", index=False, encoding="utf-8-sig")

    fwin, ffile, artifact, final_model_or_pipe, Zta = apply_real_method(src, tgt, feats, source_bundle, selected_method, selected_setting, SEED)
    ftab = target_table(ffile)
    ftab.to_csv(OUT / "target_A_P_final_method.csv", index=False, encoding="utf-8-sig")
    fwin.to_csv(OUT / "target_A_P_final_window_predictions.csv", index=False, encoding="utf-8-sig")

    # Save final adaptation/model bundle.
    final_bundle = {
        "method": selected_method,
        "setting": selected_setting,
        "selection_reason": selection_reason,
        "feature_columns": feats,
        "label_order": LABELS,
        "file_aggregation": "mean window class probabilities then argmax",
        "pseudo_labels_used": False,
        "target_fault_labels_used": False,
        "source_model_path": str(SOURCE_MODEL.relative_to(ROOT)),
        "model_or_pipeline": final_model_or_pipe,
        "adaptation_artifact": artifact,
    }
    joblib.dump(final_bundle, MODELS / "final_transfer_bundle.joblib")

    # Domain discrepancy before/after in the frozen source-standardized representation.
    sf = file_level_z(base_pipe, src, feats)
    tf = file_level_z(base_pipe, tgt, feats)
    mmd_before = weighted_mmd2(sf.to_numpy(float), tf.to_numpy(float))
    if selected_method == "A2_CORAL":
        zdf = pd.DataFrame(Zta, columns=feats)
        zdf["independent_object_id"] = tgt["independent_object_id"].astype(str).to_numpy()
        tfa = zdf.groupby("independent_object_id")[feats].mean()
        mmd_after = weighted_mmd2(sf.to_numpy(float), tfa.to_numpy(float))
        discrepancy_note = "MMD after CORAL compares source vs aligned target in frozen source-standardized space."
    elif selected_method == "A3_instance_weighting":
        ratios = np.asarray(artifact["source_importance_weights"], float)
        tmp = pd.DataFrame({"independent_object_id": src["independent_object_id"].astype(str), "ratio": ratios})
        fw = tmp.groupby("independent_object_id")["ratio"].mean().reindex(sf.index).to_numpy(float)
        mmd_after = weighted_mmd2(sf.to_numpy(float), tf.to_numpy(float), wx=fw)
        discrepancy_note = "MMD after weighting uses domain-importance-weighted source file means vs target file means; target coordinates are unchanged."
    else:
        mmd_after = mmd_before
        discrepancy_note = "No adaptation promoted; domain discrepancy is intentionally unchanged."
    domain_shift = {"mmd2_before": mmd_before, "mmd2_after_final_method": mmd_after, "decreased": bool(mmd_after < mmd_before), "note": discrepancy_note, "warning": "Lower distribution discrepancy is not treated as proof of higher diagnosis accuracy."}
    (OUT / "domain_discrepancy_before_after.json").write_text(json.dumps(domain_shift, ensure_ascii=False, indent=2), encoding="utf-8")

    # Collapse and prediction-count diagnostics (descriptive only; never used for method selection).
    counts_before = btab["pred_label"].value_counts().reindex(LABELS, fill_value=0).to_dict()
    counts_after = ftab["pred_label"].value_counts().reindex(LABELS, fill_value=0).to_dict()
    max_share_after = max(counts_after.values()) / len(ftab)
    collapse = {
        "A0_counts": {k: int(v) for k, v in counts_before.items()},
        "final_counts": {k: int(v) for k, v in counts_after.items()},
        "final_max_class_share": float(max_share_after),
        "collapse_flag_ge_0_875": bool(max_share_after >= 0.875),
        "used_for_selection": False,
        "interpretation": "A-P labels are unknown; class counts diagnose output collapse only and are not a correctness metric.",
    }
    (OUT / "target_class_collapse_check.json").write_text(json.dumps(collapse, ensure_ascii=False, indent=2), encoding="utf-8")

    # Real-target random-seed sensitivity, again descriptive only.
    stab_target, target_stab_summary = target_stability(src, tgt, feats, source_bundle, selected_method, selected_setting)
    stab_target.to_csv(OUT / "target_A_P_seed_stability.csv", index=False, encoding="utf-8-sig")
    (OUT / "stability_summary.json").write_text(json.dumps({"simulated": stab_summary, "target": target_stab_summary}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Simulated negative-transfer check for the selected method.
    base_sim = sim[(sim["method"] == "A0_no_transfer") & (sim["seed"] == SEED)].sort_values("simulated_target_load")
    final_sim = sim[(sim["method"] == selected_method) & (sim["setting"] == selected_setting) & (sim["seed"] == SEED)].sort_values("simulated_target_load")
    sim_compare = base_sim[["simulated_target_load", "macro_f1", "balanced_accuracy"]].rename(columns={"macro_f1": "A0_macro_f1", "balanced_accuracy": "A0_balanced_accuracy"}).merge(
        final_sim[["simulated_target_load", "macro_f1", "balanced_accuracy"]].rename(columns={"macro_f1": "final_macro_f1", "balanced_accuracy": "final_balanced_accuracy"}), on="simulated_target_load", how="left")
    sim_compare["macro_delta"] = sim_compare["final_macro_f1"] - sim_compare["A0_macro_f1"]
    sim_compare.to_csv(OUT / "no_transfer_vs_final_simulated.csv", index=False, encoding="utf-8-sig")
    negative_transfer = {
        "selected_method": selected_method,
        "selected_setting": selected_setting,
        "simulated_mean_macro_delta": float(sim_compare["macro_delta"].mean()),
        "simulated_worst_load_macro_delta": float(sim_compare["macro_delta"].min()),
        "frozen_rule_satisfied": bool(selected_method == "A0_no_transfer" or method_summary.loc[method_summary["method"] == selected_method, "eligible"].iloc[0]),
        "real_target_accuracy_verifiable": False,
        "real_target_negative_transfer_verifiable": False,
        "note": "Real A-P negative transfer cannot be measured without target truth; only simulated-target evidence is used for selection.",
    }
    (OUT / "negative_transfer_check.json").write_text(json.dumps(negative_transfer, ensure_ascii=False, indent=2), encoding="utf-8")

    unverified = {
        "target_truth": "unknown",
        "target_accuracy_f1": "not computable and not reported",
        "real_target_negative_transfer": "cannot be proven or disproven without truth",
        "simulated_target_scope": "load shift only; it does not reproduce all real shifts in sampling rate, RPM, sensor path, or bearing geometry",
        "pseudo_labels": "not used",
    }
    (OUT / "unverified_items.json").write_text(json.dumps(unverified, ensure_ascii=False, indent=2), encoding="utf-8")

    decision = {
        "selected_method": selected_method,
        "selected_setting": selected_setting,
        "selection_reason": selection_reason,
        "selection_used_A_P_truth": False,
        "selection_used_A_P_prediction_appearance": False,
        "pseudo_labels_used": False,
        "promotion_rule": PROMOTION,
        "real_target_prediction_counts": {k: int(v) for k, v in counts_after.items()},
        "real_target_collapse_flag": bool(collapse["collapse_flag_ge_0_875"]),
        "target_seed_label_agreement": target_stab_summary["file_label_agreement_rate_all_three"],
    }
    (OUT / "final_selection_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "step": "Q3B_STEP09",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start": git_sha(),
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "random_seed_primary": SEED,
        "stability_seeds": SEEDS,
        "source_sha256": sha256(SRC_CSV),
        "target_sha256": sha256(TGT_CSV),
        "step07_source_model_sha256": sha256(SOURCE_MODEL),
        "feature_count": len(feats),
        "coral_regs": CORAL_REGS,
        "instance_weight_clips": IW_CLIPS,
        "pseudo_labels_used": False,
        "target_labels_used": False,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Acceptance: all actual runs exist; selection is simulated-only; selected adaptation cannot violate frozen simulated rule.
    selected_eligible = bool(selected_method == "A0_no_transfer" or method_summary.loc[method_summary["method"] == selected_method, "eligible"].iloc[0])
    collapse_explained = bool((not collapse["collapse_flag_ge_0_875"]) or selected_method == "A0_no_transfer")
    checks = {
        "frozen_source_model_loaded": True,
        "source_56_target_16_inventory": src["independent_object_id"].nunique() == 56 and tgt["independent_object_id"].nunique() == 16,
        "target_truth_unknown": bool((tgt["truth_status"].astype(str) == "unknown").all()),
        "A2_and_A3_simulated_run": set(sim["method"].unique()) >= {"A0_no_transfer", "A2_CORAL", "A3_instance_weighting"},
        "selection_simulated_only": True,
        "pseudo_labels_not_used": True,
        "selected_method_obeys_frozen_rule": selected_eligible,
        "real_target_final_output_saved": len(ftab) == 16,
        "model_bundle_saved": (MODELS / "final_transfer_bundle.joblib").exists(),
        "domain_discrepancy_checked": np.isfinite(mmd_before) and np.isfinite(mmd_after),
        "stability_checked": len(stab_target) == 16 and len(stab) == 12,
        "negative_transfer_checked_on_simulated_target": np.isfinite(sim_compare["macro_delta"]).all(),
        "no_unexplained_collapse": collapse_explained,
        "target_accuracy_not_computed": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "selected_method": selected_method,
        "selected_setting": selected_setting,
        "selection_reason": selection_reason,
        "method_summary": method_summary.to_dict(orient="records"),
        "domain_discrepancy": domain_shift,
        "collapse": collapse,
        "stability": {"simulated": stab_summary, "target": target_stab_summary},
        "negative_transfer": negative_transfer,
        "target_accuracy_reported": False,
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Q3B / STEP09 unsupervised transfer",
        f"status={status}",
        f"selected_method={selected_method}",
        f"selected_setting={selected_setting}",
        f"selection_reason={selection_reason}",
        f"A0_sim_mean_macro_f1={base_sim['macro_f1'].mean():.6f}",
        f"final_sim_mean_macro_f1={final_sim['macro_f1'].mean():.6f}",
        f"sim_mean_macro_delta={sim_compare['macro_delta'].mean():.6f}",
        f"sim_worst_macro_delta={sim_compare['macro_delta'].min():.6f}",
        f"mmd2_before={mmd_before:.6f}",
        f"mmd2_after={mmd_after:.6f}",
        f"target_counts={json.dumps({k:int(v) for k,v in counts_after.items()}, ensure_ascii=False)}",
        f"target_seed_label_agreement={target_stab_summary['file_label_agreement_rate_all_three']:.6f}",
        "pseudo_labels_used=False",
        "target_accuracy_reported=False",
    ]
    (OUT / "result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
