from __future__ import annotations

from pathlib import Path
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import q1_baseline_v1 as base

BASE_OUT = ROOT / "outputs" / "q1_baseline_v1" / "formal"
OUT = ROOT / "outputs" / "q1_improvement_v1"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = base.LABELS
SEED = base.SEED
N_SPLITS = base.N_SPLITS
B2 = base.B2_FEATURES
SCALE_FREE = [
    "skewness", "kurtosis", "crest_factor", "impulse_factor",
    "shape_factor", "clearance_factor", "spectral_entropy_0_6k",
    "energy_0_500", "energy_500_1500", "energy_1500_3000", "energy_3000_6000",
]

# Pre-registered improvement criteria, fixed before this script is run.
H1_MAX_MACRO_F1_DROP = 0.02
H1_MIN_DMU_REDUCTION_FRAC = 0.10
H2_MIN_MACRO_F1_GAIN = 0.01
H2_MIN_MINFOLD_GAIN = 0.05
H2_MAX_MACRO_F1_DROP_IF_STABILITY_GAIN = 0.01
H3_MIN_B_SUPPORT = 2  # out of 3 fixed B records


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_inputs():
    files = pd.read_csv(BASE_OUT / "source_file_manifest.csv")
    src = pd.read_csv(BASE_OUT / "source_feature_windows.csv")
    tgt = pd.read_csv(BASE_OUT / "target_feature_windows.csv")
    split = pd.read_csv(BASE_OUT / "q1_split_manifest.csv")
    return files, src, tgt, split


def val_fold_map(split: pd.DataFrame) -> dict[str, int]:
    v = split[split["role"] == "validation"][["record_id", "fold"]].copy()
    if v["record_id"].duplicated().any():
        raise RuntimeError("A record appears in multiple validation folds")
    return {str(r.record_id): int(r.fold) for r in v.itertuples(index=False)}


def evaluate(files: pd.DataFrame, windows: pd.DataFrame, fold_map: dict[str, int], features: list[str], aggregation: str, name: str):
    oof_rows = []
    fold_rows = []
    all_ids = set(files["record_id"].astype(str))
    if set(fold_map) != all_ids:
        raise RuntimeError("Split manifest does not cover all source records exactly once as validation")

    for fold in range(1, N_SPLITS + 1):
        va_ids = {rid for rid, f in fold_map.items() if f == fold}
        tr_ids = all_ids - va_ids
        if tr_ids & va_ids:
            raise RuntimeError("File leakage detected")
        trw = windows[windows["record_id"].isin(tr_ids)].copy()
        vaw = windows[windows["record_id"].isin(va_ids)].copy()
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(trw[features].to_numpy(float))
        Xva = scaler.transform(vaw[features].to_numpy(float))
        if not np.isfinite(Xtr).all() or not np.isfinite(Xva).all():
            raise RuntimeError(f"Non-finite standardized values in {name} fold {fold}")
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED, solver="lbfgs")
        clf.fit(Xtr, trw["label"])
        proba = clf.predict_proba(Xva)
        classes = list(clf.classes_)
        tmp = vaw[["record_id", "label"]].copy().reset_index(drop=True)
        for ci, c in enumerate(classes):
            tmp[f"p_{c}"] = proba[:, ci]
        pcols = [f"p_{c}" for c in classes]
        gb = tmp.groupby(["record_id", "label"], as_index=False)[pcols]
        if aggregation == "mean":
            agg = gb.mean()
        elif aggregation == "median":
            agg = gb.median()
        else:
            raise ValueError(aggregation)
        for _, r in agg.iterrows():
            probs = np.asarray([r[f"p_{c}"] for c in classes], dtype=float)
            probs = probs / probs.sum()
            pred = classes[int(np.argmax(probs))]
            oof_rows.append({"experiment": name, "fold": fold, "record_id": r["record_id"], "true_label": r["label"], "pred_label": pred})

    oof = pd.DataFrame(oof_rows)
    y = oof["true_label"]
    p = oof["pred_label"]
    recalls = recall_score(y, p, labels=LABELS, average=None, zero_division=0)
    summary = {
        "experiment": name,
        "feature_count": len(features),
        "aggregation": aggregation,
        "n_files": int(len(oof)),
        "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
    }
    for lab, rec in zip(LABELS, recalls):
        summary[f"recall_{lab}"] = float(rec)
    for fold in range(1, N_SPLITS + 1):
        d = oof[oof["fold"] == fold]
        fold_rows.append({
            "experiment": name,
            "fold": fold,
            "n_files": int(len(d)),
            "macro_f1": float(f1_score(d["true_label"], d["pred_label"], labels=LABELS, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(d["true_label"], d["pred_label"])),
            "accuracy": float(accuracy_score(d["true_label"], d["pred_label"])),
        })
    return summary, pd.DataFrame(fold_rows), oof


def domain_shift(src: pd.DataFrame, tgt: pd.DataFrame, features: list[str]) -> dict:
    mu = src[features].mean(axis=0)
    sd = src[features].std(axis=0).replace(0, 1.0)
    zs = (src[features] - mu) / sd
    zt = (tgt[features] - mu) / sd
    p = len(features)
    dmu = float(np.linalg.norm(zs.mean(axis=0).to_numpy() - zt.mean(axis=0).to_numpy()) / math.sqrt(p))
    cs = np.cov(zs.to_numpy(), rowvar=False)
    ct = np.cov(zt.to_numpy(), rowvar=False)
    ds = float(np.linalg.norm(cs - ct, ord="fro") / p)
    return {"feature_count": p, "Dmu": dmu, "Dsigma": ds}


def adaptive_bsf_check(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    chosen = files[files["label"] == "B"].sort_values("filename").head(3)
    carrier_bands = [(500, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
    for _, r in chosen.iterrows():
        p = base.RAW / r["relative_path"]
        mat = loadmat(p)
        x, _ = base.select_source_de(mat, p)
        rpm = base.parse_rpm(mat, p)
        _, w = base.choose_nonoverlap_windows(x, base.SOURCE_FS)[0]
        candidates = []
        for lo, hi in carrier_bands:
            sos = butter(4, [lo, hi], btype="bandpass", fs=base.SOURCE_FS, output="sos")
            wf = sosfiltfilt(sos, w - np.mean(w))
            score = float(kurtosis(wf, fisher=False, bias=False))
            candidates.append((score, lo, hi, wf))
        score, lo, hi, wf = max(candidates, key=lambda t: t[0])
        bsf = base.bearing_freqs(rpm)["BSF"]
        ratio, hrs = base.envelope_local_ratio(wf, base.SOURCE_FS, bsf)
        rows.append({
            "record_id": r["record_id"], "rpm": rpm, "bsf_hz": bsf,
            "selected_carrier_lo_hz": lo, "selected_carrier_hi_hz": hi,
            "carrier_kurtosis": score, "adaptive_bsf_ratio": ratio,
            "harmonic_ratios": "|".join(f"{z:.6g}" for z in hrs),
            "supported_ratio_gt_1": bool(np.isfinite(ratio) and ratio > 1.0),
        })
    return pd.DataFrame(rows)


def main():
    started = datetime.now(timezone.utc).isoformat()
    files, src, tgt, split = load_inputs()
    fmap = val_fold_map(split)

    # Fair-comparison sanity checks.
    if len(files) != 56 or files["record_id"].nunique() != 56:
        raise RuntimeError("Unexpected source-file set; baseline comparability broken")
    if tgt["record_id"].nunique() != 16:
        raise RuntimeError("Unexpected target-file set")

    experiments = []
    folds = []
    oofs = []

    s0, f0, o0 = evaluate(files, src, fmap, B2, "mean", "M1_baseline_B2_mean")
    experiments.append(s0); folds.append(f0); oofs.append(o0)
    if abs(s0["macro_f1"] - 0.7020285514625138) > 1e-12:
        raise RuntimeError(f"Baseline reproduction mismatch: {s0['macro_f1']}")

    # H1: remove absolute-amplitude and absolute-frequency features.
    s1, f1, o1 = evaluate(files, src, fmap, SCALE_FREE, "mean", "H1_scale_free_features")
    experiments.append(s1); folds.append(f1); oofs.append(o1)

    # H2: change only file aggregation from mean to median, keep B2 features/model/splits.
    s2, f2, o2 = evaluate(files, src, fmap, B2, "median", "H2_median_probability_aggregation")
    experiments.append(s2); folds.append(f2); oofs.append(o2)

    eval_df = pd.DataFrame(experiments)
    fold_df = pd.concat(folds, ignore_index=True)
    oof_df = pd.concat(oofs, ignore_index=True)
    eval_df.to_csv(OUT / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT / "fold_comparison.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(OUT / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    shift_base = domain_shift(src, tgt, B2)
    shift_h1 = domain_shift(src, tgt, SCALE_FREE)
    pd.DataFrame([
        {"experiment": "M1_baseline_B2_mean", **shift_base},
        {"experiment": "H1_scale_free_features", **shift_h1},
    ]).to_csv(OUT / "domain_shift_comparison.csv", index=False, encoding="utf-8-sig")

    base_min_fold = float(f0["macro_f1"].min())
    h2_min_fold = float(f2["macro_f1"].min())
    h1_dmu_reduction = (shift_base["Dmu"] - shift_h1["Dmu"]) / shift_base["Dmu"]
    h1_keep = bool(
        s1["macro_f1"] >= s0["macro_f1"] - H1_MAX_MACRO_F1_DROP
        and h1_dmu_reduction >= H1_MIN_DMU_REDUCTION_FRAC
        and min(s1[f"recall_{lab}"] for lab in LABELS) > 0
    )
    h2_keep = bool(
        (s2["macro_f1"] >= s0["macro_f1"] + H2_MIN_MACRO_F1_GAIN)
        or (
            h2_min_fold >= base_min_fold + H2_MIN_MINFOLD_GAIN
            and s2["macro_f1"] >= s0["macro_f1"] - H2_MAX_MACRO_F1_DROP_IF_STABILITY_GAIN
        )
    )

    # H3: B mechanism only. Same three B records, same first 1 s; only envelope carrier selection changes.
    b_adapt = adaptive_bsf_check(files)
    b_adapt.to_csv(OUT / "H3_adaptive_bsf_checks.csv", index=False, encoding="utf-8-sig")
    b_support = int(b_adapt["supported_ratio_gt_1"].sum())
    h3_keep = bool(b_support >= H3_MIN_B_SUPPORT)

    hypotheses = pd.DataFrame([
        {
            "id": "H1", "problem": "Large source-target feature shift and speed/amplitude mismatch",
            "change": "Remove absolute-amplitude and absolute-frequency features; keep only dimensionless/relative features",
            "fair_control": "Same 56 files, same saved 4 folds, same logistic probe, same mean aggregation",
            "stop_rule": f"Keep only if Macro-F1 drop <= {H1_MAX_MACRO_F1_DROP:.2f}, Dmu reduction >= {H1_MIN_DMU_REDUCTION_FRAC:.0%}, all recalls > 0",
            "kept": h1_keep,
        },
        {
            "id": "H2", "problem": "Fold instability may reflect noisy windows dominating arithmetic-mean probability",
            "change": "Change file-level probability aggregation from mean to median only",
            "fair_control": "Same B2 features, same 56 files, same folds, same scaling and classifier",
            "stop_rule": "Keep for >=0.01 Macro-F1 gain, or >=0.05 min-fold gain with <=0.01 Macro-F1 loss",
            "kept": h2_keep,
        },
        {
            "id": "H3", "problem": "Baseline B/BSF mechanism support was 0/3",
            "change": "Before Hilbert envelope, choose one of four fixed carrier bands by maximum bandpassed kurtosis",
            "fair_control": "Same three B files, same first 1 s windows, same BSF formula and ratio>1 rule",
            "stop_rule": "Keep for mechanism analysis only if >=2/3 B records achieve ratio>1",
            "kept": h3_keep,
        },
    ])
    hypotheses.to_csv(OUT / "improvement_hypotheses.csv", index=False, encoding="utf-8-sig")

    # Final retention is deliberately conservative: classification route changes only if a preregistered rule passes.
    retained_classifier = "M1_baseline_B2_mean"
    if h1_keep:
        retained_classifier = "H1_scale_free_features"
    if h2_keep and not h1_keep:
        retained_classifier = "H2_median_probability_aggregation"
    if h1_keep and h2_keep:
        # They change different factors, but their combination was not preregistered/tested. Keep the stronger single-factor H1 for now.
        retained_classifier = "H1_scale_free_features"

    report = {
        "protocol": "Q1-IMPROVE-v1.0",
        "baseline_protocol": "Q1-EVAL-v1.0",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "baseline_reproduced_macro_f1": s0["macro_f1"],
        "H1": {
            "macro_f1": s1["macro_f1"], "Dmu": shift_h1["Dmu"],
            "Dmu_reduction_fraction": h1_dmu_reduction, "kept": h1_keep,
        },
        "H2": {
            "macro_f1": s2["macro_f1"], "baseline_min_fold_f1": base_min_fold,
            "H2_min_fold_f1": h2_min_fold, "kept": h2_keep,
        },
        "H3": {"B_support_count_of_3": b_support, "kept_for_mechanism_only": h3_keep},
        "retained_classifier": retained_classifier,
        "note": "No target labels were used. No random seed search was performed. Uncombined single-factor experiments only.",
    }
    (OUT / "improvement_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "result_summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Q1 failure-driven improvement ===\n")
        f.write(eval_df.to_string(index=False) + "\n\n")
        f.write("=== Domain shift ===\n")
        f.write(pd.DataFrame([
            {"experiment": "M1_baseline_B2_mean", **shift_base},
            {"experiment": "H1_scale_free_features", **shift_h1},
        ]).to_string(index=False) + "\n\n")
        f.write("=== H3 adaptive B mechanism ===\n")
        f.write(b_adapt.to_string(index=False) + "\n\n")
        f.write("=== Decisions ===\n")
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(eval_df.to_string(index=False))
    print("baseline_shift=", shift_base)
    print("H1_shift=", shift_h1)
    print("H1_keep=", h1_keep, "H2_keep=", h2_keep, "H3_keep=", h3_keep)
    print("retained_classifier=", retained_classifier)


if __name__ == "__main__":
    main()
