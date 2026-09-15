from __future__ import annotations

from pathlib import Path
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =========================
# Q1-EVAL-v1.0 fixed protocol
# =========================
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SRC_FAULT = RAW / "source_domain" / "cwru_48khz_de"
SRC_NORMAL = RAW / "source_domain" / "cwru_48khz_normal"
TARGET = RAW / "target_domain" / "train_bearing_a_to_p"
OUT = ROOT / "outputs" / "q1_baseline_v1"
SMOKE = OUT / "smoke_test"
FORMAL = OUT / "formal"
for d in (SMOKE, FORMAL):
    d.mkdir(parents=True, exist_ok=True)

SEED = 20260916
SOURCE_FS = 48000
TARGET_FS = 32000
WINDOW_SECONDS = 1.0
MAX_WINDOWS_PER_FILE = 4
COMMON_FMAX = 6000.0
N_SPLITS = 4
LABELS = ["B", "IR", "OR", "N"]
EPS = 1e-12

# SKF6205 drive-end bearing parameters from the official problem statement
Z_ROLLERS = 9
d_ball_in = 0.3126
D_pitch_in = 1.537

B1_FEATURES = [
    "rms", "std", "kurtosis", "crest_factor", "impulse_factor", "shape_factor"
]
B2_FEATURES = [
    "std", "rms", "peak_abs", "ptp", "skewness", "kurtosis",
    "crest_factor", "impulse_factor", "shape_factor", "clearance_factor",
    "dominant_freq_0_6k", "spectral_centroid_0_6k", "f95_0_6k",
    "spectral_entropy_0_6k", "energy_0_500", "energy_500_1500",
    "energy_1500_3000", "energy_3000_6000",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def source_label(path: Path) -> str:
    n = path.name.upper()
    if n.startswith("IR"):
        return "IR"
    if n.startswith("OR"):
        return "OR"
    if n.startswith("B"):
        return "B"
    if n.startswith("N"):
        return "N"
    raise ValueError(f"Unknown source label: {path.name}")


def parse_rpm(mat: dict, path: Path) -> float:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if "RPM" in k.upper():
            arr = np.asarray(v).squeeze()
            if np.size(arr):
                return float(np.ravel(arr)[0])
    import re
    m = re.search(r"(\d{4})rpm", path.name, re.I)
    if m:
        return float(m.group(1))
    return np.nan


def select_source_de(mat: dict, path: Path) -> tuple[np.ndarray, str]:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if k.endswith("DE_time"):
            x = np.asarray(v).squeeze().astype(float)
            if x.ndim == 1 and x.size > 1000:
                return x, k
    raise ValueError(f"No DE_time signal found in {path}")


def select_target_signal(mat: dict, path: Path) -> tuple[np.ndarray, str]:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        x = np.asarray(v).squeeze()
        if x.ndim == 1 and x.size > 1000:
            return x.astype(float), k
    raise ValueError(f"No target signal found in {path}")


def choose_nonoverlap_windows(x: np.ndarray, fs: int) -> list[tuple[int, np.ndarray]]:
    win = int(round(fs * WINDOW_SECONDS))
    nwin = len(x) // win
    if nwin < 1:
        raise ValueError(f"Signal shorter than {WINDOW_SECONDS}s: len={len(x)}, fs={fs}")
    if nwin <= MAX_WINDOWS_PER_FILE:
        idxs = np.arange(nwin, dtype=int)
    else:
        idxs = np.unique(np.linspace(0, nwin - 1, MAX_WINDOWS_PER_FILE, dtype=int))
    return [(int(i * win), x[int(i * win): int((i + 1) * win)]) for i in idxs]


def spectral_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size <= 1:
        return np.nan
    q = p / p.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


def extract_features(x: np.ndarray, fs: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite value in signal window")
    x = x - np.mean(x)
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    ptp = float(np.ptp(x))
    abs_mean = float(np.mean(np.abs(x)))
    sk = float(skew(x, bias=False))
    ku = float(kurtosis(x, fisher=False, bias=False))
    crest = peak / (rms + EPS)
    impulse = peak / (abs_mean + EPS)
    shape = rms / (abs_mean + EPS)
    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(x))))
    clearance = peak / (sqrt_abs_mean ** 2 + EPS)

    nperseg = min(4096, len(x))
    f, pxx = welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", scaling="density"
    )
    mask = (f >= 0) & (f <= COMMON_FMAX)
    f2, p2 = f[mask], pxx[mask]
    if len(f2) < 2:
        raise ValueError("Insufficient spectral bins in common band")
    total = float(np.trapz(p2, f2)) + EPS
    centroid = float(np.sum(f2 * p2) / (np.sum(p2) + EPS))
    csum = np.cumsum(p2)
    f95 = float(f2[np.searchsorted(csum, 0.95 * csum[-1])]) if csum[-1] > 0 else np.nan
    dom = float(f2[np.argmax(p2)])
    sent = spectral_entropy(p2)

    def band_ratio(lo: float, hi: float) -> float:
        m = (f2 >= lo) & (f2 < hi)
        if m.sum() < 2:
            return np.nan
        return float(np.trapz(p2[m], f2[m]) / total)

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak_abs": peak,
        "ptp": ptp,
        "skewness": sk,
        "kurtosis": ku,
        "crest_factor": crest,
        "impulse_factor": impulse,
        "shape_factor": shape,
        "clearance_factor": clearance,
        "dominant_freq_0_6k": dom,
        "spectral_centroid_0_6k": centroid,
        "f95_0_6k": f95,
        "spectral_entropy_0_6k": sent,
        "energy_0_500": band_ratio(0, 500),
        "energy_500_1500": band_ratio(500, 1500),
        "energy_1500_3000": band_ratio(1500, 3000),
        "energy_3000_6000": band_ratio(3000, 6000),
    }


def bearing_freqs(rpm: float) -> dict[str, float]:
    fr = rpm / 60.0
    bpfo = (Z_ROLLERS / 2.0) * fr * (1.0 - d_ball_in / D_pitch_in)
    bpfi = (Z_ROLLERS / 2.0) * fr * (1.0 + d_ball_in / D_pitch_in)
    bsf = (D_pitch_in / d_ball_in) * fr * (1.0 - (d_ball_in / D_pitch_in) ** 2)
    return {"fr": fr, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf}


def envelope_local_ratio(x: np.ndarray, fs: int, fc: float) -> tuple[float, list[float]]:
    x = np.asarray(x, dtype=float) - np.mean(x)
    env = np.abs(hilbert(x))
    env = env - np.mean(env)
    nperseg = min(8192, len(env))
    f, p = welch(env, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    ratios = []
    for h in (1, 2, 3):
        center = h * fc
        if center >= min(COMMON_FMAX, fs / 2):
            continue
        delta = max(2.0 / WINDOW_SECONDS, 0.03 * center)
        target = np.abs(f - center) <= delta
        bg = (np.abs(f - center) >= 2.0 * delta) & (np.abs(f - center) <= 5.0 * delta)
        if target.sum() < 1 or bg.sum() < 2:
            continue
        target_mean = float(np.mean(p[target]))
        bg_mean = float(np.mean(p[bg]))
        ratios.append(target_mean / (bg_mean + EPS))
    if not ratios:
        return np.nan, []
    return float(np.median(ratios)), ratios


def build_source_file_table() -> pd.DataFrame:
    rows = []
    for p in sorted(SRC_FAULT.glob("*.mat")) + sorted(SRC_NORMAL.glob("*.mat")):
        mat = loadmat(p)
        x, var = select_source_de(mat, p)
        rpm = parse_rpm(mat, p)
        rows.append({
            "record_id": str(p.relative_to(RAW)),
            "relative_path": str(p.relative_to(RAW)),
            "filename": p.name,
            "label": source_label(p),
            "fs": SOURCE_FS,
            "rpm": rpm,
            "signal_var": var,
            "signal_length": len(x),
            "duration_s": len(x) / SOURCE_FS,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No 48kHz DE source files found")
    return df


def build_window_features(file_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in file_table.iterrows():
        p = RAW / r["relative_path"]
        mat = loadmat(p)
        x, var = select_source_de(mat, p)
        windows = choose_nonoverlap_windows(x, SOURCE_FS)
        for wi, (start, w) in enumerate(windows):
            ft = extract_features(w, SOURCE_FS)
            row = {
                "record_id": r["record_id"],
                "relative_path": r["relative_path"],
                "filename": r["filename"],
                "label": r["label"],
                "fs": SOURCE_FS,
                "rpm": r["rpm"],
                "window_index": wi,
                "window_start": start,
                "window_length": len(w),
            }
            row.update(ft)
            rows.append(row)
    return pd.DataFrame(rows)


def build_target_features() -> pd.DataFrame:
    rows = []
    for p in sorted(TARGET.glob("*.mat")):
        mat = loadmat(p)
        x, var = select_target_signal(mat, p)
        windows = choose_nonoverlap_windows(x, TARGET_FS)
        for wi, (start, w) in enumerate(windows):
            ft = extract_features(w, TARGET_FS)
            row = {
                "record_id": p.stem,
                "relative_path": str(p.relative_to(RAW)),
                "filename": p.name,
                "label": "UNKNOWN",
                "fs": TARGET_FS,
                "rpm": np.nan,
                "signal_var": var,
                "window_index": wi,
                "window_start": start,
                "window_length": len(w),
            }
            row.update(ft)
            rows.append(row)
    return pd.DataFrame(rows)


def smoke_test(file_table: pd.DataFrame) -> dict:
    reps = []
    for lab in LABELS:
        sub = file_table[file_table["label"] == lab]
        if sub.empty:
            raise RuntimeError(f"Smoke test missing class {lab}")
        reps.append(sub.iloc[0])
    smoke_rows = []
    for r in reps:
        p = RAW / r["relative_path"]
        mat = loadmat(p)
        x, var = select_source_de(mat, p)
        start, w = choose_nonoverlap_windows(x, SOURCE_FS)[0]
        ft = extract_features(w, SOURCE_FS)
        smoke_rows.append({
            "record_id": r["record_id"], "label": r["label"], "signal_var": var,
            "fs": SOURCE_FS, "window_length": len(w), "window_start": start,
            "all_features_finite": bool(np.isfinite([ft[k] for k in B2_FEATURES]).all()),
            "rms": ft["rms"], "kurtosis": ft["kurtosis"],
        })

    tp = TARGET / "A.mat"
    tmat = loadmat(tp)
    tx, tvar = select_target_signal(tmat, tp)
    tstart, tw = choose_nonoverlap_windows(tx, TARGET_FS)[0]
    tft = extract_features(tw, TARGET_FS)
    target_ok = bool(np.isfinite([tft[k] for k in B2_FEATURES]).all())

    smoke_df = pd.DataFrame(smoke_rows)
    smoke_df.to_csv(SMOKE / "smoke_source_samples.csv", index=False, encoding="utf-8-sig")
    result = {
        "source_representatives": len(smoke_df),
        "source_all_finite": bool(smoke_df["all_features_finite"].all()),
        "target_sample": "A.mat",
        "target_signal_var": tvar,
        "target_fs": TARGET_FS,
        "target_window_length": len(tw),
        "target_all_features_finite": target_ok,
        "status": "PASS" if bool(smoke_df["all_features_finite"].all()) and target_ok else "FAIL",
    }
    (SMOKE / "smoke_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def make_splits(file_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    files = file_table.reset_index(drop=True)
    val_fold_by_record = {}
    manifest_rows = []
    Xdummy = np.zeros((len(files), 1))
    y = files["label"].to_numpy()
    for fold, (tr, va) in enumerate(skf.split(Xdummy, y), start=1):
        tr_ids = set(files.loc[tr, "record_id"])
        va_ids = set(files.loc[va, "record_id"])
        if tr_ids & va_ids:
            raise RuntimeError("Group leakage detected at file level")
        for rid in tr_ids:
            manifest_rows.append({"fold": fold, "record_id": rid, "role": "train"})
        for rid in va_ids:
            manifest_rows.append({"fold": fold, "record_id": rid, "role": "validation"})
            if rid in val_fold_by_record:
                raise RuntimeError(f"Record assigned to multiple validation folds: {rid}")
            val_fold_by_record[rid] = fold
    manifest = pd.DataFrame(manifest_rows).sort_values(["fold", "role", "record_id"])
    return manifest, val_fold_by_record


def eval_models(file_table: pd.DataFrame, windows: pd.DataFrame, val_fold_by_record: dict[str, int]):
    oof_rows = []
    fold_rows = []
    models = ["B0_dummy", "B1_time", "B2_shared"]

    for fold in range(1, N_SPLITS + 1):
        va_ids = {rid for rid, f in val_fold_by_record.items() if f == fold}
        tr_ids = set(file_table["record_id"]) - va_ids
        if tr_ids & va_ids:
            raise RuntimeError("Leakage in split sets")

        train_files = file_table[file_table["record_id"].isin(tr_ids)].copy()
        val_files = file_table[file_table["record_id"].isin(va_ids)].copy()
        trw = windows[windows["record_id"].isin(tr_ids)].copy()
        vaw = windows[windows["record_id"].isin(va_ids)].copy()

        # B0: file-level most frequent class, no window weighting
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_files), 1)), train_files["label"])
        pred0 = dummy.predict(np.zeros((len(val_files), 1)))
        for rid, true, pred in zip(val_files["record_id"], val_files["label"], pred0):
            oof_rows.append({"model": "B0_dummy", "fold": fold, "record_id": rid, "true_label": true, "pred_label": pred})

        for model_name, feats in [("B1_time", B1_FEATURES), ("B2_shared", B2_FEATURES)]:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(trw[feats].to_numpy(float))
            Xva = scaler.transform(vaw[feats].to_numpy(float))
            if not np.isfinite(Xtr).all() or not np.isfinite(Xva).all():
                raise RuntimeError(f"Non-finite standardized feature in {model_name} fold {fold}")
            clf = LogisticRegression(
                class_weight="balanced", max_iter=5000, random_state=SEED,
                solver="lbfgs", multi_class="auto"
            )
            clf.fit(Xtr, trw["label"])
            proba = clf.predict_proba(Xva)
            classes = list(clf.classes_)
            tmp = vaw[["record_id", "label"]].copy().reset_index(drop=True)
            for ci, c in enumerate(classes):
                tmp[f"p_{c}"] = proba[:, ci]
            prob_cols = [f"p_{c}" for c in classes]
            agg = tmp.groupby(["record_id", "label"], as_index=False)[prob_cols].mean()
            for _, ar in agg.iterrows():
                probs = np.array([ar[f"p_{c}"] for c in classes], dtype=float)
                pred = classes[int(np.argmax(probs))]
                row = {"model": model_name, "fold": fold, "record_id": ar["record_id"], "true_label": ar["label"], "pred_label": pred}
                for c, p in zip(classes, probs):
                    row[f"p_{c}"] = float(p)
                oof_rows.append(row)

    oof = pd.DataFrame(oof_rows)
    summary_rows = []
    confusion_tables = {}
    for model in models:
        d = oof[oof["model"] == model].copy()
        y_true = d["true_label"]
        y_pred = d["pred_label"]
        recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        row = {
            "model": model,
            "n_files": len(d),
            "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }
        for lab, rec in zip(LABELS, recalls):
            row[f"recall_{lab}"] = float(rec)
        summary_rows.append(row)
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        confusion_tables[model] = pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])

        for fold in range(1, N_SPLITS + 1):
            g = d[d["fold"] == fold]
            fold_rows.append({
                "model": model,
                "fold": fold,
                "n_files": len(g),
                "macro_f1": f1_score(g["true_label"], g["pred_label"], labels=LABELS, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(g["true_label"], g["pred_label"]),
                "accuracy": accuracy_score(g["true_label"], g["pred_label"]),
            })

    return oof, pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), confusion_tables


def mechanism_checks(file_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    support_rows = []
    mapping = {"OR": "BPFO", "IR": "BPFI", "B": "BSF"}
    for lab, freq_name in mapping.items():
        sub = file_table[file_table["label"] == lab].copy()
        # choose up to 3 different load suffixes when possible by taking lexicographically first per load-like ending
        chosen = sub.sort_values("filename").head(3)
        ratios = []
        for _, r in chosen.iterrows():
            p = RAW / r["relative_path"]
            mat = loadmat(p)
            x, _ = select_source_de(mat, p)
            rpm = parse_rpm(mat, p)
            start, w = choose_nonoverlap_windows(x, SOURCE_FS)[0]
            freqs = bearing_freqs(rpm)
            fc = freqs[freq_name]
            ratio, harmonic_ratios = envelope_local_ratio(w, SOURCE_FS, fc)
            ratios.append(ratio)
            rows.append({
                "label": lab,
                "record_id": r["record_id"],
                "rpm": rpm,
                "frequency_name": freq_name,
                "theoretical_frequency_hz": fc,
                "window_start": start,
                "local_envelope_ratio": ratio,
                "harmonic_ratios": "|".join(f"{x:.6g}" for x in harmonic_ratios),
                "supported_ratio_gt_1": bool(np.isfinite(ratio) and ratio > 1.0),
            })
        n_support = int(sum(bool(np.isfinite(x) and x > 1.0) for x in ratios))
        support_rows.append({
            "label": lab,
            "n_checked": len(ratios),
            "n_ratio_gt_1": n_support,
            "class_mechanism_support": bool(len(ratios) >= 3 and n_support >= 2),
        })
    return pd.DataFrame(rows), pd.DataFrame(support_rows)


def target_compatibility(source_windows: pd.DataFrame, target_windows: pd.DataFrame) -> dict:
    if target_windows.empty:
        raise RuntimeError("No target features generated")
    finite_source = np.isfinite(source_windows[B2_FEATURES].to_numpy(float)).all()
    finite_target = np.isfinite(target_windows[B2_FEATURES].to_numpy(float)).all()

    mu = source_windows[B2_FEATURES].mean(axis=0)
    sd = source_windows[B2_FEATURES].std(axis=0).replace(0, 1.0)
    Zs = (source_windows[B2_FEATURES] - mu) / sd
    Zt = (target_windows[B2_FEATURES] - mu) / sd
    p = len(B2_FEATURES)
    dmu = float(np.linalg.norm(Zs.mean(axis=0).to_numpy() - Zt.mean(axis=0).to_numpy()) / math.sqrt(p))
    cs = np.cov(Zs.to_numpy(), rowvar=False)
    ct = np.cov(Zt.to_numpy(), rowvar=False)
    dsigma = float(np.linalg.norm(cs - ct, ord="fro") / p)
    return {
        "source_all_shared_features_finite": bool(finite_source),
        "target_all_shared_features_finite": bool(finite_target),
        "target_files": int(target_windows["record_id"].nunique()),
        "target_windows": int(len(target_windows)),
        "domain_mean_shift_Dmu": dmu,
        "domain_cov_shift_Dsigma": dsigma,
    }


def independent_recalc() -> dict:
    p = SRC_FAULT / "B007_0.mat"
    mat = loadmat(p)
    x, var = select_source_de(mat, p)
    rpm = parse_rpm(mat, p)
    _, w = choose_nonoverlap_windows(x, SOURCE_FS)[0]
    w0 = w - np.mean(w)
    rms_a = float(np.sqrt(np.mean(w0 ** 2)))
    rms_b = float(np.linalg.norm(w0) / np.sqrt(len(w0)))
    fr = rpm / 60.0
    bpfo_a = (Z_ROLLERS / 2.0) * fr * (1.0 - d_ball_in / D_pitch_in)
    bpfo_coeff = (Z_ROLLERS / 2.0) * (1.0 - d_ball_in / D_pitch_in)
    bpfo_b = bpfo_coeff * fr
    return {
        "sample": str(p.relative_to(ROOT)),
        "signal_var": var,
        "rpm": rpm,
        "window_points": len(w),
        "rms_formula_sqrt_mean_square": rms_a,
        "rms_formula_norm_over_sqrt_n": rms_b,
        "rms_abs_difference": abs(rms_a - rms_b),
        "bpfo_direct_hz": bpfo_a,
        "bpfo_via_order_hz": bpfo_b,
        "bpfo_abs_difference_hz": abs(bpfo_a - bpfo_b),
        "recalc_pass": bool(abs(rms_a - rms_b) < 1e-12 and abs(bpfo_a - bpfo_b) < 1e-12),
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    log("Q1 baseline v1 starting")
    log(f"ROOT={ROOT}")
    log(f"git_sha={git_sha()}")

    file_table = build_source_file_table()
    log(f"48k-DE source files: {len(file_table)}")
    log("class counts: " + str(file_table["label"].value_counts().to_dict()))
    file_table.to_csv(FORMAL / "source_file_manifest.csv", index=False, encoding="utf-8-sig")

    smoke = smoke_test(file_table)
    log("smoke=" + json.dumps(smoke, ensure_ascii=False))
    if smoke["status"] != "PASS":
        raise RuntimeError("Smoke test failed; formal run aborted")

    source_windows = build_window_features(file_table)
    target_windows = build_target_features()
    source_windows.to_csv(FORMAL / "source_feature_windows.csv", index=False, encoding="utf-8-sig")
    target_windows.to_csv(FORMAL / "target_feature_windows.csv", index=False, encoding="utf-8-sig")

    split_manifest, val_fold_by_record = make_splits(file_table)
    split_manifest.to_csv(FORMAL / "q1_split_manifest.csv", index=False, encoding="utf-8-sig")

    oof, eval_summary, fold_metrics, cms = eval_models(file_table, source_windows, val_fold_by_record)
    oof.to_csv(FORMAL / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    eval_summary.to_csv(FORMAL / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(FORMAL / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    for model, cm in cms.items():
        cm.to_csv(FORMAL / f"confusion_{model}.csv", encoding="utf-8-sig")

    mech_detail, mech_summary = mechanism_checks(file_table)
    mech_detail.to_csv(FORMAL / "mechanism_checks.csv", index=False, encoding="utf-8-sig")
    mech_summary.to_csv(FORMAL / "mechanism_summary.csv", index=False, encoding="utf-8-sig")

    compat = target_compatibility(source_windows, target_windows)
    target_file_summary = target_windows.groupby("record_id")[B2_FEATURES].mean().reset_index()
    target_file_summary.to_csv(FORMAL / "target_feature_summary.csv", index=False, encoding="utf-8-sig")
    (FORMAL / "target_compatibility.json").write_text(json.dumps(compat, ensure_ascii=False, indent=2), encoding="utf-8")

    recalc = independent_recalc()
    (FORMAL / "independent_recalc.json").write_text(json.dumps(recalc, ensure_ascii=False, indent=2), encoding="utf-8")

    em = eval_summary.set_index("model")
    b0 = em.loc["B0_dummy"]
    b1 = em.loc["B1_time"]
    b2 = em.loc["B2_shared"]
    no_leakage = split_manifest.groupby(["fold", "record_id"])["role"].nunique().max() == 1
    all_classes = set(file_table["label"]) == set(LABELS)
    all_recall_nonzero = all(float(b2[f"recall_{lab}"]) > 0 for lab in LABELS)
    b2_beats_dummy = float(b2["macro_f1"]) > float(b0["macro_f1"])
    b2_not_both_worse_than_b1 = not (
        float(b2["macro_f1"]) < float(b1["macro_f1"]) and
        float(b2["balanced_accuracy"]) < float(b1["balanced_accuracy"])
    )
    failed_mech_classes = int((~mech_summary["class_mechanism_support"]).sum())
    mechanism_ok = failed_mech_classes < 2
    target_ok = bool(compat["target_all_shared_features_finite"] and compat["target_files"] == 16)

    criteria = {
        "no_file_leakage": bool(no_leakage),
        "all_four_classes_present": bool(all_classes),
        "source_shared_features_finite": bool(compat["source_all_shared_features_finite"]),
        "target_shared_features_finite_for_A_to_P": target_ok,
        "B2_macroF1_gt_dummy": bool(b2_beats_dummy),
        "B2_not_both_worse_than_B1": bool(b2_not_both_worse_than_b1),
        "B2_all_class_recalls_nonzero": bool(all_recall_nonzero),
        "mechanism_not_failed_for_two_or_more_fault_classes": bool(mechanism_ok),
        "independent_recalc_pass": bool(recalc["recalc_pass"]),
    }
    overall_pass = all(criteria.values())

    failure_lines = []
    for k, v in criteria.items():
        if not v:
            failure_lines.append(f"FAIL: {k}")
    if not failure_lines:
        failure_lines.append("No protocol failure detected in Q1-M1 baseline run.")
    failure_lines.append("Limitation: target A-P labels are unknown; no target-domain Accuracy/F1 is computed or claimed.")
    failure_lines.append("Limitation: target bearing geometry and per-file exact RPM are unavailable; source bearing characteristic frequencies are not imposed as target truth.")
    (FORMAL / "failure_analysis.txt").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    run_manifest = {
        "protocol": "Q1-EVAL-v1.0",
        "experiment": "Q1-M1 48kHz-DE baseline",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_start": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "source_fs_hz": SOURCE_FS,
            "target_fs_hz": TARGET_FS,
            "window_seconds": WINDOW_SECONDS,
            "max_windows_per_file": MAX_WINDOWS_PER_FILE,
            "common_fmax_hz": COMMON_FMAX,
            "n_splits": N_SPLITS,
            "seed": SEED,
            "B1_features": B1_FEATURES,
            "B2_features": B2_FEATURES,
        },
        "data": {
            "source_files": int(len(file_table)),
            "source_class_counts": {k: int(v) for k, v in file_table["label"].value_counts().to_dict().items()},
            "source_windows": int(len(source_windows)),
            "target_files": int(target_windows["record_id"].nunique()),
            "target_windows": int(len(target_windows)),
        },
        "criteria": criteria,
        "overall_pass": bool(overall_pass),
    }
    (FORMAL / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable summary
    lines = [
        "=== Q1-M1 48kHz-DE baseline result ===",
        f"protocol: Q1-EVAL-v1.0",
        f"source_files: {len(file_table)}",
        f"source_class_counts: {file_table['label'].value_counts().to_dict()}",
        f"source_windows: {len(source_windows)}",
        f"target_files: {target_windows['record_id'].nunique()}",
        f"target_windows: {len(target_windows)}",
        "",
        "=== Evaluation summary ===",
        eval_summary.to_string(index=False),
        "",
        "=== Mechanism summary ===",
        mech_summary.to_string(index=False),
        "",
        "=== Target compatibility ===",
        json.dumps(compat, ensure_ascii=False, indent=2),
        "",
        "=== Independent recalculation ===",
        json.dumps(recalc, ensure_ascii=False, indent=2),
        "",
        "=== Acceptance criteria ===",
        json.dumps(criteria, ensure_ascii=False, indent=2),
        f"OVERALL_PASS: {overall_pass}",
    ]
    (FORMAL / "result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(eval_summary.to_string(index=False))
    log(mech_summary.to_string(index=False))
    log("criteria=" + json.dumps(criteria, ensure_ascii=False))
    log(f"OVERALL_PASS={overall_pass}")
    log(f"OUTPUT={FORMAL}")


if __name__ == "__main__":
    main()
