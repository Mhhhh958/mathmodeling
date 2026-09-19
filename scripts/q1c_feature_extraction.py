# AI ASSISTANCE NOTICE
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ChatGPT；版本/型号：GPT-5.6 Sol（ChatGPT 2026-08-06更新版）；
# 开发机构/公司：OpenAI；版本发布日期：2026-08-06。
# 人工智能仅用于代码检查、调试建议与说明整理；最终算法、参数与结果由参赛队审查并由冻结复现链验证。
from __future__ import annotations

from pathlib import Path
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert, periodogram, welch
from scipy.stats import kurtosis, skew


# ============================================================
# Q1C / STEP04: windowing + full-data deterministic features
# No classifier training. No learned imputation/scaling/selection.
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SRC_FAULT = RAW / "source_domain" / "cwru_48khz_de"
SRC_NORMAL = RAW / "source_domain" / "cwru_48khz_normal"
TARGET = RAW / "target_domain" / "train_bearing_a_to_p"
OUT = ROOT / "outputs" / "q1c_feature_extraction"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_FS = 48000
TARGET_FS = 32000
WINDOW_SECONDS = 1.0
OVERLAP = 0.50
STRIDE_SECONDS = WINDOW_SECONDS * (1.0 - OVERLAP)
COMMON_FMAX = 6000.0
TARGET_RPM_APPROX = 600.0  # official statement: approximately 600 rpm; metadata only
EPS = 1e-12

# Official SKF6205 drive-end geometry used only for source-domain mechanism features.
Z_ROLLERS_6205 = 9
BALL_D_6205_IN = 0.3126
PITCH_D_6205_IN = 1.537

# SKF6203 parameters are retained only to make the separation explicit; not used for M1.
Z_ROLLERS_6203 = 9
BALL_D_6203_IN = 0.2656
PITCH_D_6203_IN = 1.122

COMMON_FEATURES = [
    "td_rms", "td_std", "td_peak_abs", "td_ptp", "td_skewness",
    "td_kurtosis", "td_crest_factor", "td_impulse_factor",
    "td_shape_factor", "td_clearance_factor", "td_zero_cross_rate",
    "fd_dominant_hz_0_6k", "fd_centroid_hz_0_6k", "fd_f95_hz_0_6k",
    "fd_entropy_0_6k", "fd_energy_ratio_0_500", "fd_energy_ratio_500_1500",
    "fd_energy_ratio_1500_3000", "fd_energy_ratio_3000_6000",
    "env_rms", "env_kurtosis", "env_dominant_hz_0_500",
    "env_centroid_hz_0_500", "env_entropy_0_500",
    "env_energy_ratio_0_50", "env_energy_ratio_50_150",
    "env_energy_ratio_150_300", "env_energy_ratio_300_500",
]

MECHANISM_FEATURES = [
    "mech_bpfo_ratio_1x", "mech_bpfo_ratio_2x", "mech_bpfo_ratio_3x",
    "mech_bpfi_ratio_1x", "mech_bpfi_ratio_2x", "mech_bpfi_ratio_3x",
    "mech_bsf_ratio_1x", "mech_bsf_ratio_2x", "mech_bsf_ratio_3x",
    "mech_bpfi_sideband_to_center_1x", "mech_bsf_sideband_to_center_1x",
]

META_COLUMNS = [
    "window_id", "domain", "original_file_id", "acquisition_id",
    "independent_object_id", "window_index_in_file", "window_start_sample",
    "window_end_sample_exclusive", "window_start_s", "window_end_s",
    "window_seconds", "stride_seconds", "overlap_fraction", "fs_hz",
    "signal_var", "channel", "sensor_position", "fault_bearing_location",
    "class_label", "truth_status", "load_hp", "rpm_value", "rpm_source",
    "fault_size_in", "or_position", "n_points_in_file",
]


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


def parse_load_hp(path: Path) -> float:
    m = re.search(r"_(\d)(?:_|\.mat$)", path.name, re.I)
    return float(m.group(1)) if m else np.nan


def parse_fault_size(path: Path) -> float:
    m = re.match(r"(?:IR|OR|B)(\d{3})", path.name, re.I)
    if not m:
        return np.nan
    return float(int(m.group(1)) / 1000.0)


def parse_or_position(path: Path) -> float:
    m = re.search(r"@(3|6|12)_", path.name, re.I)
    return float(m.group(1)) if m else np.nan


def parse_rpm(mat: dict, path: Path) -> tuple[float, str]:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if "RPM" in k.upper():
            a = np.asarray(v).squeeze()
            if np.size(a):
                return float(np.ravel(a)[0]), "mat_variable"
    m = re.search(r"(\d{4})rpm", path.name, re.I)
    if m:
        return float(m.group(1)), "filename"
    return np.nan, "unknown"


def select_source_de(mat: dict, path: Path) -> tuple[np.ndarray, str]:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if k.endswith("DE_time"):
            x = np.asarray(v).squeeze()
            if x.ndim == 1 and x.size > 1000:
                return x.astype(float), k
    raise ValueError(f"No DE_time signal found: {path}")


def select_target_signal(mat: dict, path: Path) -> tuple[np.ndarray, str]:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        x = np.asarray(v).squeeze()
        if x.ndim == 1 and x.size > 1000:
            return x.astype(float), k
    raise ValueError(f"No 1D target signal found: {path}")


def window_count(n_points: int, fs: int, window_s: float, overlap: float) -> int:
    win = int(round(window_s * fs))
    stride = int(round(win * (1.0 - overlap)))
    if win <= 0 or stride <= 0 or n_points < win:
        return 0
    return 1 + (n_points - win) // stride


def iter_windows(x: np.ndarray, fs: int):
    win = int(round(WINDOW_SECONDS * fs))
    stride = int(round(STRIDE_SECONDS * fs))
    if len(x) < win:
        return
    i = 0
    start = 0
    while start + win <= len(x):
        yield i, start, start + win, x[start:start + win]
        i += 1
        start += stride


def spectral_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size <= 1:
        return 0.0
    q = p / (p.sum() + EPS)
    return float(-(q * np.log(q + EPS)).sum() / np.log(len(q)))


def integrate_band(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    if m.sum() < 2:
        return 0.0
    return float(np.trapz(p[m], f[m]))


def extract_common_features(window: np.ndarray, fs: int) -> dict[str, float]:
    x = np.asarray(window, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite signal value")

    # STEP03 frozen policy: demean yes; no default linear detrend/bandpass/resample.
    x = x - np.mean(x)
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    ptp = float(np.ptp(x))
    abs_mean = float(np.mean(np.abs(x)))
    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(x))))
    signs = np.signbit(x)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if len(x) > 1 else 0.0

    td = {
        "td_rms": rms,
        "td_std": std,
        "td_peak_abs": peak,
        "td_ptp": ptp,
        "td_skewness": float(skew(x, bias=False)),
        "td_kurtosis": float(kurtosis(x, fisher=False, bias=False)),
        "td_crest_factor": peak / (rms + EPS),
        "td_impulse_factor": peak / (abs_mean + EPS),
        "td_shape_factor": rms / (abs_mean + EPS),
        "td_clearance_factor": peak / (sqrt_abs_mean ** 2 + EPS),
        "td_zero_cross_rate": zcr,
    }

    nperseg = min(4096, len(x))
    f, pxx = welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", scaling="density"
    )
    m = (f >= 0.0) & (f <= COMMON_FMAX)
    f2 = f[m]
    p2 = pxx[m]
    total = float(np.trapz(p2, f2)) + EPS
    csum = np.cumsum(p2)
    f95 = float(f2[np.searchsorted(csum, 0.95 * csum[-1])]) if csum[-1] > 0 else 0.0
    fd = {
        "fd_dominant_hz_0_6k": float(f2[int(np.argmax(p2))]),
        "fd_centroid_hz_0_6k": float(np.sum(f2 * p2) / (np.sum(p2) + EPS)),
        "fd_f95_hz_0_6k": f95,
        "fd_entropy_0_6k": spectral_entropy(p2),
        "fd_energy_ratio_0_500": integrate_band(f2, p2, 0, 500) / total,
        "fd_energy_ratio_500_1500": integrate_band(f2, p2, 500, 1500) / total,
        "fd_energy_ratio_1500_3000": integrate_band(f2, p2, 1500, 3000) / total,
        "fd_energy_ratio_3000_6000": integrate_band(f2, p2, 3000, 6000.000001) / total,
    }

    env = np.abs(hilbert(x))
    env_centered = env - np.mean(env)
    env_rms = float(np.sqrt(np.mean(env_centered ** 2)))
    fe, pe = periodogram(env_centered, fs=fs, window="hann", detrend=False, scaling="density")
    me = (fe >= 0.0) & (fe <= 500.0)
    fe2, pe2 = fe[me], pe[me]
    etotal = float(np.trapz(pe2, fe2)) + EPS
    envf = {
        "env_rms": env_rms,
        "env_kurtosis": float(kurtosis(env_centered, fisher=False, bias=False)),
        "env_dominant_hz_0_500": float(fe2[int(np.argmax(pe2))]),
        "env_centroid_hz_0_500": float(np.sum(fe2 * pe2) / (np.sum(pe2) + EPS)),
        "env_entropy_0_500": spectral_entropy(pe2),
        "env_energy_ratio_0_50": integrate_band(fe2, pe2, 0, 50) / etotal,
        "env_energy_ratio_50_150": integrate_band(fe2, pe2, 50, 150) / etotal,
        "env_energy_ratio_150_300": integrate_band(fe2, pe2, 150, 300) / etotal,
        "env_energy_ratio_300_500": integrate_band(fe2, pe2, 300, 500.000001) / etotal,
    }
    out = {**td, **fd, **envf}
    if not np.isfinite(np.array(list(out.values()), dtype=float)).all():
        raise ValueError("Non-finite common feature generated")
    return out


def bearing_freqs_6205(rpm: float) -> dict[str, float]:
    fr = rpm / 60.0
    ratio = BALL_D_6205_IN / PITCH_D_6205_IN
    return {
        "fr": fr,
        "BPFO": fr * Z_ROLLERS_6205 / 2.0 * (1.0 - ratio),
        "BPFI": fr * Z_ROLLERS_6205 / 2.0 * (1.0 + ratio),
        # Strictly follow the official-problem convention frozen in STEP03.
        "BSF": fr * PITCH_D_6205_IN / BALL_D_6205_IN * (1.0 - ratio ** 2),
        "FTF": fr / 2.0 * (1.0 - ratio),
    }


def band_mean_power(f: np.ndarray, p: np.ndarray, center: float, delta: float) -> float:
    m = np.abs(f - center) <= delta
    if m.sum() < 1:
        return 0.0
    return float(np.mean(p[m]))


def local_mechanism_ratio(f: np.ndarray, p: np.ndarray, center: float) -> float:
    # 1 s full-window envelope periodogram -> 1 Hz bin spacing.
    delta = max(2.0 / WINDOW_SECONDS, 0.03 * center)
    target = np.abs(f - center) <= delta
    bg = (np.abs(f - center) >= 2.0 * delta) & (np.abs(f - center) <= 5.0 * delta)
    if target.sum() < 1 or bg.sum() < 2:
        return np.nan
    return float(np.mean(p[target]) / (np.mean(p[bg]) + EPS))


def extract_source_mechanism_features(window: np.ndarray, fs: int, rpm: float) -> dict[str, float]:
    if not np.isfinite(rpm):
        return {k: np.nan for k in MECHANISM_FEATURES}
    x = np.asarray(window, dtype=float) - np.mean(window)
    env = np.abs(hilbert(x))
    env -= np.mean(env)
    f, p = periodogram(env, fs=fs, window="hann", detrend=False, scaling="density")
    q = bearing_freqs_6205(rpm)
    out = {}
    for name in ("BPFO", "BPFI", "BSF"):
        for h in (1, 2, 3):
            out[f"mech_{name.lower()}_ratio_{h}x"] = local_mechanism_ratio(f, p, h * q[name])

    # Sideband summaries are supporting descriptors, not hard rules.
    def sideband_to_center(center: float, spacing: float) -> float:
        delta_c = max(2.0 / WINDOW_SECONDS, 0.03 * center)
        e0 = band_mean_power(f, p, center, delta_c)
        em = band_mean_power(f, p, center - spacing, max(2.0 / WINDOW_SECONDS, 0.03 * max(center - spacing, 1.0)))
        ep = band_mean_power(f, p, center + spacing, max(2.0 / WINDOW_SECONDS, 0.03 * (center + spacing)))
        return float((em + ep) / (2.0 * e0 + EPS))

    out["mech_bpfi_sideband_to_center_1x"] = sideband_to_center(q["BPFI"], q["fr"])
    out["mech_bsf_sideband_to_center_1x"] = sideband_to_center(q["BSF"], q["FTF"])
    return out


def build_file_table() -> pd.DataFrame:
    rows = []
    source_paths = sorted(SRC_FAULT.glob("*.mat")) + sorted(SRC_NORMAL.glob("*.mat"))
    for p in source_paths:
        mat = loadmat(p)
        x, var = select_source_de(mat, p)
        rpm, rpm_source = parse_rpm(mat, p)
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        lab = source_label(p)
        rows.append({
            "domain": "source",
            "original_file_id": rel,
            "acquisition_id": rel,
            "independent_object_id": rel,
            "class_label": lab,
            "truth_status": "known_source_label",
            "fs_hz": SOURCE_FS,
            "signal_var": var,
            "channel": "DE_time",
            "sensor_position": "drive_end",
            "fault_bearing_location": "none" if lab == "N" else "drive_end",
            "load_hp": parse_load_hp(p),
            "rpm_value": rpm,
            "rpm_source": rpm_source,
            "fault_size_in": parse_fault_size(p),
            "or_position": parse_or_position(p),
            "n_points_in_file": int(len(x)),
            "duration_s": float(len(x) / SOURCE_FS),
        })

    for p in sorted(TARGET.glob("*.mat")):
        mat = loadmat(p)
        x, var = select_target_signal(mat, p)
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        rows.append({
            "domain": "target",
            "original_file_id": rel,
            "acquisition_id": rel,
            "independent_object_id": rel,
            "class_label": "UNKNOWN",
            "truth_status": "unknown",
            "fs_hz": TARGET_FS,
            "signal_var": var,
            "channel": "anonymous_single_channel",
            "sensor_position": "unknown",
            "fault_bearing_location": "unknown",
            "load_hp": np.nan,
            "rpm_value": TARGET_RPM_APPROX,
            "rpm_source": "problem_approximate_not_used_for_mechanism_features",
            "fault_size_in": np.nan,
            "or_position": np.nan,
            "n_points_in_file": int(len(x)),
            "duration_s": float(len(x) / TARGET_FS),
        })
    return pd.DataFrame(rows)


def read_signal_from_row(r: pd.Series) -> np.ndarray:
    p = ROOT / r["original_file_id"]
    mat = loadmat(p)
    if r["domain"] == "source":
        x, _ = select_source_de(mat, p)
    else:
        x, _ = select_target_signal(mat, p)
    return x


def build_window_design_candidates(file_table: pd.DataFrame) -> pd.DataFrame:
    configs = [
        (0.5, 0.50, "shorter alternative: higher temporal resolution, only 2 Hz spectral resolution"),
        (1.0, 0.00, "no overlap: lowest redundancy but more boundary sensitivity"),
        (1.0, 0.50, "chosen compromise: 1 Hz resolution and moderate boundary robustness"),
        (1.0, 0.75, "higher redundancy: about 4x stride density, limited independent gain"),
        (2.0, 0.50, "longer alternative: better resolution but excludes short source records"),
    ]
    rows = []
    for ws, ov, note in configs:
        for domain in ("source", "target"):
            d = file_table[file_table["domain"] == domain]
            counts = [window_count(int(r.n_points_in_file), int(r.fs_hz), ws, ov) for _, r in d.iterrows()]
            rows.append({
                "window_seconds": ws,
                "overlap_fraction": ov,
                "stride_seconds": ws * (1.0 - ov),
                "domain": domain,
                "file_count": len(d),
                "files_with_zero_windows": int(sum(c == 0 for c in counts)),
                "total_windows": int(sum(counts)),
                "fft_bin_hz_if_full_window_fft": 1.0 / ws,
                "chosen": bool(ws == WINDOW_SECONDS and abs(ov - OVERLAP) < 1e-12),
                "note": note,
            })
    return pd.DataFrame(rows)


def build_features(file_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_rows = []
    feature_rows = []
    for _, r in file_table.iterrows():
        x = read_signal_from_row(r)
        fs = int(r["fs_hz"])
        for wi, start, end, w in iter_windows(x, fs):
            window_id = f"{r['domain']}::{r['acquisition_id']}::w{wi:04d}"
            meta = {
                "window_id": window_id,
                "domain": r["domain"],
                "original_file_id": r["original_file_id"],
                "acquisition_id": r["acquisition_id"],
                "independent_object_id": r["independent_object_id"],
                "window_index_in_file": int(wi),
                "window_start_sample": int(start),
                "window_end_sample_exclusive": int(end),
                "window_start_s": float(start / fs),
                "window_end_s": float(end / fs),
                "window_seconds": WINDOW_SECONDS,
                "stride_seconds": STRIDE_SECONDS,
                "overlap_fraction": OVERLAP,
                "fs_hz": fs,
                "signal_var": r["signal_var"],
                "channel": r["channel"],
                "sensor_position": r["sensor_position"],
                "fault_bearing_location": r["fault_bearing_location"],
                "class_label": r["class_label"],
                "truth_status": r["truth_status"],
                "load_hp": r["load_hp"],
                "rpm_value": r["rpm_value"],
                "rpm_source": r["rpm_source"],
                "fault_size_in": r["fault_size_in"],
                "or_position": r["or_position"],
                "n_points_in_file": int(r["n_points_in_file"]),
            }
            common = extract_common_features(w, fs)
            if r["domain"] == "source":
                mech = extract_source_mechanism_features(w, fs, float(r["rpm_value"]))
            else:
                # Do not transfer SKF6205 geometry to target-domain bearings.
                mech = {k: np.nan for k in MECHANISM_FEATURES}
            index_rows.append(meta)
            feature_rows.append({**meta, **common, **mech})
    return pd.DataFrame(index_rows), pd.DataFrame(feature_rows)


def feature_dictionary() -> pd.DataFrame:
    rows = []
    def add(name, group, scope, definition, formula, unit, guard, missing, params, source, q2):
        rows.append({
            "feature_name": name, "group": group, "scope": scope,
            "definition": definition, "formula": formula, "unit": unit,
            "zero_denominator_guard": guard, "missing_rule": missing,
            "parameters": params, "parameter_source": source,
            "q2_default_feature": q2,
        })

    time_defs = {
        "td_rms": ("demeaned RMS", "sqrt(mean(x^2))", "signal unit"),
        "td_std": ("demeaned population standard deviation", "std(x)", "signal unit"),
        "td_peak_abs": ("maximum absolute amplitude", "max(|x|)", "signal unit"),
        "td_ptp": ("peak-to-peak amplitude", "max(x)-min(x)", "signal unit"),
        "td_skewness": ("sample skewness", "E[(x-mu)^3]/sigma^3 with bias correction", "dimensionless"),
        "td_kurtosis": ("Pearson kurtosis", "E[(x-mu)^4]/sigma^4 with bias correction", "dimensionless"),
        "td_crest_factor": ("crest factor", "max(|x|)/(RMS+eps)", "dimensionless"),
        "td_impulse_factor": ("impulse factor", "max(|x|)/(mean(|x|)+eps)", "dimensionless"),
        "td_shape_factor": ("shape factor", "RMS/(mean(|x|)+eps)", "dimensionless"),
        "td_clearance_factor": ("clearance factor", "max(|x|)/(mean(sqrt(|x|))^2+eps)", "dimensionless"),
        "td_zero_cross_rate": ("fraction of adjacent samples with sign change", "mean(signbit(x[t]) != signbit(x[t-1]))", "dimensionless"),
    }
    for n, (d, f, u) in time_defs.items():
        add(n, "time", "source_and_target", d, f, u, "eps=1e-12 where denominator exists", "raise on non-finite raw signal", "x is demeaned only", "STEP03 preprocessing policy", True)

    fd_defs = {
        "fd_dominant_hz_0_6k": ("frequency of maximum Welch PSD in 0-6 kHz", "argmax_f Pxx(f)", "Hz"),
        "fd_centroid_hz_0_6k": ("PSD spectral centroid in 0-6 kHz", "sum(f*P)/sum(P)", "Hz"),
        "fd_f95_hz_0_6k": ("frequency containing 95% cumulative PSD in 0-6 kHz", "min f: CDF_P(f)>=0.95", "Hz"),
        "fd_entropy_0_6k": ("normalized spectral entropy in 0-6 kHz", "-sum(q log q)/log(K)", "dimensionless"),
        "fd_energy_ratio_0_500": ("PSD energy ratio 0-500 Hz", "E[0,500)/E[0,6000]", "dimensionless"),
        "fd_energy_ratio_500_1500": ("PSD energy ratio 500-1500 Hz", "E[500,1500)/E[0,6000]", "dimensionless"),
        "fd_energy_ratio_1500_3000": ("PSD energy ratio 1500-3000 Hz", "E[1500,3000)/E[0,6000]", "dimensionless"),
        "fd_energy_ratio_3000_6000": ("PSD energy ratio 3000-6000 Hz", "E[3000,6000]/E[0,6000]", "dimensionless"),
    }
    for n, (d, f, u) in fd_defs.items():
        add(n, "frequency", "source_and_target", d, f, u, "eps=1e-12 for total energy/sums", "0 if empty band; raw signal non-finite raises", "Welch nperseg=min(4096,N), 50% internal Welch overlap; common band <=6 kHz", "common frequency support under M1/M2 protocol; deterministic", True)

    env_defs = {
        "env_rms": ("RMS of demeaned Hilbert envelope", "sqrt(mean((|Hilbert(x)|-mean)^2))", "signal unit"),
        "env_kurtosis": ("Pearson kurtosis of demeaned envelope", "kurtosis(env, fisher=False)", "dimensionless"),
        "env_dominant_hz_0_500": ("dominant envelope frequency 0-500 Hz", "argmax_f P_env(f)", "Hz"),
        "env_centroid_hz_0_500": ("envelope spectral centroid 0-500 Hz", "sum(f*P_env)/sum(P_env)", "Hz"),
        "env_entropy_0_500": ("normalized envelope spectral entropy 0-500 Hz", "-sum(q log q)/log(K)", "dimensionless"),
        "env_energy_ratio_0_50": ("envelope energy ratio 0-50 Hz", "E_env[0,50)/E_env[0,500]", "dimensionless"),
        "env_energy_ratio_50_150": ("envelope energy ratio 50-150 Hz", "E_env[50,150)/E_env[0,500]", "dimensionless"),
        "env_energy_ratio_150_300": ("envelope energy ratio 150-300 Hz", "E_env[150,300)/E_env[0,500]", "dimensionless"),
        "env_energy_ratio_300_500": ("envelope energy ratio 300-500 Hz", "E_env[300,500]/E_env[0,500]", "dimensionless"),
    }
    for n, (d, f, u) in env_defs.items():
        add(n, "envelope_generic", "source_and_target", d, f, u, "eps=1e-12 for energy/sums", "0 if empty band; raw signal non-finite raises", "full-window Hann periodogram; 0-500 Hz", "STEP03: envelope used as supporting weak-fault descriptor", True)

    for base in ("bpfo", "bpfi", "bsf"):
        for h in (1, 2, 3):
            n = f"mech_{base}_ratio_{h}x"
            add(n, "mechanism", "source_only", f"local envelope-power ratio near {h}x {base.upper()}", "mean(P_target_band)/(mean(P_background_band)+eps)", "dimensionless", "eps=1e-12", "NaN for target because target bearing geometry is unavailable", "delta=max(2 Hz, 3% of center); background at 2delta-5delta", "official SKF6205 geometry + STEP03 tolerance convention", False)
    add("mech_bpfi_sideband_to_center_1x", "mechanism", "source_only", "BPFI +/- shaft-frequency sideband power relative to BPFI center", "(P(BPFI-fr)+P(BPFI+fr))/(2*P(BPFI)+eps)", "dimensionless", "eps=1e-12", "NaN for target", "same local tolerance as mechanism bands", "inner-race modulation mechanism; SKF6205 source only", False)
    add("mech_bsf_sideband_to_center_1x", "mechanism", "source_only", "BSF +/- FTF sideband power relative to BSF center", "(P(BSF-FTF)+P(BSF+FTF))/(2*P(BSF)+eps)", "dimensionless", "eps=1e-12", "NaN for target", "same local tolerance as mechanism bands", "rolling-element modulation mechanism; SKF6205 source only", False)
    return pd.DataFrame(rows)


def make_q2_interface(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Keep metadata for traceability, but feature_columns below is the only model-input list.
    q2_meta = [
        "window_id", "domain", "original_file_id", "acquisition_id",
        "independent_object_id", "window_start_s", "window_end_s", "fs_hz",
        "class_label", "truth_status", "load_hp", "rpm_value", "rpm_source",
        "channel", "sensor_position",
    ]
    src = features[features["domain"] == "source"][q2_meta + COMMON_FEATURES].copy()
    tgt = features[features["domain"] == "target"][q2_meta + COMMON_FEATURES].copy()
    interface = {
        "version": "Q1C-STEP04-v1",
        "source_file": "outputs/q1c_feature_extraction/q2_source_raw.csv",
        "target_file": "outputs/q1c_feature_extraction/q2_target_raw.csv",
        "feature_columns": COMMON_FEATURES,
        "label_column": "class_label",
        "group_column": "independent_object_id",
        "window_id_column": "window_id",
        "processing_state": "raw_deterministic_features_unscaled_unimputed_unselected",
        "fit_rule": "Any learned imputer/scaler/selector must be fit on the training partition only after splitting by independent_object_id.",
        "forbidden_as_model_features": [
            "class_label", "original_file_id", "acquisition_id", "independent_object_id",
            "window_id", "load_hp", "rpm_value", "rpm_source", "channel",
            "sensor_position", "truth_status", "domain"
        ],
        "source_only_mechanism_features_excluded_from_default_q2": MECHANISM_FEATURES,
    }
    return src, tgt, interface


def validate(file_table: pd.DataFrame, index_df: pd.DataFrame, feat: pd.DataFrame,
             dictionary: pd.DataFrame, q2s: pd.DataFrame, q2t: pd.DataFrame) -> dict:
    checks = {}
    src_files = file_table[file_table.domain == "source"]
    tgt_files = file_table[file_table.domain == "target"]
    counts = src_files["class_label"].value_counts().to_dict()
    checks["m1_source_file_count_56"] = len(src_files) == 56
    checks["m1_class_counts"] = counts == {"OR": 28, "IR": 12, "B": 12, "N": 4}
    checks["target_file_count_16"] = len(tgt_files) == 16
    checks["all_files_have_windows"] = set(file_table.original_file_id) == set(index_df.original_file_id)
    checks["window_ids_unique"] = bool(index_df.window_id.is_unique)
    checks["window_traceable"] = bool((index_df.window_end_sample_exclusive <= index_df.n_points_in_file).all() and (index_df.window_start_sample >= 0).all())
    checks["group_boundary_frozen"] = bool((index_df.acquisition_id == index_df.independent_object_id).all())
    checks["one_second_physical_time"] = bool(np.allclose(index_df.window_end_s - index_df.window_start_s, WINDOW_SECONDS))
    checks["source_window_points_48000"] = bool((index_df[index_df.domain == "source"].window_end_sample_exclusive - index_df[index_df.domain == "source"].window_start_sample == 48000).all())
    checks["target_window_points_32000"] = bool((index_df[index_df.domain == "target"].window_end_sample_exclusive - index_df[index_df.domain == "target"].window_start_sample == 32000).all())
    checks["raw_feature_rows_match_index"] = len(index_df) == len(feat) and index_df.window_id.tolist() == feat.window_id.tolist()
    checks["common_features_finite"] = bool(np.isfinite(feat[COMMON_FEATURES].to_numpy(float)).all())
    checks["mechanism_missing_only_target"] = bool(feat[feat.domain == "source"][MECHANISM_FEATURES].notna().all().all() and feat[feat.domain == "target"][MECHANISM_FEATURES].isna().all().all())
    checks["dictionary_covers_all_features"] = set(COMMON_FEATURES + MECHANISM_FEATURES) == set(dictionary.feature_name)

    forbidden_tokens = ["label", "filename", "file_id", "acquisition", "object_id", "window_id", "load", "rpm_source", "truth", "domain"]
    checks["no_leakage_named_feature"] = not any(any(tok in c.lower() for tok in forbidden_tokens) for c in COMMON_FEATURES)
    checks["mechanism_not_in_q2_default"] = not bool(set(MECHANISM_FEATURES) & set(COMMON_FEATURES))
    checks["q2_row_counts_match"] = len(q2s) == int((feat.domain == "source").sum()) and len(q2t) == int((feat.domain == "target").sum())
    checks["q2_common_features_finite"] = bool(np.isfinite(q2s[COMMON_FEATURES].to_numpy(float)).all() and np.isfinite(q2t[COMMON_FEATURES].to_numpy(float)).all())
    checks["target_labels_unknown"] = bool((q2t.class_label == "UNKNOWN").all())
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_file_counts": counts,
        "source_windows": int((index_df.domain == "source").sum()),
        "target_windows": int((index_df.domain == "target").sum()),
        "total_windows": int(len(index_df)),
        "common_feature_count": len(COMMON_FEATURES),
        "source_only_mechanism_feature_count": len(MECHANISM_FEATURES),
    }


def main() -> None:
    file_table = build_file_table()
    file_table.to_csv(OUT / "file_inventory.csv", index=False, encoding="utf-8-sig")

    candidates = build_window_design_candidates(file_table)
    candidates.to_csv(OUT / "window_design_candidates.csv", index=False, encoding="utf-8-sig")

    index_df, features = build_features(file_table)
    index_df.to_csv(OUT / "window_index.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUT / "features_all_raw.csv", index=False, encoding="utf-8-sig")
    features[features.domain == "source"].to_csv(OUT / "features_source_raw.csv", index=False, encoding="utf-8-sig")
    features[features.domain == "target"].to_csv(OUT / "features_target_raw.csv", index=False, encoding="utf-8-sig")

    dictionary = feature_dictionary()
    dictionary.to_csv(OUT / "feature_dictionary.csv", index=False, encoding="utf-8-sig")

    q2s, q2t, interface = make_q2_interface(features)
    q2s.to_csv(OUT / "q2_source_raw.csv", index=False, encoding="utf-8-sig")
    q2t.to_csv(OUT / "q2_target_raw.csv", index=False, encoding="utf-8-sig")
    (OUT / "q2_interface.json").write_text(json.dumps(interface, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reconciliation at file and class/domain levels.
    recon_rows = []
    for domain, d in file_table.groupby("domain"):
        subw = index_df[index_df.domain == domain]
        recon_rows.append({
            "scope": domain,
            "files_before": len(d),
            "files_after_with_at_least_one_window": subw.original_file_id.nunique(),
            "windows_after": len(subw),
        })
    for lab, d in file_table[file_table.domain == "source"].groupby("class_label"):
        subw = index_df[(index_df.domain == "source") & (index_df.class_label == lab)]
        recon_rows.append({
            "scope": f"source_class_{lab}",
            "files_before": len(d),
            "files_after_with_at_least_one_window": subw.original_file_id.nunique(),
            "windows_after": len(subw),
        })
    pd.DataFrame(recon_rows).to_csv(OUT / "count_reconciliation.csv", index=False, encoding="utf-8-sig")

    # Physical rationale summary based on actual source RPMs.
    srpm = file_table[(file_table.domain == "source") & np.isfinite(file_table.rpm_value)].rpm_value.astype(float)
    qmin = bearing_freqs_6205(float(srpm.min()))
    qmax = bearing_freqs_6205(float(srpm.max()))
    window_design = {
        "chosen_window_seconds": WINDOW_SECONDS,
        "chosen_overlap_fraction": OVERLAP,
        "chosen_stride_seconds": STRIDE_SECONDS,
        "source_points_per_window": int(SOURCE_FS * WINDOW_SECONDS),
        "target_points_per_window": int(TARGET_FS * WINDOW_SECONDS),
        "same_physical_time_not_same_point_count": True,
        "fft_resolution_hz_full_1s_window": 1.0,
        "source_rpm_min": float(srpm.min()),
        "source_rpm_max": float(srpm.max()),
        "source_rotations_per_window_min": float(srpm.min() / 60.0 * WINDOW_SECONDS),
        "source_rotations_per_window_max": float(srpm.max() / 60.0 * WINDOW_SECONDS),
        "target_approx_rotations_per_window": float(TARGET_RPM_APPROX / 60.0 * WINDOW_SECONDS),
        "source_bpfo_cycles_per_window_range": [qmin["BPFO"] * WINDOW_SECONDS, qmax["BPFO"] * WINDOW_SECONDS],
        "source_bpfi_cycles_per_window_range": [qmin["BPFI"] * WINDOW_SECONDS, qmax["BPFI"] * WINDOW_SECONDS],
        "source_bsf_cycles_per_window_range": [qmin["BSF"] * WINDOW_SECONDS, qmax["BSF"] * WINDOW_SECONDS],
        "why_not_0_5s": "Only 0.5 s physical context and 2 Hz full-window spectral resolution; target has only about 5 shaft rotations per window.",
        "why_not_2s": "Would discard at least one short M1 source record and reduce available windows; not acceptable for full-data extraction.",
        "why_50pct_overlap": "Reduces boundary/phase sensitivity with 0.5 s stride while avoiding the roughly 4x redundancy of 75% overlap. Correlated windows remain inside one independent_object_id group.",
        "group_rule": "Split by independent_object_id before any learned preprocessing/model fitting; windows from one raw MAT may never cross train/test.",
    }
    (OUT / "window_design.json").write_text(json.dumps(window_design, ensure_ascii=False, indent=2), encoding="utf-8")

    report = validate(file_table, index_df, features, dictionary, q2s, q2t)
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    answer_material = f"""# 问题1C答案素材（由实际全量提取生成）

## 窗口设计
- 采用 {WINDOW_SECONDS:.1f} s 物理时长窗口、{OVERLAP:.0%} 重叠，步长 {STRIDE_SECONDS:.1f} s。
- 源域48 kHz对应每窗48000点；目标域32 kHz对应每窗32000点。相同点数并不代表相同物理时间，因此不采用固定点数跨域切片。
- 源域实际转速范围为 {srpm.min():.0f}-{srpm.max():.0f} rpm，1 s内约包含 {srpm.min()/60:.1f}-{srpm.max()/60:.1f} 个轴转周期；目标域题面约600 rpm，1 s约10个轴转周期。
- 1 s满窗FFT的名义频率分辨率为1 Hz。0.5 s方案仅2 Hz且目标域仅约5转；2 s方案会使短源文件无法贡献窗口；因此选择1 s。
- 50%重叠用于减小事件跨窗边界的敏感性；所有相关窗口仍以原始MAT为独立分组边界，后续不得跨训练/测试。

## 特征体系
- 共域可用特征：{len(COMMON_FEATURES)}个，包括时域冲击统计、0-6 kHz频谱形态/能量、0-500 Hz通用包络特征。
- 源域机理特征：{len(MECHANISM_FEATURES)}个，依据SKF6205和逐文件RPM计算BPFO/BPFI/BSF谐波及必要侧带；目标域因缺少同等完整轴承几何参数而保持缺失，并明确不进入问题2默认共同特征集。
- 所有窗口先按步骤03结论只去均值；不统一线性去趋势、不默认带通、不在本步骤强制重采样。
- 原始确定性特征与需要学习的插补、缩放、筛选严格分离。本步骤没有在全数据上拟合任何标准化器或特征选择器。

## 数据接口
- 源域：`outputs/q1c_feature_extraction/q2_source_raw.csv`
- 目标域：`outputs/q1c_feature_extraction/q2_target_raw.csv`
- 读取规则：`outputs/q1c_feature_extraction/q2_interface.json`
- 分组字段：`independent_object_id`
- 标签字段：`class_label`（目标域全部UNKNOWN）
- 模型输入字段必须严格读取接口中的`feature_columns`，不得把文件ID、类别、工况元数据等作为特征。

## 实际数量
- 源文件：{len(file_table[file_table.domain=='source'])}；源窗口：{len(index_df[index_df.domain=='source'])}
- 目标文件：{len(file_table[file_table.domain=='target'])}；目标窗口：{len(index_df[index_df.domain=='target'])}
- 总窗口：{len(index_df)}
- STEP04自动验收状态：{report['status']}
"""
    (OUT / "q1_answer_material.md").write_text(answer_material, encoding="utf-8")

    manifest = {
        "step": "Q1C_STEP04",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "window_seconds": WINDOW_SECONDS,
        "overlap_fraction": OVERLAP,
        "stride_seconds": STRIDE_SECONDS,
        "source_fs": SOURCE_FS,
        "target_fs": TARGET_FS,
        "common_fmax_hz": COMMON_FMAX,
        "source_files": int((file_table.domain == "source").sum()),
        "target_files": int((file_table.domain == "target").sum()),
        "source_windows": int((index_df.domain == "source").sum()),
        "target_windows": int((index_df.domain == "target").sum()),
        "common_feature_count": len(COMMON_FEATURES),
        "mechanism_feature_count": len(MECHANISM_FEATURES),
        "learned_preprocessing_applied": False,
        "validation_status": report["status"],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = (
        "Q1C / STEP04 full-data feature extraction\n"
        f"status={report['status']}\n"
        f"window={WINDOW_SECONDS:.1f}s overlap={OVERLAP:.0%} stride={STRIDE_SECONDS:.1f}s\n"
        f"source_files={manifest['source_files']} source_windows={manifest['source_windows']}\n"
        f"target_files={manifest['target_files']} target_windows={manifest['target_windows']}\n"
        f"common_features={len(COMMON_FEATURES)} source_only_mechanism_features={len(MECHANISM_FEATURES)}\n"
        "learned_preprocessing_applied=False\n"
        "Q2 split rule: group by independent_object_id before any learned preprocessing.\n"
    )
    (OUT / "result_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    if report["status"] != "PASS":
        raise SystemExit("STEP04 validation failed; inspect validation_report.json")


if __name__ == "__main__":
    main()
