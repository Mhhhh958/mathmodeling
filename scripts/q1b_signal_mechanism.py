# AI ASSISTANCE NOTICE
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ChatGPT；版本/型号：GPT-5.6 Sol（ChatGPT 2026-08-06更新版）；
# 开发机构/公司：OpenAI；版本发布日期：2026-08-06。
# 人工智能仅用于代码检查、调试建议与说明整理；最终算法、参数与结果由参赛队审查并由冻结复现链验证。
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import platform
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, hilbert, spectrogram, welch
from scipy.stats import kurtosis


# =========================
# Q1B / STEP03 fixed protocol
# =========================
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "source_domain"
SRC_FAULT = RAW / "cwru_48khz_de"
SRC_NORMAL = RAW / "cwru_48khz_normal"
OUT = ROOT / "outputs" / "q1b_signal_mechanism"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SOURCE_FS = 48000
WINDOW_SECONDS = 1.0
WINDOW_POINTS = int(SOURCE_FS * WINDOW_SECONDS)
EPS = 1e-12
COMMON_FMAX = 6000.0
ENVELOPE_FMAX = 1000.0

# Official problem-statement SKF6205 (drive-end) geometry.
Z_ROLLERS = 9
d_ball_in = 0.3126
D_pitch_in = 1.537

EXPECTED_ORDERS = {
    "BPFO": 3.584775536759922,
    "BPFI": 5.415224463240078,
    "BSF": 4.713443401429695,
    "FTF": 0.39830839297332465,
}

EXPECTED_COUNTS = {"OR": 28, "IR": 12, "B": 12, "N": 4}

REPRESENTATIVES = [
    ("OR_typical", SRC_FAULT / "OR007@12_0.mat", "OR"),
    ("IR_typical", SRC_FAULT / "IR007_0.mat", "IR"),
    ("B_difficult", SRC_FAULT / "B007_0.mat", "B"),
    ("N_typical", SRC_NORMAL / "N_1_(1772rpm).mat", "N"),
    ("N_difficult", SRC_NORMAL / "N_0.mat", "N"),
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def source_label(path: Path) -> str:
    name = path.name.upper()
    if name.startswith("OR"):
        return "OR"
    if name.startswith("IR"):
        return "IR"
    if name.startswith("B"):
        return "B"
    if name.startswith("N"):
        return "N"
    raise ValueError(f"Unknown label for {path.name}")


def read_source_de(path: Path) -> tuple[np.ndarray, str, float, str]:
    mat = loadmat(path)

    signal_key = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if k.endswith("DE_time"):
            a = np.asarray(v).squeeze()
            if a.ndim == 1 and a.size > 1000:
                signal_key = k
                break
    if signal_key is None:
        raise RuntimeError(f"No DE_time in {path}")

    x = np.asarray(mat[signal_key]).squeeze().astype(float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite values in {path}")

    rpm = np.nan
    rpm_source = "missing"
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if "RPM" in k.upper():
            a = np.asarray(v).squeeze()
            if np.size(a):
                rpm = float(np.ravel(a)[0])
                rpm_source = "mat_variable"
                break

    if not np.isfinite(rpm):
        m = re.search(r"(\d{4})rpm", path.name, re.I)
        if m:
            rpm = float(m.group(1))
            rpm_source = "filename"

    return x, signal_key, rpm, rpm_source


def bearing_orders() -> dict[str, float]:
    bpfo = (Z_ROLLERS / 2.0) * (1.0 - d_ball_in / D_pitch_in)
    bpfi = (Z_ROLLERS / 2.0) * (1.0 + d_ball_in / D_pitch_in)
    bsf = (D_pitch_in / d_ball_in) * (1.0 - (d_ball_in / D_pitch_in) ** 2)
    ftf = 0.5 * (1.0 - d_ball_in / D_pitch_in)
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


def bearing_freqs(rpm: float) -> dict[str, float]:
    if not np.isfinite(rpm):
        raise ValueError("RPM required for mechanism frequency calculation")
    fr = rpm / 60.0
    orders = bearing_orders()
    return {
        "fr": fr,
        "BPFO": orders["BPFO"] * fr,
        "BPFI": orders["BPFI"] * fr,
        "BSF": orders["BSF"] * fr,
        "FTF": orders["FTF"] * fr,
    }


def envelope_local_ratio(x: np.ndarray, fs: int, fc: float) -> tuple[float, list[float]]:
    """Exactly matches the existing q1_baseline_v1 mechanism-check definition."""
    xc = np.asarray(x, dtype=float) - np.mean(x)
    env = np.abs(hilbert(xc))
    env = env - np.mean(env)
    nperseg = min(8192, len(env))
    f, p = welch(
        env,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )
    ratios: list[float] = []
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


def mechanism_centers(label: str, q: dict[str, float]) -> dict[str, float]:
    centers: dict[str, float] = {}
    if label == "OR":
        for h in (1, 2, 3):
            centers[f"{h}xBPFO"] = h * q["BPFO"]
    elif label == "IR":
        for h in (1, 2, 3):
            c = h * q["BPFI"]
            centers[f"{h}xBPFI"] = c
            centers[f"{h}xBPFI-fr"] = c - q["fr"]
            centers[f"{h}xBPFI+fr"] = c + q["fr"]
    elif label == "B":
        for h in (1, 2, 3):
            c = h * q["BSF"]
            centers[f"{h}xBSF"] = c
            centers[f"{h}xBSF-FTF"] = c - q["FTF"]
            centers[f"{h}xBSF+FTF"] = c + q["FTF"]
    return centers


def audit_m1() -> pd.DataFrame:
    paths = sorted(SRC_FAULT.glob("*.mat")) + sorted(SRC_NORMAL.glob("*.mat"))
    rows = []
    for p in paths:
        x, var, rpm, rpm_source = read_source_de(p)
        rows.append({
            "relative_path": str(p.relative_to(ROOT)),
            "filename": p.name,
            "label": source_label(p),
            "fs_hz": SOURCE_FS,
            "signal_var": var,
            "n_points": int(len(x)),
            "duration_s": float(len(x) / SOURCE_FS),
            "rpm": rpm,
            "rpm_source": rpm_source,
            "at_least_1s": bool(len(x) >= WINDOW_POINTS),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "m1_file_audit.csv", index=False, encoding="utf-8-sig")
    return df


def save_waveform_compare(role: str, raw: np.ndarray, demeaned: np.ndarray, linear_dt: np.ndarray) -> Path:
    t = np.arange(len(raw)) / SOURCE_FS
    nshow = min(len(raw), int(0.2 * SOURCE_FS))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t[:nshow], raw[:nshow])
    axes[0].set_title(f"{role}: raw waveform")
    axes[0].set_ylabel("amplitude")
    axes[1].plot(t[:nshow], demeaned[:nshow])
    axes[1].set_title("demeaned (used)")
    axes[1].set_ylabel("amplitude")
    axes[2].plot(t[:nshow], linear_dt[:nshow])
    axes[2].set_title("linear detrended (diagnostic only, not used by default)")
    axes[2].set_xlabel("time / s")
    axes[2].set_ylabel("amplitude")
    fig.tight_layout()
    path = FIG / f"{role}_waveform_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_spectrum(role: str, demeaned: np.ndarray) -> Path:
    nperseg = min(4096, len(demeaned))
    f, pxx = welch(
        demeaned,
        fs=SOURCE_FS,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    m = f <= COMMON_FMAX
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(f[m], pxx[m])
    ax.set_xlim(0, COMMON_FMAX)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("PSD")
    ax.set_title(f"{role}: demeaned spectrum (Welch, 0-6 kHz)")
    fig.tight_layout()
    path = FIG / f"{role}_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_envelope(role: str, label: str, raw: np.ndarray, q: dict[str, float]) -> Path:
    xc = raw - np.mean(raw)
    env = np.abs(hilbert(xc))
    env = env - np.mean(env)
    nperseg = min(8192, len(env))
    f, pxx = welch(
        env,
        fs=SOURCE_FS,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    m = f <= ENVELOPE_FMAX
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(f[m], pxx[m])
    for name, center in mechanism_centers(label, q).items():
        if 0 < center <= ENVELOPE_FMAX:
            ax.axvline(center, linestyle="--", alpha=0.40)
    ax.set_xlim(0, ENVELOPE_FMAX)
    ax.set_xlabel("envelope frequency / Hz")
    ax.set_ylabel("envelope PSD")
    ax.set_title(f"{role}: Hilbert envelope spectrum (no band-pass)")
    fig.tight_layout()
    path = FIG / f"{role}_envelope.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_spectrogram(role: str, demeaned: np.ndarray) -> Path:
    f, t, sxx = spectrogram(
        demeaned,
        fs=SOURCE_FS,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        scaling="spectrum",
        mode="magnitude",
    )
    m = f <= COMMON_FMAX
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pcm = ax.pcolormesh(t, f[m], 20.0 * np.log10(sxx[m] + EPS), shading="auto")
    fig.colorbar(pcm, ax=ax, label="magnitude / dB")
    ax.set_xlabel("time / s")
    ax.set_ylabel("frequency / Hz")
    ax.set_title(f"{role}: spectrogram")
    fig.tight_layout()
    path = FIG / f"{role}_spectrogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def representative_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    summary_rows = []
    pre_rows = []
    mech_rows = []
    images: list[Path] = []

    for role, path, label in REPRESENTATIVES:
        x, signal_var, rpm, rpm_source = read_source_de(path)
        if len(x) < WINDOW_POINTS:
            raise RuntimeError(f"Representative shorter than 1 s: {path}")
        if not np.isfinite(rpm):
            raise RuntimeError(f"Representative RPM missing: {path}")

        raw = x[:WINDOW_POINTS].copy()
        demeaned = raw - np.mean(raw)
        linear_dt = detrend(raw, type="linear")
        q = bearing_freqs(rpm)

        raw_mean = float(np.mean(raw))
        demean_mean = float(np.mean(demeaned))
        raw_std = float(np.std(raw))
        demean_std = float(np.std(demeaned))
        linear_std = float(np.std(linear_dt))
        const_shift_error = float(np.max(np.abs((raw - demeaned) - raw_mean)))
        std_rel_change = float(abs(demean_std - raw_std) / (raw_std + EPS))
        linear_vs_demean_l2 = float(
            np.linalg.norm(linear_dt - demeaned) / (np.linalg.norm(demeaned) + EPS)
        )
        demean_pass = bool(
            abs(demean_mean) <= 1e-10 * max(1.0, raw_std)
            and std_rel_change <= 1e-12
            and const_shift_error <= 1e-10 * max(1.0, abs(raw_mean))
        )

        pre_rows.append({
            "role": role,
            "filename": path.name,
            "raw_mean": raw_mean,
            "demeaned_mean": demean_mean,
            "raw_std": raw_std,
            "demeaned_std": demean_std,
            "linear_detrended_std": linear_std,
            "demean_std_relative_change": std_rel_change,
            "demean_constant_shift_max_error": const_shift_error,
            "linear_vs_demean_relative_l2": linear_vs_demean_l2,
            "demean_check_pass": demean_pass,
            "default_use_demean": True,
            "default_use_linear_detrend": False,
            "default_use_bandpass": False,
            "default_use_resample": False,
        })

        ku = float(kurtosis(demeaned, fisher=False, bias=False))
        rms = float(np.sqrt(np.mean(demeaned ** 2)))
        summary_rows.append({
            "role": role,
            "relative_path": str(path.relative_to(ROOT)),
            "filename": path.name,
            "label": label,
            "fs_hz": SOURCE_FS,
            "rpm": rpm,
            "rpm_source": rpm_source,
            "signal_var": signal_var,
            "n_points_total": len(x),
            "window_points": len(raw),
            "window_seconds": len(raw) / SOURCE_FS,
            "rms": rms,
            "kurtosis": ku,
            "fr_hz": q["fr"],
            "bpfo_hz": q["BPFO"],
            "bpfi_hz": q["BPFI"],
            "bsf_hz": q["BSF"],
            "ftf_hz": q["FTF"],
        })

        if label in {"OR", "IR", "B"}:
            freq_name = {"OR": "BPFO", "IR": "BPFI", "B": "BSF"}[label]
            ratio, hrs = envelope_local_ratio(raw, SOURCE_FS, q[freq_name])
            mech_rows.append({
                "label": label,
                "record_id": str(path.relative_to(ROOT / "data" / "raw")),
                "rpm": rpm,
                "frequency_name": freq_name,
                "theoretical_frequency_hz": q[freq_name],
                "window_start": 0,
                "local_envelope_ratio": ratio,
                "harmonic_ratios": "|".join(f"{v:.6g}" for v in hrs),
                "supported_ratio_gt_1": bool(np.isfinite(ratio) and ratio > 1.0),
            })

        images.append(save_waveform_compare(role, raw, demeaned, linear_dt))
        images.append(save_spectrum(role, demeaned))
        images.append(save_envelope(role, label, raw, q))
        if "difficult" in role:
            images.append(save_spectrogram(role, demeaned))

    summary = pd.DataFrame(summary_rows)
    pre = pd.DataFrame(pre_rows)
    mech = pd.DataFrame(mech_rows)
    summary.to_csv(OUT / "representative_signal_summary.csv", index=False, encoding="utf-8-sig")
    pre.to_csv(OUT / "preprocessing_checks.csv", index=False, encoding="utf-8-sig")
    mech.to_csv(OUT / "mechanism_recalc.csv", index=False, encoding="utf-8-sig")
    return summary, pre, mech, images


def make_frequency_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "role": r["role"],
            "filename": r["filename"],
            "rpm": r["rpm"],
            "fr_hz": r["fr_hz"],
            "bpfo_hz": r["bpfo_hz"],
            "bpfi_hz": r["bpfi_hz"],
            "bsf_hz": r["bsf_hz"],
            "ftf_hz": r["ftf_hz"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mechanism_frequency_table.csv", index=False, encoding="utf-8-sig")
    return df


def crosscheck_existing(mech: pd.DataFrame) -> pd.DataFrame:
    baseline_path = ROOT / "outputs" / "q1_baseline_v1" / "formal" / "mechanism_checks.csv"
    rows = []
    if not baseline_path.exists():
        for _, r in mech.iterrows():
            rows.append({
                "record_id": r["record_id"],
                "baseline_found": False,
                "frequency_abs_diff_hz": np.nan,
                "ratio_abs_diff": np.nan,
                "match_pass": False,
            })
    else:
        baseline = pd.read_csv(baseline_path)
        for _, r in mech.iterrows():
            b = baseline[baseline["record_id"] == r["record_id"]]
            if b.empty:
                rows.append({
                    "record_id": r["record_id"],
                    "baseline_found": False,
                    "frequency_abs_diff_hz": np.nan,
                    "ratio_abs_diff": np.nan,
                    "match_pass": False,
                })
                continue
            br = b.iloc[0]
            fd = abs(float(r["theoretical_frequency_hz"]) - float(br["theoretical_frequency_hz"]))
            rd = abs(float(r["local_envelope_ratio"]) - float(br["local_envelope_ratio"]))
            rows.append({
                "record_id": r["record_id"],
                "baseline_found": True,
                "frequency_abs_diff_hz": fd,
                "ratio_abs_diff": rd,
                "match_pass": bool(fd <= 1e-9 and rd <= 1e-6),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "crosscheck_existing_outputs.csv", index=False, encoding="utf-8-sig")
    return df


def image_checks(images: list[Path]) -> pd.DataFrame:
    rows = []
    for p in images:
        size = p.stat().st_size if p.exists() else 0
        readable = False
        shape = ""
        if p.exists() and size > 0:
            try:
                arr = plt.imread(p)
                readable = bool(arr.ndim in (2, 3) and arr.shape[0] > 100 and arr.shape[1] > 100)
                shape = "x".join(map(str, arr.shape))
            except Exception:
                readable = False
        rows.append({
            "file": str(p.relative_to(ROOT)),
            "exists": p.exists(),
            "bytes": size,
            "readable": readable,
            "shape": shape,
            "image_pass": bool(p.exists() and size > 10000 and readable),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "figure_checks.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    # 1. Full M1 audit: all 56 independent source files.
    audit = audit_m1()
    counts = audit["label"].value_counts().to_dict()
    count_pass = bool(len(audit) == 56 and all(counts.get(k, 0) == v for k, v in EXPECTED_COUNTS.items()))
    readable_pass = bool(audit["at_least_1s"].all() and audit["signal_var"].astype(str).str.endswith("DE_time").all())

    # 2. Independent formula check against frozen expected SKF6205 orders.
    orders = bearing_orders()
    order_diffs = {k: abs(orders[k] - EXPECTED_ORDERS[k]) for k in EXPECTED_ORDERS}
    formula_pass = bool(all(v <= 1e-12 for v in order_diffs.values()))
    order_df = pd.DataFrame([
        {"name": k, "calculated_order": orders[k], "expected_order": EXPECTED_ORDERS[k], "abs_diff": order_diffs[k]}
        for k in ("BPFO", "BPFI", "BSF", "FTF")
    ])
    order_df.to_csv(OUT / "skf6205_formula_check.csv", index=False, encoding="utf-8-sig")

    # 3. Representative and difficult samples.
    summary, pre, mech, images = representative_analysis()
    make_frequency_table(summary)
    pre_pass = bool(pre["demean_check_pass"].all())

    # 4. Reproduce existing actual mechanism-check evidence.
    cross = crosscheck_existing(mech)
    cross_pass = bool(len(cross) == 3 and cross["match_pass"].all())

    # 5. Validate generated PNG evidence itself.
    fig_checks = image_checks(images)
    figure_pass = bool(len(fig_checks) == 17 and fig_checks["image_pass"].all())

    # 6. Existing adaptive-BSF evidence is trace-only; no hard pass criterion.
    h3_path = ROOT / "outputs" / "q1_improvement_v1" / "H3_adaptive_bsf_checks.csv"
    h3_note = {"exists": h3_path.exists()}
    if h3_path.exists():
        h3 = pd.read_csv(h3_path)
        h3_note.update({
            "n_rows": int(len(h3)),
            "adaptive_bsf_ratio_values": [float(x) for x in h3.get("adaptive_bsf_ratio", pd.Series(dtype=float)).dropna().tolist()],
            "all_adaptive_bsf_ratio_lt_1": bool((h3["adaptive_bsf_ratio"] < 1.0).all()) if "adaptive_bsf_ratio" in h3 else None,
        })

    overall_pass = bool(count_pass and readable_pass and formula_pass and pre_pass and cross_pass and figure_pass)

    manifest = {
        "protocol": "Q1B-STEP03-v1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "source_fs_hz": SOURCE_FS,
        "window_seconds": WINDOW_SECONDS,
        "m1_total_files": int(len(audit)),
        "m1_class_counts": {k: int(counts.get(k, 0)) for k in EXPECTED_COUNTS},
        "representatives": [x[1].name for x in REPRESENTATIVES],
        "preprocessing_policy": {
            "demean": True,
            "linear_detrend_default": False,
            "bandpass_default": False,
            "resample_in_step03": False,
            "envelope": "Hilbert envelope for mechanism evidence; no forced band-pass",
        },
        "h3_adaptive_bsf_trace": h3_note,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "checks": {
            "m1_count_and_class_mapping": count_pass,
            "m1_de_readable_and_at_least_1s": readable_pass,
            "skf6205_formula_implementation": formula_pass,
            "demean_preserves_variability": pre_pass,
            "reproduces_existing_mechanism_checks": cross_pass,
            "all_required_figures_generated_and_readable": figure_pass,
        },
        "evidence": {
            "m1_file_audit": "outputs/q1b_signal_mechanism/m1_file_audit.csv",
            "representative_summary": "outputs/q1b_signal_mechanism/representative_signal_summary.csv",
            "preprocessing_checks": "outputs/q1b_signal_mechanism/preprocessing_checks.csv",
            "mechanism_frequency_table": "outputs/q1b_signal_mechanism/mechanism_frequency_table.csv",
            "mechanism_recalc": "outputs/q1b_signal_mechanism/mechanism_recalc.csv",
            "baseline_crosscheck": "outputs/q1b_signal_mechanism/crosscheck_existing_outputs.csv",
            "figure_checks": "outputs/q1b_signal_mechanism/figure_checks.csv",
            "figures_directory": "outputs/q1b_signal_mechanism/figures/",
        },
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Q1B / STEP03 signal mechanism validation",
        f"status={report['status']}",
        f"M1 files={len(audit)}; counts={manifest['m1_class_counts']}",
        f"formula_pass={formula_pass}",
        f"preprocessing_pass={pre_pass}",
        f"baseline_crosscheck_pass={cross_pass}",
        f"figure_pass={figure_pass}; figure_count={len(fig_checks)}",
        "Policy: demean=yes; linear_detrend_default=no; bandpass_default=no; resample_step03=no.",
        "Mechanism frequencies are supporting evidence, not hard label rules.",
    ]
    (OUT / "result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
