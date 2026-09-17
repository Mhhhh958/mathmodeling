from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, hilbert, spectrogram, welch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "source_domain"
OUT = ROOT / "outputs" / "q1b_signal_mechanism_export"
OUT.mkdir(parents=True, exist_ok=True)

FS = 48000
WINDOW_SECONDS = 1.0
N = int(FS * WINDOW_SECONDS)
COMMON_FMAX = 6000.0
ENVELOPE_FMAX = 1000.0
EPS = 1e-12

# Frozen STEP03-A representatives. Do not change without reopening A-stage selection.
SAMPLES = [
    ("OR_typical", "OR", RAW / "cwru_48khz_de" / "OR007@12_0.mat"),
    ("IR_typical", "IR", RAW / "cwru_48khz_de" / "IR007_0.mat"),
    ("B_difficult", "B", RAW / "cwru_48khz_de" / "B007_0.mat"),
    ("N_typical", "N", RAW / "cwru_48khz_normal" / "N_1_(1772rpm).mat"),
    ("N_difficult", "N", RAW / "cwru_48khz_normal" / "N_0.mat"),
]

FROZEN_STD = {
    "OR_typical": 0.22117714591722754,
    "IR_typical": 0.5805873751719882,
    "B_difficult": 0.14710492669108596,
    "N_typical": 0.06502069031187659,
    "N_difficult": 0.07253491456775636,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_de(path: Path) -> tuple[np.ndarray, str]:
    mat = loadmat(path)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if k.endswith("DE_time"):
            a = np.asarray(v).squeeze()
            if a.ndim == 1 and a.size >= N:
                x = a.astype(float)
                if not np.isfinite(x).all():
                    raise RuntimeError(f"non-finite values: {path}")
                return x, k
    raise RuntimeError(f"no usable DE_time: {path}")


wave_rows = []
psd_rows = []
env_rows = []
spec_rows = []
input_rows = []

for role, label, path in SAMPLES:
    x, signal_var = read_de(path)
    raw = x[:N].copy()
    demeaned = raw - np.mean(raw)
    linear_dt = detrend(raw, type="linear")

    # Frozen-value consistency check against STEP03-A.
    std = float(np.std(raw))
    if abs(std - FROZEN_STD[role]) > 1e-10:
        raise AssertionError(f"frozen std mismatch for {role}: {std} vs {FROZEN_STD[role]}")

    t = np.arange(N, dtype=float) / FS
    for i in range(N):
        wave_rows.append((role, label, path.name, signal_var, i, t[i], raw[i], demeaned[i], linear_dt[i]))

    nperseg = min(4096, len(demeaned))
    f_psd, p_psd = welch(
        demeaned,
        fs=FS,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    mask = f_psd <= COMMON_FMAX
    for ff, pp in zip(f_psd[mask], p_psd[mask]):
        psd_rows.append((role, label, path.name, float(ff), float(pp)))

    env = np.abs(hilbert(demeaned))
    env = env - np.mean(env)
    nperseg_env = min(8192, len(env))
    f_env, p_env = welch(
        env,
        fs=FS,
        nperseg=nperseg_env,
        noverlap=nperseg_env // 2,
        detrend="constant",
        scaling="density",
    )
    mask_env = f_env <= ENVELOPE_FMAX
    for ff, pp in zip(f_env[mask_env], p_env[mask_env]):
        env_rows.append((role, label, path.name, float(ff), float(pp)))

    if role == "B_difficult":
        f_sp, t_sp, sxx = spectrogram(
            demeaned,
            fs=FS,
            window="hann",
            nperseg=2048,
            noverlap=1536,
            scaling="spectrum",
            mode="magnitude",
        )
        keep = f_sp <= COMMON_FMAX
        mag_db = 20.0 * np.log10(sxx[keep] + EPS)
        for fi, ff in enumerate(f_sp[keep]):
            for ti, tt in enumerate(t_sp):
                spec_rows.append((role, label, path.name, float(tt), float(ff), float(mag_db[fi, ti])))

    input_rows.append({
        "role": role,
        "class_label": label,
        "relative_path": str(path.relative_to(ROOT)),
        "signal_var": signal_var,
        "fs_hz": FS,
        "window_start_sample": 0,
        "window_end_sample_exclusive": N,
        "window_seconds": WINDOW_SECONDS,
        "raw_file_sha256": sha256_file(path),
        "raw_std_first_1s": std,
    })

pd.DataFrame(
    wave_rows,
    columns=["role","class_label","filename","signal_var","sample_index","time_s","raw","demeaned","linear_detrended"],
).to_csv(OUT / "representative_waveforms.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

pd.DataFrame(
    psd_rows,
    columns=["role","class_label","filename","frequency_hz","psd"],
).to_csv(OUT / "representative_psd_0_6k.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

pd.DataFrame(
    env_rows,
    columns=["role","class_label","filename","frequency_hz","envelope_psd"],
).to_csv(OUT / "representative_envelope_psd_0_1k.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

pd.DataFrame(
    spec_rows,
    columns=["role","class_label","filename","time_s","frequency_hz","magnitude_db"],
).to_csv(OUT / "B_difficult_spectrogram_0_6k.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

pd.DataFrame(input_rows).to_csv(OUT / "input_trace.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

readme = """# STEP03-A supplemental plot-data export\n\nThis directory only exports frozen source data for STEP03-B figure composition.\nIt does not train a classifier, change sample selection, tune parameters, or change any STEP03-A conclusion.\n\nFrozen protocol:\n- five representatives fixed in STEP03-A;\n- first 1.0 s (48000 points), DE_time, 48 kHz;\n- demean = raw - mean(raw); linear detrend exported only for comparison;\n- Welch PSD: nperseg=4096, 50% overlap, density, 0-6 kHz;\n- Hilbert envelope PSD: no forced band-pass, nperseg=8192, 50% overlap, 0-1 kHz;\n- B_difficult spectrogram: Hann, nperseg=2048, noverlap=1536, spectrum magnitude, 0-6 kHz.\n"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

manifest = {
    "step": "03-A-supplement",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "plot-source-data export only; no retraining/reselection/retuning",
    "source_fs_hz": FS,
    "window_seconds": WINDOW_SECONDS,
    "samples": input_rows,
    "parameters": {
        "demean": True,
        "linear_detrend": "exported for comparison only; not default preprocessing",
        "bandpass": None,
        "resample": None,
        "welch_psd": {"nperseg": 4096, "noverlap": 2048, "scaling": "density", "fmax_hz": 6000},
        "envelope_psd": {"hilbert": True, "forced_bandpass": False, "nperseg": 8192, "noverlap": 4096, "fmax_hz": 1000},
        "B_spectrogram": {"window": "hann", "nperseg": 2048, "noverlap": 1536, "scaling": "spectrum", "mode": "magnitude", "fmax_hz": 6000},
    },
}

for fn in [
    "representative_waveforms.csv",
    "representative_psd_0_6k.csv",
    "representative_envelope_psd_0_1k.csv",
    "B_difficult_spectrogram_0_6k.csv",
    "input_trace.csv",
    "README.md",
]:
    fp = OUT / fn
    manifest.setdefault("output_files", []).append({
        "name": fn,
        "bytes": fp.stat().st_size,
        "sha256": sha256_file(fp),
    })

(OUT / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("STEP03A_SUPPLEMENT_EXPORT=PASS")
print(json.dumps({"files": len(manifest["output_files"]), "out": str(OUT)}, ensure_ascii=False))
