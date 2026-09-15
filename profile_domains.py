from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from scipy.signal import welch

ROOT = Path("data/raw")
OUT_DIR = Path("outputs/data_profile")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_FMAX = 6000.0
WINDOW_SECONDS = 1.0
MAX_WINDOWS_PER_FILE = 4

def infer_fs(path):
    s = str(path).lower()
    if "12khz" in s:
        return 12000
    if "48khz" in s:
        return 48000
    if "target_domain" in s:
        return 32000
    raise ValueError(f"无法判断采样率: {path}")

def source_label(filename):
    name = filename.upper()
    if name.startswith("IR"):
        return "IR"
    if name.startswith("OR"):
        return "OR"
    if name.startswith("B"):
        return "B"
    if name.startswith("N"):
        return "N"
    return "UNKNOWN"

def select_signal(mat, path):
    keys = [k for k in mat.keys() if not k.startswith("__")]
    lowpath = str(path).lower()

    # 目标域：A-P 每个文件只有一个同名字母信号
    if "target_domain" in lowpath:
        for k in keys:
            arr = np.asarray(mat[k]).squeeze()
            if arr.ndim == 1 and arr.size > 1000:
                return arr.astype(float), k
        raise ValueError(f"目标域未找到有效信号: {path}")

    # 源域：根据目录选择近端传感器
    if "12khz_fe" in lowpath:
        preferred = ["FE_time"]
    else:
        preferred = ["DE_time"]

    for suffix in preferred:
        for k in keys:
            if k.endswith(suffix):
                arr = np.asarray(mat[k]).squeeze()
                if arr.ndim == 1 and arr.size > 1000:
                    return arr.astype(float), k

    # 兜底：寻找任意 DE/FE 信号
    for k in keys:
        if ("DE_time" in k or "FE_time" in k):
            arr = np.asarray(mat[k]).squeeze()
            if arr.ndim == 1 and arr.size > 1000:
                return arr.astype(float), k

    raise ValueError(f"源域未找到有效振动信号: {path}")

def parse_rpm(mat, path):
    for k in mat:
        if "RPM" in k.upper():
            arr = np.asarray(mat[k]).squeeze()
            if arr.size:
                try:
                    return float(arr.flat[0])
                except:
                    pass

    m = re.search(r"(\d{4})rpm", path.name, re.I)
    if m:
        return float(m.group(1))

    return np.nan

def spectral_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) <= 1:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def features(x, fs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    mean = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    abs_mean = np.mean(np.abs(x))
    ptp = np.ptp(x)

    eps = 1e-12

    crest = peak / (rms + eps)
    impulse = peak / (abs_mean + eps)
    shape = rms / (abs_mean + eps)

    sqrt_abs_mean = np.mean(np.sqrt(np.abs(x)))
    clearance = peak / (sqrt_abs_mean ** 2 + eps)

    sk = skew(x, bias=False)
    ku = kurtosis(x, fisher=False, bias=False)

    # Welch PSD
    nperseg = min(4096, len(x))
    f, pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant"
    )

    mask = (f >= 0) & (f <= COMMON_FMAX)
    f2 = f[mask]
    p2 = pxx[mask]

    total = np.trapz(p2, f2) + eps

    centroid = np.sum(f2 * p2) / (np.sum(p2) + eps)

    csum = np.cumsum(p2)
    if csum[-1] > 0:
        f95 = f2[np.searchsorted(csum, 0.95 * csum[-1])]
    else:
        f95 = np.nan

    dom_freq = f2[np.argmax(p2)] if len(p2) else np.nan
    spec_ent = spectral_entropy(p2)

    def band_energy(lo, hi):
        m = (f2 >= lo) & (f2 < hi)
        if np.sum(m) < 2:
            return np.nan
        return np.trapz(p2[m], f2[m]) / total

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
        "dominant_freq_0_6k": dom_freq,
        "spectral_centroid_0_6k": centroid,
        "f95_0_6k": f95,
        "spectral_entropy_0_6k": spec_ent,
        "energy_0_500": band_energy(0, 500),
        "energy_500_1500": band_energy(500, 1500),
        "energy_1500_3000": band_energy(1500, 3000),
        "energy_3000_6000": band_energy(3000, 6000),
    }

rows = []
errors = []

files = sorted(ROOT.rglob("*.mat"))

for path in files:
    try:
        fs = infer_fs(path)
        mat = loadmat(path)
        signal, signal_var = select_signal(mat, path)
        rpm = parse_rpm(mat, path)

        if "target_domain" in str(path).lower():
            domain = "target"
            subgroup = "target_32khz"
            label = "UNKNOWN"
        else:
            domain = "source"

            low = str(path).lower()
            if "12khz_de" in low:
                subgroup = "source_12khz_de"
            elif "12khz_fe" in low:
                subgroup = "source_12khz_fe"
            elif "48khz_de" in low:
                subgroup = "source_48khz_de"
            elif "48khz_normal" in low:
                subgroup = "source_48khz_normal"
            else:
                subgroup = "source_other"

            label = source_label(path.name)

        win_len = int(fs * WINDOW_SECONDS)
        nwin = len(signal) // win_len
        nuse = min(nwin, MAX_WINDOWS_PER_FILE)

        if nuse == 0:
            raise ValueError(
                f"有效长度不足1秒: len={len(signal)}, fs={fs}"
            )

        # 均匀选窗口，避免只看信号开头
        if nwin <= MAX_WINDOWS_PER_FILE:
            starts = [i * win_len for i in range(nwin)]
        else:
            idx = np.linspace(
                0,
                nwin - 1,
                MAX_WINDOWS_PER_FILE,
                dtype=int
            )
            starts = [int(i * win_len) for i in idx]

        for widx, start in enumerate(starts):
            x = signal[start:start + win_len]

            ft = features(x, fs)

            row = {
                "domain": domain,
                "subgroup": subgroup,
                "file": path.name,
                "relative_path": str(path.relative_to(ROOT)),
                "label": label,
                "fs": fs,
                "rpm": rpm,
                "signal_var": signal_var,
                "signal_length": len(signal),
                "window_index": widx,
                "window_start": start
            }

            row.update(ft)
            rows.append(row)

    except Exception as e:
        errors.append({
            "file": str(path.relative_to(ROOT)),
            "error": repr(e)
        })

df = pd.DataFrame(rows)
err_df = pd.DataFrame(errors)

detail_path = OUT_DIR / "domain_profile_windows.csv"
df.to_csv(detail_path, index=False, encoding="utf-8-sig")

if len(err_df):
    err_df.to_csv(
        OUT_DIR / "domain_profile_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

feature_cols = [
    "std",
    "rms",
    "peak_abs",
    "ptp",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "clearance_factor",
    "dominant_freq_0_6k",
    "spectral_centroid_0_6k",
    "f95_0_6k",
    "spectral_entropy_0_6k",
    "energy_0_500",
    "energy_500_1500",
    "energy_1500_3000",
    "energy_3000_6000"
]

summary = (
    df.groupby(["domain", "subgroup"])[feature_cols]
      .agg(["mean", "std", "median"])
)

summary.to_csv(
    OUT_DIR / "domain_profile_group_summary.csv",
    encoding="utf-8-sig"
)

# ---------- 域相似度 ----------
# 为避免幅值尺度主导距离，只使用较稳健的无量纲/频谱特征
sim_features = [
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "spectral_centroid_0_6k",
    "f95_0_6k",
    "spectral_entropy_0_6k",
    "energy_0_500",
    "energy_500_1500",
    "energy_1500_3000",
    "energy_3000_6000"
]

X = df[sim_features].replace([np.inf, -np.inf], np.nan)
med = X.median()
X = X.fillna(med)

mu = X.mean()
sd = X.std().replace(0, 1)

Z = (X - mu) / sd

zdf = df[["domain", "subgroup"]].copy()
for c in sim_features:
    zdf[c] = Z[c]

target_center = (
    zdf[zdf["domain"] == "target"][sim_features]
    .mean()
    .values
)

similarity_rows = []

for subgroup, g in zdf[zdf["domain"] == "source"].groupby("subgroup"):
    center = g[sim_features].mean().values
    dist = float(np.linalg.norm(center - target_center))

    similarity_rows.append({
        "source_subgroup": subgroup,
        "distance_to_target": dist,
        "n_windows": len(g)
    })

sim_df = pd.DataFrame(similarity_rows)
sim_df = sim_df.sort_values("distance_to_target")
sim_df.to_csv(
    OUT_DIR / "source_target_similarity.csv",
    index=False,
    encoding="utf-8-sig"
)

# ---------- 文本报告 ----------
report_path = OUT_DIR / "domain_profile_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("=== E题阶段1.1：源域/目标域数据画像 ===\n\n")

    f.write(f"成功处理 MAT 文件数: {df['relative_path'].nunique()}\n")
    f.write(f"产生 1秒窗口数: {len(df)}\n")
    f.write(f"处理失败文件数: {len(errors)}\n\n")

    f.write("=== 各数据组窗口数量 ===\n")
    counts = (
        df.groupby(["domain", "subgroup"])
          .size()
          .reset_index(name="windows")
    )

    for _, r in counts.iterrows():
        f.write(
            f"{r['domain']} / {r['subgroup']}: "
            f"{r['windows']}\n"
        )

    f.write("\n=== 源域类别窗口数量 ===\n")
    s = df[df["domain"] == "source"]
    label_counts = (
        s.groupby(["subgroup", "label"])
         .size()
         .reset_index(name="windows")
    )

    for _, r in label_counts.iterrows():
        f.write(
            f"{r['subgroup']} / {r['label']}: "
            f"{r['windows']}\n"
        )

    f.write("\n=== 源域到目标域的初步特征距离 ===\n")
    f.write("说明：距离越小，仅表示在本次选取的统计/频谱特征下越接近；\n")
    f.write("不能单独据此决定最终源域筛选方案。\n\n")

    for _, r in sim_df.iterrows():
        f.write(
            f"{r['source_subgroup']}: "
            f"distance={r['distance_to_target']:.6f}, "
            f"windows={int(r['n_windows'])}\n"
        )

    f.write("\n=== 目标域 A-P 的基础统计（按文件均值） ===\n")

    target_file = (
        df[df["domain"] == "target"]
        .groupby("file")[[
            "rms",
            "kurtosis",
            "crest_factor",
            "spectral_centroid_0_6k",
            "spectral_entropy_0_6k"
        ]]
        .mean()
    )

    f.write(target_file.to_string())
    f.write("\n")

    if errors:
        f.write("\n=== 失败文件 ===\n")
        for e in errors:
            f.write(f"{e['file']}: {e['error']}\n")
    else:
        f.write("\n失败文件: 无\n")

print("阶段1.1运行完成。")
print()
print("请优先把这个文件发给我：")
print(report_path.resolve())
print()
print("如果方便，再一起发：")
print((OUT_DIR / "source_target_similarity.csv").resolve())
