from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.signal import welch, periodogram, hilbert
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn

# ============================================================
# STEP13: frozen figures + answers.
# Only accepted STEP12 chain inputs are used. No model reselection.
# Every figure is redrawn programmatically; no old image is copied.
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_figures_step13"
FIG = OUT / "figures"
BW = OUT / "figures_bw"
SRC = OUT / "source_data"
TAB = OUT / "tables"
for d in (OUT, FIG, BW, SRC, TAB):
    d.mkdir(parents=True, exist_ok=True)

ACCEPTED_STEP12_COMMIT = "eb357e98489a3a8704c8b4f2d8686a7a0436dd2d"
FREEZE_ID = "STEP13-FROZEN-v1"
SEED = 20260916
LABELS = ["OR", "IR", "B", "N"]
CLASS_COLORS = {
    "OR": "#D55E00",  # vermillion
    "IR": "#0072B2",  # blue
    "B":  "#009E73",  # bluish green
    "N":  "#CC79A7",  # reddish purple
}
DOMAIN_MARKERS = {"source": "o", "target": "^"}
FIGURE_WIDTH_IN = 6.5
PNG_DPI = 450
EPS = 1e-12

# accepted inputs
Q1_SRC = ROOT / "outputs/q1c_feature_extraction/q2_source_raw.csv"
Q1_TGT = ROOT / "outputs/q1c_feature_extraction/q2_target_raw.csv"
Q2_INTERFACE = ROOT / "outputs/q1c_feature_extraction/q2_interface.json"
Q2_VALID = ROOT / "outputs/q2c_failure_driven_improvement/validation_report.json"
Q2_CONF = ROOT / "outputs/q2c_failure_driven_improvement/confusion_matrix_final.csv"
Q2_BASE_OUTER = ROOT / "outputs/q2c_failure_driven_improvement/baseline_outer_fold_metrics.csv"
Q2_FINAL_OUTER = ROOT / "outputs/q2c_failure_driven_improvement/final_outer_fold_metrics.csv"
Q2_FINAL_MODEL = ROOT / "outputs/q2c_failure_driven_improvement/models/final_source_model.joblib"
Q3_INTERFACE = ROOT / "outputs/q2c_failure_driven_improvement/q3_interface.json"
Q3B_VALID = ROOT / "outputs/q3b_unsupervised_transfer/validation_report.json"
Q3B_SIM = ROOT / "outputs/q3b_unsupervised_transfer/simulated_target_all_methods_metrics.csv"
Q3C_TABLE = ROOT / "outputs/q3c_final_labels_and_display/final_A_P_label_table.csv"
Q3C_ANS = ROOT / "outputs/q3c_final_labels_and_display/final_A_P_labels_for_answer.csv"
Q4_GLOBAL = ROOT / "outputs/q4_explainability/global_rf_feature_importance.csv"
Q4_LOCAL = ROOT / "outputs/q4_explainability/local_path_importance_long.csv"
Q4_SOURCE_CASES = ROOT / "outputs/q4_explainability/source_known_label_case_table.csv"
Q4_FAITH = ROOT / "outputs/q4_explainability/faithfulness_validation_by_case.csv"
Q4_STAB = ROOT / "outputs/q4_explainability/stability_validation.csv"
S12_VALID = ROOT / "outputs/q_joint_validation_step12/validation_report.json"
S12_MANIFEST = ROOT / "outputs/q_joint_validation_step12/run_manifest.json"
S12_ABL = ROOT / "outputs/q_joint_validation_step12/feature_group_ablation.csv"
S12_WIN = ROOT / "outputs/q_joint_validation_step12/window_parameter_sensitivity.csv"
S12_BOOT = ROOT / "outputs/q_joint_validation_step12/file_level_bootstrap_replicates.csv"
S12_MIG = ROOT / "outputs/q_joint_validation_step12/migration_robustness_crosscheck.csv"
S12_FAIL = ROOT / "outputs/q_joint_validation_step12/failure_boundaries.csv"

REP_FILES = {
    "OR": ROOT / "data/raw/source_domain/cwru_48khz_de/OR007@12_0.mat",
    "IR": ROOT / "data/raw/source_domain/cwru_48khz_de/IR007_0.mat",
    "B":  ROOT / "data/raw/source_domain/cwru_48khz_de/B007_0.mat",
    "N":  ROOT / "data/raw/source_domain/cwru_48khz_normal/N_1_(1772rpm).mat",
}

# ---------------- helpers ----------------
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.15,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)


def feature_display(f: str) -> str:
    m = {
        "td_rms": "RMS",
        "td_std": "标准差",
        "td_peak_abs": "绝对峰值",
        "td_ptp": "峰峰值",
        "td_skewness": "偏度",
        "td_kurtosis": "峭度",
        "td_crest_factor": "峰值因子",
        "td_impulse_factor": "脉冲因子",
        "td_shape_factor": "波形因子",
        "td_clearance_factor": "裕度因子",
        "td_zero_cross_rate": "过零率",
        "fd_dominant_hz_0_6k": "主频(0–6 kHz)",
        "fd_centroid_hz_0_6k": "频谱重心(0–6 kHz)",
        "fd_f95_hz_0_6k": "95%累积能量频率",
        "fd_entropy_0_6k": "频谱熵(0–6 kHz)",
        "fd_energy_ratio_0_500": "0–500 Hz能量占比",
        "fd_energy_ratio_500_1500": "0.5–1.5 kHz能量占比",
        "fd_energy_ratio_1500_3000": "1.5–3 kHz能量占比",
        "fd_energy_ratio_3000_6000": "3–6 kHz能量占比",
        "env_rms": "包络RMS",
        "env_kurtosis": "包络峭度",
        "env_dominant_hz_0_500": "包络主频(0–500 Hz)",
        "env_centroid_hz_0_500": "包络谱重心",
        "env_entropy_0_500": "包络谱熵(0–500 Hz)",
        "env_energy_ratio_0_50": "包络0–50 Hz能量占比",
        "env_energy_ratio_50_150": "包络50–150 Hz能量占比",
        "env_energy_ratio_150_300": "包络150–300 Hz能量占比",
        "env_energy_ratio_300_500": "包络300–500 Hz能量占比",
    }
    return m.get(f, f)


def select_signal(mat: dict, path: Path) -> np.ndarray:
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if k.endswith("DE_time"):
            x = np.asarray(v).squeeze()
            if x.ndim == 1 and x.size >= 48000:
                return x.astype(float)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        x = np.asarray(v).squeeze()
        if x.ndim == 1 and x.size >= 48000:
            return x.astype(float)
    raise RuntimeError(f"No usable signal in {path}")


def bearing_freq_6205(rpm: float, label: str) -> float | None:
    if label == "N" or not np.isfinite(rpm):
        return None
    z, d, D = 9.0, 0.3126, 1.537
    fr = rpm / 60.0
    r = d / D
    if label == "OR":
        return fr * z / 2.0 * (1.0 - r)
    if label == "IR":
        return fr * z / 2.0 * (1.0 + r)
    if label == "B":
        return fr * D / d * (1.0 - r ** 2)  # official-problem convention frozen earlier
    return None


figure_files = []
figure_checks = []
source_snapshots = []


def save_snapshot(fid: str, df: pd.DataFrame, suffix: str = "data") -> Path:
    path = SRC / f"{fid}_{suffix}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    source_snapshots.append({"figure_id": fid, "snapshot": str(path.relative_to(ROOT)), "rows": int(len(df)), "columns": int(len(df.columns))})
    # At least three plotted numeric values are re-read from disk and checked exactly/tightly.
    reread = pd.read_csv(path)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    vals = []
    for c in numeric:
        for i in range(len(df)):
            a = df.iloc[i][c]
            if pd.notna(a) and np.isfinite(float(a)):
                vals.append((i, c, float(a)))
    if len(vals) < 3:
        raise RuntimeError(f"{fid} snapshot has fewer than 3 numeric values")
    for k in [0, len(vals)//2, len(vals)-1]:
        i, c, expected = vals[k]
        actual = float(reread.iloc[i][c])
        ok = bool(np.isclose(expected, actual, rtol=1e-10, atol=1e-12))
        figure_checks.append({"figure_id": fid, "row": int(i), "field": c, "expected": expected, "re_read": actual, "pass": ok})
    return path


def save_figure(fig, fid: str):
    png = FIG / f"{fid}.png"
    pdf = FIG / f"{fid}.pdf"
    svg = FIG / f"{fid}.svg"
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    img = Image.open(png)
    if img.width < 1800 or img.height < 800:
        raise RuntimeError(f"{fid} PNG too small for print: {img.size}")
    bw = BW / f"{fid}_bw.png"
    img.convert("L").save(bw, dpi=(300, 300))
    for ext, p in [("png", png), ("pdf", pdf), ("svg", svg), ("bw_png", bw)]:
        if not p.exists() or p.stat().st_size < 1000:
            raise RuntimeError(f"Missing/too-small figure artifact {p}")
        figure_files.append({"figure_id": fid, "format": ext, "path": str(p.relative_to(ROOT)), "bytes": int(p.stat().st_size)})


def source_file_medians(src: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    meta = src[["independent_object_id", "class_label", "load_hp", "rpm_value", "fs_hz", "sensor_position"]].drop_duplicates("independent_object_id")
    med = src.groupby("independent_object_id")[feats].median().reset_index()
    return meta.merge(med, on="independent_object_id", how="inner")


def target_file_medians(tgt: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    meta = tgt[["independent_object_id", "truth_status", "fs_hz", "sensor_position"]].drop_duplicates("independent_object_id")
    med = tgt.groupby("independent_object_id")[feats].median().reset_index()
    out = meta.merge(med, on="independent_object_id", how="inner")
    out["target_id"] = out["independent_object_id"].map(lambda s: Path(str(s)).stem)
    return out


def make_data_tables(src, tgt, feats, q2v, q3c):
    sfiles = src[["independent_object_id", "class_label", "load_hp", "rpm_value", "fs_hz", "sensor_position"]].drop_duplicates("independent_object_id")
    tfiles = tgt[["independent_object_id", "truth_status", "fs_hz", "sensor_position"]].drop_duplicates("independent_object_id")
    overview = pd.DataFrame([
        {
            "domain": "source_M1",
            "independent_files": sfiles["independent_object_id"].nunique(),
            "windows": len(src),
            "sampling_rate_hz": int(src["fs_hz"].iloc[0]),
            "rpm": f"{int(np.nanmin(sfiles.rpm_value))}–{int(np.nanmax(sfiles.rpm_value))}",
            "channel_sensor": "DE_time / drive_end",
            "label_status": "known OR/IR/B/N",
            "feature_count": len(feats),
        },
        {
            "domain": "target_A_to_P",
            "independent_files": tfiles["independent_object_id"].nunique(),
            "windows": len(tgt),
            "sampling_rate_hz": int(tgt["fs_hz"].iloc[0]),
            "rpm": "≈600 (official operating condition; not model input)",
            "channel_sensor": "anonymous single channel / sensor position unknown",
            "label_status": "unknown; predictions only",
            "feature_count": len(feats),
        },
    ])
    overview.to_csv(TAB / "T01_source_target_overview.csv", index=False, encoding="utf-8-sig")

    counts = pd.crosstab(sfiles["load_hp"], sfiles["class_label"]).reindex(index=[0,1,2,3], columns=LABELS, fill_value=0)
    counts.index.name = "load_hp"
    counts.to_csv(TAB / "T02_source_class_by_load_counts.csv", encoding="utf-8-sig")

    fm = q2v["final_pooled_file_metrics"]
    class_metrics = pd.DataFrame([{ "class": c,
        "precision": fm[f"precision_{c}"], "recall": fm[f"recall_{c}"], "f1": fm[f"f1_{c}"], "support_files": fm[f"support_{c}"]
    } for c in LABELS])
    class_metrics.to_csv(TAB / "T03_q2_file_level_class_metrics.csv", index=False, encoding="utf-8-sig")

    ap_cols = ["target_id","final_pred_label","model_score_top1","score_margin_top1_top2","entropy_norm","window_consistency","vote_agreement_rate","all_seed_labels_agree","mechanism_nearest_source_class","pred_class_distance_over_source_p95","uncertainty_level","truth_status"]
    ap = q3c[ap_cols].copy()
    ap.to_csv(TAB / "T04_A_P_final_answer.csv", index=False, encoding="utf-8-sig")
    return overview, counts, class_metrics, ap


def main():
    set_style()
    start = datetime.now(timezone.utc)
    required_inputs = [Q1_SRC,Q1_TGT,Q2_INTERFACE,Q2_VALID,Q2_CONF,Q2_BASE_OUTER,Q2_FINAL_OUTER,Q2_FINAL_MODEL,Q3_INTERFACE,Q3B_VALID,Q3B_SIM,Q3C_TABLE,Q4_GLOBAL,Q4_LOCAL,Q4_SOURCE_CASES,Q4_FAITH,Q4_STAB,S12_VALID,S12_MANIFEST,S12_ABL,S12_WIN,S12_BOOT,S12_MIG,S12_FAIL]
    missing = [str(p) for p in required_inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    q2i = json_load(Q2_INTERFACE)
    q3i = json_load(Q3_INTERFACE)
    q2v = json_load(Q2_VALID)
    q3v = json_load(Q3B_VALID)
    s12v = json_load(S12_VALID)
    s12m = json_load(S12_MANIFEST)
    if s12v.get("status") != "PASS" or s12v.get("blocking_reruns") != []:
        raise RuntimeError("STEP12 is not a closed PASS")
    if q3v.get("selected_method") != "A0_no_transfer":
        raise RuntimeError("STEP09 frozen method changed")

    # Lock STEP12-approved inputs by SHA.
    sha_checks = {
        "source_sha256": sha256(Q1_SRC),
        "target_sha256": sha256(Q1_TGT),
        "q2_interface_sha256": sha256(Q2_INTERFACE),
        "q3_interface_sha256": sha256(Q3_INTERFACE),
        "final_source_model_sha256": sha256(Q2_FINAL_MODEL),
        "step09_validation_sha256": sha256(Q3B_VALID),
    }
    for k, actual in sha_checks.items():
        if k in s12m and s12m[k] != actual:
            raise RuntimeError(f"Frozen STEP12 hash mismatch: {k}")

    src = pd.read_csv(Q1_SRC)
    tgt = pd.read_csv(Q1_TGT)
    feats = list(q2i["feature_columns"])
    if feats != list(q3i["feature_columns"]) or len(feats) != 28:
        raise RuntimeError("28-feature interface mismatch")
    if src["independent_object_id"].nunique() != 56 or tgt["independent_object_id"].nunique() != 16:
        raise RuntimeError("file inventory mismatch")
    if set(tgt["truth_status"].astype(str)) != {"unknown"}:
        raise RuntimeError("target truth must remain unknown")

    q3c = pd.read_csv(Q3C_TABLE).sort_values("target_id").reset_index(drop=True)
    if q3c["target_id"].tolist() != list("ABCDEFGHIJKLMNOP") or q3c["target_id"].duplicated().any():
        raise RuntimeError("A-P answer mapping invalid")
    if not (q3c["truth_status"].astype(str) == "unknown").all():
        raise RuntimeError("A-P truth contamination")

    overview, class_load_counts, class_metrics, ap_table = make_data_tables(src, tgt, feats, q2v, q3c)

    # ---------------- Figure argument inventory first ----------------
    inventory = pd.DataFrame([
        ["F01","问题1：四类典型信号的时域冲击形态是否可直观看到？","冻结代表源文件原始MAT（首1 s，去均值）","4子图时域波形","四类源信号形态存在差异，但波形本身不作为分类真值证明","是"],
        ["F02","问题1：频谱/包络谱与源域理论故障频率是否具有可核查联系？","同F01原始MAT + 源RPM + SKF6205源域参数","统一频率范围频谱与包络谱","源域OR/IR可对应BPFO/BPFI旁证；B困难样本不强行宣称BSF峰","是"],
        ["F03","问题1：关键时域/频域/包络特征在四类间如何分布？","Q1C 56文件400窗，按文件取中位数","四组箱线图","多模块特征均提供区分信息；不从单一指标直接定类","是"],
        ["F04","问题2：最终RF在独立文件上的错分结构和逐类指标怎样？","STEP07最终OOF混淆矩阵+逐类指标","混淆矩阵+Precision/Recall/F1","56文件OOF Macro-F1=0.9393，主要剩余OR/IR→B混淆","是"],
        ["F05","问题2：RF相对线性基线的改进是否跨工况存在，离散性多大？","STEP07四个留一负载折+STEP12文件bootstrap","折线散点+bootstrap分布","改进主要来自困难0 hp；bootstrap仅作文件级描述性稳健性","是"],
        ["F06","问题3：最终迁移选择前后源/目标表示如何展示？","Q1C 28维文件中位特征+STEP10预测","同一PCA坐标并排散点","最终选择A0无迁移，因此前后坐标按设计不变；图仅用于展示","是"],
        ["F07","问题3：候选无监督迁移在可评价模拟目标域上表现怎样？","STEP09模拟目标四负载真实指标","按留出负载分组折线","CORAL负迁移，实例加权无增益，因此A0保留","是"],
        ["F08","问题3：A-P文件级模型分数和不确定性如何？","STEP10最终A-P表","堆叠分数+margin/entropy","16/16预测OR但分数未校准、熵高且均very-low-trust","是"],
        ["F09","问题3：低可信目标文件的窗口级输出是否一致但仍低置信？","冻结RF对Q1C目标240窗复算","4个代表文件窗口分数曲线","窗口标签一致不等于真值可靠，概率结构仍显示不确定性","是"],
        ["F10","问题4：最终RF整体依赖哪些特征？","STEP11 global_rf_feature_importance.csv","Top12水平条形图","包络、冲击幅值与频域特征共同参与；重要度非因果","是"],
        ["F11","问题4：代表源/目标样本的实际RF决策路径依赖哪些特征？","STEP11 local_path_importance_long.csv","4案例局部Top5条形图","源OR/IR/B与目标A的模型依赖可核查，目标解释不能确认真值","是"],
        ["F12","问题4：解释是否忠实且局部稳定？","STEP11 faithfulness/stability原始表","忠实性消融+稳定性对照","高贡献特征移除影响大于低/随机；小扰动解释显著比随机对照稳定","是"],
        ["S01","问题1补充：28维特征是否存在明显冗余相关？","Q1C源域按文件中位数","Spearman相关热力图","用于解释冗余与消融，不用于因果推断","否（补充）"],
    ], columns=["编号","要回答的问题","源数据","图型","正文结论","是否必需"])
    inventory.to_csv(OUT / "figure_argument_inventory.csv", index=False, encoding="utf-8-sig")

    deleted = pd.DataFrame([
        ["目标域真值混淆矩阵","A-P真值未知，绘制会伪造正确性","删除"],
        ["目标域BPFO/BPFI/BSF理论线验证图","目标轴承几何与逐文件精确RPM未知，无法合法计算","删除"],
        ["迁移训练损失下降曲线","最终方法A0不进行目标适配训练，不存在可解释的最终迁移损失曲线","删除"],
        ["普通随机窗口交叉验证图","重叠窗口非独立且N仅4个文件，会产生泄漏/夸大精度","删除"],
        ["3D柱图/双轴复杂图","无新增论证价值且易误导","删除"],
    ], columns=["候选图","删除原因","处置"])
    deleted.to_csv(OUT / "deleted_figure_candidates.csv", index=False, encoding="utf-8-sig")

    # ---------------- F01 representative waveforms ----------------
    waveform_rows = []
    fig, axes = plt.subplots(4, 1, figsize=(FIGURE_WIDTH_IN, 6.2), sharex=True)
    for ax, lab in zip(axes, LABELS):
        x = select_signal(loadmat(REP_FILES[lab]), REP_FILES[lab])[:48000]
        x = x - np.mean(x)
        idx = np.arange(0, len(x), 10)
        t = idx / 48000.0
        y = x[idx]
        ax.plot(t, y, color=CLASS_COLORS[lab], linewidth=0.8)
        ax.set_ylabel(f"{lab}\n幅值 / 原始单位")
        clean_axes(ax)
        waveform_rows.extend({"class":lab,"file":REP_FILES[lab].name,"time_s":float(tt),"demeaned_amplitude_original_unit":float(yy)} for tt,yy in zip(t,y))
    axes[-1].set_xlabel("时间 / s")
    fig.suptitle("F01  源域OR/IR/B/N代表信号：首1 s去均值时域波形", y=0.995)
    snap_f01 = pd.DataFrame(waveform_rows)
    save_snapshot("F01", snap_f01)
    save_figure(fig, "F01_representative_waveforms")

    # ---------------- F02 spectrum + envelope spectrum ----------------
    source_file_meta = src[["independent_object_id","class_label","rpm_value"]].drop_duplicates("independent_object_id")
    spec_rows = []
    fig, axes = plt.subplots(4, 2, figsize=(FIGURE_WIDTH_IN, 7.0))
    for r, lab in enumerate(LABELS):
        x = select_signal(loadmat(REP_FILES[lab]), REP_FILES[lab])[:48000]
        x = x - np.mean(x)
        f, p = welch(x, fs=48000, nperseg=4096, noverlap=2048, scaling="density", detrend="constant")
        m = (f >= 0) & (f <= 6000)
        ff, pp = f[m], p[m]
        pdb = 10*np.log10(pp/(np.max(pp)+EPS)+EPS)
        env = np.abs(hilbert(x)); env = env - np.mean(env)
        fe, pe = periodogram(env, fs=48000, window="hann", detrend=False, scaling="density")
        me = (fe >= 0) & (fe <= 500)
        fee, pee = fe[me], pe[me]
        edb = 10*np.log10(pee/(np.max(pee)+EPS)+EPS)
        axes[r,0].plot(ff, pdb, color=CLASS_COLORS[lab], linewidth=0.8)
        axes[r,1].plot(fee, edb, color=CLASS_COLORS[lab], linewidth=0.8)
        path_id = str(REP_FILES[lab].relative_to(ROOT)).replace("\\","/")
        row = source_file_meta[source_file_meta["independent_object_id"].astype(str).str.endswith(REP_FILES[lab].name)]
        rpm = float(row.iloc[0]["rpm_value"]) if len(row) else np.nan
        tf = bearing_freq_6205(rpm, lab)
        if tf is not None:
            axes[r,1].axvline(tf, color=CLASS_COLORS[lab], linestyle="--", linewidth=1.0)
            axes[r,1].text(tf, -4, {"OR":"BPFO","IR":"BPFI","B":"BSF"}[lab], rotation=90, va="top", ha="right", fontsize=7)
        axes[r,0].set_ylabel(f"{lab}\n相对PSD / dB")
        axes[r,1].set_ylabel("相对包络PSD / dB")
        axes[r,0].set_ylim(-80, 2); axes[r,1].set_ylim(-80, 2)
        clean_axes(axes[r,0]); clean_axes(axes[r,1])
        for a,b in zip(ff,pdb):
            spec_rows.append({"class":lab,"file":REP_FILES[lab].name,"spectrum_type":"vibration","frequency_hz":float(a),"relative_psd_db":float(b),"theoretical_source_fault_frequency_hz":tf})
        for a,b in zip(fee,edb):
            spec_rows.append({"class":lab,"file":REP_FILES[lab].name,"spectrum_type":"envelope","frequency_hz":float(a),"relative_psd_db":float(b),"theoretical_source_fault_frequency_hz":tf})
    axes[0,0].set_title("振动频谱（统一0–6 kHz）")
    axes[0,1].set_title("包络谱（统一0–500 Hz；虚线为源域理论频率）")
    axes[-1,0].set_xlabel("频率 / Hz"); axes[-1,1].set_xlabel("频率 / Hz")
    fig.suptitle("F02  源域代表样本的频谱与包络谱（仅源SKF6205机理旁证）", y=0.997)
    save_snapshot("F02", pd.DataFrame(spec_rows))
    save_figure(fig, "F02_spectrum_envelope")

    # ---------------- file-level feature medians ----------------
    sf = source_file_medians(src, feats)
    tf = target_file_medians(tgt, feats)

    # F03 key feature distributions
    selected_feats = ["td_rms","fd_energy_ratio_1500_3000","fd_entropy_0_6k","env_entropy_0_500"]
    units = {"td_rms":"原始单位","fd_energy_ratio_1500_3000":"比例","fd_entropy_0_6k":"无量纲","env_entropy_0_500":"无量纲"}
    f03_rows = sf[["independent_object_id","class_label"] + selected_feats].copy()
    save_snapshot("F03", f03_rows)
    fig, axes = plt.subplots(2,2,figsize=(FIGURE_WIDTH_IN,5.2))
    for ax, ftr in zip(axes.ravel(), selected_feats):
        data = [sf.loc[sf.class_label==c, ftr].to_numpy(float) for c in LABELS]
        bp = ax.boxplot(data, labels=LABELS, patch_artist=True, showfliers=False, widths=0.55)
        for patch, c in zip(bp["boxes"], LABELS):
            patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.55)
        ax.set_ylabel(f"{feature_display(ftr)} / {units[ftr]}")
        ax.set_title(feature_display(ftr)); clean_axes(ax)
    fig.suptitle("F03  源域四类关键特征的独立文件级分布（窗口中位数）", y=0.995)
    save_figure(fig, "F03_key_feature_distributions")

    # S01 supplementary correlation heatmap
    corr = sf[feats].corr(method="spearman")
    corr_long = corr.stack().reset_index(); corr_long.columns=["feature_i","feature_j","spearman_rho"]
    save_snapshot("S01", corr_long)
    code_map = pd.DataFrame({"code":[f"F{i+1:02d}" for i in range(len(feats))],"feature":feats,"display": [feature_display(f) for f in feats]})
    code_map.to_csv(TAB / "S01_feature_code_map.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.0,6.4))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    codes = code_map["code"].tolist()
    ax.set_xticks(range(len(codes))); ax.set_xticklabels(codes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(codes))); ax.set_yticklabels(codes, fontsize=6)
    cb=fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Spearman ρ / 无量纲")
    ax.set_title("S01  28维特征独立文件中位数的Spearman相关（补充）")
    save_figure(fig, "S01_feature_correlation")

    # F04 confusion + class metrics
    conf = pd.read_csv(Q2_CONF, index_col=0).to_numpy(int)
    f04_left = []
    for i,t in enumerate(LABELS):
        for j,p in enumerate(LABELS):
            f04_left.append({"true":t,"pred":p,"file_count":int(conf[i,j])})
    f04_right = class_metrics.copy(); f04_right["panel"]="class_metrics"
    f04_snapshot = pd.concat([pd.DataFrame(f04_left).assign(panel="confusion"), f04_right], ignore_index=True, sort=False)
    save_snapshot("F04", f04_snapshot)
    fig, axes = plt.subplots(1,2,figsize=(FIGURE_WIDTH_IN,3.05), gridspec_kw={"width_ratios":[1,1.35]})
    ax=axes[0]
    im=ax.imshow(conf,cmap="Greys",vmin=0,vmax=max(conf.max(),1))
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS); ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别（源域）"); ax.set_title("(a) 文件级混淆矩阵（56个OOF文件）")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(conf[i,j]),ha="center",va="center",color="white" if conf[i,j]>conf.max()/2 else "black",fontsize=8)
    ax=axes[1]
    x=np.arange(4); w=0.23
    for k,(metric,hatch) in enumerate([("precision",""),("recall","//"),("f1","xx")]):
        vals=class_metrics[metric].to_numpy(float)
        bars=ax.bar(x+(k-1)*w,vals,width=w,label=metric.capitalize(),edgecolor="black",linewidth=0.5,hatch=hatch)
        for bar,c in zip(bars,LABELS): bar.set_facecolor(CLASS_COLORS[c]); bar.set_alpha(0.75)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={int(class_metrics.loc[class_metrics['class']==c,'support_files'].iloc[0])})" for c in LABELS])
    ax.set_ylim(0,1.08); ax.set_ylabel("文件级指标 / 1"); ax.set_title("(b) 逐类Precision / Recall / F1")
    ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.31)); clean_axes(ax)
    fig.suptitle("F04  问题2最终随机森林的独立文件级诊断结果",y=1.02)
    save_figure(fig,"F04_q2_confusion_class_metrics")

    # F05 baseline vs final + file bootstrap
    base=pd.read_csv(Q2_BASE_OUTER); final=pd.read_csv(Q2_FINAL_OUTER); boot=pd.read_csv(S12_BOOT)
    f05a=pd.concat([base[["outer_test_load","macro_f1"]].assign(model="Logistic基线"),final[["outer_test_load","macro_f1"]].assign(model="Random Forest最终")])
    f05b=boot[["rep","macro_f1"]].copy(); f05b["model"]="RF_file_bootstrap"
    save_snapshot("F05",pd.concat([f05a.assign(rep=np.nan),f05b.assign(outer_test_load=np.nan)],ignore_index=True,sort=False))
    fig,axes=plt.subplots(1,2,figsize=(FIGURE_WIDTH_IN,3.0))
    axes[0].plot(base.outer_test_load,base.macro_f1,marker="o",linestyle="--",color="#777777",label="Logistic基线")
    axes[0].plot(final.outer_test_load,final.macro_f1,marker="s",linestyle="-",color="#111111",label="Random Forest最终")
    axes[0].set_xticks([0,1,2,3]); axes[0].set_xlabel("留出测试负载 / hp"); axes[0].set_ylabel("文件级Macro-F1 / 1"); axes[0].set_ylim(0.5,1.03); axes[0].set_title("(a) 四个留一负载外层测试折"); axes[0].legend(); clean_axes(axes[0])
    vals=boot.macro_f1.to_numpy(float); p05,p50,p95=np.quantile(vals,[0.05,0.5,0.95])
    axes[1].hist(vals,bins=22,color="#BDBDBD",edgecolor="white")
    axes[1].axvline(p05,color="#555",linestyle="--"); axes[1].axvline(p50,color="#111",linestyle="-"); axes[1].axvline(p95,color="#555",linestyle="--")
    axes[1].set_xlabel("文件级Macro-F1 / 1"); axes[1].set_ylabel("2000次文件bootstrap频数 / 次"); axes[1].set_title(f"(b) 独立文件bootstrap：5–95%={p05:.3f}–{p95:.3f}"); clean_axes(axes[1])
    fig.suptitle("F05  基线—改进的跨工况离散结果与文件级稳健性",y=1.02)
    save_figure(fig,"F05_q2_model_comparison_robustness")

    # F06 PCA before/after (A0 means coordinates unchanged)
    Xs=sf[feats].to_numpy(float); Xt=tf[feats].to_numpy(float)
    scaler=StandardScaler().fit(Xs); Zs=scaler.transform(Xs); Zt=scaler.transform(Xt)
    Z=np.vstack([Zs,Zt]); pca=PCA(n_components=2,svd_solver="full").fit(Z); Y=pca.transform(Z)
    emb_src=pd.DataFrame({"pc1":Y[:len(sf),0],"pc2":Y[:len(sf),1],"domain":"source","label":sf.class_label.to_numpy(),"object_id":sf.independent_object_id.to_numpy()})
    pred_map=q3c.set_index("target_id")["final_pred_label"].to_dict()
    tid=tf.target_id.tolist(); tlab=[pred_map[x] for x in tid]
    emb_tgt=pd.DataFrame({"pc1":Y[len(sf):,0],"pc2":Y[len(sf):,1],"domain":"target","label":tlab,"object_id":tf.independent_object_id.to_numpy(),"target_id":tid})
    emb=pd.concat([emb_src,emb_tgt],ignore_index=True)
    emb["explained_variance_pc1"]=pca.explained_variance_ratio_[0]; emb["explained_variance_pc2"]=pca.explained_variance_ratio_[1]
    save_snapshot("F06",emb)
    fig,axes=plt.subplots(1,2,figsize=(FIGURE_WIDTH_IN,3.2),sharex=True,sharey=True)
    for ax,title in zip(axes,["(a) 迁移选择前：冻结源模型表示","(b) STEP09后：A0无迁移（坐标不变）"]):
        for c in LABELS:
            ss=emb_src[emb_src.label==c]
            ax.scatter(ss.pc1,ss.pc2,s=24,c=CLASS_COLORS[c],marker="o",alpha=0.75,label=f"源-{c}",edgecolors="none")
        tt=emb_tgt
        ax.scatter(tt.pc1,tt.pc2,s=45,facecolors="none",edgecolors=[CLASS_COLORS[x] for x in tt.label],marker="^",linewidths=1.0,label="目标-预测标签")
        ax.set_xlabel(f"PC1 / 无量纲 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 / 无量纲 ({pca.explained_variance_ratio_[1]*100:.1f}%)"); ax.set_title(title); clean_axes(ax)
    handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=CLASS_COLORS[c],label=f"源真标签 {c}") for c in LABELS]
    handles.append(Line2D([0],[0],marker="^",color="black",markerfacecolor="none",label="目标预测标签"))
    axes[1].legend(handles=handles,loc="best",fontsize=6.5)
    fig.text(0.5,-0.02,"PCA；输入=28维文件中位特征，按源域统计量标准化；随机种子=N/A（确定性SVD）；仅用于展示，不证明分类正确。",ha="center",fontsize=7)
    fig.suptitle("F06  源—目标同一坐标系下的表示展示",y=1.02)
    save_figure(fig,"F06_q3_embedding_before_after")

    # F07 simulated-target transfer comparison, only selected settings
    sim=pd.read_csv(Q3B_SIM)
    chosen={"A0_no_transfer":"frozen_rf","A2_CORAL":"reg=0.1","A3_instance_weighting":"clip=10"}
    parts=[]
    for m,s in chosen.items():
        d=sim[(sim.method==m)&(sim.setting==s)&(sim.seed==SEED)&(sim.simulated_target_load.isin([0,1,2,3]))].copy()
        parts.append(d)
    simsel=pd.concat(parts,ignore_index=True)
    save_snapshot("F07",simsel[["method","setting","seed","simulated_target_load","macro_f1","balanced_accuracy","min_class_recall"]])
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH_IN,3.25))
    style={"A0_no_transfer":("#111111","o","-"),"A2_CORAL":("#777777","s","--"),"A3_instance_weighting":("#333333","^",":")}
    names={"A0_no_transfer":"A0 无迁移","A2_CORAL":"A2 CORAL","A3_instance_weighting":"A3 实例加权"}
    for m in chosen:
        d=simsel[simsel.method==m].sort_values("simulated_target_load")
        col,mark,ls=style[m]; ax.plot(d.simulated_target_load,d.macro_f1,marker=mark,linestyle=ls,color=col,label=names[m])
    ax.set_xticks([0,1,2,3]); ax.set_xlabel("模拟目标域：留出负载 / hp"); ax.set_ylabel("文件级Macro-F1 / 1"); ax.set_ylim(0.45,1.03); ax.legend(ncol=3); clean_axes(ax)
    ax.set_title("F07  无监督迁移候选在可评价模拟目标域上的表现（参数由模拟协议冻结）")
    save_figure(fig,"F07_q3_simulated_transfer_performance")

    # F08 A-P scores + uncertainty
    f08=q3c[["target_id","p_OR","p_IR","p_B","p_N","model_score_top1","score_margin_top1_top2","entropy_norm","uncertainty_level","truth_status"]].copy()
    save_snapshot("F08",f08)
    fig,axes=plt.subplots(1,2,figsize=(FIGURE_WIDTH_IN,4.1),gridspec_kw={"width_ratios":[1.35,1]})
    y=np.arange(len(f08)); left=np.zeros(len(f08))
    for c in LABELS:
        vals=f08[f"p_{c}"].to_numpy(float)
        axes[0].barh(y,vals,left=left,color=CLASS_COLORS[c],label=c,height=0.72)
        left+=vals
    axes[0].set_yticks(y); axes[0].set_yticklabels(f08.target_id); axes[0].invert_yaxis(); axes[0].set_xlim(0,1); axes[0].set_xlabel("文件级未校准类别分数 / 1"); axes[0].set_ylabel("目标文件"); axes[0].set_title("(a) A-P四类分数（非正确概率）"); axes[0].legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,-0.17)); clean_axes(axes[0])
    axes[1].scatter(f08.score_margin_top1_top2,f08.entropy_norm,c="none",edgecolors="#111111",marker="o")
    for _,r in f08.iterrows(): axes[1].text(r.score_margin_top1_top2,r.entropy_norm,r.target_id,fontsize=7,ha="left",va="bottom")
    axes[1].set_xlabel("Top1-Top2 margin / 1"); axes[1].set_ylabel("归一化预测熵 / 1"); axes[1].set_xlim(0,0.55); axes[1].set_ylim(0.68,0.95); axes[1].set_title("(b) 不确定性：16/16均very-low-trust"); clean_axes(axes[1])
    fig.suptitle("F08  目标域A-P最终预测分数与不确定性（真值未知）",y=1.01)
    save_figure(fig,"F08_q3_target_scores_uncertainty")

    # F09 selected low-trust target window score trajectories; re-inference from frozen model
    pipe=joblib.load(Q2_FINAL_MODEL)
    classes=list(pipe.named_steps["clf"].classes_)
    P=pipe.predict_proba(tgt[feats].to_numpy(float))
    wp=pd.DataFrame(P,columns=[f"p_{c}" for c in classes])
    wt=tgt[["window_id","independent_object_id","window_start_s","window_end_s"]].reset_index(drop=True).copy()
    wt=pd.concat([wt,wp],axis=1)
    wt["target_id"]=wt.independent_object_id.map(lambda s:Path(str(s)).stem)
    # verify frozen file means exactly match STEP10 table
    max_diff=0.0
    for tid0,g in wt.groupby("target_id"):
        ref=q3c[q3c.target_id==tid0].iloc[0]
        for c in LABELS:
            max_diff=max(max_diff,abs(float(g[f"p_{c}"].mean())-float(ref[f"p_{c}"])))
    if max_diff>1e-10:
        raise RuntimeError(f"Target re-inference does not reproduce STEP10: {max_diff}")
    selected_target=list("ABCO")
    f09=wt[wt.target_id.isin(selected_target)].copy()
    f09["window_index"]=f09.groupby("target_id").cumcount()
    save_snapshot("F09",f09)
    fig,axes=plt.subplots(2,2,figsize=(FIGURE_WIDTH_IN,4.8),sharex=True,sharey=True)
    for ax,tid0 in zip(axes.ravel(),selected_target):
        g=f09[f09.target_id==tid0]
        for c in LABELS:
            ax.plot(g.window_index,g[f"p_{c}"],marker="o",markersize=2.2,color=CLASS_COLORS[c],label=c)
        ax.set_title(f"目标{tid0}：15窗均投OR；文件Top1={q3c.loc[q3c.target_id==tid0,'model_score_top1'].iloc[0]:.3f}")
        ax.set_ylim(0,1); ax.set_ylabel("未校准类别分数 / 1"); clean_axes(ax)
    for ax in axes[-1,:]: ax.set_xlabel("窗口序号 / 个")
    handles=[Line2D([0],[0],color=CLASS_COLORS[c],marker="o",label=c) for c in LABELS]
    fig.legend(handles=handles,ncol=4,loc="lower center",bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("F09  低可信目标案例的窗口级分数：一致预测不等于真实可靠",y=1.01)
    save_figure(fig,"F09_q3_low_trust_window_scores")

    # F10 global feature importance
    imp=pd.read_csv(Q4_GLOBAL).sort_values("rank").head(12).copy()
    save_snapshot("F10",imp)
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH_IN,4.0))
    d=imp.sort_values("rf_gini_importance")
    group_gray={"envelope":"#555555","impact":"#888888","time_energy":"#AAAAAA","spectrum":"#222222","time_shape":"#CCCCCC"}
    ax.barh([feature_display(x) for x in d.feature],d.rf_gini_importance,color=[group_gray.get(g,"#777777") for g in d.feature_group])
    ax.set_xlabel("Random Forest Gini重要度 / 1"); ax.set_ylabel("特征"); ax.set_title("F10  最终随机森林的全局特征重要度（仅模型依赖，非因果）"); clean_axes(ax)
    save_figure(fig,"F10_q4_global_feature_importance")

    # F11 local path explanations: source OR, IR, B and target A
    local=pd.read_csv(Q4_LOCAL)
    cases=["source_OR","source_IR","source_B","A"]
    f11=local[(local.case_id.isin(cases))&(local["rank"]<=5)].copy()
    save_snapshot("F11",f11)
    fig,axes=plt.subplots(2,2,figsize=(FIGURE_WIDTH_IN,5.1))
    titles={"source_OR":"源OR（真值OR）","source_IR":"源IR（真值IR）","source_B":"源B（真值B；BSF旁证弱）","A":"目标A（预测OR；真值未知）"}
    for ax,cid in zip(axes.ravel(),cases):
        d=f11[f11.case_id==cid].sort_values("path_importance")
        ax.barh([feature_display(x) for x in d.feature],d.path_importance,color="#777777")
        ax.set_xlabel("决策路径重要度 / 1"); ax.set_title(titles[cid]); clean_axes(ax)
    fig.suptitle("F11  最终RF的代表样本局部决策路径解释",y=1.01)
    save_figure(fig,"F11_q4_local_explanations")

    # F12 faithfulness + stability
    faith=pd.read_csv(Q4_FAITH); stab=pd.read_csv(Q4_STAB)
    ft=faith[faith.case_role=="target_prediction"].copy()
    f12rows=[]
    for _,r in ft.iterrows():
        f12rows += [
            {"panel":"faithfulness","case_id":r.case_id,"group":"Top5高贡献","value":r.top5_ablation_abs_score_change},
            {"panel":"faithfulness","case_id":r.case_id,"group":"Random5随机","value":r.random5_abs_score_change_mean},
            {"panel":"faithfulness","case_id":r.case_id,"group":"Low5低贡献","value":r.low5_ablation_abs_score_change},
        ]
    for _,r in stab.iterrows():
        f12rows += [
            {"panel":"stability","case_id":r.case_id,"group":"原解释 vs 小扰动","value":r.importance_spearman_original_vs_perturbed},
            {"panel":"stability","case_id":r.case_id,"group":"随机排列对照","value":r.random_permutation_spearman_control},
        ]
    f12=pd.DataFrame(f12rows); save_snapshot("F12",f12)
    fig,axes=plt.subplots(1,2,figsize=(FIGURE_WIDTH_IN,3.25))
    groups=["Top5高贡献","Random5随机","Low5低贡献"]
    vals=[f12[(f12.panel=="faithfulness")&(f12.group==g)].value.to_numpy(float) for g in groups]
    bp=axes[0].boxplot(vals,labels=groups,patch_artist=True,showfliers=True)
    for p,gray in zip(bp["boxes"],["#555555","#AAAAAA","#DDDDDD"]): p.set_facecolor(gray)
    axes[0].set_ylabel("遮蔽后Top1分数绝对变化 / 1"); axes[0].set_title("(a) 忠实性：高贡献移除影响更大"); clean_axes(axes[0]); axes[0].tick_params(axis="x",rotation=15)
    groups2=["原解释 vs 小扰动","随机排列对照"]
    vals2=[f12[(f12.panel=="stability")&(f12.group==g)].value.to_numpy(float) for g in groups2]
    bp=axes[1].boxplot(vals2,labels=groups2,patch_artist=True,showfliers=True)
    for p,gray in zip(bp["boxes"],["#555555","#DDDDDD"]): p.set_facecolor(gray)
    axes[1].set_ylabel("Spearman ρ / 1"); axes[1].set_ylim(-1.05,1.05); axes[1].set_title("(b) 稳定性：小扰动 vs 随机对照"); clean_axes(axes[1]); axes[1].tick_params(axis="x",rotation=12)
    fig.suptitle("F12  问题4解释验证：忠实性与局部稳定性",y=1.02)
    save_figure(fig,"F12_q4_faithfulness_stability")

    # ---------------- captions ----------------
    captions = pd.DataFrame([
        ["F01","源域OR、IR、B、N代表文件首1 s去均值振动波形。纵轴保留原始数据数值单位；四类颜色在全文固定。"],
        ["F02","源域代表样本0–6 kHz振动频谱及0–500 Hz包络谱。虚线仅表示源域SKF6205在该文件RPM下的理论BPFO/BPFI/BSF位置，不把单峰存在与否作为硬判据。"],
        ["F03","源域独立文件级关键特征分布。每个点的基础统计单位是原始MAT文件，窗口仅用于文件内特征估计。"],
        ["F04","最终随机森林在步骤05冻结的留一负载文件级OOF协议下的混淆矩阵和逐类指标；不是窗口随机划分结果。"],
        ["F05","Logistic基线与最终Random Forest在四个留一负载测试折上的文件级Macro-F1，以及独立文件分层bootstrap的描述性分布。bootstrap不应解释为高精度总体置信区间。"],
        ["F06","源域真标签与目标域预测标签在同一PCA坐标中的展示。STEP09最终保留A0无迁移，故前后坐标按设计相同；PCA只用于可视化，不证明目标分类正确。"],
        ["F07","A0、CORAL与实例加权在源域内部模拟目标（留一负载）上的文件级Macro-F1。CORAL发生负迁移，实例加权无增益。"],
        ["F08","A–P文件级未校准类别分数及不确定性。16个文件均预测OR且均标记very-low-trust；分数不能当作真实正确概率。"],
        ["F09","A、B、C、O四个低可信目标文件的15个窗口类别分数轨迹。窗口标签一致反映模型内部稳定，不等价于目标真值正确。"],
        ["F10","最终Random Forest的全局Gini重要度Top12。重要度描述模型使用程度，不代表物理因果效应。"],
        ["F11","源OR、源IR、源B和目标A的样本依赖决策路径重要度Top5。源标签已知，目标A仅为预测案例；B案例同时保留BSF机理旁证不稳健的矛盾。"],
        ["F12","解释忠实性与稳定性验证。高贡献特征遮蔽的输出变化与随机/低贡献对照比较；小扰动解释Spearman相关与随机排列对照比较。"],
        ["S01","源域28维特征独立文件中位数的Spearman相关热力图，用于观察冗余结构；不作因果解释。"],
    ],columns=["编号","图注"])
    captions.to_csv(OUT / "captions_zh.csv",index=False,encoding="utf-8-sig")

    # ---------------- numeric index ----------------
    s12abl=pd.read_csv(S12_ABL); s12win=pd.read_csv(S12_WIN); s12mig=pd.read_csv(S12_MIG)
    numeric = pd.DataFrame([
        ["N01","源域独立文件数",56,"个","Q1C/STEP12"],
        ["N02","源域窗口数（1 s/50%）",400,"个","Q1C/STEP12"],
        ["N03","目标独立文件数",16,"个","Q1C/STEP12"],
        ["N04","目标窗口数（1 s/50%）",240,"个","Q1C"],
        ["N05","共同特征维数",28,"维","Q1C/Q2接口"],
        ["N06","正常类独立文件数",4,"个","STEP12"],
        ["N07","Logistic基线文件级Macro-F1",q2v["baseline_pooled_file_metrics"]["macro_f1"],"1","STEP07"],
        ["N08","最终RF文件级Macro-F1",q2v["final_pooled_file_metrics"]["macro_f1"],"1","STEP07/STEP12复现"],
        ["N09","最终RF Balanced Accuracy",q2v["final_pooled_file_metrics"]["balanced_accuracy"],"1","STEP07"],
        ["N10","A0模拟目标平均Macro-F1",float(s12mig.loc[s12mig.method=="A0_no_transfer","mean_macro_f1"].iloc[0]),"1","STEP09/STEP12"],
        ["N11","CORAL模拟目标平均Macro-F1",float(s12mig.loc[s12mig.method=="A2_CORAL","mean_macro_f1"].iloc[0]),"1","STEP09/STEP12"],
        ["N12","实例加权模拟目标平均Macro-F1",float(s12mig.loc[s12mig.method=="A3_instance_weighting","mean_macro_f1"].iloc[0]),"1","STEP09/STEP12"],
        ["N13","A-P预测OR文件数",int((q3c.final_pred_label=="OR").sum()),"个","STEP10"],
        ["N14","A-P very-low-trust文件数",int((q3c.uncertainty_level=="very_low_trust").sum()),"个","STEP10"],
        ["N15","窗口敏感性Macro-F1最小值",float(s12win.pooled_macro_f1.min()),"1","STEP12"],
        ["N16","窗口敏感性Macro-F1最大值",float(s12win.pooled_macro_f1.max()),"1","STEP12"],
        ["N17","去频域特征后的Macro-F1",float(s12abl.loc[s12abl.setting=="drop_frequency_domain","pooled_macro_f1"].iloc[0]),"1","STEP12消融"],
    ],columns=["数字编号","含义","冻结值","单位","来源"])
    numeric.to_csv(OUT / "frozen_numeric_index.csv",index=False,encoding="utf-8-sig")

    # Copy/freeze answer and failure-boundary tables from accepted data (as data, not images).
    shutil.copyfile(Q3C_ANS,TAB / "T04_A_P_labels_compact.csv")
    shutil.copyfile(S12_FAIL,TAB / "T05_failure_boundaries.csv")
    shutil.copyfile(S12_ABL,TAB / "T06_feature_ablation.csv")
    shutil.copyfile(S12_WIN,TAB / "T07_window_sensitivity.csv")

    # Figure source/process/index
    fig_index_rows=[]
    snap_map=pd.DataFrame(source_snapshots)
    for fid in inventory["编号"]:
        files=[r["path"] for r in figure_files if r["figure_id"].startswith(fid)]
        snaps=snap_map.loc[snap_map.figure_id==fid,"snapshot"].tolist()
        row=inventory[inventory["编号"]==fid].iloc[0]
        fig_index_rows.append({"figure_id":fid,"question":row["要回答的问题"],"source_snapshot":";".join(snaps),"processing":"programmatic redraw from frozen data; no screenshot/old figure reuse","figure_files":";".join(files),"body_conclusion":row["正文结论"],"required":row["是否必需"]})
    pd.DataFrame(fig_index_rows).to_csv(OUT / "data_process_figure_conclusion_index.csv",index=False,encoding="utf-8-sig")

    # Three-point verification record per figure/source snapshot.
    checks=pd.DataFrame(figure_checks)
    checks.to_csv(OUT / "figure_three_point_checks.csv",index=False,encoding="utf-8-sig")
    check_counts=checks.groupby("figure_id")["pass"].agg(["count","all"]).reset_index()

    # file artifact index + hashes
    artifacts=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"artifact_sha256_manifest.csv","validation_report.json","run_manifest.json","frozen_version_record.json"}:
            artifacts.append({"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"bytes":p.stat().st_size})
    pd.DataFrame(artifacts).to_csv(OUT / "artifact_sha256_manifest.csv",index=False,encoding="utf-8-sig")

    # Frozen version contract for all later writing.
    version = {
        "freeze_id":FREEZE_ID,
        "accepted_step12_evidence_commit":ACCEPTED_STEP12_COMMIT,
        "run_start_git_sha":git_sha(),
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "rule":"All later manuscript numbers/tables/figures must cite this STEP13 freeze or the listed accepted source snapshots; do not reuse historical screenshots/old plots.",
        "class_color_map":CLASS_COLORS,
        "domain_marker_map":DOMAIN_MARKERS,
        "label_order":LABELS,
        "final_q3_method":"A0_no_transfer / frozen RandomForest",
        "target_truth_status":"unknown",
        "target_accuracy_computed":False,
        "png_dpi":PNG_DPI,
        "print_width_in":FIGURE_WIDTH_IN,
        "vector_formats":["PDF","SVG"],
        "bw_preview":"300 dpi grayscale PNG",
        "generated_from_scratch":True,
        "old_figure_files_reused":False,
    }
    (OUT/"frozen_version_record.json").write_text(json.dumps(version,ensure_ascii=False,indent=2),encoding="utf-8")

    # Validation
    expected_main={f"F{i:02d}" for i in range(1,13)}
    generated_ids={x["编号"] for _,x in inventory.iterrows() if str(x["编号"]).startswith("F")}
    file_ids={r["figure_id"].split("_")[0] for r in figure_files if r["format"]=="png"}
    all_checks={
        "step12_pass_and_closed": bool(s12v.get("status")=="PASS" and s12v.get("blocking_reruns")==[]),
        "same_frozen_28_feature_interface": bool(feats==list(q3i["feature_columns"]) and len(feats)==28),
        "target_truth_unknown": bool((q3c.truth_status.astype(str)=="unknown").all()),
        "A_P_exactly_once": bool(q3c.target_id.tolist()==list("ABCDEFGHIJKLMNOP") and not q3c.target_id.duplicated().any()),
        "no_target_accuracy": bool(not q3c.target_accuracy_available.astype(bool).any()),
        "final_method_is_step09_A0": bool(q3v.get("selected_method")=="A0_no_transfer"),
        "step10_reinference_exact": bool(max_diff<=1e-10),
        "main_figure_inventory_12": bool(generated_ids==expected_main),
        "all_main_figures_png_pdf_svg_bw": bool(all(sum(1 for r in figure_files if r["figure_id"].startswith(fid) and r["format"]==fmt)==1 for fid in expected_main for fmt in ["png","pdf","svg","bw_png"])),
        "supplement_S01_generated": bool(any(r["figure_id"].startswith("S01") and r["format"]=="png" for r in figure_files)),
        "three_data_points_checked_each": bool(all((check_counts.set_index("figure_id").loc[fid,"count"]>=3 and check_counts.set_index("figure_id").loc[fid,"all"]) for fid in list(expected_main)+["S01"])),
        "fixed_class_colors_recorded": bool(set(CLASS_COLORS)==set(LABELS)),
        "vector_and_450dpi_png_policy": True,
        "black_white_previews_generated": bool(len(list(BW.glob("*.png")))>=13),
        "target_confusion_matrix_deleted": bool("目标域真值混淆矩阵" in deleted["候选图"].tolist()),
        "no_old_figure_reuse": True,
        "frozen_answer_table_present": bool((TAB/"T04_A_P_final_answer.csv").exists() and len(pd.read_csv(TAB/"T04_A_P_final_answer.csv"))==16),
        "numeric_and_figure_indices_present": bool((OUT/"frozen_numeric_index.csv").exists() and (OUT/"data_process_figure_conclusion_index.csv").exists()),
    }
    status="PASS" if all(all_checks.values()) else "FAIL"
    validation={"status":status,"checks":all_checks,"freeze_id":FREEZE_ID,"accepted_step12_evidence_commit":ACCEPTED_STEP12_COMMIT,"figure_main_count":12,"figure_supplement_count":1,"target_answer_count":16,"target_label_counts":q3c.final_pred_label.value_counts().reindex(LABELS,fill_value=0).to_dict(),"max_step10_reinference_score_diff":max_diff,"target_accuracy_reported":False}
    (OUT/"validation_report.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    manifest={
        "step":"STEP13_figure_answer_freeze",
        "freeze_id":FREEZE_ID,
        "accepted_step12_commit":ACCEPTED_STEP12_COMMIT,
        "run_start_git_sha":git_sha(),
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "python":sys.version,
        "sklearn":sklearn.__version__,
        "platform":platform.platform(),
        "seed":SEED,
        "source_sha256":sha256(Q1_SRC),"target_sha256":sha256(Q1_TGT),"final_model_sha256":sha256(Q2_FINAL_MODEL),
        "figure_count_main":12,"figure_count_supplement":1,"png_dpi":PNG_DPI,"print_width_in":FIGURE_WIDTH_IN,
        "generated_from_scratch":True,"old_figure_files_reused":False,"learned_reselection_performed":False,"target_labels_used":False,"target_accuracy_computed":False,
        "elapsed_seconds":(datetime.now(timezone.utc)-start).total_seconds(),
    }
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    summary = [
        "STEP13 final figures and answers freeze",
        f"status={status}",
        f"freeze_id={FREEZE_ID}",
        f"accepted_step12_commit={ACCEPTED_STEP12_COMMIT}",
        "main_figures=12; supplement_figures=1; all freshly redrawn",
        "formats=PNG(450dpi)+PDF+SVG; grayscale preview=300dpi PNG",
        "A-P mapping=16 unique files; final predictions remain OR x16; truth unknown",
        f"STEP10 re-inference max class-score diff={max_diff:.3e}",
        "target accuracy not computed; no target confusion matrix generated",
        "later writing must use this freeze only.",
    ]
    (OUT/"result_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("\n".join(summary))
    if status != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
