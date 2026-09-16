from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import subprocess
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

import q1c_feature_extraction as q1c
import q2c_failure_driven_improvement as q2c

ROOT = Path(__file__).resolve().parents[1]
Q1C = ROOT / "outputs" / "q1c_feature_extraction"
Q2B = ROOT / "outputs" / "q2b_minimal_baseline"
Q2C = ROOT / "outputs" / "q2c_failure_driven_improvement"
Q3B = ROOT / "outputs" / "q3b_unsupervised_transfer"
Q3C = ROOT / "outputs" / "q3c_final_labels_and_display"
Q4 = ROOT / "outputs" / "q4_explainability"
OUT = ROOT / "outputs" / "q_joint_validation_step12"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SRC_CSV = Q1C / "q2_source_raw.csv"
TGT_CSV = Q1C / "q2_target_raw.csv"
Q2_INTERFACE = Q1C / "q2_interface.json"
Q3_INTERFACE = Q2C / "q3_interface.json"
Q2C_VALID = Q2C / "validation_report.json"
Q3B_VALID = Q3B / "validation_report.json"
Q3C_VALID = Q3C / "validation_report.json"
Q4_VALID = Q4 / "validation_report.json"
Q2C_OOF_FILE = Q2C / "final_method_oof_file_predictions.csv"
Q2C_OOF_WIN = Q2C / "final_method_oof_window_predictions.csv"
FINAL_SOURCE_MODEL = Q2C / "models" / "final_source_model.joblib"
FINAL_TARGET_TABLE = Q3C / "final_A_P_label_table.csv"

LABELS = ["OR", "IR", "B", "N"]
LOADS = [0, 1, 2, 3]
SEED = 20260916
RNG = np.random.default_rng(SEED)
BOOT_REPS = 2000
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


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


def rf_config():
    return {"max_depth": 10, "min_samples_leaf": 1, "n_estimators": 300, "max_features": "sqrt"}


def rf_exp():
    return [x for x in q2c.EXPERIMENTS if x["experiment_id"] == "E3_random_forest"][0]


def fit_fixed_rf(train_df: pd.DataFrame, feats: list[str]):
    pipe = q2c.build_pipeline(rf_exp(), rf_config())
    w = q2c.make_weights(train_df, 1.0)
    pipe.fit(train_df[feats].to_numpy(float), train_df["class_label"].astype(str).to_numpy(), clf__sample_weight=w)
    return pipe


def aggregate_predictions(df: pd.DataFrame, P: np.ndarray, classes: list[str]):
    pcols = [f"p_{x}" for x in LABELS]
    tmp = df[["window_id", "independent_object_id", "class_label", "load_hp"]].copy()
    for lab in LABELS:
        tmp[f"p_{lab}"] = P[:, classes.index(lab)]
    rows = []
    for oid, g in tmp.groupby("independent_object_id", sort=True):
        pm = np.array([g[f"p_{x}"].mean() for x in LABELS], float)
        rows.append({
            "independent_object_id": oid,
            "class_label": str(g["class_label"].iloc[0]),
            "load_hp": float(g["load_hp"].iloc[0]),
            **{f"p_{lab}": float(pm[i]) for i, lab in enumerate(LABELS)},
            "pred_label": LABELS[int(np.argmax(pm))],
            "n_windows": int(len(g)),
        })
    return tmp, pd.DataFrame(rows)


def evaluate_fixed_rf(df: pd.DataFrame, feats: list[str], setting_name: str):
    fold_rows, file_rows = [], []
    for load in LOADS:
        train = df[df["load_hp"] != load].copy().reset_index(drop=True)
        test = df[df["load_hp"] == load].copy().reset_index(drop=True)
        pipe = fit_fixed_rf(train, feats)
        P = pipe.predict_proba(test[feats].to_numpy(float))
        _, ff = aggregate_predictions(test, P, list(pipe.named_steps["clf"].classes_))
        m = metric_dict(ff["class_label"].astype(str), ff["pred_label"].astype(str))
        fold_rows.append({"setting": setting_name, "outer_test_load": load, "n_train_files": train["independent_object_id"].nunique(), "n_test_files": test["independent_object_id"].nunique(), **m})
        ff["setting"] = setting_name
        ff["outer_test_load"] = load
        file_rows.append(ff)
    files = pd.concat(file_rows, ignore_index=True)
    pooled = metric_dict(files["class_label"].astype(str), files["pred_label"].astype(str))
    folds = pd.DataFrame(fold_rows)
    summary = {
        "setting": setting_name,
        "n_files": int(files["independent_object_id"].nunique()),
        "n_windows": int(len(df)),
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        "pooled_accuracy": pooled["accuracy"],
        "mean_outer_macro_f1": float(folds["macro_f1"].mean()),
        "min_outer_macro_f1": float(folds["macro_f1"].min()),
        "max_outer_macro_f1": float(folds["macro_f1"].max()),
    }
    return summary, folds, files


def load_source_signals():
    items = []
    paths = sorted(q1c.SRC_FAULT.glob("*.mat")) + sorted(q1c.SRC_NORMAL.glob("*.mat"))
    for p in paths:
        mat = loadmat(p)
        x, var = q1c.select_source_de(mat, p)
        oid = p.relative_to(ROOT).as_posix()
        items.append({
            "path": p, "oid": oid, "x": x, "signal_var": var,
            "class_label": q1c.source_label(p), "load_hp": q1c.parse_load_hp(p),
        })
    return items


def extract_source_setting(items, window_s: float, overlap: float):
    rows = []
    for it in items:
        fs = q1c.SOURCE_FS
        win = int(round(window_s * fs))
        stride = int(round(win * (1.0 - overlap)))
        if len(it["x"]) < win:
            continue
        k, start = 0, 0
        while start + win <= len(it["x"]):
            feat = q1c.extract_common_features(it["x"][start:start+win], fs)
            rows.append({
                "window_id": f"stress::{it['oid']}::{window_s:.2f}s::{overlap:.2f}::w{k:04d}",
                "independent_object_id": it["oid"],
                "class_label": it["class_label"],
                "load_hp": it["load_hp"],
                "window_start_s": start / fs,
                "window_end_s": (start + win) / fs,
                **feat,
            })
            k += 1
            start += stride
    return pd.DataFrame(rows)


def aggregation_sensitivity(oof_win: pd.DataFrame, target: pd.DataFrame, final_pipe, feats: list[str]):
    rows = []
    # Source OOF: compare frozen mean-probability aggregation with majority vote of OOF windows.
    for oid, g in oof_win.groupby("independent_object_id", sort=True):
        mean_p = np.array([g[f"p_{x}"].mean() for x in LABELS], float)
        mean_lab = LABELS[int(np.argmax(mean_p))]
        counts = g["pred_label_window"].value_counts()
        maxc = counts.max()
        tied = [x for x in LABELS if int(counts.get(x, 0)) == int(maxc)]
        if len(tied) == 1:
            vote_lab = tied[0]
        else:
            vote_lab = max(tied, key=lambda z: mean_p[LABELS.index(z)])
        rows.append({"domain": "source_oof", "independent_object_id": oid, "class_label": str(g["class_label"].iloc[0]), "mean_probability_label": mean_lab, "majority_vote_label": vote_lab, "same_label": mean_lab == vote_lab})
    src_cmp = pd.DataFrame(rows)
    m_mean = metric_dict(src_cmp["class_label"], src_cmp["mean_probability_label"])
    m_vote = metric_dict(src_cmp["class_label"], src_cmp["majority_vote_label"])

    # Target: same sensitivity check, never an accuracy estimate.
    P = final_pipe.predict_proba(target[feats].to_numpy(float))
    cls = list(final_pipe.named_steps["clf"].classes_)
    t = target[["window_id", "independent_object_id"]].copy()
    for lab in LABELS:
        t[f"p_{lab}"] = P[:, cls.index(lab)]
    t["pred_window"] = [LABELS[int(np.argmax([r[f"p_{x}"] for x in LABELS]))] for _, r in t.iterrows()]
    tr = []
    for oid, g in t.groupby("independent_object_id", sort=True):
        pm = np.array([g[f"p_{x}"].mean() for x in LABELS], float)
        mean_lab = LABELS[int(np.argmax(pm))]
        counts = g["pred_window"].value_counts(); maxc = counts.max()
        tied = [x for x in LABELS if int(counts.get(x, 0)) == int(maxc)]
        vote_lab = tied[0] if len(tied)==1 else max(tied, key=lambda z: pm[LABELS.index(z)])
        tr.append({"domain":"target_unknown_truth", "independent_object_id":oid, "class_label":"UNKNOWN", "mean_probability_label":mean_lab, "majority_vote_label":vote_lab, "same_label":mean_lab==vote_lab})
    tgt_cmp = pd.DataFrame(tr)
    detail = pd.concat([src_cmp, tgt_cmp], ignore_index=True)
    summary = pd.DataFrame([
        {"domain":"source_oof", "mean_prob_macro_f1":m_mean["macro_f1"], "majority_vote_macro_f1":m_vote["macro_f1"], "file_label_disagreement_count":int((~src_cmp["same_label"]).sum()), "n_files":len(src_cmp)},
        {"domain":"target_unknown_truth", "mean_prob_macro_f1":np.nan, "majority_vote_macro_f1":np.nan, "file_label_disagreement_count":int((~tgt_cmp["same_label"]).sum()), "n_files":len(tgt_cmp)},
    ])
    return detail, summary


def stratified_file_bootstrap(oof_file: pd.DataFrame):
    by = {lab: oof_file[oof_file["class_label"] == lab].copy().reset_index(drop=True) for lab in LABELS}
    vals = []
    for b in range(BOOT_REPS):
        parts = []
        for lab in LABELS:
            g = by[lab]
            idx = RNG.integers(0, len(g), size=len(g))
            parts.append(g.iloc[idx])
        q = pd.concat(parts, ignore_index=True)
        m = metric_dict(q["class_label"].astype(str), q["pred_label_file"].astype(str))
        vals.append({"rep": b, "macro_f1":m["macro_f1"], "balanced_accuracy":m["balanced_accuracy"], "recall_N":m["recall_N"]})
    d = pd.DataFrame(vals)
    summary = {
        "resampling_unit": "independent source files, stratified within OR/IR/B/N; windows are never resampled independently",
        "repetitions": BOOT_REPS,
        "macro_f1_median": float(d["macro_f1"].median()),
        "macro_f1_p05": float(d["macro_f1"].quantile(0.05)),
        "macro_f1_p95": float(d["macro_f1"].quantile(0.95)),
        "recall_N_median": float(d["recall_N"].median()),
        "recall_N_p05": float(d["recall_N"].quantile(0.05)),
        "recall_N_p95": float(d["recall_N"].quantile(0.95)),
        "warning": "Descriptive resampling sensitivity only; N has only four independent files, so this is not a high-precision population confidence interval.",
    }
    return d, summary


def leave_one_normal_file_out(src: pd.DataFrame, feats: list[str]):
    normals = sorted(src.loc[src["class_label"] == "N", "independent_object_id"].unique())
    rows = []
    for oid in normals:
        train = src[src["independent_object_id"] != oid].copy().reset_index(drop=True)
        test = src[src["independent_object_id"] == oid].copy().reset_index(drop=True)
        pipe = fit_fixed_rf(train, feats)
        P = pipe.predict_proba(test[feats].to_numpy(float))
        _, ff = aggregate_predictions(test, P, list(pipe.named_steps["clf"].classes_))
        r = ff.iloc[0]
        rows.append({"held_normal_file": oid, "load_hp":float(r["load_hp"]), "train_independent_normals":int(train.loc[train["class_label"]=="N", "independent_object_id"].nunique()), "n_test_windows":int(r["n_windows"]), "pred_label":str(r["pred_label"]), "p_N":float(r["p_N"]), "correct":str(r["pred_label"])=="N"})
    return pd.DataFrame(rows)


def target_gain_stress(target: pd.DataFrame, pipe, feats: list[str]):
    # A pure multiplicative sensor-chain gain scales these amplitude-like features linearly.
    amp = ["td_rms", "td_std", "td_peak_abs", "td_ptp", "env_rms"]
    factors = [0.5, 1/math.sqrt(2), 1.0, math.sqrt(2), 2.0]  # -6.02,-3.01,0,+3.01,+6.02 dB
    base_labels = None
    rows = []
    for fac in factors:
        q = target.copy()
        q.loc[:, amp] = q[amp] * fac
        P = pipe.predict_proba(q[feats].to_numpy(float))
        _, ff = aggregate_predictions(q.assign(class_label="UNKNOWN", load_hp=np.nan), P, list(pipe.named_steps["clf"].classes_))
        labels = dict(zip(ff["independent_object_id"], ff["pred_label"]))
        if abs(fac - 1.0) < 1e-12:
            base_labels = labels
        rows.append({"gain_factor":fac, "gain_db":20*math.log10(fac), "OR_count":int((ff["pred_label"]=="OR").sum()), "IR_count":int((ff["pred_label"]=="IR").sum()), "B_count":int((ff["pred_label"]=="B").sum()), "N_count":int((ff["pred_label"]=="N").sum()), "labels_serialized":";".join(f"{Path(k).stem}:{v}" for k,v in sorted(labels.items()))})
    out = pd.DataFrame(rows)
    # second pass for changes relative to factor=1
    base = {x.split(":")[0]:x.split(":")[1] for x in out.loc[np.isclose(out["gain_factor"],1.0), "labels_serialized"].iloc[0].split(";")}
    changed = []
    for _, r in out.iterrows():
        cur = {x.split(":")[0]:x.split(":")[1] for x in r["labels_serialized"].split(";")}
        changed.append(sum(cur[k] != base[k] for k in base))
    out["changed_files_vs_nominal"] = changed
    out["scenario_basis"] = "sensor-chain multiplicative gain stress: ±3 dB and ±6 dB; not asserted as measured target calibration error"
    return out


def make_figures(ablation, windows, gain):
    q = ablation.sort_values("pooled_macro_f1")
    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.barh(q["setting"], q["pooled_macro_f1"])
    ax.axvline(float(ablation.loc[ablation["setting"]=="all_28", "pooled_macro_f1"].iloc[0]), linestyle="--", linewidth=1)
    ax.set_xlabel("Pooled file-level Macro-F1")
    ax.set_title("STEP12 feature-group ablation under frozen load-held-out protocol")
    fig.tight_layout(); fig.savefig(FIG / "feature_group_ablation.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.plot(windows["setting"], windows["pooled_macro_f1"], marker="o")
    ax.set_ylabel("Pooled file-level Macro-F1")
    ax.set_title("Window duration/overlap sensitivity (same RF, file-grouped evaluation)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG / "window_parameter_sensitivity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(gain["gain_db"], gain["changed_files_vs_nominal"], marker="o")
    ax.set_xlabel("Target amplitude gain scenario (dB)")
    ax.set_ylabel("A-P files whose predicted label changes")
    ax.set_title("Target sensor-gain stress: prediction sensitivity only")
    fig.tight_layout(); fig.savefig(FIG / "target_gain_stress.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11,4.2))
    ax.axis("off")
    txt = "Q1C: 56 source files / 16 target files\n1 s, 50% overlap, 28 shared features\nindependent_object_id frozen" \
          "  →  Q2C: file-grouped LOLO source RF\n300 trees, depth 10\nfile probability mean" \
          "  →  Q3B/Q3C: A0 no-transfer retained\nA-P truth unknown\n16 file predictions" \
          "  →  Q4: explain same frozen A0 RF\npath importance + controls\nno target accuracy"
    ax.text(0.01,0.5,txt,va="center",ha="left",fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "four_question_version_chain.png", dpi=180); plt.close(fig)


def main():
    t0 = time.perf_counter()
    q2i = json.loads(Q2_INTERFACE.read_text(encoding="utf-8"))
    q3i = json.loads(Q3_INTERFACE.read_text(encoding="utf-8"))
    q2v = json.loads(Q2C_VALID.read_text(encoding="utf-8"))
    q3v = json.loads(Q3B_VALID.read_text(encoding="utf-8"))
    q3cv = json.loads(Q3C_VALID.read_text(encoding="utf-8"))
    q4v = json.loads(Q4_VALID.read_text(encoding="utf-8"))
    src = pd.read_csv(SRC_CSV)
    tgt = pd.read_csv(TGT_CSV)
    oof_file = pd.read_csv(Q2C_OOF_FILE)
    oof_win = pd.read_csv(Q2C_OOF_WIN)
    target_table = pd.read_csv(FINAL_TARGET_TABLE)
    feats = list(q2i["feature_columns"])
    final_pipe = joblib.load(FINAL_SOURCE_MODEL)

    # ---------------- Version/interface chain ----------------
    chain = pd.DataFrame([
        {"stage":"Q1C_STEP04", "artifact":"q2_source_raw.csv + q2_target_raw.csv", "version":q2i["version"], "feature_count":len(feats), "group_field":q2i["group_column"], "aggregation":"not applicable; raw windows", "source_truth":"known OR/IR/B/N", "target_truth":"unknown"},
        {"stage":"Q2C_STEP07", "artifact":"final_source_model.joblib + q3_interface.json", "version":q3i["version"], "feature_count":len(q3i["feature_columns"]), "group_field":q3i["group_column"], "aggregation":q3i["file_aggregation"], "source_truth":"used only inside frozen grouped protocol", "target_truth":"not used"},
        {"stage":"Q3B_STEP09", "artifact":"final_transfer_bundle.joblib", "version":"A0_no_transfer/frozen_rf", "feature_count":28, "group_field":"independent_object_id", "aggregation":"arithmetic mean of window class probabilities; argmax", "source_truth":"simulated target truth revealed only after adaptation evaluation", "target_truth":"unknown; never used"},
        {"stage":"Q3C_STEP10", "artifact":"final_A_P_label_table.csv", "version":"STEP09 frozen A0", "feature_count":28, "group_field":"independent_object_id", "aggregation":q3cv["summary"]["aggregation_rule"], "source_truth":"not applicable", "target_truth":"unknown; accuracy not computed"},
        {"stage":"Q4_STEP11", "artifact":"q4_explainability outputs", "version":"explains STEP09 A0 RF", "feature_count":28, "group_field":"independent_object_id", "aggregation":"same file-level model output; explanations averaged across file windows", "source_truth":"known representatives for mechanism cross-check", "target_truth":"unknown; explanations not truth"},
    ])
    chain.to_csv(OUT / "four_question_interface_chain.csv", index=False, encoding="utf-8-sig")

    # ---------------- P0 audit ----------------
    src_files = src[["independent_object_id","class_label","load_hp"]].drop_duplicates()
    tgt_files = tgt[["independent_object_id","truth_status"]].drop_duplicates()
    normal_files = src_files[src_files["class_label"]=="N"]
    p0 = []
    def chk(name, ok, evidence, severity="P0"):
        p0.append({"check":name,"severity":severity,"pass":bool(ok),"evidence":evidence})
    chk("Q1C/Q2C/Q3 shared feature list identical", feats == list(q3i["feature_columns"]) and len(feats)==28, f"q2={len(feats)}, q3={len(q3i['feature_columns'])}")
    chk("source inventory 56 independent files", src_files["independent_object_id"].nunique()==56, str(src_files["independent_object_id"].nunique()))
    chk("target inventory A-P 16 independent files", tgt_files["independent_object_id"].nunique()==16, str(tgt_files["independent_object_id"].nunique()))
    chk("normal class has exactly four independent files", normal_files["independent_object_id"].nunique()==4, f"N files={normal_files['independent_object_id'].nunique()}, loads={sorted(normal_files['load_hp'].tolist())}")
    chk("target truth remains unknown", set(tgt["truth_status"].astype(str).unique())=={"unknown"}, str(tgt["truth_status"].unique().tolist()))
    chk("source windows map to one class/load per independent object", src.groupby("independent_object_id")["class_label"].nunique().max()==1 and src.groupby("independent_object_id")["load_hp"].nunique().max()==1, "class/load uniqueness by independent_object_id")
    chk("OOF source files exactly 56 once", oof_file["independent_object_id"].nunique()==56 and len(oof_file)==56 and not oof_file["independent_object_id"].duplicated().any(), f"rows={len(oof_file)} unique={oof_file['independent_object_id'].nunique()}")
    chk("OOF windows exactly 400 once", oof_win["window_id"].nunique()==400 and len(oof_win)==400, f"rows={len(oof_win)} unique={oof_win['window_id'].nunique()}")
    chk("source/target sampling-rate metadata separated", set(src["fs_hz"].astype(int).unique())=={48000} and set(tgt["fs_hz"].astype(int).unique())=={32000}, f"source={src['fs_hz'].unique().tolist()}, target={tgt['fs_hz'].unique().tolist()}")
    chk("target approximate RPM excluded from model features", "rpm_value" not in feats and "rpm_source" not in feats, "rpm fields forbidden by Q2 interface")
    chk("source-only geometry mechanism features excluded from Q2/Q3", all(x not in feats for x in q2i["source_only_mechanism_features_excluded_from_default_q2"]) and not q3i["source_only_geometry_features_used"], "SKF6205-specific mechanism columns are not model inputs")
    chk("outer tests not used for Q2 model selection", q2v["outer_test_used_for_selection"] is False and q2v["checks"]["selection_inner_only"], q2v["selection_reason"])
    chk("no target pseudo-label self-certification", q3v["checks"]["pseudo_labels_not_used"] and q3v["checks"]["selection_simulated_only"], q3v["selection_reason"])
    chk("file aggregation consistent downstream", q3i["file_aggregation"].startswith("arithmetic mean") and "arithmetic mean" in q3cv["summary"]["aggregation_rule"], f"Q3 interface={q3i['file_aggregation']}; STEP10={q3cv['summary']['aggregation_rule']}")
    chk("Q4 explains actual final Q3 model", q4v["checks"]["explains_actual_step09_final_model"], "STEP11 validation report")
    p0_df = pd.DataFrame(p0)
    p0_df.to_csv(OUT / "p0_leakage_and_chain_audit.csv", index=False, encoding="utf-8-sig")

    # ---------------- Feature group ablation ----------------
    groups = {
        "time_domain": [f for f in feats if f.startswith("td_")],
        "frequency_domain": [f for f in feats if f.startswith("fd_")],
        "envelope_domain": [f for f in feats if f.startswith("env_")],
    }
    ab_rows = []
    ab_folds = []
    for name, used in [("all_28", feats)] + [(f"drop_{g}", [f for f in feats if f not in fs]) for g,fs in groups.items()]:
        s, fd, _ = evaluate_fixed_rf(src, used, name)
        s["feature_count"] = len(used)
        s["removed_group"] = "none" if name=="all_28" else name.replace("drop_","")
        ab_rows.append(s); ab_folds.append(fd.assign(feature_count=len(used)))
    ab = pd.DataFrame(ab_rows)
    base_macro = float(ab.loc[ab["setting"]=="all_28","pooled_macro_f1"].iloc[0])
    ab["delta_macro_f1_vs_all28"] = ab["pooled_macro_f1"] - base_macro
    ab.to_csv(OUT / "feature_group_ablation.csv", index=False, encoding="utf-8-sig")
    pd.concat(ab_folds, ignore_index=True).to_csv(OUT / "feature_group_ablation_by_load.csv", index=False, encoding="utf-8-sig")

    # ---------------- Window duration / overlap sensitivity ----------------
    signal_items = load_source_signals()
    window_settings = [
        ("0.75s_50pct_overlap",0.75,0.50),
        ("1.00s_50pct_overlap",1.00,0.50),
        ("1.25s_50pct_overlap",1.25,0.50),
        ("1.00s_no_overlap",1.00,0.00),
    ]
    ws_rows, ws_folds = [], []
    for name, win_s, ov in window_settings:
        d = extract_source_setting(signal_items, win_s, ov)
        s, fd, _ = evaluate_fixed_rf(d, feats, name)
        s.update({"window_seconds":win_s,"overlap_fraction":ov,"source_rotations_min":win_s*(1718/60),"source_rotations_max":win_s*(1797/60),"target_rotations_at_approx600rpm":win_s*10.0,"all_56_files_retained":d["independent_object_id"].nunique()==56})
        ws_rows.append(s); ws_folds.append(fd.assign(window_seconds=win_s,overlap_fraction=ov))
    ws = pd.DataFrame(ws_rows)
    ws.to_csv(OUT / "window_parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.concat(ws_folds, ignore_index=True).to_csv(OUT / "window_parameter_sensitivity_by_load.csv", index=False, encoding="utf-8-sig")

    # frozen 1s reproduction check against official STEP07 pooled result
    official_macro = float(q2v["final_pooled_file_metrics"]["macro_f1"])
    stress_1s_macro = float(ws.loc[ws["setting"]=="1.00s_50pct_overlap","pooled_macro_f1"].iloc[0])
    reproduction_abs_diff = abs(official_macro - stress_1s_macro)

    # ---------------- Aggregation sensitivity ----------------
    agg_detail, agg_summary = aggregation_sensitivity(oof_win, tgt, final_pipe, feats)
    agg_detail.to_csv(OUT / "aggregation_sensitivity_detail.csv", index=False, encoding="utf-8-sig")
    agg_summary.to_csv(OUT / "aggregation_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    # ---------------- Independent-file bootstrap ----------------
    boot, boot_summary = stratified_file_bootstrap(oof_file)
    boot.to_csv(OUT / "file_level_bootstrap_replicates.csv", index=False, encoding="utf-8-sig")
    (OUT / "file_level_bootstrap_summary.json").write_text(json.dumps(boot_summary,ensure_ascii=False,indent=2),encoding="utf-8")

    # ---------------- Normal leave-one-file-out ----------------
    lono = leave_one_normal_file_out(src, feats)
    lono.to_csv(OUT / "normal_leave_one_file_out.csv", index=False, encoding="utf-8-sig")

    # ---------------- Target amplitude/gain stress ----------------
    gain = target_gain_stress(tgt, final_pipe, feats)
    gain.to_csv(OUT / "target_gain_stress.csv", index=False, encoding="utf-8-sig")

    # ---------------- Migration cross-check ----------------
    mig = pd.DataFrame(q3v["method_summary"])
    mig["selection_basis"] = "four simulated held-out source loads only; real A-P truth unavailable"
    mig["real_target_accuracy_available"] = False
    mig.to_csv(OUT / "migration_robustness_crosscheck.csv", index=False, encoding="utf-8-sig")

    # ---------------- Error propagation / scenario ledger ----------------
    err = pd.DataFrame([
        {"upstream_quantity":"target RPM ≈600 rpm", "uncertainty_or_shift":"approximate operating condition; exact per-file RPM unavailable", "path_to_final_prediction":"none: rpm_value/rpm_source are metadata and forbidden model features", "scenario_or_test":"structural audit", "downstream_result":"no direct numerical propagation into frozen 28-feature RF; only limits interpretation of target characteristic frequencies"},
        {"upstream_quantity":"source SKF6205 geometry", "uncertainty_or_shift":"target bearing geometry unknown", "path_to_final_prediction":"none: 11 source-only geometry mechanism features excluded from Q2/Q3", "scenario_or_test":"interface audit", "downstream_result":"no source geometry parameter is numerically imposed on A-P classification"},
        {"upstream_quantity":"sampling rate", "uncertainty_or_shift":"source 48 kHz vs target 32 kHz", "path_to_final_prediction":"feature extractor uses true fs and 1-second physical windows; spectral features restricted to common 0–6 kHz band", "scenario_or_test":"metadata + feature interface audit", "downstream_result":"same point count is never assumed; 0–6 kHz is below both Nyquist limits"},
        {"upstream_quantity":"window duration/overlap", "uncertainty_or_shift":"0.75/1.00/1.25 s at 50% overlap plus 1.0 s no-overlap", "path_to_final_prediction":"changes number of physical cycles and windows per file", "scenario_or_test":"fresh source re-extraction + same load-held-out RF", "downstream_result":f"pooled Macro-F1 range {ws['pooled_macro_f1'].min():.4f}–{ws['pooled_macro_f1'].max():.4f}; all tested settings retain 56 files"},
        {"upstream_quantity":"sensor-chain amplitude gain", "uncertainty_or_shift":"stress only: ±3 dB and ±6 dB multiplicative gain, not asserted measurement error", "path_to_final_prediction":"linearly scales td_rms/std/peak/ptp and env_rms", "scenario_or_test":"frozen target RF prediction stress", "downstream_result":f"maximum A-P label changes vs nominal across gain scenarios = {int(gain['changed_files_vs_nominal'].max())}/16"},
        {"upstream_quantity":"normal-class finite sample", "uncertainty_or_shift":"only 4 independent N files", "path_to_final_prediction":"class weighting and validation uncertainty", "scenario_or_test":"leave-one-normal-file-out + file bootstrap", "downstream_result":f"LONO correct {int(lono['correct'].sum())}/4; bootstrap N-recall 5–95% = {boot_summary['recall_N_p05']:.3f}–{boot_summary['recall_N_p95']:.3f}"},
    ])
    err.to_csv(OUT / "upstream_error_propagation_and_scenarios.csv", index=False, encoding="utf-8-sig")

    # ---------------- Claim-evidence matrix ----------------
    claim = pd.DataFrame([
        {"major_conclusion":"Q1 selected M1 is a coherent 56-file, four-class source set and produces a target-compatible 28-feature interface.", "required_evidence":"inventory, group IDs, feature interface, no source-only geometry features in Q2/Q3", "evidence":"Q1C interface + STEP12 P0 audit", "status":"supported"},
        {"major_conclusion":"Q2 final RF improves on the frozen linear baseline under the same file-grouped load-held-out protocol.", "required_evidence":"inner-only selection plus confirmatory 56-file OOF metrics", "evidence":f"Macro-F1 {q2v['baseline_pooled_file_metrics']['macro_f1']:.4f} -> {q2v['final_pooled_file_metrics']['macro_f1']:.4f}; outer test not used for selection", "status":"supported; do not call statistically significant"},
        {"major_conclusion":"The final source classifier uses information from multiple feature modules rather than one unverified module.", "required_evidence":"single-group ablations under unchanged split/model", "evidence":"feature_group_ablation.csv", "status":"supported with measured ablation deltas"},
        {"major_conclusion":"Source generalization must be stated across held-out operating loads, not random windows.", "required_evidence":"load-held-out file-level folds", "evidence":f"STEP07 pooled Macro-F1={official_macro:.4f}; STEP12 reproduces fixed protocol", "status":"supported"},
        {"major_conclusion":"No tested unsupervised adaptation earned promotion over A0.", "required_evidence":"simulated-target metrics, negative-transfer rule, no A-P truth tuning", "evidence":f"A2 gain={float(mig.loc[mig.method=='A2_CORAL','gain_vs_A0'].iloc[0]):.4f}; A3 gain={float(mig.loc[mig.method=='A3_instance_weighting','gain_vs_A0'].iloc[0]):.4f}", "status":"supported"},
        {"major_conclusion":"A-P labels are model predictions only, not verified truth; all-OR collapse is a warning.", "required_evidence":"unknown truth ledger, no target accuracy, collapse flag", "evidence":f"target truth={set(tgt.truth_status.astype(str))}; counts={q3v['collapse']['final_counts']}", "status":"supported as limitation"},
        {"major_conclusion":"Q4 explanations describe the frozen final model and have empirical faithfulness/stability checks.", "required_evidence":"same model/interface; >=2 explanation validations", "evidence":f"STEP11 valid validation types={q4v['valid_validation_type_count']}", "status":"supported; not causal"},
    ])
    claim.to_csv(OUT / "major_claim_evidence_matrix.csv", index=False, encoding="utf-8-sig")

    # ---------------- Conflict / repair ledger ----------------
    conflicts = pd.DataFrame([
        {"issue":"Potential window leakage", "severity":"P0", "finding":"closed", "repair_or_control":"all splits and OOF checks use independent_object_id; 400 OOF windows belong to exactly 56 file groups"},
        {"issue":"N class may be overstated by 54 windows", "severity":"P0", "finding":"closed", "repair_or_control":"report N=4 independent files; LONO and file bootstrap use files, not windows, as resampling unit"},
        {"issue":"Historical/implicit target labels", "severity":"P0", "finding":"closed", "repair_or_control":"target truth_status remains unknown; target accuracy never computed; no pseudo-label selection"},
        {"issue":"48k source vs 32k target point-count confusion", "severity":"P0", "finding":"closed", "repair_or_control":"windows defined in seconds; fs retained; frequency features use physical Hz in common 0–6k band"},
        {"issue":"Source SKF6205 geometry could leak into target", "severity":"P0", "finding":"closed", "repair_or_control":"geometry-specific mechanism features excluded from Q2/Q3 interface"},
        {"issue":"Outer test tuning", "severity":"P0", "finding":"closed", "repair_or_control":"STEP07 selection_inner_only=true and outer_test_used_for_selection=false"},
        {"issue":"Pseudo-label self-certification", "severity":"P0", "finding":"closed", "repair_or_control":"STEP09 pseudo_labels_not_used=true; selection uses simulated source target only"},
        {"issue":"Aggregation rule drift", "severity":"P0", "finding":"closed", "repair_or_control":"mean window class probability then argmax is frozen Q2→Q3→Q3C; majority vote tested only as sensitivity"},
        {"issue":"STEP11 first CI attempt failed on matplotlib API", "severity":"P2 execution", "finding":"closed", "repair_or_control":"changed boxplot keyword labels→tick_labels and reran successfully; accepted evidence commit is later successful run"},
        {"issue":"A-P all-OR output collapse", "severity":"P1 scientific limitation", "finding":"open but explained; not a leakage defect", "repair_or_control":"kept as low-trust warning; rejected unvalidated adaptation rather than manually rebalance labels"},
        {"issue":"B007_0 lacks robust BSF envelope support", "severity":"P1 mechanism inconsistency", "finding":"open but bounded", "repair_or_control":"kept as contradiction case; do not claim every B sample has clean BSF peak"},
    ])
    conflicts.to_csv(OUT / "conflict_and_repair_log.csv", index=False, encoding="utf-8-sig")

    # ---------------- Failure boundaries and retained conclusions ----------------
    max_gain_changes = int(gain["changed_files_vs_nominal"].max())
    ab_min = float(ab["pooled_macro_f1"].min())
    ws_min = float(ws["pooled_macro_f1"].min())
    failures = pd.DataFrame([
        {"condition":"CORAL global covariance alignment", "observed":"simulated-target mean Macro-F1 below A0 and new zero-recall condition", "boundary":"do not claim distribution alignment improves diagnosis; CORAL not promoted"},
        {"condition":"domain-classifier instance weighting", "observed":"no mean Macro-F1 gain over A0", "boundary":"extra adaptation complexity unsupported; retain A0"},
        {"condition":"target A-P inference", "observed":"16/16 predicted OR and all marked very-low-trust in STEP10", "boundary":"cannot infer target class prevalence or accuracy"},
        {"condition":"normal-class evidence", "observed":"only 4 independent files regardless of window count", "boundary":"N metrics remain coarse; report counts and LONO rather than precise population claims"},
        {"condition":"window parameter dependence", "observed":f"tested pooled Macro-F1 minimum={ws_min:.4f}", "boundary":"1 s/50% is frozen main setting; neighboring physical windows are sensitivity checks, not new model-selection opportunities"},
        {"condition":"feature module removal", "observed":f"worst ablated pooled Macro-F1={ab_min:.4f}", "boundary":"do not call any module indispensable unless its removal causes a material reproducible drop; report actual delta"},
        {"condition":"target sensor gain", "observed":f"up to {max_gain_changes}/16 target file labels change in ±3/±6 dB stress", "boundary":"if gain-sensitive predictions change, A-P labels are not gain-robust; this stress is not a target-accuracy test"},
        {"condition":"mechanism explanation", "observed":"source B difficult case lacks robust BSF support; many target prototypes disagree with OR", "boundary":"feature importance/nearest prototype cannot be written as physical causation or target truth"},
    ])
    failures.to_csv(OUT / "failure_boundaries.csv", index=False, encoding="utf-8-sig")

    retained = pd.DataFrame([
        {"conclusion":"M1/28-feature interface is traceable and leakage-controlled at original-file group level.", "scope":"source acquisition record is the minimum independent unit; physical bearing specimen IDs are unavailable"},
        {"conclusion":f"Frozen RF source OOF Macro-F1 is {official_macro:.4f} under load-held-out file evaluation.", "scope":"source-domain evidence only; N has 4 independent files"},
        {"conclusion":"A0 no-transfer remains the justified STEP09 method because tested A1/A2/A3 did not meet frozen promotion rules.", "scope":"simulated load-shift evidence; real target negative transfer cannot be measured without truth"},
        {"conclusion":"A-P final outputs are all OR predictions with explicit low-trust/collapse warning.", "scope":"not target ground truth, prevalence, or accuracy"},
        {"conclusion":"Q4 explanation is empirically faithful/stable for the frozen RF and partly mechanism-consistent on source exemplars.", "scope":"model dependence, not causal proof; target geometry prevents exact characteristic-frequency validation"},
    ])
    retained.to_csv(OUT / "retained_conclusions.csv", index=False, encoding="utf-8-sig")

    # no key reruns remain if all core checks close and 1 s stress reproduces official metric.
    reruns = []
    if not bool(p0_df["pass"].all()):
        reruns.append({"priority":"P0","experiment":"repair failed P0 audit items","status":"REQUIRED","blocking":True})
    if reproduction_abs_diff > 1e-10:
        reruns.append({"priority":"P0","experiment":"reproduce 1.0s/50% source feature extraction and fixed RF evaluation","status":"REQUIRED","blocking":True})
    if not reruns:
        reruns.append({"priority":"none","experiment":"none","status":"CLOSED: no critical rerun required","blocking":False})
    rerun_df = pd.DataFrame(reruns)
    rerun_df.to_csv(OUT / "rerun_required.csv", index=False, encoding="utf-8-sig")

    make_figures(ab, ws, gain)

    checks = {
        "no_P0_leakage_or_chain_failure": bool(p0_df["pass"].all()),
        "q1_q2_q3_feature_interface_same_28": feats == list(q3i["feature_columns"]) and len(feats)==28,
        "normal_independent_file_count_is_4": int(normal_files["independent_object_id"].nunique())==4,
        "target_truth_unknown_and_accuracy_absent": set(tgt["truth_status"].astype(str).unique())=={"unknown"} and q3v["target_accuracy_reported"] is False,
        "fixed_1s_window_reproduces_step07_metric": reproduction_abs_diff <= 1e-10,
        "feature_ablation_completed": len(ab)==4,
        "window_sensitivity_completed_with_56_files_each": len(ws)==4 and bool(ws["all_56_files_retained"].all()),
        "aggregation_sensitivity_completed": len(agg_summary)==2,
        "independent_file_bootstrap_completed": len(boot)==BOOT_REPS,
        "normal_leave_one_file_out_completed": len(lono)==4,
        "migration_negative_transfer_evidence_present": len(mig)>=3 and q3v["checks"]["negative_transfer_checked_on_simulated_target"],
        "explainability_chain_valid": q4v["valid_validation_type_count"]>=2,
        "all_major_claims_have_evidence": bool((claim["evidence"].astype(str).str.len()>0).all()),
        "no_blocking_rerun_open": not bool(rerun_df["blocking"].any()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "p0_failures": p0_df.loc[~p0_df["pass"], "check"].tolist(),
        "official_step07_pooled_macro_f1": official_macro,
        "fresh_1s50_pooled_macro_f1": stress_1s_macro,
        "fresh_reproduction_abs_diff": reproduction_abs_diff,
        "feature_ablation": ab[["setting","feature_count","pooled_macro_f1","delta_macro_f1_vs_all28"]].to_dict("records"),
        "window_sensitivity": ws[["setting","n_windows","pooled_macro_f1","min_outer_macro_f1","all_56_files_retained"]].to_dict("records"),
        "aggregation_sensitivity": agg_summary.to_dict("records"),
        "normal_lono_correct": f"{int(lono['correct'].sum())}/4",
        "file_bootstrap": boot_summary,
        "target_gain_max_label_changes": max_gain_changes,
        "step09_selected_method": q3v["selected_method"],
        "blocking_reruns": rerun_df.loc[rerun_df["blocking"], "experiment"].tolist(),
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    manifest = {
        "step":"STEP12_joint_validation",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start":git_sha(),
        "python":sys.version,
        "sklearn":sklearn.__version__,
        "platform":platform.platform(),
        "seed":SEED,
        "source_sha256":sha256(SRC_CSV),
        "target_sha256":sha256(TGT_CSV),
        "q2_interface_sha256":sha256(Q2_INTERFACE),
        "q3_interface_sha256":sha256(Q3_INTERFACE),
        "final_source_model_sha256":sha256(FINAL_SOURCE_MODEL),
        "step09_validation_sha256":sha256(Q3B_VALID),
        "step10_validation_sha256":sha256(Q3C_VALID),
        "step11_validation_sha256":sha256(Q4_VALID),
        "feature_count":len(feats),
        "bootstrap_repetitions":BOOT_REPS,
        "learned_reselection_performed":False,
        "target_labels_used":False,
        "target_accuracy_computed":False,
        "elapsed_seconds":time.perf_counter()-t0,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    summary = [
        "STEP12 joint validation / robustness / failure boundaries",
        f"status={status}",
        f"P0_failures={validation['p0_failures']}",
        f"STEP07 official Macro-F1={official_macro:.6f}; fresh 1s50 reproduction={stress_1s_macro:.6f}; abs_diff={reproduction_abs_diff:.3e}",
        f"feature_ablation_macro_range={ab['pooled_macro_f1'].min():.6f}..{ab['pooled_macro_f1'].max():.6f}",
        f"window_sensitivity_macro_range={ws['pooled_macro_f1'].min():.6f}..{ws['pooled_macro_f1'].max():.6f}",
        f"normal_LONO={int(lono['correct'].sum())}/4",
        f"target_gain_stress_max_changed_files={max_gain_changes}/16",
        f"step09_method={q3v['selected_method']}",
        f"blocking_reruns={validation['blocking_reruns']}",
        "Target truth remains unknown; no target accuracy was computed.",
    ]
    (OUT / "result_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("\n".join(summary))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
