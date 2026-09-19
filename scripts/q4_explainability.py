# AI ASSISTANCE NOTICE
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ChatGPT；版本/型号：GPT-5.6 Sol（ChatGPT 2026-08-06更新版）；
# 开发机构/公司：OpenAI；版本发布日期：2026-08-06。
# 人工智能仅用于代码检查、调试建议与说明整理；最终算法、参数与结果由参赛队审查并由冻结复现链验证。
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
Q1C = ROOT / "outputs" / "q1c_feature_extraction"
Q1B = ROOT / "outputs" / "q1b_signal_mechanism"
Q2C = ROOT / "outputs" / "q2c_failure_driven_improvement"
Q3B = ROOT / "outputs" / "q3b_unsupervised_transfer"
Q3C = ROOT / "outputs" / "q3c_final_labels_and_display"
OUT = ROOT / "outputs" / "q4_explainability"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SRC_CSV = Q1C / "q2_source_raw.csv"
TGT_CSV = Q1C / "q2_target_raw.csv"
Q3_INTERFACE = Q2C / "q3_interface.json"
SOURCE_OOF = Q2C / "final_method_oof_file_predictions.csv"
FINAL_BUNDLE = Q3B / "models" / "final_transfer_bundle.joblib"
Q3B_DECISION = Q3B / "final_selection_decision.json"
FINAL_TARGET_TABLE = Q3C / "final_A_P_label_table.csv"
Q3C_MECH = Q3C / "mechanism_side_evidence.csv"
Q1B_MECH = Q1B / "mechanism_recalc.csv"

LABELS = ["OR", "IR", "B", "N"]
SEED = 20260916
EPS = 1e-12
TOPK = 5
STABILITY_REPS = 5
STABILITY_NOISE_FRAC = 0.005
RANDOM_CONTROL_REPS = 30

FEATURE_GROUP = {
    "td_rms": "time_energy", "td_std": "time_energy", "td_peak_abs": "impact",
    "td_ptp": "impact", "td_skewness": "time_shape", "td_kurtosis": "impact",
    "td_crest_factor": "impact", "td_impulse_factor": "impact", "td_shape_factor": "time_shape",
    "td_clearance_factor": "impact", "td_zero_cross_rate": "time_shape",
    "fd_dominant_hz_0_6k": "spectrum", "fd_centroid_hz_0_6k": "spectrum",
    "fd_f95_hz_0_6k": "spectrum", "fd_entropy_0_6k": "spectrum",
    "fd_energy_ratio_0_500": "spectrum", "fd_energy_ratio_500_1500": "spectrum",
    "fd_energy_ratio_1500_3000": "spectrum", "fd_energy_ratio_3000_6000": "spectrum",
    "env_rms": "envelope", "env_kurtosis": "envelope", "env_dominant_hz_0_500": "envelope",
    "env_centroid_hz_0_500": "envelope", "env_entropy_0_500": "envelope",
    "env_energy_ratio_0_50": "envelope", "env_energy_ratio_50_150": "envelope",
    "env_energy_ratio_150_300": "envelope", "env_energy_ratio_300_500": "envelope",
}
MECHANISM_RELATED = {f for f, g in FEATURE_GROUP.items() if g in {"impact", "spectrum", "envelope"}}
SOURCE_CASES = {
    "source_OR": "data/raw/source_domain/cwru_48khz_de/OR007@12_0.mat",
    "source_IR": "data/raw/source_domain/cwru_48khz_de/IR007_0.mat",
    "source_B": "data/raw/source_domain/cwru_48khz_de/B007_0.mat",
    "source_N": "data/raw/source_domain/cwru_48khz_normal/N_1_(1772rpm).mat",
}
TARGET_CASE_IDS = ["A", "B", "C", "O"]


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


def predict_file(pipe, df: pd.DataFrame, feats: list[str]):
    P = pipe.predict_proba(df[feats].to_numpy(float))
    classes = list(pipe.named_steps["clf"].classes_)
    pm = P.mean(axis=0)
    j = int(np.argmax(pm))
    return {
        "label": str(classes[j]),
        "score": float(pm[j]),
        "prob": {str(c): float(pm[k]) for k, c in enumerate(classes)},
    }


def source_reference_median(src: pd.DataFrame, feats: list[str]):
    # Equal-file reference: mean each acquisition first, then median across 56 files.
    fm = src.groupby("independent_object_id")[feats].mean()
    return fm.median(axis=0), fm.std(axis=0, ddof=1).replace(0, 1.0)


def node_impurity_decrease(tree, node: int):
    left = int(tree.children_left[node]); right = int(tree.children_right[node])
    if left < 0 or right < 0:
        return 0.0
    n = float(tree.weighted_n_node_samples[node])
    nl = float(tree.weighted_n_node_samples[left]); nr = float(tree.weighted_n_node_samples[right])
    return max(0.0, n * float(tree.impurity[node]) - nl * float(tree.impurity[left]) - nr * float(tree.impurity[right]))


def path_importance_matrix(pipe, Xraw: np.ndarray):
    imp = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    Z = scaler.transform(imp.transform(np.asarray(Xraw, float)))
    n, p = Z.shape
    A = np.zeros((n, p), float)
    for est in clf.estimators_:
        path = est.decision_path(Z)
        tr = est.tree_
        for i in range(n):
            nodes = path.indices[path.indptr[i]:path.indptr[i + 1]]
            for node in nodes:
                f = int(tr.feature[node])
                if f >= 0:
                    A[i, f] += node_impurity_decrease(tr, int(node))
    rs = A.sum(axis=1, keepdims=True)
    A = np.divide(A, np.where(rs > EPS, rs, 1.0))
    return A


def file_path_importance(pipe, df: pd.DataFrame, feats: list[str]):
    A = path_importance_matrix(pipe, df[feats].to_numpy(float))
    v = A.mean(axis=0)
    s = float(v.sum())
    if s > EPS:
        v = v / s
    return v, A


def ablate_feature_set(pipe, df: pd.DataFrame, feats: list[str], ref: pd.Series, feature_set: list[str], explained_label: str):
    X = df[feats].to_numpy(float).copy()
    for f in feature_set:
        X[:, feats.index(f)] = float(ref[f])
    P = pipe.predict_proba(X)
    classes = list(pipe.named_steps["clf"].classes_)
    pm = P.mean(axis=0)
    score = float(pm[classes.index(explained_label)])
    label = str(classes[int(np.argmax(pm))])
    return score, label


def explain_one_file(case_id: str, role: str, df: pd.DataFrame, pipe, feats, ref, rng):
    base = predict_file(pipe, df, feats)
    v, Awin = file_path_importance(pipe, df, feats)
    order = np.argsort(v)[::-1]
    top_idx = list(order[:TOPK])
    low_idx = list(order[-TOPK:])
    top = [feats[i] for i in top_idx]
    low = [feats[i] for i in low_idx]
    base_score = float(base["score"])
    top_score, top_label = ablate_feature_set(pipe, df, feats, ref, top, base["label"])
    low_score, low_label = ablate_feature_set(pipe, df, feats, ref, low, base["label"])
    pool = [i for i in range(len(feats)) if i not in set(top_idx + low_idx)]
    random_changes = []
    random_labels = []
    random_sets = []
    for _ in range(RANDOM_CONTROL_REPS):
        idx = rng.choice(pool, size=TOPK, replace=False).tolist() if len(pool) >= TOPK else rng.choice(len(feats), size=TOPK, replace=False).tolist()
        fs = [feats[i] for i in idx]
        sc, lab = ablate_feature_set(pipe, df, feats, ref, fs, base["label"])
        random_changes.append(abs(base_score - sc))
        random_labels.append(lab)
        random_sets.append(";".join(fs))
    long_rows = []
    for rank, i in enumerate(order, start=1):
        long_rows.append({
            "case_id": case_id, "case_role": role, "feature": feats[i], "feature_group": FEATURE_GROUP.get(feats[i], "other"),
            "path_importance": float(v[i]), "rank": int(rank), "explained_label": base["label"],
        })
    summary = {
        "case_id": case_id, "case_role": role, "n_windows": int(len(df)), "explained_label": base["label"],
        "base_score": base_score,
        "top_features": ";".join(top), "top_groups": ";".join(FEATURE_GROUP.get(f, "other") for f in top),
        "low_features": ";".join(low),
        "top5_mechanism_related_count": int(sum(f in MECHANISM_RELATED for f in top)),
        "top5_ablation_abs_score_change": float(abs(base_score - top_score)), "top5_ablation_signed_score_drop": float(base_score - top_score),
        "top5_ablation_label": top_label, "low5_ablation_abs_score_change": float(abs(base_score - low_score)),
        "low5_ablation_signed_score_drop": float(base_score - low_score), "low5_ablation_label": low_label,
        "random5_abs_score_change_mean": float(np.mean(random_changes)), "random5_abs_score_change_std": float(np.std(random_changes, ddof=1)),
        "random5_label_change_rate": float(np.mean([x != base["label"] for x in random_labels])),
        "random_control_example": random_sets[0],
    }
    return pd.DataFrame(long_rows), summary, v, Awin


def stability_for_case(case_id, role, df, pipe, feats, ref, source_sd, original_v, rng):
    X0 = df[feats].to_numpy(float)
    rows = []
    for rep in range(STABILITY_REPS):
        noise = rng.normal(0.0, STABILITY_NOISE_FRAC, size=X0.shape) * source_sd[feats].to_numpy(float)[None, :]
        d = df.copy()
        d.loc[:, feats] = X0 + noise
        pred = predict_file(pipe, d, feats)
        v, _ = file_path_importance(pipe, d, feats)
        rho = spearmanr(original_v, v).statistic
        if not np.isfinite(rho):
            rho = 0.0
        perm = rng.permutation(original_v)
        rr = spearmanr(original_v, perm).statistic
        if not np.isfinite(rr):
            rr = 0.0
        rows.append({
            "case_id": case_id, "case_role": role, "rep": rep,
            "noise_scale_fraction_of_source_file_sd": STABILITY_NOISE_FRAC,
            "importance_spearman_original_vs_perturbed": float(rho),
            "random_permutation_spearman_control": float(rr),
            "perturbed_pred_label": pred["label"],
        })
    return rows


def make_global_importance_figure(pipe, feats):
    clf = pipe.named_steps["clf"]
    vals = np.asarray(clf.feature_importances_, float)
    tab = pd.DataFrame({"feature": feats, "feature_group": [FEATURE_GROUP.get(f, "other") for f in feats], "rf_gini_importance": vals})
    tab = tab.sort_values("rf_gini_importance", ascending=False).reset_index(drop=True)
    tab["rank"] = np.arange(1, len(tab) + 1)
    tab.to_csv(OUT / "global_rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    q = tab.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.barh(q["feature"], q["rf_gini_importance"])
    ax.set_xlabel("Random-forest impurity importance")
    ax.set_title("Final STEP09 model: global feature usage\n(descriptive split importance, not causal effect)")
    fig.tight_layout()
    fig.savefig(FIG / "global_rf_feature_importance.png", dpi=180)
    plt.close(fig)
    return tab


def make_faithfulness_figure(faith):
    cats = ["top5_ablation_abs_score_change", "low5_ablation_abs_score_change", "random5_abs_score_change_mean"]
    labels = ["High path-importance", "Low path-importance", "Random control"]
    target = faith[faith["case_role"] == "target_prediction"]
    source = faith[faith["case_role"] == "source_known_label"]
    vals_t = [float(target[c].median()) for c in cats]
    vals_s = [float(source[c].median()) for c in cats]
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x - w/2, vals_s, width=w, label="source known-label cases")
    ax.bar(x + w/2, vals_t, width=w, label="target predicted files")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("Median absolute change in explained-class score")
    ax.set_title("Faithfulness control: ablate high-use vs low-use/random features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "faithfulness_ablation_controls.png", dpi=180)
    plt.close(fig)


def make_stability_figure(stab):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = [stab["importance_spearman_original_vs_perturbed"].to_numpy(float), stab["random_permutation_spearman_control"].to_numpy(float)]
    ax.boxplot(data, labels=["0.5% source-SD perturbation", "random permutation control"], showmeans=True)
    ax.set_ylabel("Spearman correlation of 28-feature explanation")
    ax.set_title("Explanation stability under small input perturbations")
    ax.axhline(0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "explanation_stability.png", dpi=180)
    plt.close(fig)


def make_case_heatmap(local_long, case_ids):
    q = local_long[local_long["case_id"].isin(case_ids)].pivot(index="case_id", columns="feature", values="path_importance")
    q = q.reindex(case_ids)
    # Show the 14 features with highest mean local usage among the selected cases.
    cols = q.mean(axis=0).sort_values(ascending=False).head(14).index.tolist()
    vals = q[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(vals, aspect="auto")
    ax.set_yticks(range(len(q)), q.index)
    ax.set_xticks(range(len(cols)), cols, rotation=65, ha="right")
    ax.set_title("Local decision-path importance for representative source/target cases")
    fig.colorbar(im, ax=ax, label="normalized path impurity usage")
    fig.tight_layout()
    fig.savefig(FIG / "representative_case_path_importance.png", dpi=180)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    interface = json.loads(Q3_INTERFACE.read_text(encoding="utf-8"))
    decision = json.loads(Q3B_DECISION.read_text(encoding="utf-8"))
    feats = list(interface["feature_columns"])
    if len(feats) != 28:
        raise RuntimeError("Frozen 28-feature interface changed")
    src = pd.read_csv(SRC_CSV)
    tgt = pd.read_csv(TGT_CSV)
    source_oof = pd.read_csv(SOURCE_OOF)
    target_final = pd.read_csv(FINAL_TARGET_TABLE)
    target_mech = pd.read_csv(Q3C_MECH)
    q1b_mech = pd.read_csv(Q1B_MECH)
    bundle = joblib.load(FINAL_BUNDLE)
    if decision["selected_method"] != "A0_no_transfer" or bundle["method"] != "A0_no_transfer":
        raise RuntimeError("STEP09 final method changed; STEP11 must explain the actual final model")
    pipe = bundle["model_or_pipeline"]
    if list(bundle["feature_columns"]) != feats:
        raise RuntimeError("STEP09 feature interface changed")
    if set(tgt["truth_status"].astype(str).unique()) != {"unknown"}:
        raise RuntimeError("Target truth boundary changed")

    ref, source_sd = source_reference_median(src, feats)
    global_imp = make_global_importance_figure(pipe, feats)

    # Explanations are generated for all 16 target files plus the four STEP03 source representatives.
    cases = []
    for tid in list("ABCDEFGHIJKLMNOP"):
        oid = f"data/raw/target_domain/train_bearing_a_to_p/{tid}.mat"
        d = tgt[tgt["independent_object_id"].astype(str) == oid].sort_values("window_start_s").copy()
        if len(d) != 15:
            raise RuntimeError(f"Target {tid} window count changed: {len(d)}")
        cases.append((tid, "target_prediction", d, "UNKNOWN"))
    for cid, oid in SOURCE_CASES.items():
        d = src[src["independent_object_id"].astype(str) == oid].sort_values("window_start_s").copy()
        if len(d) == 0:
            raise RuntimeError(f"Missing source representative {oid}")
        cases.append((cid, "source_known_label", d, str(d["class_label"].iloc[0])))

    rng = np.random.default_rng(SEED)
    local_parts, faith_rows, stability_rows = [], [], []
    original_vectors = {}
    for cid, role, d, truth in cases:
        ll, fs, v, _ = explain_one_file(cid, role, d, pipe, feats, ref, rng)
        fs["known_or_unknown_label"] = truth
        if role == "target_prediction":
            expected = str(target_final.loc[target_final["target_id"].astype(str) == cid, "final_pred_label"].iloc[0])
            if fs["explained_label"] != expected:
                raise RuntimeError(f"Target {cid} explanation label does not reproduce STEP10")
        local_parts.append(ll)
        faith_rows.append(fs)
        original_vectors[cid] = v
        stability_rows.extend(stability_for_case(cid, role, d, pipe, feats, ref, source_sd, v, rng))

    local_long = pd.concat(local_parts, ignore_index=True)
    faith = pd.DataFrame(faith_rows)
    stab = pd.DataFrame(stability_rows)
    local_long.to_csv(OUT / "local_path_importance_long.csv", index=False, encoding="utf-8-sig")
    faith.to_csv(OUT / "faithfulness_validation_by_case.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(OUT / "stability_validation.csv", index=False, encoding="utf-8-sig")

    # Attach source OOF behavior and STEP03 exact mechanism check where available.
    source_case_rows = []
    oof_map = source_oof.set_index("independent_object_id")
    q1map = {Path(str(r.record_id)).name: r for r in q1b_mech.itertuples(index=False)}
    for cid, oid in SOURCE_CASES.items():
        frow = faith[faith["case_id"] == cid].iloc[0]
        true_lab = str(src.loc[src["independent_object_id"].astype(str) == oid, "class_label"].iloc[0])
        oof = oof_map.loc[oid] if oid in oof_map.index else None
        name = Path(oid).name
        qm = q1map.get(name)
        source_case_rows.append({
            "case_id": cid, "independent_object_id": oid, "true_label": true_lab,
            "final_model_explained_label": frow["explained_label"], "final_model_score": float(frow["base_score"]),
            "source_oof_pred_label": str(oof["pred_label_file"]) if oof is not None else "NA",
            "source_oof_confidence": float(oof["confidence_top1"]) if oof is not None else np.nan,
            "top5_path_features": frow["top_features"], "top5_mechanism_related_count": int(frow["top5_mechanism_related_count"]),
            "step03_frequency_name": str(qm.frequency_name) if qm is not None else "not_checked",
            "step03_theoretical_frequency_hz": float(qm.theoretical_frequency_hz) if qm is not None else np.nan,
            "step03_envelope_ratio": float(qm.local_envelope_ratio) if qm is not None else np.nan,
            "step03_mechanism_supported": bool(qm.supported_ratio_gt_1) if qm is not None else False,
            "interpretation_boundary": "final model explanation; source truth is known, but the final model has been trained on all source files",
        })
    source_cases = pd.DataFrame(source_case_rows)
    source_cases.to_csv(OUT / "source_known_label_case_table.csv", index=False, encoding="utf-8-sig")

    # Target case table: predictions remain unknown truth; mechanism prototype evidence is explanatory only.
    target_summary = faith[faith["case_role"] == "target_prediction"].copy()
    target_summary = target_summary.merge(target_final[["target_id", "model_score_top1", "score_margin_top1_top2", "entropy_norm", "window_consistency", "uncertainty_level"]], left_on="case_id", right_on="target_id", how="left")
    target_summary = target_summary.merge(target_mech[["target_id", "mechanism_nearest_source_class", "mechanism_pred_class_similarity_index", "mechanism_model_prototype_agree", "pred_class_distance_over_source_p95"]], on="target_id", how="left")
    target_summary["truth_status"] = "unknown"
    target_summary["explanation_boundary"] = "explains frozen model output only; cannot confirm target truth"
    target_summary.to_csv(OUT / "target_A_P_explanation_summary.csv", index=False, encoding="utf-8-sig")

    # Faithfulness validation with low and random controls.
    faith["top_exceeds_low"] = faith["top5_ablation_abs_score_change"] > faith["low5_ablation_abs_score_change"] + 1e-12
    faith["top_exceeds_random_mean"] = faith["top5_ablation_abs_score_change"] > faith["random5_abs_score_change_mean"] + 1e-12
    faith["top_exceeds_both"] = faith["top_exceeds_low"] & faith["top_exceeds_random_mean"]
    target_faith = faith[faith["case_role"] == "target_prediction"]
    faith_summary = {
        "definition": "Features are ranked by local RF decision-path impurity usage; validation independently replaces the top-5 raw features by the file-balanced source median and compares output change with bottom-5 and 30 random 5-feature controls.",
        "target_median_top5_abs_score_change": float(target_faith["top5_ablation_abs_score_change"].median()),
        "target_median_low5_abs_score_change": float(target_faith["low5_ablation_abs_score_change"].median()),
        "target_median_random5_abs_score_change": float(target_faith["random5_abs_score_change_mean"].median()),
        "target_fraction_top_exceeds_both_controls": float(target_faith["top_exceeds_both"].mean()),
    }
    faith_summary["pass"] = bool(
        faith_summary["target_median_top5_abs_score_change"] > faith_summary["target_median_low5_abs_score_change"]
        and faith_summary["target_median_top5_abs_score_change"] > faith_summary["target_median_random5_abs_score_change"]
        and faith_summary["target_fraction_top_exceeds_both_controls"] >= 0.50
    )
    (OUT / "faithfulness_summary.json").write_text(json.dumps(faith_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stability validation: small perturbations vs random permutation control.
    stab = stab.merge(faith[["case_id", "explained_label"]], on="case_id", how="left")
    stab["label_stable"] = stab["perturbed_pred_label"].astype(str) == stab["explained_label"].astype(str)
    stability_summary = {
        "input_perturbation": f"independent Gaussian noise, sd={STABILITY_NOISE_FRAC:.3%} of source independent-file feature SD",
        "median_explanation_spearman": float(stab["importance_spearman_original_vs_perturbed"].median()),
        "median_random_control_spearman": float(stab["random_permutation_spearman_control"].median()),
        "prediction_label_stability_rate": float(stab["label_stable"].mean()),
        "n_case_repetitions": int(len(stab)),
    }
    stability_summary["pass"] = bool(
        stability_summary["median_explanation_spearman"] >= 0.60
        and stability_summary["prediction_label_stability_rate"] >= 0.90
        and stability_summary["median_explanation_spearman"] > stability_summary["median_random_control_spearman"] + 0.30
    )
    (OUT / "stability_summary.json").write_text(json.dumps(stability_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Mechanism-consistency validation is anchored only where STEP03 had known source labels and an exact frequency check.
    orrow = source_cases[source_cases["true_label"] == "OR"].iloc[0]
    irrow = source_cases[source_cases["true_label"] == "IR"].iloc[0]
    mech_summary = {
        "scope": "Known source representatives only for exact BPFO/BPFI validation; target geometry is unknown, so target exact characteristic-frequency validation is prohibited.",
        "source_OR_step03_supported": bool(orrow["step03_mechanism_supported"]),
        "source_IR_step03_supported": bool(irrow["step03_mechanism_supported"]),
        "source_OR_top5_mechanism_related_count": int(orrow["top5_mechanism_related_count"]),
        "source_IR_top5_mechanism_related_count": int(irrow["top5_mechanism_related_count"]),
        "source_B_step03_note": "B007_0 was deliberately retained as a difficult/contradictory case: STEP03 BSF envelope support was false, so it is not used to manufacture a positive consistency claim.",
    }
    mech_summary["pass"] = bool(
        mech_summary["source_OR_step03_supported"] and mech_summary["source_IR_step03_supported"]
        and mech_summary["source_OR_top5_mechanism_related_count"] >= 1
        and mech_summary["source_IR_top5_mechanism_related_count"] >= 1
    )
    (OUT / "mechanism_consistency_summary.json").write_text(json.dumps(mech_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Contradictions/failures are kept, not hidden.
    contradictions = []
    brow = source_cases[source_cases["true_label"] == "B"].iloc[0]
    contradictions.append({
        "case_id": "source_B", "case_type": "source_known_label", "model_output": brow["final_model_explained_label"],
        "contradiction": "Known B source file is correctly class-labelled, but STEP03 exact BSF-envelope support was not robust.",
        "interpretation": "Feature/model explanation must not be turned into a claim that every B sample shows a clean BSF peak."
    })
    for tid in ["A", "B", "G", "O"]:
        r = target_mech[target_mech["target_id"].astype(str) == tid].iloc[0]
        contradictions.append({
            "case_id": tid, "case_type": "target_prediction", "model_output": str(target_final.loc[target_final["target_id"] == tid, "final_pred_label"].iloc[0]),
            "contradiction": f"Frozen model predicts OR, while mechanism-feature nearest source prototype is {r['mechanism_nearest_source_class']}.",
            "interpretation": "Target truth is unknown; disagreement is an uncertainty signal, not evidence that either label is the true class."
        })
    contradictions_df = pd.DataFrame(contradictions)
    contradictions_df.to_csv(OUT / "failure_and_contradiction_cases.csv", index=False, encoding="utf-8-sig")

    make_faithfulness_figure(faith)
    make_stability_figure(stab)
    make_case_heatmap(local_long, ["source_OR", "source_IR", "source_B", "source_N", "A", "B", "C", "O"])

    validation_types = {
        "faithfulness": bool(faith_summary["pass"]),
        "stability": bool(stability_summary["pass"]),
        "mechanism_consistency": bool(mech_summary["pass"]),
    }
    n_valid = int(sum(validation_types.values()))
    analysis = {
        "chosen_explanation_scope": [
            "pre-hoc: physically motivated time/spectrum/envelope feature design inherited from Problem 1",
            "post-hoc: final RandomForest global split importance and sample-dependent decision-path importance",
        ],
        "why_no_forced_transfer_process_explanation": "STEP09 selected A0_no_transfer; A2 CORAL and A3 instance weighting were rejected. There is no promoted target transform or adapted shared representation to explain as if it were used.",
        "faithfulness": faith_summary,
        "stability": stability_summary,
        "mechanism_consistency": mech_summary,
        "valid_validation_type_count": n_valid,
        "target_truth": "unknown",
        "causal_claim": "prohibited; importance and ablation measure model dependence/sensitivity, not physical causation",
        "limitations": [
            "The final target model is the frozen source RandomForest with no promoted domain adaptation; all A-P remain out-of-support/low-trust STEP10 predictions.",
            "RandomForest impurity/path importance can favor features used repeatedly by tree splits and can distribute importance across correlated features.",
            "Median-feature ablation creates synthetic feature combinations and tests model dependence, not a physically realizable intervention.",
            "Source known-label representatives were used to train the final all-source model; their final-model explanations are illustrative, not held-out performance estimates. OOF predictions are reported separately.",
            "Target bearing geometry and exact per-file speed are unavailable; source SKF6205 BPFO/BPFI/BSF frequencies are not imposed on target explanations.",
            "A-P explanations cannot establish target ground truth or target accuracy.",
        ],
    }
    (OUT / "problem4_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "explains_actual_step09_final_model": decision["selected_method"] == "A0_no_transfer" and bundle["method"] == "A0_no_transfer",
        "frozen_28_feature_interface": len(feats) == 28 and list(bundle["feature_columns"]) == feats,
        "all_16_target_files_explained": set(target_summary["target_id"].astype(str)) == set("ABCDEFGHIJKLMNOP") and len(target_summary) == 16,
        "source_known_label_cases_included": len(source_cases) == 4 and set(source_cases["true_label"]) == set(LABELS),
        "faithfulness_with_controls_saved": (OUT / "faithfulness_validation_by_case.csv").exists(),
        "stability_with_random_control_saved": (OUT / "stability_validation.csv").exists(),
        "mechanism_consistency_checked": (OUT / "mechanism_consistency_summary.json").exists(),
        "at_least_two_validation_types_pass": n_valid >= 2,
        "contradiction_cases_saved": len(contradictions_df) >= 3,
        "explanation_figures_saved": all((FIG / x).exists() for x in ["global_rf_feature_importance.png", "faithfulness_ablation_controls.png", "explanation_stability.png", "representative_case_path_importance.png"]),
        "target_truth_unknown": set(tgt["truth_status"].astype(str)) == {"unknown"},
        "target_accuracy_not_computed": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {"status": status, "checks": checks, "validation_types": validation_types, "valid_validation_type_count": n_valid}
    (OUT / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "step": "Q4_STEP11",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start": git_sha(),
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "source_sha256": sha256(SRC_CSV),
        "target_sha256": sha256(TGT_CSV),
        "step09_bundle_sha256": sha256(FINAL_BUNDLE),
        "step10_target_table_sha256": sha256(FINAL_TARGET_TABLE),
        "feature_count": len(feats),
        "random_seed": SEED,
        "path_importance_definition": "normalized impurity decrease of decision nodes traversed by each sample, averaged over 300 RF trees and file windows",
        "faithfulness_ablation_reference": "median of 56 source independent-file mean feature vectors",
        "stability_noise_fraction_source_file_sd": STABILITY_NOISE_FRAC,
        "target_labels_used": False,
        "target_accuracy_computed": False,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Q4 / STEP11 explainability",
        f"status={status}",
        "final_model=A0_no_transfer RandomForest from STEP09",
        f"validation_types={json.dumps(validation_types, ensure_ascii=False, sort_keys=True)}",
        f"valid_validation_type_count={n_valid}",
        f"faithfulness_target_median_top5_change={faith_summary['target_median_top5_abs_score_change']:.6f}",
        f"faithfulness_target_median_low5_change={faith_summary['target_median_low5_abs_score_change']:.6f}",
        f"faithfulness_target_median_random5_change={faith_summary['target_median_random5_abs_score_change']:.6f}",
        f"stability_median_spearman={stability_summary['median_explanation_spearman']:.6f}",
        f"stability_label_rate={stability_summary['prediction_label_stability_rate']:.6f}",
        "target_truth_known=False",
        "target_accuracy_reported=False",
        "causal_claim=False",
    ]
    (OUT / "result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
