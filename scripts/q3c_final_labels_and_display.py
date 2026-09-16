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
from sklearn.decomposition import PCA

import q3a_domain_diagnosis as q3a

ROOT = Path(__file__).resolve().parents[1]
Q1C = ROOT / "outputs" / "q1c_feature_extraction"
Q2C = ROOT / "outputs" / "q2c_failure_driven_improvement"
Q3B = ROOT / "outputs" / "q3b_unsupervised_transfer"
OUT = ROOT / "outputs" / "q3c_final_labels_and_display"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SRC_CSV = Q1C / "q2_source_raw.csv"
TGT_CSV = Q1C / "q2_target_raw.csv"
Q3_INTERFACE = Q2C / "q3_interface.json"
SOURCE_OOF = Q2C / "final_method_oof_file_predictions.csv"
FINAL_BUNDLE = Q3B / "models" / "final_transfer_bundle.joblib"
Q3B_A0 = Q3B / "target_A_P_A0_no_transfer.csv"
Q3B_FINAL = Q3B / "target_A_P_final_method.csv"
Q3B_SEED_STABILITY = Q3B / "target_A_P_seed_stability.csv"
Q3B_DECISION = Q3B / "final_selection_decision.json"

LABELS = ["OR", "IR", "B", "N"]
EXPECTED_TARGET_IDS = list("ABCDEFGHIJKLMNOP")
EPS = 1e-12
MECHANISM_FEATURES = [
    "td_kurtosis", "td_crest_factor", "td_impulse_factor",
    "fd_energy_ratio_0_500", "fd_energy_ratio_500_1500",
    "fd_energy_ratio_1500_3000", "fd_energy_ratio_3000_6000",
    "env_kurtosis", "env_energy_ratio_0_50", "env_energy_ratio_50_150",
    "env_energy_ratio_150_300", "env_energy_ratio_300_500",
]


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


def entropy_rows(P):
    P = np.clip(np.asarray(P, float), EPS, 1.0)
    return -np.sum(P * np.log(P), axis=1) / np.log(P.shape[1])


def predict_frozen(bundle, tgt, feats):
    pipe = bundle["model_or_pipeline"]
    P = pipe.predict_proba(tgt[feats].to_numpy(float))
    cls = list(pipe.named_steps["clf"].classes_)
    win, files = q3a.aggregate_probabilities(tgt, P, cls)
    files["target_id"] = files["independent_object_id"].map(lambda x: Path(str(x)).stem)
    files["truth_status"] = "unknown"
    return win, files.sort_values("target_id").reset_index(drop=True)


def source_uncertainty_reference(source_oof):
    pcols = [f"p_{x}" for x in LABELS]
    q = source_oof.copy()
    q["entropy_norm"] = entropy_rows(q[pcols].to_numpy(float))
    correct = q[q["pred_label_file"].astype(str) == q["class_label"].astype(str)].copy()
    if len(correct) < 10:
        raise RuntimeError("Insufficient correct source OOF files for uncertainty reference")
    return {
        "reference_population": "correct source OOF independent files only",
        "n_reference_files": int(len(correct)),
        "confidence_q10": float(correct["confidence_top1"].quantile(0.10)),
        "margin_q10": float(correct["margin_top1_top2"].quantile(0.10)),
        "entropy_q90": float(correct["entropy_norm"].quantile(0.90)),
        "use": "Descriptive source-derived uncertainty thresholds only; not calibrated target correctness probabilities.",
    }


def file_feature_space(src, tgt, feats):
    sf = src.groupby(["independent_object_id", "class_label"], as_index=False)[feats].mean()
    tf = tgt.groupby("independent_object_id", as_index=False)[feats].mean()
    mu = sf[feats].mean().to_numpy(float)
    sd = sf[feats].std(ddof=1).to_numpy(float)
    sd = np.where(sd > EPS, sd, 1.0)
    Zs = (sf[feats].to_numpy(float) - mu) / sd
    Zt = (tf[feats].to_numpy(float) - mu) / sd
    return sf, tf, Zs, Zt


def prototype_evidence(src, tgt, feats, pred_files):
    sf, tf, Zs, Zt = file_feature_space(src, tgt, feats)
    labels_s = sf["class_label"].astype(str).to_numpy()
    centroids, radii = {}, {}
    for lab in LABELS:
        z = Zs[labels_s == lab]
        c = z.mean(axis=0)
        centroids[lab] = c
        d = np.linalg.norm(z - c, axis=1) / math.sqrt(len(feats))
        radii[lab] = float(np.percentile(d, 95))
    midx = [feats.index(f) for f in MECHANISM_FEATURES]
    mech_centroids = {lab: centroids[lab][midx] for lab in LABELS}
    pred_map = dict(zip(pred_files["independent_object_id"].astype(str), pred_files["pred_label"].astype(str)))
    rows = []
    for i, oid in enumerate(tf["independent_object_id"].astype(str)):
        pred = pred_map[oid]
        all_d = {lab: float(np.linalg.norm(Zt[i] - centroids[lab]) / math.sqrt(len(feats))) for lab in LABELS}
        nearest = min(all_d, key=all_d.get)
        pred_ratio = float(all_d[pred] / max(radii[pred], EPS))
        mech_d = {lab: float(np.linalg.norm(Zt[i, midx] - mech_centroids[lab]) / math.sqrt(len(midx))) for lab in LABELS}
        mech_nearest = min(mech_d, key=mech_d.get)
        diffs = np.abs(Zt[i, midx] - mech_centroids[pred])
        order_small = np.argsort(diffs)[:3]
        order_large = np.argsort(diffs)[-3:][::-1]
        rows.append({
            "target_id": Path(oid).stem,
            "independent_object_id": oid,
            "pred_label": pred,
            **{f"distance_all28_to_{lab}": all_d[lab] for lab in LABELS},
            "nearest_all28_source_class": nearest,
            "pred_class_distance_over_source_p95": pred_ratio,
            **{f"mechanism_distance_to_{lab}": mech_d[lab] for lab in LABELS},
            "mechanism_nearest_source_class": mech_nearest,
            "mechanism_pred_class_similarity_index": float(1.0 / (1.0 + mech_d[pred])),
            "mechanism_model_prototype_agree": bool(pred == mech_nearest),
            "three_most_prototype_consistent_mechanism_features": ";".join(f"{MECHANISM_FEATURES[j]}(|zdiff|={diffs[j]:.3f})" for j in order_small),
            "three_largest_mechanism_feature_deviations": ";".join(f"{MECHANISM_FEATURES[j]}(|zdiff|={diffs[j]:.3f})" for j in order_large),
            "evidence_role": "explanatory side evidence only; not target truth",
        })
    return pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True), sf, tf, Zs, Zt


def make_transfer_display(sf, tf, Zs, Zt, final_table):
    pca = PCA(n_components=2, random_state=0).fit(Zs)
    Cs, Ct = pca.transform(Zs), pca.transform(Zt)
    source_labels = sf["class_label"].astype(str).to_numpy()
    pred_map = dict(zip(final_table["target_id"], final_table["final_pred_label"]))
    target_ids = [Path(x).stem for x in tf["independent_object_id"].astype(str)]
    target_labels = [pred_map[x] for x in target_ids]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    titles = ["Before adaptation: frozen source model", "After STEP09 selection: A0 no-transfer (unchanged)"]
    for ax, title in zip(axes, titles):
        for lab in LABELS:
            m = source_labels == lab
            ax.scatter(Cs[m, 0], Cs[m, 1], label=f"source truth {lab}", alpha=0.55, s=34)
        for lab in LABELS:
            m = np.array(target_labels) == lab
            if np.any(m):
                ax.scatter(Ct[m, 0], Ct[m, 1], marker="x", s=70, linewidths=1.7, label=f"target predicted {lab}")
        for j, tid in enumerate(target_ids):
            ax.annotate(tid, (Ct[j, 0], Ct[j, 1]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("PC1 (source-fitted 28-feature space)")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1].text(0.02, 0.02, "A2/A3 were not promoted; no target transform is applied.", transform=axes[1].transAxes, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("STEP10 transfer display: source symbols are true labels; target x symbols are model predictions")
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.savefig(FIG / "transfer_display_before_after.png", dpi=180)
    plt.close(fig)
    return [float(x) for x in pca.explained_variance_ratio_]


def make_prototype_heatmap(evidence):
    vals = evidence[[f"mechanism_distance_to_{x}" for x in LABELS]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    im = ax.imshow(vals, aspect="auto")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(evidence)), evidence["target_id"])
    ax.set_xlabel("Source class prototype (mechanism-related features)")
    ax.set_ylabel("Target file")
    ax.set_title("A-P mechanism-feature distance to source class prototypes\n(smaller is more similar; explanatory only)")
    fig.colorbar(im, ax=ax, label="standardized prototype distance")
    fig.tight_layout()
    fig.savefig(FIG / "mechanism_prototype_distance_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    interface = json.loads(Q3_INTERFACE.read_text(encoding="utf-8"))
    decision = json.loads(Q3B_DECISION.read_text(encoding="utf-8"))
    feats = list(interface["feature_columns"])
    src, tgt = pd.read_csv(SRC_CSV), pd.read_csv(TGT_CSV)
    source_oof = pd.read_csv(SOURCE_OOF)
    a0_saved, final_saved = pd.read_csv(Q3B_A0), pd.read_csv(Q3B_FINAL)
    seed_stability = pd.read_csv(Q3B_SEED_STABILITY)
    bundle = joblib.load(FINAL_BUNDLE)
    if decision["selected_method"] != "A0_no_transfer" or bundle["method"] != "A0_no_transfer" or bundle["setting"] != "frozen_rf":
        raise RuntimeError("STEP09 frozen method/setting changed")
    if list(bundle["feature_columns"]) != feats or len(feats) != 28:
        raise RuntimeError("Frozen STEP09 feature interface changed")
    if set(tgt["truth_status"].astype(str).unique()) != {"unknown"}:
        raise RuntimeError("Target truth boundary changed")

    win, files = predict_frozen(bundle, tgt, feats)
    files = files.rename(columns={"pred_label": "final_pred_label", "confidence": "model_score_top1", "margin": "score_margin_top1_top2"})
    files["score_interpretation"] = "uncalibrated class score; not a true correctness probability"
    chk = files.merge(final_saved[["target_id", "pred_label", "p_OR", "p_IR", "p_B", "p_N"]], on="target_id", suffixes=("", "_step09"))
    for lab in LABELS:
        if not np.allclose(chk[f"p_{lab}"], chk[f"p_{lab}_step09"], atol=1e-12, rtol=0):
            raise RuntimeError(f"STEP10 probability mismatch vs STEP09 for {lab}")
    if not (chk["final_pred_label"].astype(str) == chk["pred_label"].astype(str)).all():
        raise RuntimeError("STEP10 label mismatch vs STEP09")

    diff = a0_saved[["target_id", "pred_label", "p_OR", "p_IR", "p_B", "p_N"]].merge(
        files[["target_id", "final_pred_label", "p_OR", "p_IR", "p_B", "p_N"]], on="target_id", suffixes=("_A0", "_final"))
    diff["label_changed_A0_to_final"] = diff["pred_label"].astype(str) != diff["final_pred_label"].astype(str)
    diff["probability_L1_difference"] = sum(np.abs(diff[f"p_{lab}_A0"] - diff[f"p_{lab}_final"]) for lab in LABELS)
    diff["interpretation"] = "STEP09 selected A0; final strategy intentionally has no adaptation change"
    diff.to_csv(OUT / "no_transfer_vs_final_difference.csv", index=False, encoding="utf-8-sig")

    pred_cols = [c for c in seed_stability.columns if c.startswith("pred_seed_")]
    seed_stability["vote_agreement_rate"] = seed_stability[pred_cols].apply(lambda r: r.value_counts(normalize=True).iloc[0], axis=1)
    seed_stability["vote_label"] = seed_stability[pred_cols].apply(lambda r: r.value_counts().index[0], axis=1)
    seed_stability["vote_interpretation"] = "agreement across three fixed RF random seeds; not correctness probability"

    evidence, sf, tf, Zs, Zt = prototype_evidence(src, tgt, feats, files.rename(columns={"final_pred_label":"pred_label"}))
    evidence.to_csv(OUT / "mechanism_side_evidence.csv", index=False, encoding="utf-8-sig")
    ref = source_uncertainty_reference(source_oof)
    (OUT / "source_reference_uncertainty_thresholds.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")

    table = files[["target_id", "independent_object_id", "truth_status", "n_windows", "p_OR", "p_IR", "p_B", "p_N", "final_pred_label", "model_score_top1", "score_margin_top1_top2", "entropy_norm", "window_consistency", "score_interpretation"]].copy()
    table = table.merge(seed_stability[["target_id", "vote_label", "vote_agreement_rate", "all_seed_labels_agree"]], on="target_id", how="left")
    table = table.merge(evidence[["target_id", "nearest_all28_source_class", "pred_class_distance_over_source_p95", "mechanism_nearest_source_class", "mechanism_pred_class_similarity_index", "mechanism_model_prototype_agree", "three_most_prototype_consistent_mechanism_features", "three_largest_mechanism_feature_deviations"]], on="target_id", how="left")
    table["flag_low_source_like_score"] = table["model_score_top1"] < ref["confidence_q10"]
    table["flag_small_margin"] = table["score_margin_top1_top2"] < ref["margin_q10"]
    table["flag_high_entropy"] = table["entropy_norm"] > ref["entropy_q90"]
    table["flag_outside_predicted_class_p95_support"] = table["pred_class_distance_over_source_p95"] > 1.0
    table["flag_prototype_conflict"] = ~table["mechanism_model_prototype_agree"].astype(bool)
    global_collapse = float(table["final_pred_label"].value_counts(normalize=True).max()) >= 0.875
    table["global_collapse_warning"] = global_collapse
    local_count = table[["flag_low_source_like_score", "flag_small_margin", "flag_high_entropy"]].sum(axis=1)
    table["uncertainty_level"] = np.where(table["flag_prototype_conflict"] | (table["pred_class_distance_over_source_p95"] >= 5.0) | (local_count >= 2), "very_low_trust", np.where(table["flag_outside_predicted_class_p95_support"] | (local_count >= 1), "low_trust", "source_like"))
    def reasons(r):
        out = []
        if r["flag_low_source_like_score"]: out.append("top score below source-correct OOF q10")
        if r["flag_small_margin"]: out.append("margin below source-correct OOF q10")
        if r["flag_high_entropy"]: out.append("entropy above source-correct OOF q90")
        if r["flag_outside_predicted_class_p95_support"]: out.append("outside predicted-class source p95 support")
        if r["flag_prototype_conflict"]: out.append("mechanism prototype nearest class disagrees with model")
        if global_collapse: out.append("global A-P output collapse warning")
        return "; ".join(out) if out else "no source-derived warning triggered"
    table["uncertainty_reasons"] = table.apply(reasons, axis=1)
    table["target_accuracy_available"] = False
    table = table.sort_values("target_id").reset_index(drop=True)
    table.to_csv(OUT / "final_A_P_label_table.csv", index=False, encoding="utf-8-sig")
    table[["target_id", "final_pred_label"]].to_csv(OUT / "final_A_P_labels_for_answer.csv", index=False, encoding="utf-8-sig")
    low = table[table["uncertainty_level"].isin(["low_trust", "very_low_trust"])].copy()
    low.to_csv(OUT / "low_confidence_cases.csv", index=False, encoding="utf-8-sig")

    explained_var = make_transfer_display(sf, tf, Zs, Zt, table)
    make_prototype_heatmap(evidence)
    analysis = {
        "frozen_step09_method": bundle["method"], "frozen_step09_setting": bundle["setting"],
        "aggregation_rule": "arithmetic mean of 15 window class scores per A-P file, then argmax",
        "label_mapping": LABELS,
        "target_predictions": {r.target_id:r.final_pred_label for r in table[["target_id","final_pred_label"]].itertuples(index=False)},
        "target_prediction_counts": {k:int(v) for k,v in table["final_pred_label"].value_counts().reindex(LABELS, fill_value=0).to_dict().items()},
        "global_collapse_warning": bool(global_collapse),
        "all_files_seed_vote_agreement": bool((table["vote_agreement_rate"] == 1.0).all()),
        "mean_window_consistency": float(table["window_consistency"].mean()),
        "low_trust_file_count": int(len(low)),
        "very_low_trust_files": table.loc[table["uncertainty_level"] == "very_low_trust", "target_id"].tolist(),
        "source_fitted_pca_explained_variance_ratio": explained_var,
        "migration_display_note": "STEP09 promoted no adaptation; before/final panels are intentionally identical. This is method-selection evidence, not target-correctness evidence.",
        "mechanism_evidence_note": "Prototype similarities use impact/spectral/envelope features and source true classes only; explanatory, not target truth.",
        "score_note": "RF class scores are uncalibrated and not real correctness probabilities.",
        "target_accuracy_reported": False,
    }
    (OUT / "problem3_result_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "step":"Q3C_STEP10", "generated_utc":datetime.now(timezone.utc).isoformat(), "git_sha_at_run_start":git_sha(),
        "python":sys.version, "sklearn":sklearn.__version__, "platform":platform.platform(),
        "source_sha256":sha256(SRC_CSV), "target_sha256":sha256(TGT_CSV), "q3_interface_sha256":sha256(Q3_INTERFACE),
        "step09_final_bundle_sha256":sha256(FINAL_BUNDLE), "step09_decision_sha256":sha256(Q3B_DECISION),
        "feature_count":len(feats), "mechanism_feature_count":len(MECHANISM_FEATURES),
        "aggregation_rule":"mean window class scores by independent_object_id; argmax", "manual_label_adjustment":False,
        "target_labels_used":False, "target_accuracy_computed":False, "elapsed_seconds":float(time.perf_counter()-t0),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = table["target_id"].tolist()
    checks = {
        "step09_frozen_bundle_only": bundle["method"] == "A0_no_transfer" and bundle["setting"] == "frozen_rf",
        "exact_A_to_P_once_each": ids == EXPECTED_TARGET_IDS and table["target_id"].nunique() == 16,
        "valid_label_mapping": set(table["final_pred_label"]).issubset(set(LABELS)),
        "fixed_file_aggregation": bool((table["n_windows"] == 15).all()),
        "step09_predictions_reproduced": bool((chk["final_pred_label"].astype(str) == chk["pred_label"].astype(str)).all()),
        "no_manual_ratio_adjustment": True,
        "scores_marked_uncalibrated": bool(table["score_interpretation"].str.contains("not a true correctness probability", regex=False).all()),
        "no_transfer_vs_final_recorded": len(diff) == 16 and float(diff["probability_L1_difference"].max()) < 1e-12,
        "cross_seed_vote_recorded": bool(table["vote_agreement_rate"].notna().all()),
        "mechanism_side_evidence_each_file": len(evidence) == 16 and evidence["target_id"].nunique() == 16,
        "uncertainty_flags_source_derived": True,
        "transfer_visualization_saved": (FIG / "transfer_display_before_after.png").exists(),
        "prototype_visualization_saved": (FIG / "mechanism_prototype_distance_heatmap.png").exists(),
        "evidence_identity_clear": True,
        "target_truth_unknown": bool((table["truth_status"].astype(str) == "unknown").all()),
        "target_accuracy_not_computed": not bool(table["target_accuracy_available"].any()),
    }
    status = "PASS" if all(bool(v) for v in checks.values()) else "FAIL"
    (OUT / "validation_report.json").write_text(json.dumps({"status":status,"checks":checks,"summary":analysis}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Q3C / STEP10 final A-P labels and transfer display", f"status={status}", f"frozen_method={bundle['method']}", "aggregation=mean of 15 window scores then argmax", f"prediction_counts={json.dumps(analysis['target_prediction_counts'], ensure_ascii=False)}", f"all_seed_votes_agree={analysis['all_files_seed_vote_agreement']}", f"low_trust_file_count={analysis['low_trust_file_count']}", f"very_low_trust_files={','.join(analysis['very_low_trust_files'])}", f"global_collapse_warning={analysis['global_collapse_warning']}", "target_accuracy_reported=False", "manual_label_adjustment=False"]
    (OUT / "result_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    if status != "PASS": raise SystemExit(2)


if __name__ == "__main__":
    main()
