from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, platform, subprocess, sys, time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "outputs" / "q1c_feature_extraction"
BASE_DIR = ROOT / "outputs" / "q2b_minimal_baseline"
SRC_CSV = IN_DIR / "q2_source_raw.csv"
TGT_CSV = IN_DIR / "q2_target_raw.csv"
INTERFACE_JSON = IN_DIR / "q2_interface.json"
OUT = ROOT / "outputs" / "q2c_failure_driven_improvement"
MODELS = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 20260916
LABELS = ["OR", "IR", "B", "N"]
LOADS = [0, 1, 2, 3]
C_GRID = [0.1, 1.0, 10.0]
RF_GRID = [
    {"max_depth": 10, "min_samples_leaf": 1},
    {"max_depth": 10, "min_samples_leaf": 3},
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 3},
]

EXPERIMENTS = [
    {
        "experiment_id": "E0_baseline_logreg_fullbalance",
        "hypothesis": "Reference baseline from STEP06.",
        "failure_reason": "Provides the frozen comparator; OR014@6 was repeatedly confused with B and 0 hp was the weakest condition.",
        "change": "None: 28 common features, L2 logistic regression, full inverse class-file balancing.",
        "why_may_help": "Comparator only.",
        "added_assumption": "Linear decision surfaces after standardization.",
        "fair_comparison": "Same STEP05 load-group splits, seed, features, aggregation and training-boundary preprocessing.",
        "stop_rule": "No iterative change; fixed comparator.",
        "model": "logreg", "class_alpha": 1.0, "corr_threshold": None, "complexity_rank": 0,
    },
    {
        "experiment_id": "E1_logreg_sqrt_class_balance",
        "hypothesis": "Full inverse class balancing may over-emphasize B/N and contribute to OR->B false positives.",
        "failure_reason": "STEP06 had four OR files predicted as B; B recall=1.0 but B precision=0.75.",
        "change": "Only class-file weight exponent changes from alpha=1.0 to alpha=0.5; per-file normalization is unchanged.",
        "why_may_help": "Reduces minority over-weighting while still compensating imbalance, potentially improving OR/B boundary stability.",
        "added_assumption": "Square-root inverse class frequency is sufficient imbalance correction.",
        "fair_comparison": "Same 28 features, L2 logistic model, C grid, splits and seed; only weighting strength changes.",
        "stop_rule": "Promote only by predeclared inner-validation rule; outer tests are not used for promotion.",
        "model": "logreg", "class_alpha": 0.5, "corr_threshold": None, "complexity_rank": 1,
    },
    {
        "experiment_id": "E2_logreg_corr95",
        "hypothesis": "Highly redundant amplitude/shape features may destabilize linear coefficients across loads.",
        "failure_reason": "Systematic OR014@6->B errors persisted at all four loads, suggesting a stable boundary problem rather than one random window.",
        "change": "Only add unsupervised absolute-correlation pruning at |r|>0.95 fitted inside each training boundary.",
        "why_may_help": "Removes near-duplicate predictors and can reduce coefficient instability without adding target-specific assumptions.",
        "added_assumption": "One feature from a highly correlated pair is sufficient for the linear baseline.",
        "fair_comparison": "Same logistic model, class weights, C grid, splits and seed; pruning is fitted on training data only.",
        "stop_rule": "Threshold fixed at 0.95 before running; no threshold search.",
        "model": "logreg", "class_alpha": 1.0, "corr_threshold": 0.95, "complexity_rank": 2,
    },
    {
        "experiment_id": "E3_random_forest",
        "hypothesis": "OR/B separation may require modest nonlinear interactions among the same 28 interpretable features.",
        "failure_reason": "Four OR014@6 files were predicted as B by the linear baseline under all loads.",
        "change": "Only classifier family changes to 300-tree Random Forest; same common features and sample-weight rule.",
        "why_may_help": "Trees can represent nonlinear feature interactions without imposing an artificial feature order.",
        "added_assumption": "Piecewise nonlinear interactions generalize across held-out loads.",
        "fair_comparison": "Same 28 features, splits, seed, file/class weights and file-level probability aggregation; four RF settings were frozen in STEP05 scope.",
        "stop_rule": "No model-family expansion beyond this RF grid; promotion uses inner validation only.",
        "model": "rf", "class_alpha": 1.0, "corr_threshold": None, "complexity_rank": 3,
    },
]

# Precommitted promotion rule: selection NEVER reads outer-test metrics.
PROMOTION = {
    "min_inner_macro_f1_gain": 0.01,
    "max_mean_min_recall_drop": 0.05,
    "min_outer_contexts_nonworse": 3,
    "context_nonworse_tolerance": 0.005,
    "tie_order": ["inner_macro_f1", "inner_balanced_accuracy", "inner_min_class_recall", "lower_complexity_rank"],
}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_model():
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


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


def file_meta(df):
    g = df[["independent_object_id", "class_label", "load_hp"]].drop_duplicates().copy()
    if g["independent_object_id"].duplicated().any():
        raise RuntimeError("independent_object_id maps to multiple file metadata rows")
    return g.sort_values("independent_object_id").reset_index(drop=True)


def make_weights(train_df, alpha):
    # Each file contributes equal total mass before class balancing. Class factor nc^-alpha
    # is normalized so total sample-weight mass equals number of independent files.
    nwin = train_df.groupby("independent_object_id").size().to_dict()
    meta = train_df[["independent_object_id", "class_label"]].drop_duplicates()
    class_n = meta.groupby("class_label")["independent_object_id"].nunique().to_dict()
    n_files = int(meta["independent_object_id"].nunique())
    raw_factor = {c: float(class_n[c]) ** (-float(alpha)) for c in LABELS}
    total_file_mass_raw = sum(class_n[c] * raw_factor[c] for c in LABELS)
    norm = n_files / total_file_mass_raw
    out = []
    for _, r in train_df.iterrows():
        out.append(norm * raw_factor[r["class_label"]] / nwin[r["independent_object_id"]])
    w = np.asarray(out, float)
    if not np.isclose(w.sum(), n_files, atol=1e-9):
        raise RuntimeError("sample-weight normalization failed")
    return w


def corr_prune_features(train_df, feats, threshold):
    if threshold is None:
        return list(feats)
    X = train_df[feats].copy()
    med = X.median(axis=0)
    X = X.fillna(med)
    corr = X.corr().abs()
    keep, dropped = [], []
    for f in feats:
        if any(float(corr.loc[f, k]) > float(threshold) for k in keep):
            dropped.append(f)
        else:
            keep.append(f)
    if len(keep) < 4:
        raise RuntimeError("correlation pruning removed too many features")
    return keep


def config_key(cfg):
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def config_grid(exp):
    if exp["model"] == "logreg":
        return [{"C": float(c)} for c in C_GRID]
    return [dict(x, n_estimators=300, max_features="sqrt") for x in RF_GRID]


def build_pipeline(exp, cfg):
    if exp["model"] == "logreg":
        clf = LogisticRegression(
            C=float(cfg["C"]), penalty="l2", solver="lbfgs", max_iter=5000,
            random_state=SEED,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]), max_depth=cfg["max_depth"],
            min_samples_leaf=int(cfg["min_samples_leaf"]), max_features=cfg["max_features"],
            random_state=SEED, n_jobs=1,
        )
    # Keep the same deterministic preprocessing family for the fair classifier comparison.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def fit_model(train_df, base_feats, exp, cfg):
    used = corr_prune_features(train_df, base_feats, exp["corr_threshold"])
    pipe = build_pipeline(exp, cfg)
    X = train_df[used].to_numpy(float)
    y = train_df["class_label"].astype(str).to_numpy()
    w = make_weights(train_df, exp["class_alpha"])
    pipe.fit(X, y, clf__sample_weight=w)
    return pipe, used


def predict_window_table(pipe, used, df, exp_id, context, cfg):
    p = pipe.predict_proba(df[used].to_numpy(float))
    classes = list(pipe.named_steps["clf"].classes_)
    out = df[["window_id", "independent_object_id", "class_label", "load_hp"]].copy().reset_index(drop=True)
    out["experiment_id"] = exp_id
    out["context"] = str(context)
    out["config"] = config_key(cfg)
    for lab in LABELS:
        out[f"p_{lab}"] = p[:, classes.index(lab)]
    out["pred_label_window"] = out[[f"p_{x}" for x in LABELS]].idxmax(axis=1).str.replace("p_", "", regex=False)
    return out


def aggregate_files(win):
    pcols = [f"p_{x}" for x in LABELS]
    keys = ["experiment_id", "context", "config", "independent_object_id", "class_label", "load_hp"]
    g = win.groupby(keys, as_index=False)[pcols].mean()
    counts = win.groupby("independent_object_id").size().rename("n_windows").reset_index()
    g = g.merge(counts, on="independent_object_id", how="left")
    g["pred_label_file"] = g[pcols].idxmax(axis=1).str.replace("p_", "", regex=False)
    arr = np.sort(g[pcols].to_numpy(float), axis=1)
    g["confidence_top1"] = np.max(g[pcols].to_numpy(float), axis=1)
    g["margin_top1_top2"] = arr[:, -1] - arr[:, -2]
    return g


def choose_cfg_for_context(df, feats, exp, outer_test_load):
    dev_loads = [x for x in LOADS if x != outer_test_load]
    rows = []
    for cfg in config_grid(exp):
        for val_load in dev_loads:
            tr_loads = [x for x in dev_loads if x != val_load]
            tr = df[df["load_hp"].isin(tr_loads)].copy()
            va = df[df["load_hp"] == val_load].copy()
            if set(tr["independent_object_id"]) & set(va["independent_object_id"]):
                raise RuntimeError("inner group leakage")
            pipe, used = fit_model(tr, feats, exp, cfg)
            wf = predict_window_table(pipe, used, va, exp["experiment_id"], f"outer{outer_test_load}_inner{val_load}", cfg)
            ff = aggregate_files(wf)
            m = metric_dict(ff["class_label"], ff["pred_label_file"])
            rows.append({
                "experiment_id": exp["experiment_id"], "outer_test_load": outer_test_load,
                "inner_val_load": val_load, "config": config_key(cfg), "n_features_used": len(used), **m,
            })
    detail = pd.DataFrame(rows)
    sm = detail.groupby(["experiment_id", "outer_test_load", "config"], as_index=False).agg(
        inner_macro_f1=("macro_f1", "mean"),
        inner_balanced_accuracy=("balanced_accuracy", "mean"),
        inner_min_class_recall=("min_class_recall", "mean"),
        mean_features_used=("n_features_used", "mean"),
    )
    sm = sm.sort_values(
        ["inner_macro_f1", "inner_balanced_accuracy", "inner_min_class_recall", "config"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = json.loads(sm.iloc[0]["config"])
    return best, detail, sm


def run_all_inner(df, feats):
    all_detail, all_cfg_summary, chosen_rows = [], [], []
    chosen = {}
    for exp in EXPERIMENTS:
        eid = exp["experiment_id"]
        chosen[eid] = {}
        for outer in LOADS:
            cfg, det, sm = choose_cfg_for_context(df, feats, exp, outer)
            chosen[eid][outer] = cfg
            all_detail.append(det); all_cfg_summary.append(sm)
            top = sm.iloc[0]
            chosen_rows.append({
                "experiment_id": eid, "outer_test_load": outer, "selected_config": config_key(cfg),
                "inner_macro_f1": float(top["inner_macro_f1"]),
                "inner_balanced_accuracy": float(top["inner_balanced_accuracy"]),
                "inner_min_class_recall": float(top["inner_min_class_recall"]),
                "mean_features_used": float(top["mean_features_used"]),
            })
    return pd.concat(all_detail, ignore_index=True), pd.concat(all_cfg_summary, ignore_index=True), pd.DataFrame(chosen_rows), chosen


def select_experiment(chosen_inner):
    exp_meta = {x["experiment_id"]: x for x in EXPERIMENTS}
    agg = chosen_inner.groupby("experiment_id", as_index=False).agg(
        inner_macro_f1=("inner_macro_f1", "mean"),
        inner_balanced_accuracy=("inner_balanced_accuracy", "mean"),
        inner_min_class_recall=("inner_min_class_recall", "mean"),
        mean_features_used=("mean_features_used", "mean"),
    )
    base_id = EXPERIMENTS[0]["experiment_id"]
    base = agg[agg["experiment_id"] == base_id].iloc[0]
    rows = []
    for _, r in agg.iterrows():
        eid = r["experiment_id"]
        if eid == base_id:
            nonworse = 4
            eligible = True
        else:
            c = chosen_inner[chosen_inner["experiment_id"] == eid].sort_values("outer_test_load")
            b = chosen_inner[chosen_inner["experiment_id"] == base_id].sort_values("outer_test_load")
            nonworse = int(np.sum(c["inner_macro_f1"].to_numpy() >= b["inner_macro_f1"].to_numpy() - PROMOTION["context_nonworse_tolerance"]))
            eligible = (
                float(r["inner_macro_f1"] - base["inner_macro_f1"]) >= PROMOTION["min_inner_macro_f1_gain"]
                and float(r["inner_min_class_recall"] - base["inner_min_class_recall"]) >= -PROMOTION["max_mean_min_recall_drop"]
                and nonworse >= PROMOTION["min_outer_contexts_nonworse"]
            )
        rows.append({
            **r.to_dict(),
            "inner_macro_gain_vs_baseline": float(r["inner_macro_f1"] - base["inner_macro_f1"]),
            "inner_bal_acc_gain_vs_baseline": float(r["inner_balanced_accuracy"] - base["inner_balanced_accuracy"]),
            "inner_min_recall_gain_vs_baseline": float(r["inner_min_class_recall"] - base["inner_min_class_recall"]),
            "contexts_nonworse_vs_baseline": nonworse,
            "eligible_for_promotion": bool(eligible),
            "complexity_rank": int(exp_meta[eid]["complexity_rank"]),
        })
    tab = pd.DataFrame(rows)
    candidates = tab[(tab["experiment_id"] != base_id) & tab["eligible_for_promotion"]].copy()
    if len(candidates) == 0:
        selected = base_id
        reason = "No improvement hypothesis met the predeclared inner-validation promotion rule; retain baseline."
    else:
        candidates = candidates.sort_values(
            ["inner_macro_f1", "inner_balanced_accuracy", "inner_min_class_recall", "complexity_rank"],
            ascending=[False, False, False, True],
        )
        selected = str(candidates.iloc[0]["experiment_id"])
        reason = "Selected strictly from predeclared inner-validation criteria; outer-test metrics were not read by selection code."
    tab["selected_by_inner_only"] = tab["experiment_id"] == selected
    return selected, reason, tab


def run_outer_for_experiment(df, feats, exp, chosen_cfg):
    all_w, all_f, mrows = [], [], []
    for outer in LOADS:
        dev = df[df["load_hp"] != outer].copy()
        te = df[df["load_hp"] == outer].copy()
        if set(dev["independent_object_id"]) & set(te["independent_object_id"]):
            raise RuntimeError("outer group leakage")
        cfg = chosen_cfg[exp["experiment_id"]][outer]
        pipe, used = fit_model(dev, feats, exp, cfg)
        wf = predict_window_table(pipe, used, te, exp["experiment_id"], f"outer_test_{outer}", cfg)
        ff = aggregate_files(wf)
        m = metric_dict(ff["class_label"], ff["pred_label_file"])
        all_w.append(wf); all_f.append(ff)
        mrows.append({"experiment_id": exp["experiment_id"], "outer_test_load": outer, "config": config_key(cfg), "n_features_used": len(used), **m})
    w = pd.concat(all_w, ignore_index=True)
    f = pd.concat(all_f, ignore_index=True)
    pooled = metric_dict(f["class_label"], f["pred_label_file"])
    return w, f, pd.DataFrame(mrows), pooled


def select_global_cfg(df, feats, exp):
    rows = []
    for cfg in config_grid(exp):
        for val_load in LOADS:
            tr = df[df["load_hp"] != val_load].copy()
            va = df[df["load_hp"] == val_load].copy()
            pipe, used = fit_model(tr, feats, exp, cfg)
            wf = predict_window_table(pipe, used, va, exp["experiment_id"], f"global_lolo_{val_load}", cfg)
            ff = aggregate_files(wf)
            m = metric_dict(ff["class_label"], ff["pred_label_file"])
            rows.append({"config": config_key(cfg), "val_load": val_load, "n_features_used": len(used), **m})
    det = pd.DataFrame(rows)
    sm = det.groupby("config", as_index=False).agg(
        mean_macro_f1=("macro_f1", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_min_class_recall=("min_class_recall", "mean"),
        mean_features_used=("n_features_used", "mean"),
    ).sort_values(["mean_macro_f1", "mean_balanced_accuracy", "mean_min_class_recall", "config"], ascending=[False, False, False, True]).reset_index(drop=True)
    return json.loads(sm.iloc[0]["config"]), det, sm


def complexity_and_timing(model_bundle, df):
    pipe = model_bundle["pipeline"]
    feats = model_bundle["feature_columns"]
    clf = pipe.named_steps["clf"]
    model_path = MODELS / "final_source_model.joblib"
    model_bytes = int(model_path.stat().st_size)
    if isinstance(clf, LogisticRegression):
        complexity = {
            "classifier": "LogisticRegression",
            "trainable_parameter_count": int(clf.coef_.size + clf.intercept_.size),
            "n_trees": None, "total_tree_nodes": None, "total_tree_leaves": None,
        }
    else:
        complexity = {
            "classifier": "RandomForestClassifier",
            "trainable_parameter_count": None,
            "n_trees": int(len(clf.estimators_)),
            "total_tree_nodes": int(sum(t.tree_.node_count for t in clf.estimators_)),
            "total_tree_leaves": int(sum(t.tree_.n_leaves for t in clf.estimators_)),
        }

    groups = [g for _, g in df.groupby("independent_object_id")]
    # Warm-up.
    for g in groups[:3]:
        _ = pipe.predict_proba(g[feats].to_numpy(float)).mean(axis=0)
    durations = []
    repeats = 10
    for _ in range(repeats):
        for g in groups:
            t0 = time.perf_counter()
            _ = pipe.predict_proba(g[feats].to_numpy(float)).mean(axis=0)
            durations.append((time.perf_counter() - t0) * 1000.0)
    return {
        **complexity,
        "input_feature_count": int(len(feats)),
        "serialized_model_bytes": model_bytes,
        "inference_ms_per_file_median": float(np.median(durations)),
        "inference_ms_per_file_mean": float(np.mean(durations)),
        "inference_ms_per_file_p95": float(np.percentile(durations, 95)),
        "timing_repeats_per_file": repeats,
        "timing_batch": "one independent file at a time; all its precomputed feature windows",
        "timing_includes": "pipeline imputation/scaling, classifier predict_proba, mean probability aggregation",
        "timing_excludes": "raw MAT loading, signal windowing, feature extraction, CSV I/O",
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def manual_macro_f1(y_true, y_pred):
    vals = []
    for lab in LABELS:
        tp = sum((a == lab and b == lab) for a, b in zip(y_true, y_pred))
        fp = sum((a != lab and b == lab) for a, b in zip(y_true, y_pred))
        fn = sum((a == lab and b != lab) for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(vals))


def main():
    started = time.perf_counter()
    interface = json.loads(INTERFACE_JSON.read_text(encoding="utf-8"))
    feats = list(interface["feature_columns"])
    forbidden = set(interface["forbidden_as_model_features"])
    if forbidden.intersection(feats):
        raise RuntimeError(f"Leakage feature in interface: {forbidden.intersection(feats)}")
    df = pd.read_csv(SRC_CSV)
    if len(df) != 400 or len(feats) != 28:
        raise RuntimeError("STEP04 input shape changed")
    ft = file_meta(df)
    counts = ft["class_label"].value_counts().to_dict()
    if len(ft) != 56 or counts != {"OR": 28, "B": 12, "IR": 12, "N": 4}:
        raise RuntimeError(f"Unexpected source inventory: {len(ft)}, {counts}")
    for load in LOADS:
        c = ft[ft["load_hp"] == load]["class_label"].value_counts().to_dict()
        if c != {"OR": 7, "B": 3, "IR": 3, "N": 1}:
            raise RuntimeError(f"Load {load} coverage changed: {c}")

    # Record hypotheses before any improvement result is evaluated.
    hypo_cols = ["experiment_id", "hypothesis", "failure_reason", "change", "why_may_help", "added_assumption", "fair_comparison", "stop_rule"]
    pd.DataFrame([{k: e[k] for k in hypo_cols} for e in EXPERIMENTS]).to_csv(OUT / "improvement_hypotheses.csv", index=False, encoding="utf-8-sig")
    (OUT / "precommitted_selection_rule.json").write_text(json.dumps(PROMOTION, ensure_ascii=False, indent=2), encoding="utf-8")

    # Phase A: development only. Every candidate is screened only by inner validation.
    inner_detail, inner_cfg_summary, chosen_inner, chosen_cfg = run_all_inner(df, feats)
    inner_detail.to_csv(OUT / "inner_all_experiments_detail.csv", index=False, encoding="utf-8-sig")
    inner_cfg_summary.to_csv(OUT / "inner_config_summary.csv", index=False, encoding="utf-8-sig")
    chosen_inner.to_csv(OUT / "inner_selected_config_by_context.csv", index=False, encoding="utf-8-sig")
    selected_id, selection_reason, experiment_summary = select_experiment(chosen_inner)
    experiment_summary.to_csv(OUT / "experiment_selection_summary.csv", index=False, encoding="utf-8-sig")
    failed = experiment_summary[(experiment_summary["experiment_id"] != selected_id) & (experiment_summary["experiment_id"] != EXPERIMENTS[0]["experiment_id"])].copy()
    failed["failure_status"] = np.where(failed["eligible_for_promotion"], "eligible_but_not_selected", "failed_predeclared_promotion_rule")
    failed.to_csv(OUT / "failed_experiments.csv", index=False, encoding="utf-8-sig")

    selection = {
        "selected_experiment_id": selected_id,
        "selection_reason": selection_reason,
        "selection_data_source": "inner_validation_only",
        "outer_test_metrics_used_for_selection": False,
        "promotion_rule": PROMOTION,
    }
    (OUT / "selection_decision.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    exp_map = {e["experiment_id"]: e for e in EXPERIMENTS}
    base_exp = exp_map[EXPERIMENTS[0]["experiment_id"]]
    selected_exp = exp_map[selected_id]

    # Phase B: one confirmatory outer evaluation for baseline and the inner-selected method only.
    bw, bf, bm, bp = run_outer_for_experiment(df, feats, base_exp, chosen_cfg)
    if selected_id == base_exp["experiment_id"]:
        sw, sf, sm, sp = bw.copy(), bf.copy(), bm.copy(), dict(bp)
    else:
        sw, sf, sm, sp = run_outer_for_experiment(df, feats, selected_exp, chosen_cfg)
    bw.to_csv(OUT / "baseline_oof_window_predictions.csv", index=False, encoding="utf-8-sig")
    bf.to_csv(OUT / "baseline_oof_file_predictions.csv", index=False, encoding="utf-8-sig")
    sw.to_csv(OUT / "final_method_oof_window_predictions.csv", index=False, encoding="utf-8-sig")
    sf.to_csv(OUT / "final_method_oof_file_predictions.csv", index=False, encoding="utf-8-sig")
    bm.assign(role="baseline").to_csv(OUT / "baseline_outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sm.assign(role="final_inner_selected").to_csv(OUT / "final_outer_fold_metrics.csv", index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame([
        {"role": "baseline", "experiment_id": base_exp["experiment_id"], **bp},
        {"role": "final_inner_selected", "experiment_id": selected_id, **sp},
    ])
    comparison.to_csv(OUT / "baseline_vs_final_pooled.csv", index=False, encoding="utf-8-sig")

    for tag, ff in [("baseline", bf), ("final", sf)]:
        cm = pd.DataFrame(confusion_matrix(ff["class_label"], ff["pred_label_file"], labels=LABELS), index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
        cm.to_csv(OUT / f"confusion_matrix_{tag}.csv", encoding="utf-8-sig")

    # Baseline reproduction against STEP06, if available.
    baseline_repro = {"available": False}
    old_metrics = BASE_DIR / "metrics_summary.json"
    if old_metrics.exists():
        old = json.loads(old_metrics.read_text(encoding="utf-8"))["file_level_pooled_oof"]
        baseline_repro = {
            "available": True,
            "step06_macro_f1": float(old["macro_f1"]),
            "step07_rerun_macro_f1": float(bp["macro_f1"]),
            "abs_diff": abs(float(old["macro_f1"]) - float(bp["macro_f1"])),
        }
        baseline_repro["pass"] = baseline_repro["abs_diff"] <= 1e-12
    (OUT / "baseline_reproduction.json").write_text(json.dumps(baseline_repro, ensure_ascii=False, indent=2), encoding="utf-8")

    # Error transitions and explicit key failure tracking.
    bsmall = bf[["independent_object_id", "class_label", "load_hp", "pred_label_file"]].rename(columns={"pred_label_file": "baseline_pred"})
    ssmall = sf[["independent_object_id", "pred_label_file"]].rename(columns={"pred_label_file": "final_pred"})
    trans = bsmall.merge(ssmall, on="independent_object_id", how="inner")
    trans["baseline_correct"] = trans["baseline_pred"] == trans["class_label"]
    trans["final_correct"] = trans["final_pred"] == trans["class_label"]
    trans["transition"] = np.select(
        [~trans["baseline_correct"] & trans["final_correct"], trans["baseline_correct"] & ~trans["final_correct"], ~trans["baseline_correct"] & ~trans["final_correct"]],
        ["corrected", "new_error", "persistent_error"], default="correct_both",
    )
    trans.to_csv(OUT / "error_transition_table.csv", index=False, encoding="utf-8-sig")
    final_mis = sf[sf["pred_label_file"] != sf["class_label"]].copy()
    final_mis.to_csv(OUT / "final_misclassified_files.csv", index=False, encoding="utf-8-sig")
    key_ids = [x for x in trans["independent_object_id"] if "OR014@6_" in x or x.endswith("IR014_0.mat") or x.endswith("IR007_3.mat") or x.endswith("N_0.mat")]
    key = trans[trans["independent_object_id"].isin(key_ids)].copy()
    key.to_csv(OUT / "key_failure_cases_comparison.csv", index=False, encoding="utf-8-sig")

    # Phase C: choose one global source-only configuration by 4-load LOLO CV, then fit all 56 source files.
    global_cfg, global_detail, global_summary = select_global_cfg(df, feats, selected_exp)
    global_detail.to_csv(OUT / "final_global_lolo_detail.csv", index=False, encoding="utf-8-sig")
    global_summary.to_csv(OUT / "final_global_config_selection.csv", index=False, encoding="utf-8-sig")
    tfit = time.perf_counter()
    final_pipe, final_feats = fit_model(df, feats, selected_exp, global_cfg)
    full_train_seconds = time.perf_counter() - tfit
    bundle = {
        "pipeline": final_pipe,
        "feature_columns": final_feats,
        "base_feature_columns": feats,
        "labels": LABELS,
        "selected_experiment_id": selected_id,
        "global_config": global_cfg,
        "class_alpha": selected_exp["class_alpha"],
        "corr_threshold": selected_exp["corr_threshold"],
        "random_seed": SEED,
        "aggregation": "mean window class probabilities per independent file",
        "training_scope": "all 56 M1 source independent acquisition files / 400 STEP04 windows",
    }
    joblib.dump(bundle, MODELS / "final_source_model.joblib")

    complexity = complexity_and_timing(bundle, df)
    complexity["full_source_fit_seconds"] = float(full_train_seconds)
    complexity["sklearn_version"] = sklearn.__version__
    (OUT / "complexity_report.json").write_text(json.dumps(complexity, ensure_ascii=False, indent=2), encoding="utf-8")

    q3_interface = {
        "version": "Q2C-STEP07-v1",
        "model_path": "outputs/q2c_failure_driven_improvement/models/final_source_model.joblib",
        "selected_experiment_id": selected_id,
        "global_config": global_cfg,
        "source_feature_file": "outputs/q1c_feature_extraction/q2_source_raw.csv",
        "target_feature_file": "outputs/q1c_feature_extraction/q2_target_raw.csv",
        "feature_columns": final_feats,
        "base_feature_columns": feats,
        "label_order": LABELS,
        "group_column": "independent_object_id",
        "window_id_column": "window_id",
        "file_aggregation": "arithmetic mean of window class probabilities; argmax for file label",
        "preprocessing_inside_saved_pipeline": True,
        "source_only_geometry_features_used": False,
        "target_truth_status": "unknown",
        "target_label_usage_allowed": False,
        "problem3_contract": "Problem3 may use this frozen source representation/classifier as its source-side baseline. Any target-domain adaptation must not use target labels and must preserve A-P file grouping.",
    }
    (OUT / "q3_interface.json").write_text(json.dumps(q3_interface, ensure_ascii=False, indent=2), encoding="utf-8")

    # Independent recalculation: manual Macro-F1 and one file aggregation.
    manual = manual_macro_f1(sf["class_label"].astype(str).tolist(), sf["pred_label_file"].astype(str).tolist())
    sample_id = "data/raw/source_domain/cwru_48khz_de/B007_0.mat"
    wsample = sw[sw["independent_object_id"] == sample_id]
    fsample = sf[sf["independent_object_id"] == sample_id].iloc[0]
    pcols = [f"p_{x}" for x in LABELS]
    meanp = wsample[pcols].mean().to_dict()
    recalc = {
        "manual_macro_f1": manual,
        "stored_macro_f1": float(sp["macro_f1"]),
        "macro_f1_abs_diff": abs(manual - float(sp["macro_f1"])),
        "sample_independent_object_id": sample_id,
        "sample_window_count": int(len(wsample)),
        "recalc_mean_probabilities": {k.replace("p_", ""): float(v) for k, v in meanp.items()},
        "stored_file_probabilities": {x: float(fsample[f"p_{x}"]) for x in LABELS},
        "recalc_pred_label": max(LABELS, key=lambda x: meanp[f"p_{x}"]),
        "stored_pred_label": str(fsample["pred_label_file"]),
    }
    recalc["max_probability_abs_diff"] = float(max(abs(meanp[f"p_{x}"] - float(fsample[f"p_{x}"])) for x in LABELS))
    recalc["pass"] = recalc["macro_f1_abs_diff"] <= 1e-12 and recalc["max_probability_abs_diff"] <= 1e-12 and recalc["recalc_pred_label"] == recalc["stored_pred_label"]
    (OUT / "recalculation_evidence.json").write_text(json.dumps(recalc, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "step": "Q2C_STEP07",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run_start": git_sha(),
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "random_seed": SEED,
        "input_source_csv": str(SRC_CSV.relative_to(ROOT)),
        "input_source_sha256": sha256(SRC_CSV),
        "input_interface_json": str(INTERFACE_JSON.relative_to(ROOT)),
        "input_interface_sha256": sha256(INTERFACE_JSON),
        "feature_count_base": len(feats),
        "experiments": [e["experiment_id"] for e in EXPERIMENTS],
        "selection_uses_outer_test_metrics": False,
        "selected_experiment_id": selected_id,
        "global_config": global_cfg,
        "final_feature_count": len(final_feats),
        "elapsed_seconds_total": float(time.perf_counter() - started),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "source_56_files_400_windows": len(df) == 400 and len(ft) == 56,
        "frozen_load_coverage": all(ft[ft["load_hp"] == l]["class_label"].value_counts().to_dict() == {"OR": 7, "B": 3, "IR": 3, "N": 1} for l in LOADS),
        "no_feature_name_leakage": not bool(forbidden.intersection(feats)),
        "all_hypotheses_predeclared": len(EXPERIMENTS) == 4 and (OUT / "precommitted_selection_rule.json").exists(),
        "selection_inner_only": selection["outer_test_metrics_used_for_selection"] is False,
        "failed_experiments_preserved": (OUT / "failed_experiments.csv").exists(),
        "baseline_reproduced": (not baseline_repro.get("available")) or bool(baseline_repro.get("pass")),
        "final_oof_56_once": len(sf) == 56 and sf["independent_object_id"].nunique() == 56,
        "final_oof_400_windows_once": len(sw) == 400 and sw["window_id"].nunique() == 400,
        "independent_recalc_pass": bool(recalc["pass"]),
        "final_model_saved": (MODELS / "final_source_model.joblib").exists(),
        "q3_interface_saved": (OUT / "q3_interface.json").exists(),
        "complexity_measured": complexity["serialized_model_bytes"] > 0 and complexity["inference_ms_per_file_median"] >= 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "selected_experiment_id": selected_id,
        "selection_reason": selection_reason,
        "baseline_pooled_file_metrics": bp,
        "final_pooled_file_metrics": sp,
        "outer_test_used_for_selection": False,
        "note": "Outer results are confirmatory for the inner-selected method; failed candidates were retained at inner-validation stage and were not promoted by test peeking.",
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Q2C / STEP07 failure-driven improvement and freeze",
        f"status={status}",
        f"selected_experiment={selected_id}",
        f"selection_reason={selection_reason}",
        f"baseline_macro_f1={bp['macro_f1']:.6f}",
        f"final_macro_f1={sp['macro_f1']:.6f}",
        f"baseline_balanced_accuracy={bp['balanced_accuracy']:.6f}",
        f"final_balanced_accuracy={sp['balanced_accuracy']:.6f}",
        f"baseline_misclassified_files={int((bf['pred_label_file'] != bf['class_label']).sum())}",
        f"final_misclassified_files={int((sf['pred_label_file'] != sf['class_label']).sum())}",
        f"final_feature_count={len(final_feats)}",
        f"final_global_config={config_key(global_cfg)}",
        f"inference_ms_per_file_median={complexity['inference_ms_per_file_median']:.6f}",
        f"manual_recalc_pass={recalc['pass']}",
        "outer_test_used_for_selection=False",
    ]
    (OUT / "result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
