# STEP07-A 根据真实失败改进并定型

- run_id: run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb
- Base: 06-A LogisticRegression, 26 common features
- H1: RandomForest, same 26 common features and identical 05-A grouped protocol
- H2: LR + train-only |Pearson r|>0.95 redundancy pruning, inner-fold ablation only
- final: RandomForestClassifier, 300 trees, max_depth=8, min_samples_leaf=1, max_features=sqrt, file_and_class_balanced, seed=20260919
- final model SHA256: 4ee408ac96054e3b86e1c62c7a7afedc4baabd792390b1ad8bbb338da404d685

H2 failed to improve: 12 paired inner-fold Macro-F1 delta mean=-0.012408, median=0, range=-0.138023 to +0.043182. It was stopped without threshold retuning and never opened the outer tests.

H1 passed the pre-outer gate: selected RF inner Macro-F1 mean 0.934935 vs Base 0.783667 (+0.151268); mean minimum-class recall 0.85 vs 0.672222 (+0.177778). H1 was then evaluated once on the same four frozen outer tests.

Paired outer Macro-F1 deltas Final-Base: +0.495130, -0.063492, +0.190476, +0.072727. Mean +0.173710, median +0.131602, min-max -0.063492 to +0.495130; 3/4 folds improved. This is not a statistical significance claim.

Pooled file-level OOF:
- Base Macro-F1=0.818452, Accuracy=0.795918, 10 file errors.
- Final RF Macro-F1=0.967500, Accuracy=0.959184, 2 file errors.
- Final recalls OR/IR/B/N=0.904762/1/1/1.
- Confusion matrix rows true OR/IR/B/N, cols pred OR/IR/B/N:
  [[19,1,1,0],[0,12,0,0],[0,0,12,0],[0,0,0,4]].
- Remaining errors: OR014@3_1→IR and OR021@3_2→B.
- 9 Base errors became correct, 1 Base-correct file became wrong, and 1 file remained wrong.

Complexity on GitHub Actions ubuntu-24.04 CPU, using already-extracted 26-D windows:
- Base LR: 108 coefficient/intercept parameters; median model 3,148 B; median feature-ready single-file inference 0.138 ms.
- Final RF: 300 trees, 11,596 nodes, model 275,708 B; all-source fit 0.354 s; outer-model median feature-ready single-file inference 7.283 ms.
Raw signal preprocessing/feature extraction is excluded from these inference timings.

Q3 inherits only the same 26 target-compatible common features; source-only mechanism features are excluded. Scores remain uncalibrated model scores.

PASS_TOKEN: 07-A-V2026.09.20-2e401561-通过
NEXT_ALLOWED: 07-B
