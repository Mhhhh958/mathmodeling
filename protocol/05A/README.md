# STEP05-A 训练测试划分与评价协议冻结摘要

- run_id: run_05-A_20260919T192338074034Z_f82972c7_ba6432cf
- input feature data: FEAT-23d45f5649dcd5f1
- source independent files: 49; source windows: 733
- file counts: OR=21, IR=12, B=12, N=4
- outer protocol: 4 grouped outer folds; each fold tests exactly one independent N file
- inner protocol: within each outer train pool, 3 grouped inner folds; each validation fold contains exactly one of the remaining N files
- all train/validation/test group intersections: 0
- primary metric: file-level Macro-F1
- file aggregation: mean per-window class probabilities within file, then argmax
- window metrics: auxiliary only
- outer score reporting: four fold values + median + min–max; not a confidence interval
- baseline grid: LR 6 configs + RF 6 configs = 12
- budget: 36 inner fits per outer fold; 144 inner fits total; 4 outer refits; no adaptive grid expansion
- no model trained in 05-A; no Word edit; no five-ledger edit
- GitHub Actions check: 105953253659 / success
- PASS_TOKEN: 05-A-V2026.09.20-338d8e27-通过
- NEXT_ALLOWED: 05-B
