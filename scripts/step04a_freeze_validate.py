from __future__ import annotations
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd

# Trigger-only revision for final STEP04-A freeze workflow; validation logic unchanged.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs' / 'q1c_feature_extraction'
OUT = ROOT / 'outputs' / 'step04a_freeze'
OUT.mkdir(parents=True, exist_ok=True)

feat = pd.read_csv(SRC / 'features_all_raw.csv')
src = pd.read_csv(SRC / 'features_source_raw.csv')
tgt = pd.read_csv(SRC / 'features_target_raw.csv')
idx = pd.read_csv(SRC / 'window_index.csv')
dic = pd.read_csv(SRC / 'feature_dictionary.csv')
interface = json.loads((SRC / 'q2_interface.json').read_text(encoding='utf-8'))
manifest = json.loads((SRC / 'run_manifest.json').read_text(encoding='utf-8'))
validation = json.loads((SRC / 'validation_report.json').read_text(encoding='utf-8'))

common = interface['feature_columns']
mechanism = interface['source_only_mechanism_features_excluded_from_default_q2']
checks = {}
checks['window_id_unique'] = bool(idx['window_id'].is_unique)
checks['independent_object_id_present'] = bool(idx['independent_object_id'].notna().all())
checks['group_equals_acquisition'] = bool((idx['independent_object_id'] == idx['acquisition_id']).all())
checks['physical_time_1s'] = bool(np.allclose(idx['window_end_s'] - idx['window_start_s'], 1.0))
checks['source_points_48000'] = bool(((idx.loc[idx.domain=='source','window_end_sample_exclusive'] - idx.loc[idx.domain=='source','window_start_sample']) == 48000).all())
checks['target_points_32000'] = bool(((idx.loc[idx.domain=='target','window_end_sample_exclusive'] - idx.loc[idx.domain=='target','window_start_sample']) == 32000).all())
checks['all_target_unknown'] = bool((tgt['class_label'].astype(str) == 'UNKNOWN').all())
checks['q2_interface_no_leakage'] = not bool(set(interface['forbidden_as_model_features']) & set(common))
checks['learned_preprocessing_not_applied'] = manifest.get('learned_preprocessing_applied') is False

quality_rows = []
for domain, df in [('source', src), ('target', tgt)]:
    for col in common + mechanism:
        if col not in df.columns:
            continue
        a = pd.to_numeric(df[col], errors='coerce')
        n = len(a)
        n_nan = int(a.isna().sum())
        arr = a.to_numpy(dtype=float)
        n_inf = int(np.isinf(arr).sum())
        finite = a.replace([np.inf, -np.inf], np.nan).dropna()
        nunique = int(finite.nunique())
        constant = bool(nunique <= 1 and len(finite) > 0)
        if len(finite):
            q1, med, q3 = [float(x) for x in finite.quantile([0.25,0.5,0.75]).tolist()]
            iqr = q3-q1
            lo, hi = q1-10*iqr, q3+10*iqr
            extreme_n = int(((finite < lo) | (finite > hi)).sum()) if iqr > 0 else 0
            minv, maxv = float(finite.min()), float(finite.max())
        else:
            q1=med=q3=iqr=minv=maxv=np.nan; extreme_n=0
        quality_rows.append({'domain':domain,'feature':col,'n_rows':n,'nan_count':n_nan,'inf_count':n_inf,'finite_unique_count':nunique,'constant_feature':constant,'q1':q1,'median':med,'q3':q3,'iqr':iqr,'min':minv,'max':maxv,'extreme_count_10IQR_rule':extreme_n,'expected_missing':'target_mechanism_only' if domain=='target' and col in mechanism else 'none'})
q = pd.DataFrame(quality_rows)
q.to_csv(OUT/'feature_quality_audit.csv', index=False, encoding='utf-8-sig')
common_q = q[q.feature.isin(common)]
checks['common_features_no_nan'] = bool((common_q.nan_count == 0).all())
checks['common_features_no_inf'] = bool((common_q.inf_count == 0).all())
checks['common_features_nonconstant'] = bool((~common_q.constant_feature).all())
src_mech_q = q[(q.domain=='source') & q.feature.isin(mechanism)]
tgt_mech_q = q[(q.domain=='target') & q.feature.isin(mechanism)]
checks['source_mechanism_finite'] = bool(((src_mech_q.nan_count==0) & (src_mech_q.inf_count==0)).all())
checks['target_mechanism_missing_by_design'] = bool((tgt_mech_q.nan_count == tgt_mech_q.n_rows).all())

unit_map = dic.set_index('feature_name')['unit'].to_dict()
unit_rows=[]
for col in common:
    unit_rows.append({'feature':col,'unit':unit_map.get(col,''),'source_definition':'same q1c deterministic formula','target_definition':'same q1c deterministic formula','consistent':True})
pd.DataFrame(unit_rows).to_csv(OUT/'unit_consistency.csv', index=False, encoding='utf-8-sig')
checks['unit_dictionary_complete_for_common'] = all(bool(unit_map.get(c,'')) for c in common)

leak = pd.DataFrame({'column': interface['forbidden_as_model_features'],'forbidden_from_feature_columns': [c not in common for c in interface['forbidden_as_model_features']]})
leak.to_csv(OUT/'leakage_audit.csv', index=False, encoding='utf-8-sig')
checks['forbidden_columns_all_excluded'] = bool(leak.forbidden_from_feature_columns.all())

candidates = pd.read_csv(SRC/'window_design_candidates.csv')
candidates.to_csv(OUT/'window_candidate_range_for_step12.csv', index=False, encoding='utf-8-sig')

copy_names = ['window_index.csv','features_all_raw.csv','features_source_raw.csv','features_target_raw.csv','feature_dictionary.csv','count_reconciliation.csv','window_design_candidates.csv','q1_answer_material.md','q2_interface.json','q2_source_raw.csv','q2_target_raw.csv','run_manifest.json','validation_report.json','result_summary.txt','file_inventory.csv']
for name in copy_names:
    shutil.copy2(SRC/name, OUT/name)
shutil.copy2(ROOT/'scripts'/'q1c_feature_extraction.py', OUT/'q1c_feature_extraction.py')

status = 'PASS' if all(checks.values()) and validation.get('status')=='PASS' else 'FAIL'
summary = {'step':'04-A','status':status,'input_q1c_run_start_commit':manifest.get('git_sha_at_run_start'),'q1c_evidence_commit':'9ee74c77c6b4e2963efa8917335d4da3af7536d3','window_seconds':manifest['window_seconds'],'overlap_fraction':manifest['overlap_fraction'],'source_files':manifest['source_files'],'target_files':manifest['target_files'],'source_windows':manifest['source_windows'],'target_windows':manifest['target_windows'],'common_feature_count':manifest['common_feature_count'],'mechanism_feature_count':manifest['mechanism_feature_count'],'checks':checks,'extreme_value_rule':'descriptive flag only: outside [Q1-10*IQR, Q3+10*IQR]; no deletion or clipping in STEP04-A','learned_preprocessing':'none; any imputation/scaling/selection must be fit after group split on training partition only'}
(OUT/'step04a_validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

rows=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='MANIFEST_SHA256.csv':
        rows.append({'file':p.name,'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(OUT/'MANIFEST_SHA256.csv', index=False, encoding='utf-8-sig')
print('STEP04A_FREEZE_STATUS='+status)
print(json.dumps(summary,ensure_ascii=False))
