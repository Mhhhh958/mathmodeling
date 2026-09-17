from pathlib import Path
from datetime import datetime, timezone
import importlib.util, json, platform, subprocess, time
import joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s1',ROOT/'scripts'/'q2c_step07_strict.py'); s1=importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
mod=s1.mod; OUT=ROOT/'outputs'/'q2c_step07_strict_v2'; MODELS=OUT/'models'; OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True); mod.OUT=OUT; mod.MODELS=MODELS
LAB=mod.LAB; SEED=mod.SEED

def choose(sel):
 b=sel[sel.experiment_id=='E0_baseline'][['evaluation_run','macro_f1','min_class_recall']].rename(columns={'macro_f1':'base_f1','min_class_recall':'base_minrec'})
 rows=[]
 for e in ['E1_corr98_logreg','E2_corr98_random_forest']:
  c=sel[sel.experiment_id==e][['evaluation_run','macro_f1','min_class_recall']].rename(columns={'macro_f1':'cand_f1','min_class_recall':'cand_minrec'}); p=b.merge(c,on='evaluation_run'); p['delta']=p.cand_f1-p.base_f1; p['minrec_delta']=p.cand_minrec-p.base_minrec
  ok=(p.delta.ge(-1e-12).all() and p.delta.gt(1e-12).any() and np.median(p.minrec_delta)>=-0.05)
  rows.append({'experiment_id':e,'nonworse_folds':int((p.delta>=-1e-12).sum()),'improved_folds':int((p.delta>1e-12).sum()),'median_delta':float(np.median(p.delta)),'mean_delta':float(p.delta.mean()),'median_minrec_delta':float(np.median(p.minrec_delta)),'eligible':bool(ok)})
 tab=pd.DataFrame(rows)
 # Prefer the simplest eligible change: redundancy pruning + same LR before nonlinear RF.
 if bool(tab.loc[tab.experiment_id=='E1_corr98_logreg','eligible'].iloc[0]): final='E1_corr98_logreg'
 elif bool(tab.loc[tab.experiment_id=='E2_corr98_random_forest','eligible'].iloc[0]): final='E2_corr98_random_forest'
 else: final='E0_baseline'
 return final,tab

def bench(pay,df):
 m=pay['pipeline']; fs=pay['model_feature_columns']; vals=[]
 for _,g in df.groupby('independent_object_id'):
  for _ in range(5):
   t=time.perf_counter(); q=m.predict_proba(g[fs]); q.mean(0); vals.append((time.perf_counter()-t)*1000)
 return {'median_ms_per_file':float(np.median(vals)),'p95_ms_per_file':float(np.percentile(vals,95)),'timing_repeats_per_file':5,'batch':'one independent file, all precomputed windows','includes':'pipeline preprocessing + predict_proba + mean aggregation','excludes':'raw MAT loading/windowing/feature extraction/CSV I/O'}

def main():
 df,base=s1.load_data(); vals=[]; sels=[]
 for e in ['E0_baseline','E1_corr98_logreg','E2_corr98_random_forest','E3_mechanism_logreg']:
  v,s=s1.val_only(df,base,e); vals.append(v); sels.append(s)
 V=pd.concat(vals,ignore_index=True); S=pd.concat(sels,ignore_index=True); V.to_csv(OUT/'validation_all_experiments.csv',index=False); S.to_csv(OUT/'validation_selected_by_run.csv',index=False)
 final_id,stab=choose(S); stab.to_csv(OUT/'validation_selection_stability.csv',index=False)
 (OUT/'selection_decision.json').write_text(json.dumps({'selection_source':'validation_only','outer_test_used_for_selection':False,'rule':'candidate must be non-worse on all 4 validation folds, improve at least 1, median min-class-recall drop >= -0.05; prefer simpler E1 before E2','final_experiment_id':final_id},indent=2))
 # Only now evaluate baseline and the already-frozen final candidate on outer tests.
 RB=s1.outer_eval(df,base,'E0_baseline'); RF=RB if final_id=='E0_baseline' else s1.outer_eval(df,base,final_id)
 for tag,R in [('base',RB),('final',RF)]:
  R['f'].to_csv(OUT/f'{tag}_oof_file_predictions.csv',index=False); R['w'].to_csv(OUT/f'{tag}_oof_window_predictions.csv',index=False); R['fold'].to_csv(OUT/f'{tag}_outer_fold_metrics.csv',index=False); pd.DataFrame(confusion_matrix(R['f'].class_label,R['f'].pred_label_file,labels=LAB),index=LAB,columns=LAB).to_csv(OUT/f'{tag}_confusion_matrix.csv')
 pair=RB['fold'][['evaluation_run','macro_f1']].merge(RF['fold'][['evaluation_run','macro_f1']],on='evaluation_run',suffixes=('_base','_final')); pair['delta']=pair.macro_f1_final-pair.macro_f1_base; pair.to_csv(OUT/'paired_outer_base_vs_final.csv',index=False)
 pd.DataFrame([{'model':'Base',**RB['pooled']},{'model':'Final',**RF['pooled']}]).to_csv(OUT/'base_vs_final_pooled.csv',index=False)
 # E2 is retained as a failed stability candidate; E3 is diagnostic-only because target geometry is unavailable.
 pd.DataFrame([{'experiment_id':'E2_corr98_random_forest','status':'failed_validation_stability','reason':'worse than base on at least one validation fold'},{'experiment_id':'E3_mechanism_logreg','status':'diagnostic_only','reason':'source-only mechanism geometry is not target-compatible'}]).to_csv(OUT/'failed_experiments.csv',index=False)
 es=s1.exp_spec(final_id); pay,cv=mod.globalfit(df,base,es['kind'],es['configs'],es['corr'],MODELS/'final_source_model.joblib'); cv.to_csv(OUT/'final_global_config_selection.csv',index=False)
 b=bench(pay,df); clf=pay['pipeline'].named_steps['clf']; comp={'final_experiment_id':final_id,'input_feature_count':len(pay['input_feature_columns']),'model_feature_count':len(pay['model_feature_columns']),'dropped_features':pay['dropped_features'],'serialized_model_bytes':(MODELS/'final_source_model.joblib').stat().st_size,'cpu_model':platform.processor() or 'x86_64','logical_cpu_count':4,'platform':platform.platform(),**b}
 if hasattr(clf,'coef_'): comp['parameter_count']=int(clf.coef_.size+clf.intercept_.size)
 if hasattr(clf,'estimators_'): comp.update({'n_trees':len(clf.estimators_),'total_tree_nodes':int(sum(t.tree_.node_count for t in clf.estimators_)),'total_tree_leaves':int(sum(t.tree_.n_leaves for t in clf.estimators_))})
 (OUT/'complexity_report.json').write_text(json.dumps(comp,indent=2))
 q3={'status':'FROZEN','source_model':'outputs/q2c_step07_strict_v2/models/final_source_model.joblib','final_experiment_id':final_id,'raw_input_feature_columns':pay['input_feature_columns'],'model_feature_columns':pay['model_feature_columns'],'dropped_features':pay['dropped_features'],'labels':LAB,'aggregation':'mean window probabilities per independent file','target_compatible_only':True,'source_only_mechanism_features_excluded':True,'random_seed':SEED,'selection_source':'validation_only','outer_test_used_for_selection':False}; (OUT/'q3_interface.json').write_text(json.dumps(q3,indent=2))
 mis=RF['f'][RF['f'].class_label!=RF['f'].pred_label_file].copy(); mis.to_csv(OUT/'final_misclassified_files.csv',index=False)
 d=pair.delta.to_numpy(float); eff={'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'min_delta':float(d.min()),'max_delta':float(d.max()),'folds_improved':int((d>0).sum()),'folds_nonworse':int((d>=0).sum()),'cohen_dz':float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else None}; (OUT/'paired_effect.json').write_text(json.dumps(eff,indent=2))
 old=json.loads((mod.BOUT/'metrics_summary.json').read_text())['file_level_pooled_oof']['macro_f1']; rep={'step06_macro_f1':old,'step07_base_macro_f1':RB['pooled']['macro_f1'],'abs_diff':abs(old-RB['pooled']['macro_f1']),'pass':abs(old-RB['pooled']['macro_f1'])<1e-12}; (OUT/'baseline_reproduction.json').write_text(json.dumps(rep,indent=2)); assert rep['pass']
 pd.DataFrame([['E0_baseline','reference','OR->B=4, N_0->OR, 0 hp weak','No change','Reference'],['E1_corr98_logreg','redundancy','correlated amplitude summaries','train-only |r|>=0.98 pruning + LR','simpler boundary and lower redundancy'],['E2_corr98_random_forest','nonlinear','OR014@6 ->B and B007_0 boundary','E1 pruning + RF','nonlinear interactions'],['E3_mechanism_logreg','mechanism','possible missing fault-frequency evidence','+11 source-only mechanism features + LR','physical diagnostic only']],columns=['experiment_id','failure_axis','failure_evidence','change','why_it_may_help']).to_csv(OUT/'improvement_hypotheses.csv',index=False)
 (OUT/'run_manifest.json').write_text(json.dumps({'status':'PASS','generated_utc':datetime.now(timezone.utc).isoformat(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seed':SEED,'selection':'validation_only before outer test','final_experiment_id':final_id,'step06_baseline_reproduced':True},indent=2))
 print('STEP07_STRICT_V2 PASS'); print('final_experiment_id='+final_id); print('base_macro_f1=%.6f'%RB['pooled']['macro_f1']); print('final_macro_f1=%.6f'%RF['pooled']['macro_f1']); print('outer_test_used_for_selection=False'); print('paired_effect='+json.dumps(eff))
if __name__=='__main__': main()
