from pathlib import Path
from datetime import datetime, timezone
import json, platform, time, hashlib, subprocess, sys
import joblib, numpy as np, pandas as pd
import importlib.util
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
SRCMOD=ROOT/'scripts'/'q2c_failure_driven_improvement.py'
spec=importlib.util.spec_from_file_location('q2c_base',SRCMOD); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
OUT=ROOT/'outputs'/'q2c_step07_strict'; MODELS=OUT/'models'; OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
mod.OUT=OUT; mod.MODELS=MODELS
LAB=mod.LAB; RUNS=mod.RUNS; SEED=mod.SEED
assert SEED==2025
assert RUNS=={'R1':{'train':['F3','F4'],'validation':['F2'],'test':['F1']},'R2':{'train':['F4','F1'],'validation':['F3'],'test':['F2']},'R3':{'train':['F1','F2'],'validation':['F4'],'test':['F3']},'R4':{'train':['F2','F3'],'validation':['F1'],'test':['F4']}}

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_data():
 iface=json.loads(mod.IFACE.read_text()); base=list(iface['feature_columns']); df=pd.read_csv(mod.SRC); ma=pd.read_csv(mod.SRCALL,usecols=['window_id']+mod.MECH); df=df.merge(ma,on='window_id',validate='one_to_one'); df['base_fold']=[mod.fold(g,c,l) for g,c,l in zip(df.independent_object_id,df.class_label,df.load_hp)]; return df,base

def exp_spec(eid):
 if eid=='E0_baseline': return dict(kind='lr',configs=mod.LRC,corr=False,mech=False)
 if eid=='E1_corr98_logreg': return dict(kind='lr',configs=mod.LRC,corr=True,mech=False)
 if eid=='E2_corr98_random_forest': return dict(kind='rf',configs=mod.RFC,corr=True,mech=False)
 if eid=='E3_mechanism_logreg': return dict(kind='lr',configs=mod.LRC,corr=False,mech=True)
 raise KeyError(eid)

def val_only(df,base,eid):
 s=exp_spec(eid); rows=[]; selected=[]
 for run,sp in RUNS.items():
  tr=df[df.base_fold.isin(sp['train'])].copy(); va=df[df.base_fold.isin(sp['validation'])].copy(); fs=base+(mod.MECH if s['mech'] else []); drop=[]
  if s['corr']: fs,drop=mod.corrset(tr,fs)
  cand=[]
  for c in s['configs']:
   m=mod.pipe(s['kind'],c); m.fit(tr[fs],tr.class_label,clf__sample_weight=mod.wt(tr)); fv=mod.agg(mod.pred(m,va,fs,run,'validation',eid,c['id'])); r={'evaluation_run':run,'experiment_id':eid,'config_id':c['id'],'config_order':c['ord'],'n_features':len(fs),'dropped':'|'.join(drop),**mod.met(fv.class_label,fv.pred_label_file)}; rows.append(r); cand.append(r)
  b=pd.DataFrame(cand).sort_values(['macro_f1','min_class_recall','config_order'],ascending=[False,False,True]).iloc[0]; selected.append(dict(b))
 return pd.DataFrame(rows),pd.DataFrame(selected)

def select_from_validation(sel):
 # Frozen before any outer-test access: E1/E3 are diagnostics only. E2 is eligible only if its best validation score is non-worse in >=3/4 runs, median delta >0, and median min-recall does not drop >0.05.
 b=sel[sel.experiment_id=='E0_baseline'][['evaluation_run','macro_f1','min_class_recall']].rename(columns={'macro_f1':'base_f1','min_class_recall':'base_minrec'})
 c=sel[sel.experiment_id=='E2_corr98_random_forest'][['evaluation_run','macro_f1','min_class_recall']].rename(columns={'macro_f1':'cand_f1','min_class_recall':'cand_minrec'})
 p=b.merge(c,on='evaluation_run'); p['delta']=p.cand_f1-p.base_f1; p['minrec_delta']=p.cand_minrec-p.base_minrec
 checks={'median_validation_delta_gt_0':bool(np.median(p.delta)>0),'validation_folds_nonworse_ge_3':bool((p.delta>=0).sum()>=3),'median_minrec_drop_ge_minus_0_05':bool(np.median(p.minrec_delta)>=-0.05)}
 accept=all(checks.values()); return ('E2_corr98_random_forest' if accept else 'E0_baseline'),p,checks

def outer_eval(df,base,eid):
 s=exp_spec(eid); WA=[]; FA=[]; FM=[]; cfgrows=[]; fsrows=[]
 for run,sp in RUNS.items():
  tr=df[df.base_fold.isin(sp['train'])].copy(); va=df[df.base_fold.isin(sp['validation'])].copy(); te=df[df.base_fold.isin(sp['test'])].copy(); fs=base+(mod.MECH if s['mech'] else []); drop=[]
  if s['corr']: fs,drop=mod.corrset(tr,fs)
  cand=[]
  for c in s['configs']:
   m=mod.pipe(s['kind'],c); m.fit(tr[fs],tr.class_label,clf__sample_weight=mod.wt(tr)); fv=mod.agg(mod.pred(m,va,fs,run,'validation',eid,c['id'])); cand.append({'config_id':c['id'],'config_order':c['ord'],**mod.met(fv.class_label,fv.pred_label_file)})
  best=pd.DataFrame(cand).sort_values(['macro_f1','min_class_recall','config_order'],ascending=[False,False,True]).iloc[0]; c=next(z for z in s['configs'] if z['id']==best.config_id)
  dev=pd.concat([tr,va],ignore_index=True); fs=base+(mod.MECH if s['mech'] else []); drop=[]
  if s['corr']: fs,drop=mod.corrset(dev,fs)
  m=mod.pipe(s['kind'],c); m.fit(dev[fs],dev.class_label,clf__sample_weight=mod.wt(dev)); ww=mod.pred(m,te,fs,run,'test',eid,c['id']); ff=mod.agg(ww); WA.append(ww); FA.append(ff); FM.append({'evaluation_run':run,'experiment_id':eid,'selected_config_id':c['id'],'n_features':len(fs),'dropped':'|'.join(drop),**mod.met(ff.class_label,ff.pred_label_file)}); cfgrows.append({'evaluation_run':run,'experiment_id':eid,'selected_config_id':c['id']}); fsrows.append({'evaluation_run':run,'experiment_id':eid,'n_features':len(fs),'features':'|'.join(fs),'dropped':'|'.join(drop)})
 W=pd.concat(WA,ignore_index=True); F=pd.concat(FA,ignore_index=True); return {'w':W,'f':F,'fold':pd.DataFrame(FM),'cfg':pd.DataFrame(cfgrows),'fs':pd.DataFrame(fsrows),'pooled':mod.met(F.class_label,F.pred_label_file)}

def benchmark(pay,df):
 m=pay['pipeline']; fs=pay['model_feature_columns']; vals=[]
 for _,g in df.groupby('independent_object_id'):
  for _ in range(5):
   t=time.perf_counter(); q=m.predict_proba(g[fs]); q.mean(axis=0); vals.append((time.perf_counter()-t)*1000)
 return {'median_ms_per_file':float(np.median(vals)),'p95_ms_per_file':float(np.percentile(vals,95)),'timing_repeats_per_file':5,'batch':'one independent file, all precomputed windows','includes':'pipeline preprocessing + predict_proba + mean aggregation','excludes':'raw MAT loading/windowing/feature extraction/CSV I/O'}

def main():
 df,base=load_data(); assert len(base)==28 and len(df)==400 and df.independent_object_id.nunique()==56
 hyp=pd.DataFrame([
 ['E0_baseline','reference','OR->B=4, N_0->OR, 0 hp weak','No change','Reference','same Step05 folds/seed/aggregation','reference'],
 ['E1_corr98_logreg','redundancy','highly correlated amplitude summaries','train-only |r|>=0.98 pruning + LR','reduce linear redundancy','same folds/seed/LR grid','diagnostic; stop after one threshold'],
 ['E2_corr98_random_forest','nonlinear boundary','OR014@6 systematic ->B; B007_0 near boundary','E1 pruning + small RF grid','nonlinear interactions may separate OR/B','same folds/seed/aggregation; selected only from validation','eligible; stop after 3 RF configs'],
 ['E3_mechanism_logreg','mechanism','generic features may miss fault-frequency evidence','add 11 source-only mechanism features + LR','test physical harmonic evidence','same folds/seed/LR grid','diagnostic only; not Q3-eligible']
 ],columns=['experiment_id','failure_axis','failure_evidence','change','why_it_may_help','fair_comparison','stop_rule']); hyp.to_csv(OUT/'improvement_hypotheses.csv',index=False)
 vals=[]; sels=[]
 for e in ['E0_baseline','E1_corr98_logreg','E2_corr98_random_forest','E3_mechanism_logreg']:
  v,s=val_only(df,base,e); vals.append(v); sels.append(s)
 V=pd.concat(vals,ignore_index=True); S=pd.concat(sels,ignore_index=True); V.to_csv(OUT/'validation_all_experiments.csv',index=False); S.to_csv(OUT/'validation_selected_by_run.csv',index=False)
 final_id,pairv,checks=select_from_validation(S); pairv.to_csv(OUT/'validation_paired_base_vs_candidate.csv',index=False); (OUT/'selection_decision.json').write_text(json.dumps({'selection_source':'validation_only','outer_test_used_for_selection':False,'eligible':['E0_baseline','E2_corr98_random_forest'],'diagnostic_only':['E1_corr98_logreg','E3_mechanism_logreg'],'checks':checks,'final_experiment_id':final_id},indent=2))
 # Outer tests are touched only after final_id is frozen. Baseline is re-evaluated only as the reference comparator.
 RB=outer_eval(df,base,'E0_baseline'); RF=RB if final_id=='E0_baseline' else outer_eval(df,base,final_id)
 for tag,R in [('base',RB),('final',RF)]:
  R['f'].to_csv(OUT/f'{tag}_oof_file_predictions.csv',index=False); R['w'].to_csv(OUT/f'{tag}_oof_window_predictions.csv',index=False); R['fold'].to_csv(OUT/f'{tag}_outer_fold_metrics.csv',index=False); pd.DataFrame(confusion_matrix(R['f'].class_label,R['f'].pred_label_file,labels=LAB),index=LAB,columns=LAB).to_csv(OUT/f'{tag}_confusion_matrix.csv')
 pair=RB['fold'][['evaluation_run','macro_f1']].merge(RF['fold'][['evaluation_run','macro_f1']],on='evaluation_run',suffixes=('_base','_final')); pair['delta']=pair.macro_f1_final-pair.macro_f1_base; pair.to_csv(OUT/'paired_outer_base_vs_final.csv',index=False)
 pooled=pd.DataFrame([{'model':'Base',**RB['pooled']},{'model':'Final',**RF['pooled']}]); pooled.to_csv(OUT/'base_vs_final_pooled.csv',index=False)
 # Retain validation-only failures; do not evaluate their outer tests.
 fail=pd.DataFrame([{'experiment_id':'E1_corr98_logreg','status':'diagnostic_not_selected','reason':'not eligible for final; redundancy test only'},{'experiment_id':'E3_mechanism_logreg','status':'diagnostic_not_selected','reason':'source-only geometry makes it non-transferable to target'}]); fail.to_csv(OUT/'failed_experiments.csv',index=False)
 # Fit final source model using all 56 files; hyperparameters selected by grouped 4-fold CV internal to source only.
 es=exp_spec(final_id); pay,cv=mod.globalfit(df,base,es['kind'],es['configs'],es['corr'],MODELS/'final_source_model.joblib'); cv.to_csv(OUT/'final_global_config_selection.csv',index=False)
 bench=benchmark(pay,df); clf=pay['pipeline'].named_steps['clf']; comp={'final_experiment_id':final_id,'input_feature_count':len(pay['input_feature_columns']),'model_feature_count':len(pay['model_feature_columns']),'serialized_model_bytes':(MODELS/'final_source_model.joblib').stat().st_size,'cpu_model':platform.processor() or 'unknown','logical_cpu_count':4,'platform':platform.platform(),**bench}
 if hasattr(clf,'coef_'): comp['parameter_count']=int(clf.coef_.size+clf.intercept_.size)
 if hasattr(clf,'estimators_'): comp.update({'n_trees':len(clf.estimators_),'total_tree_nodes':int(sum(t.tree_.node_count for t in clf.estimators_)),'total_tree_leaves':int(sum(t.tree_.n_leaves for t in clf.estimators_))})
 (OUT/'complexity_report.json').write_text(json.dumps(comp,indent=2))
 q3={'status':'FROZEN','source_model':'outputs/q2c_step07_strict/models/final_source_model.joblib','final_experiment_id':final_id,'raw_input_feature_columns':pay['input_feature_columns'],'model_feature_columns':pay['model_feature_columns'],'dropped_features':pay['dropped_features'],'labels':LAB,'aggregation':'mean window probabilities per independent file','target_compatible_only':True,'source_only_mechanism_features_excluded':True,'random_seed':SEED,'selection_source':'validation_only','outer_test_used_for_selection':False}; (OUT/'q3_interface.json').write_text(json.dumps(q3,indent=2))
 # Misclassification explanation for selected final.
 mis=RF['f'][RF['f'].class_label!=RF['f'].pred_label_file].copy(); mis.to_csv(OUT/'final_misclassified_files.csv',index=False)
 d=pair.delta.to_numpy(float); effect={'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'min_delta':float(d.min()),'max_delta':float(d.max()),'folds_improved':int((d>0).sum()),'folds_nonworse':int((d>=0).sum()),'cohen_dz':float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1)>0 else None}; (OUT/'paired_effect.json').write_text(json.dumps(effect,indent=2))
 old=json.loads((mod.BOUT/'metrics_summary.json').read_text())['file_level_pooled_oof']['macro_f1']; rep={'step06_macro_f1':old,'step07_base_macro_f1':RB['pooled']['macro_f1'],'abs_diff':abs(old-RB['pooled']['macro_f1']),'pass':abs(old-RB['pooled']['macro_f1'])<1e-12}; (OUT/'baseline_reproduction.json').write_text(json.dumps(rep,indent=2)); assert rep['pass']
 manifest={'status':'PASS','generated_utc':datetime.now(timezone.utc).isoformat(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seed':SEED,'selection':'validation_only before outer test','final_experiment_id':final_id,'source_csv_sha256':sha(mod.SRC),'step06_baseline_reproduced':True}; (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('STEP07_STRICT PASS'); print('final_experiment_id='+final_id); print('base_macro_f1=%.6f'%RB['pooled']['macro_f1']); print('final_macro_f1=%.6f'%RF['pooled']['macro_f1']); print('outer_test_used_for_selection=False'); print('paired_effect='+json.dumps(effect))

if __name__=='__main__': main()
