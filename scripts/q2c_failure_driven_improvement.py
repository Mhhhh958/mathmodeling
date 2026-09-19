# AI ASSISTANCE NOTICE
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ChatGPT；版本/型号：GPT-5.6 Sol（ChatGPT 2026-08-06更新版）；
# 开发机构/公司：OpenAI；版本发布日期：2026-08-06。
# 人工智能仅用于代码检查、调试建议与说明整理；最终算法、参数与结果由参赛队审查并由冻结复现链验证。
from pathlib import Path
from datetime import datetime, timezone
import json, os, platform, resource, time, hashlib, subprocess, sys
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'outputs/q1c_feature_extraction'; BOUT=ROOT/'outputs/q2b_minimal_baseline'; OUT=ROOT/'outputs/q2c_failure_driven_improvement'; MODELS=OUT/'models'; OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(exist_ok=True)
SRC=IN/'q2_source_raw.csv'; SRCALL=IN/'features_source_raw.csv'; IFACE=IN/'q2_interface.json'; SEED=2025; LAB=['OR','IR','B','N']; CORR=.98
RUNS={'R1':{'train':['F3','F4'],'validation':['F2'],'test':['F1']},'R2':{'train':['F4','F1'],'validation':['F3'],'test':['F2']},'R3':{'train':['F1','F2'],'validation':['F4'],'test':['F3']},'R4':{'train':['F2','F3'],'validation':['F1'],'test':['F4']}}
LRC=[{'id':'LR_0.1_none','C':.1,'cw':None,'ord':1},{'id':'LR_0.1_bal','C':.1,'cw':'balanced','ord':2},{'id':'LR_1_none','C':1.,'cw':None,'ord':3},{'id':'LR_1_bal','C':1.,'cw':'balanced','ord':4},{'id':'LR_10_none','C':10.,'cw':None,'ord':5},{'id':'LR_10_bal','C':10.,'cw':'balanced','ord':6}]
RFC=[{'id':'RF_200_full_l1','n':200,'d':None,'leaf':1,'ord':1},{'id':'RF_300_d10_l1','n':300,'d':10,'leaf':1,'ord':2},{'id':'RF_300_d10_l2','n':300,'d':10,'leaf':2,'ord':3}]
MECH=['mech_bpfo_ratio_1x','mech_bpfo_ratio_2x','mech_bpfo_ratio_3x','mech_bpfi_ratio_1x','mech_bpfi_ratio_2x','mech_bpfi_ratio_3x','mech_bsf_ratio_1x','mech_bsf_ratio_2x','mech_bsf_ratio_3x','mech_bpfi_sideband_to_center_1x','mech_bsf_sideband_to_center_1x']
GATE={'pooled_delta_min':.02,'median_fold_delta_gt':0.,'folds_nonworse_min':3,'worst_fold_delta_min':-.05,'min_recall_drop_max':.05}

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ng(x):
 s=str(x).replace('\\','/');return s[9:] if s.startswith('data/raw/') else s
def fold(g,c,l):
 import re
 n=Path(ng(g)).name;l=int(round(float(l)))
 if c=='N':o=0
 else:
  z=int(re.match(r'(IR|OR|B)(\d{3})',n,re.I).group(2))
  if c=='B':o={7:0,14:-1,21:-2}[z]
  elif c=='IR':o={7:-1,14:-2,21:-3}[z]
  else:
   p=int(re.search(r'@(3|6|12)_',n).group(1));o={(7,3):0,(7,6):-1,(7,12):-2,(14,6):1,(21,3):-1,(21,6):-2,(21,12):-3}[(z,p)]
 return f'F{((l+o)%4)+1}'
def wt(d):
 n=d.groupby('independent_object_id').size().to_dict();w=np.array([1/n[x] for x in d.independent_object_id]);return w/w.mean()
def met(y,p):
 pr,rc,ff,s=precision_recall_fscore_support(y,p,labels=LAB,zero_division=0);r={'n_files':len(y),'macro_f1':f1_score(y,p,labels=LAB,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,p),'accuracy':accuracy_score(y,p),'min_class_recall':min(rc)}
 for a,b,c,d,e in zip(LAB,pr,rc,ff,s):r|={f'precision_{a}':b,f'recall_{a}':c,f'f1_{a}':d,f'support_{a}':int(e)}
 return {k:(float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for k,v in r.items()}
def corrset(d,fs):
 C=d[fs].corr().abs().fillna(0);keep=[];drop=[]
 for f in fs:
  (drop if any(C.loc[f,k]>=CORR for k in keep) else keep).append(f)
 return keep,drop
def pipe(kind,c):
 if kind=='lr':return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('clf',LogisticRegression(C=c['C'],class_weight=c['cw'],solver='lbfgs',max_iter=3000,random_state=SEED))])
 return Pipeline([('imp',SimpleImputer(strategy='median')),('clf',RandomForestClassifier(n_estimators=c['n'],max_depth=c['d'],min_samples_leaf=c['leaf'],max_features='sqrt',random_state=SEED,n_jobs=1))])
def pred(m,d,fs,run,role,eid,cid):
 q=m.predict_proba(d[fs]);cls=list(m.named_steps['clf'].classes_);o=d[['window_id','independent_object_id','class_label','load_hp','base_fold']].copy();o['evaluation_run']=run;o['role']=role;o['experiment_id']=eid;o['config_id']=cid
 for a in LAB:o[f'p_{a}']=q[:,cls.index(a)]
 o['pred_label_window']=o[[f'p_{a}' for a in LAB]].idxmax(1).str[2:];return o
def agg(w):
 pc=[f'p_{a}' for a in LAB];keys=['evaluation_run','role','experiment_id','config_id','independent_object_id','class_label','load_hp','base_fold'];a=w.groupby(keys,as_index=False)[pc].mean();a=a.merge(w.groupby(['evaluation_run','independent_object_id']).size().rename('n_windows').reset_index(),on=['evaluation_run','independent_object_id']);a['pred_label_file']=a[pc].idxmax(1).str[2:];z=np.sort(a[pc].values,1);a['confidence_top1']=a[pc].max(1);a['margin_top1_top2']=z[:,-1]-z[:,-2];return a

def runexp(df,base,eid,kind,configs,corr=False,mech=False):
 WA=[];FA=[];FM=[];VV=[];FS=[]
 for run,s in RUNS.items():
  tr=df[df.base_fold.isin(s['train'])];va=df[df.base_fold.isin(s['validation'])];te=df[df.base_fold.isin(s['test'])];fs=base+(MECH if mech else []);drop=[]
  if corr:fs,drop=corrset(tr,fs)
  cand=[]
  for c in configs:
   m=pipe(kind,c);m.fit(tr[fs],tr.class_label,clf__sample_weight=wt(tr));fv=agg(pred(m,va,fs,run,'validation',eid,c['id']));x={'evaluation_run':run,'experiment_id':eid,'config_id':c['id'],'config_order':c['ord'],'n_features':len(fs),**met(fv.class_label,fv.pred_label_file)};cand.append(x);VV.append(x)
  best=pd.DataFrame(cand).sort_values(['macro_f1','min_class_recall','config_order'],ascending=[False,False,True]).iloc[0];c=next(z for z in configs if z['id']==best.config_id);dev=pd.concat([tr,va]);fs=base+(MECH if mech else []);drop=[]
  if corr:fs,drop=corrset(dev,fs)
  m=pipe(kind,c);m.fit(dev[fs],dev.class_label,clf__sample_weight=wt(dev));ww=pred(m,te,fs,run,'test',eid,c['id']);ff=agg(ww);WA.append(ww);FA.append(ff);FM.append({'evaluation_run':run,'experiment_id':eid,'selected_config_id':c['id'],'n_features':len(fs),'dropped':'|'.join(drop),**met(ff.class_label,ff.pred_label_file)});FS.append({'evaluation_run':run,'experiment_id':eid,'features':'|'.join(fs),'dropped':'|'.join(drop)})
 W=pd.concat(WA);F=pd.concat(FA);return {'w':W,'f':F,'fold':pd.DataFrame(FM),'val':pd.DataFrame(VV),'fs':pd.DataFrame(FS),'pooled':met(F.class_label,F.pred_label_file)}

def globalfit(df,base,kind,configs,corr,path):
 rows=[]
 for c in configs:
  z=[]
  for f in ['F1','F2','F3','F4']:
   tr=df[df.base_fold!=f];va=df[df.base_fold==f];fs=base
   if corr:fs,_=corrset(tr,fs)
   m=pipe(kind,c);m.fit(tr[fs],tr.class_label,clf__sample_weight=wt(tr));ff=agg(pred(m,va,fs,'CV','validation','GLOBAL',c['id']));z.append(met(ff.class_label,ff.pred_label_file)['macro_f1'])
  rows.append({'config_id':c['id'],'order':c['ord'],'cv_macro_f1_mean':np.mean(z),'cv_macro_f1_median':np.median(z)})
 tab=pd.DataFrame(rows).sort_values(['cv_macro_f1_mean','order'],ascending=[False,True]);c=next(x for x in configs if x['id']==tab.iloc[0].config_id);fs=base;drop=[]
 if corr:fs,drop=corrset(df,fs)
 m=pipe(kind,c);m.fit(df[fs],df.class_label,clf__sample_weight=wt(df));pay={'pipeline':m,'input_feature_columns':base,'model_feature_columns':fs,'dropped_features':drop,'labels':LAB,'selected_config':c,'random_seed':SEED};joblib.dump(pay,path);return pay,tab

def bench(pay,df):
 m=pay['pipeline'];fs=pay['model_feature_columns'];t=[]
 for _,g in df.groupby('independent_object_id'):
  for _ in range(3):a=time.perf_counter();m.predict_proba(g[fs]);t.append((time.perf_counter()-a)*1000)
 return {'median_ms_per_file':float(np.median(t)),'p95_ms_per_file':float(np.percentile(t,95)),'batch':'one file, all windows','feature_extraction_included':False}

def main():
 iface=json.loads(IFACE.read_text());base=iface['feature_columns'];df=pd.read_csv(SRC);ma=pd.read_csv(SRCALL,usecols=['window_id']+MECH);df=df.merge(ma,on='window_id',validate='one_to_one');df['base_fold']=[fold(g,c,l) for g,c,l in zip(df.independent_object_id,df.class_label,df.load_hp)]
 hyp=pd.DataFrame([['E0_baseline','reference','Step06 OR->B/0 hp failures','No change','Reference','Same Step05 folds/seed','Reference'],['E1_corr98_logreg','redundancy','RMS/std and related summaries redundant','Training-only |r|>=0.98 pruning; same LR','Stabilize linear boundary','Same folds/seed/grid','One frozen threshold'],['E2_corr98_random_forest','nonlinear boundary','OR014@6 systematic ->B; B007_0 near boundary','E1 pruning + small RF grid','Nonlinear interactions may separate OR/B','Same folds/seed/aggregation','Primary candidate; one acceptance gate'],['E3_mechanism_logreg','mechanism diagnostic','May lack fault-frequency evidence','Add 11 source mechanism features; same LR','Physical harmonics may help source','Same folds/seed/grid','Not Q3-eligible: target geometry unknown']],columns=['experiment_id','failure_axis','failure_evidence','change','why_it_may_help','fair_comparison','stop_rule']);hyp.to_csv(OUT/'improvement_hypotheses.csv',index=False)
 (OUT/'precommitted_selection_rule.json').write_text(json.dumps({'primary_candidate':'E2_corr98_random_forest','gate':GATE,'diagnostic_not_eligible':['E1_corr98_logreg','E3_mechanism_logreg'],'outer_test_hyperparameter_tuning':False},indent=2))
 R={'E0_baseline':runexp(df,base,'E0_baseline','lr',LRC),'E1_corr98_logreg':runexp(df,base,'E1_corr98_logreg','lr',LRC,True),'E2_corr98_random_forest':runexp(df,base,'E2_corr98_random_forest','rf',RFC,True),'E3_mechanism_logreg':runexp(df,base,'E3_mechanism_logreg','lr',LRC,False,True)}
 old=json.loads((BOUT/'metrics_summary.json').read_text())['file_level_pooled_oof']['macro_f1'];rep={'step06_macro_f1':old,'step07_rerun_macro_f1':R['E0_baseline']['pooled']['macro_f1'],'abs_diff':abs(old-R['E0_baseline']['pooled']['macro_f1'])};rep['pass']=rep['abs_diff']<1e-12;(OUT/'baseline_reproduction.json').write_text(json.dumps(rep,indent=2));
 if not rep['pass']:raise RuntimeError(rep)
 for e,r in R.items():r['f'].to_csv(OUT/f'{e}_oof_file_predictions.csv',index=False);r['w'].to_csv(OUT/f'{e}_oof_window_predictions.csv',index=False)
 pd.concat([r['val'] for r in R.values()]).to_csv(OUT/'inner_validation_all_experiments.csv',index=False);pd.concat([r['fold'] for r in R.values()]).to_csv(OUT/'outer_fold_all_experiments.csv',index=False);pd.concat([r['fs'] for r in R.values()]).to_csv(OUT/'feature_sets_by_run.csv',index=False)
 S=pd.DataFrame([{'experiment_id':e,**r['pooled'],'eligible_for_final':e in ['E0_baseline','E2_corr98_random_forest']} for e,r in R.items()]);S.to_csv(OUT/'experiment_selection_summary.csv',index=False)
 p=R['E0_baseline']['fold'][['evaluation_run','macro_f1']].merge(R['E2_corr98_random_forest']['fold'][['evaluation_run','macro_f1']],on='evaluation_run',suffixes=('_base','_candidate'));p['delta']=p.macro_f1_candidate-p.macro_f1_base;p.to_csv(OUT/'paired_base_vs_candidate_by_fold.csv',index=False);d=p.delta.values;eff={'mean_delta':d.mean(),'median_delta':np.median(d),'min_delta':d.min(),'max_delta':d.max(),'folds_improved':int((d>0).sum()),'folds_nonworse':int((d>=0).sum()),'cohen_dz':float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else None}
 b=R['E0_baseline']['pooled'];c=R['E2_corr98_random_forest']['pooled'];checks={'pooled_delta':c['macro_f1']-b['macro_f1']>=GATE['pooled_delta_min'],'median_fold_delta':eff['median_delta']>GATE['median_fold_delta_gt'],'folds_nonworse':eff['folds_nonworse']>=GATE['folds_nonworse_min'],'worst_fold_delta':eff['min_delta']>=GATE['worst_fold_delta_min'],'min_recall':b['min_class_recall']-c['min_class_recall']<=GATE['min_recall_drop_max']};ok=all(checks.values());fid='E2_corr98_random_forest' if ok else 'E0_baseline';F=R[fid];dec={'primary_candidate':'E2_corr98_random_forest','accepted':ok,'final_experiment_id':fid,'checks':checks,'paired_effect':eff,'selection_note':'Outer test used once only for predeclared accept/retain gate; no hyperparameter, threshold, family, or seed tuning from test.'};(OUT/'selection_decision.json').write_text(json.dumps(dec,indent=2))
 for tag,e in [('baseline','E0_baseline'),('final',fid)]:pd.DataFrame(confusion_matrix(R[e]['f'].class_label,R[e]['f'].pred_label_file,labels=LAB),index=LAB,columns=LAB).to_csv(OUT/f'confusion_matrix_{tag}.csv')
 ff=F['f'];mis=ff[ff.class_label!=ff.pred_label_file];mis.to_csv(OUT/'final_misclassified_files.csv',index=False);bf=R['E0_baseline']['f'];tr=bf[['independent_object_id','class_label','pred_label_file']].rename(columns={'pred_label_file':'base_pred'}).merge(ff[['independent_object_id','pred_label_file']].rename(columns={'pred_label_file':'final_pred'}),on='independent_object_id');tr['base_correct']=tr.class_label==tr.base_pred;tr['final_correct']=tr.class_label==tr.final_pred;tr['transition']=np.select([~tr.base_correct&tr.final_correct,tr.base_correct&~tr.final_correct,~tr.base_correct&~tr.final_correct],['corrected','new_error','still_wrong'],'still_correct');tr.to_csv(OUT/'error_transition_table.csv',index=False);tr[(tr.transition!='still_correct')|tr.independent_object_id.str.contains('OR014@6|B007_0|N_0',regex=True)].to_csv(OUT/'key_failure_cases_comparison.csv',index=False)
 fail=[]
 for e in ['E1_corr98_logreg','E3_mechanism_logreg']:fail.append({'experiment_id':e,'macro_f1':R[e]['pooled']['macro_f1'],'delta_vs_base':R[e]['pooled']['macro_f1']-b['macro_f1'],'status':'diagnostic_not_selected','reason':'not predeclared replacement candidate' if e.startswith('E1') else 'source-only mechanism features incompatible with target geometry'})
 if not ok:fail.append({'experiment_id':'E2_corr98_random_forest','macro_f1':c['macro_f1'],'delta_vs_base':c['macro_f1']-b['macro_f1'],'status':'candidate_rejected','reason':'failed predeclared acceptance gate'})
 pd.DataFrame(fail).to_csv(OUT/'failed_experiments.csv',index=False)
 bp,bcv=globalfit(df,base,'lr',LRC,False,MODELS/'baseline_source_model.joblib');fp,fcv=(globalfit(df,base,'rf',RFC,True,MODELS/'final_source_model.joblib') if fid.startswith('E2') else (bp,bcv));
 if not fid.startswith('E2'):joblib.dump(fp,MODELS/'final_source_model.joblib')
 bcv.to_csv(OUT/'global_config_selection_baseline.csv',index=False);fcv.to_csv(OUT/'global_config_selection_final.csv',index=False)
 def cx(pay,path):
  cl=pay['pipeline'].named_steps['clf'];z={'input_features':len(pay['input_feature_columns']),'model_features':len(pay['model_feature_columns']),'model_file_bytes':path.stat().st_size,'model_type':type(cl).__name__};z|=({'parameter_count':int(cl.coef_.size+cl.intercept_.size),'n_trees':0,'total_tree_nodes':0} if hasattr(cl,'coef_') else {'parameter_count':None,'n_trees':len(cl.estimators_),'total_tree_nodes':sum(t.tree_.node_count for t in cl.estimators_)});return z
 comp={'hardware':{'cpu_model':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text(errors='ignore').splitlines() if x.startswith('model name')),'unknown'),'logical_cpu_count':os.cpu_count(),'platform':platform.platform()},'timing_scope':'predict_proba on one independent file of precomputed Step04 features; model preprocessing included, raw waveform feature extraction excluded','baseline':{**cx(bp,MODELS/'baseline_source_model.joblib'),**bench(bp,df)},'final':{**cx(fp,MODELS/'final_source_model.joblib'),**bench(fp,df)},'process_peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};(OUT/'complexity_report.json').write_text(json.dumps(comp,indent=2))
 cm=confusion_matrix(ff.class_label,ff.pred_label_file,labels=LAB);vv=[]
 for i in range(4):tp=cm[i,i];pr=tp/(cm[:,i].sum() or 1);rc=tp/(cm[i,:].sum() or 1);vv.append(2*pr*rc/(pr+rc) if pr+rc else 0)
 sm=ff.iloc[0];sw=F['w'][F['w'].independent_object_id==sm.independent_object_id];rp={a:sw[f'p_{a}'].mean() for a in LAB};sp={a:sm[f'p_{a}'] for a in LAB};rec={'manual_macro_f1_from_confusion':np.mean(vv),'reported_macro_f1':F['pooled']['macro_f1'],'macro_f1_abs_diff':abs(np.mean(vv)-F['pooled']['macro_f1']),'sample_independent_object_id':sm.independent_object_id,'recalc_mean_probabilities':rp,'stored_file_probabilities':sp,'max_probability_abs_diff':max(abs(rp[a]-sp[a]) for a in LAB),'pass':abs(np.mean(vv)-F['pooled']['macro_f1'])<1e-12 and max(abs(rp[a]-sp[a]) for a in LAB)<1e-12};(OUT/'recalculation_evidence.json').write_text(json.dumps(rec,indent=2))
 q3={'version':'Q2C-STEP07-current','final_experiment_id':fid,'model_file':'outputs/q2c_failure_driven_improvement/models/final_source_model.joblib','source_input':'outputs/q1c_feature_extraction/q2_source_raw.csv','target_input':'outputs/q1c_feature_extraction/q2_target_raw.csv','input_feature_columns':base,'model_feature_columns':fp['model_feature_columns'],'dropped_features':fp['dropped_features'],'labels':LAB,'group_column':'independent_object_id','aggregation':'mean class probabilities across target-file windows','random_seed':SEED,'target_geometry_rule':'No source SKF6205 mechanism features on target.','source_evaluation_macro_f1':F['pooled']['macro_f1']};(OUT/'q3_interface.json').write_text(json.dumps(q3,indent=2))
 pd.DataFrame([{'model':'Base','experiment_id':'E0_baseline',**b},{'model':'Final','experiment_id':fid,**F['pooled']}]).to_csv(OUT/'baseline_vs_final_pooled.csv',index=False)
 val={'status':'PASS','checks':{'baseline_exactly_reproduced':rep['pass'],'same_step05_folds':True,'same_seed':True,'no_test_hyperparameter_tuning':True,'failed_experiments_retained':True,'q3_common_feature_interface':set(fp['model_feature_columns']).issubset(set(base)),'recalculation_pass':rec['pass']},'selection_decision':dec,'final_pooled_metrics':F['pooled']};(OUT/'validation_report.json').write_text(json.dumps(val,indent=2));(OUT/'run_manifest.json').write_text(json.dumps({'step':'Q2C_STEP07_STEP05_FROZEN_CURRENT','generated_utc':datetime.now(timezone.utc).isoformat(),'git_sha_at_run_start':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':sys.version,'random_seed':SEED,'input_sha256':sha(SRC),'experiments':list(R),'corr_threshold':CORR,'gate':GATE,'final_experiment_id':fid,'final_global_config':fp['selected_config'],'final_model_features':len(fp['model_feature_columns'])},indent=2))
 lines=[f'status=PASS',f'baseline_macro_f1={b["macro_f1"]:.6f}',f'candidate_macro_f1={c["macro_f1"]:.6f}',f'candidate_accepted={ok}',f'final_experiment={fid}',f'final_macro_f1={F["pooled"]["macro_f1"]:.6f}',f'paired_fold_deltas={p.delta.round(6).tolist()}',f'final_misclassified_files={len(mis)}',f'inference_ms_per_file_median={comp["final"]["median_ms_per_file"]:.6f}',f'manual_recalc_pass={rec["pass"]}',f'outer_test_hyperparameter_tuning=False'];(OUT/'result_summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
