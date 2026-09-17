from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, re, subprocess, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'outputs'/'q1c_feature_extraction'; OUT=ROOT/'outputs'/'q2b_minimal_baseline'; MODELS=OUT/'models'
OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
SRC=IN/'q2_source_raw.csv'; IFACE=IN/'q2_interface.json'
SEED=2025; LABELS=['OR','IR','B','N']; EPS=1e-12
RUNS={'R1':{'train':['F3','F4'],'validation':['F2'],'test':['F1']},'R2':{'train':['F4','F1'],'validation':['F3'],'test':['F2']},'R3':{'train':['F1','F2'],'validation':['F4'],'test':['F3']},'R4':{'train':['F2','F3'],'validation':['F1'],'test':['F4']}}
CONFIGS=[
 {'config_id':'M1_C0.1_none','C':.1,'class_weight':None,'order':1}, {'config_id':'M1_C0.1_bal','C':.1,'class_weight':'balanced','order':2},
 {'config_id':'M1_C1_none','C':1.,'class_weight':None,'order':3}, {'config_id':'M1_C1_bal','C':1.,'class_weight':'balanced','order':4},
 {'config_id':'M1_C10_none','C':10.,'class_weight':None,'order':5}, {'config_id':'M1_C10_bal','C':10.,'class_weight':'balanced','order':6}]
STEP05_HASHES={'base_fold_csv':'be0b423d5e3cfead8a1b2dc470a23e8db071204497ba78ef818a03bed1e97385','train_val_test_csv':'541b6fa43fa766a6842533d57a09d57777d15688e3e9a0e71807b272be07ff32','protocol_json':'9f3c720c2891de66be598a91808a0cb8b7d184471413fd71953936e995bda904'}

def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def git_sha():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 except:return 'UNKNOWN'

def norm_group(x):
 s=str(x).replace('\\','/'); return s[9:] if s.startswith('data/raw/') else s

def frozen_fold(group, cls, load):
 # Exact STEP05 rule reconstructed from the frozen 56-file list; used only to reproduce the split, never as a model feature.
 name=Path(norm_group(group)).name; load=int(round(float(load)))
 if cls=='N': off=0
 else:
  m=re.match(r'(IR|OR|B)(\d{3})',name,re.I)
  if not m: raise ValueError(name)
  size=int(m.group(2))
  if cls=='B': off={7:0,14:-1,21:-2}[size]
  elif cls=='IR': off={7:-1,14:-2,21:-3}[size]
  else:
   pm=re.search(r'@(3|6|12)_',name); pos=int(pm.group(1)) if pm else None
   off={(7,3):0,(7,6):-1,(7,12):-2,(14,6):1,(21,3):-1,(21,6):-2,(21,12):-3}[(size,pos)]
 return f'F{((load+off)%4)+1}'

def met(y,p):
 pr,rc,f,s=precision_recall_fscore_support(y,p,labels=LABELS,zero_division=0)
 d={'n_files':int(len(y)),'macro_f1':float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'accuracy':float(accuracy_score(y,p)),'min_class_recall':float(np.min(rc))}
 for a,b,c,e,n in zip(LABELS,pr,rc,f,s): d.update({f'precision_{a}':float(b),f'recall_{a}':float(c),f'f1_{a}':float(e),f'support_{a}':int(n)})
 return d

def weights(d):
 n=d.groupby('independent_object_id').size().to_dict(); w=np.array([1/n[g] for g in d.independent_object_id],float); return w/w.mean()

def pipe(cfg):
 return Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('clf',LogisticRegression(C=cfg['C'],class_weight=cfg['class_weight'],solver='lbfgs',max_iter=3000,random_state=SEED))])

def fit(d,feats,cfg):
 p=pipe(cfg); p.fit(d[feats].to_numpy(float),d.class_label.astype(str).to_numpy(),clf__sample_weight=weights(d)); return p

def winpred(p,d,feats,run,role,cfg):
 q=p.predict_proba(d[feats].to_numpy(float)); cls=list(p.named_steps['clf'].classes_)
 o=d[['window_id','independent_object_id','class_label','load_hp','base_fold']].copy().reset_index(drop=True); o['evaluation_run']=run; o['role']=role; o['selected_config_id']=cfg['config_id']; o['selected_C']=cfg['C']; o['selected_class_weight']='None' if cfg['class_weight'] is None else str(cfg['class_weight'])
 for a in LABELS:o[f'p_{a}']=q[:,cls.index(a)]
 o['pred_label_window']=o[[f'p_{a}' for a in LABELS]].idxmax(axis=1).str.replace('p_','',regex=False); return o

def fileagg(w):
 pc=[f'p_{a}' for a in LABELS]; keys=['evaluation_run','role','selected_config_id','selected_C','selected_class_weight','independent_object_id','class_label','load_hp','base_fold']
 a=w.groupby(keys,as_index=False)[pc].mean(); c=w.groupby(['evaluation_run','independent_object_id']).size().rename('n_windows').reset_index(); a=a.merge(c,on=['evaluation_run','independent_object_id'],how='left')
 a['pred_label_file']=a[pc].idxmax(axis=1).str.replace('p_','',regex=False); z=np.sort(a[pc].to_numpy(float),axis=1); a['confidence_top1']=a[pc].max(axis=1); a['margin_top1_top2']=z[:,-1]-z[:,-2]; return a

def smoke(df,feats):
 reps=[df.loc[df.class_label==a,'independent_object_id'].drop_duplicates().iloc[0] for a in LABELS]; d=df[df.independent_object_id.isin(reps)].copy(); cfg={'config_id':'smoke','C':1.,'class_weight':None}; p=fit(d,feats,cfg); q=p.predict_proba(d[feats].to_numpy(float)); ok=len(feats)==28 and q.shape==(len(d),4) and np.allclose(q.sum(1),1,atol=1e-9)
 x={'status':'PASS' if ok else 'FAIL','representative_groups':[norm_group(v) for v in reps],'rows':int(len(d)),'feature_count':len(feats),'probability_shape':list(q.shape),'row_sum_max_abs_error':float(np.max(np.abs(q.sum(1)-1)))}; (OUT/'smoke_test.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');
 if not ok:raise RuntimeError('smoke failed')
 return x

def manual_macro(cm):
 v=[]
 for i in range(4):
  tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; v.append(2*p*r/(p+r) if p+r else 0)
 return float(np.mean(v))

def main():
 iface=json.loads(IFACE.read_text(encoding='utf-8')); feats=list(iface['feature_columns']); forbidden=set(iface['forbidden_as_model_features']); df=pd.read_csv(SRC)
 if len(feats)!=28 or forbidden.intersection(feats):raise RuntimeError('feature interface/leakage failure')
 if len(df)!=400 or not np.isfinite(df[feats].to_numpy(float)).all():raise RuntimeError('input feature table failure')
 df['group_norm']=df.independent_object_id.map(norm_group); df['base_fold']=[frozen_fold(g,c,l) for g,c,l in zip(df.independent_object_id,df.class_label,df.load_hp)]
 fm=df[['group_norm','independent_object_id','class_label','load_hp','base_fold']].drop_duplicates(); cc=fm.class_label.value_counts().to_dict()
 if len(fm)!=56 or cc!={'OR':28,'B':12,'IR':12,'N':4}:raise RuntimeError(f'file inventory {len(fm)} {cc}')
 fc=fm.groupby(['base_fold','class_label']).size().unstack(fill_value=0)
 for f in ['F1','F2','F3','F4']:
  if {a:int(fc.loc[f,a]) for a in LABELS}!={'OR':7,'IR':3,'B':3,'N':1}:raise RuntimeError(f'fold coverage {f}')
 sm=smoke(df,feats)
 roles=[]
 for r,spec in RUNS.items():
  for _,z in fm.iterrows():
   role='train' if z.base_fold in spec['train'] else ('validation' if z.base_fold in spec['validation'] else 'test'); roles.append({'evaluation_run':r,'role':role,'base_fold':z.base_fold,'independent_object_id':z.group_norm,'class_label':z.class_label,'load_hp':z.load_hp})
 pd.DataFrame(roles).to_csv(OUT/'frozen_train_validation_test_manifest.csv',index=False,encoding='utf-8-sig')
 sel=[]; allw=[]; allf=[]; rm=[]; mm=[]
 for r,spec in RUNS.items():
  tr=df[df.base_fold.isin(spec['train'])].copy(); va=df[df.base_fold.isin(spec['validation'])].copy(); te=df[df.base_fold.isin(spec['test'])].copy(); sets=[set(x.group_norm) for x in [tr,va,te]]
  if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise RuntimeError(f'group leakage {r}')
  cand=[]
  for cfg in CONFIGS:
   p=fit(tr,feats,cfg); fv=fileagg(winpred(p,va,feats,r,'validation',cfg)); row={'evaluation_run':r,'config_id':cfg['config_id'],'C':cfg['C'],'class_weight':'None' if cfg['class_weight'] is None else str(cfg['class_weight']),'config_order':cfg['order'],**met(fv.class_label,fv.pred_label_file)}; cand.append(row); sel.append(row)
  c=pd.DataFrame(cand).sort_values(['macro_f1','min_class_recall','config_order'],ascending=[False,False,True]).iloc[0]; cfg=next(x for x in CONFIGS if x['config_id']==c.config_id)
  dev=pd.concat([tr,va],ignore_index=True); p=fit(dev,feats,cfg); mp=MODELS/f'logreg_{r}_{cfg["config_id"]}.joblib'; joblib.dump({'pipeline':p,'feature_columns':feats,'labels':LABELS,'selected_config':cfg,'evaluation_run':r,'random_seed':SEED,'split':spec},mp)
  wt=winpred(p,te,feats,r,'test',cfg); ft=fileagg(wt); allw.append(wt); allf.append(ft); rm.append({'evaluation_run':r,'train_folds':'+'.join(spec['train']),'validation_fold':spec['validation'][0],'test_fold':spec['test'][0],'selected_config_id':cfg['config_id'],'selected_C':cfg['C'],'selected_class_weight':'None' if cfg['class_weight'] is None else str(cfg['class_weight']),**met(ft.class_label,ft.pred_label_file)}); mm.append({'evaluation_run':r,'model_file':str(mp.relative_to(ROOT)),'selected_config_id':cfg['config_id']})
 pd.DataFrame(sel).to_csv(OUT/'validation_config_scores.csv',index=False,encoding='utf-8-sig'); runm=pd.DataFrame(rm); runm.to_csv(OUT/'run_file_metrics.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(mm).to_csv(OUT/'saved_models_manifest.csv',index=False,encoding='utf-8-sig')
 w=pd.concat(allw,ignore_index=True); f=pd.concat(allf,ignore_index=True); w.to_csv(OUT/'oof_window_predictions.csv',index=False,encoding='utf-8-sig'); f.to_csv(OUT/'oof_file_predictions.csv',index=False,encoding='utf-8-sig')
 if len(f)!=56 or f.independent_object_id.map(norm_group).nunique()!=56 or len(w)!=400 or w.window_id.nunique()!=400:raise RuntimeError('OOF coverage')
 pooled=met(f.class_label,f.pred_label_file); wax={'n_windows':400,'macro_f1':float(f1_score(w.class_label,w.pred_label_window,labels=LABELS,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(w.class_label,w.pred_label_window)),'accuracy':float(accuracy_score(w.class_label,w.pred_label_window))}
 cm=confusion_matrix(f.class_label,f.pred_label_file,labels=LABELS); pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(OUT/'confusion_matrix_file_counts.csv',encoding='utf-8-sig'); pct=cm/cm.sum(1,keepdims=True); pd.DataFrame(pct,index=LABELS,columns=LABELS).to_csv(OUT/'confusion_matrix_file_row_pct.csv',encoding='utf-8-sig')
 pr,rc,ff,su=precision_recall_fscore_support(f.class_label,f.pred_label_file,labels=LABELS,zero_division=0); pd.DataFrame({'class_label':LABELS,'precision':pr,'recall':rc,'f1':ff,'support_files':su}).to_csv(OUT/'per_class_file_metrics.csv',index=False,encoding='utf-8-sig')
 mis=f[f.class_label!=f.pred_label_file].copy().sort_values(['class_label','pred_label_file','load_hp','independent_object_id']); mis.to_csv(OUT/'misclassified_files.csv',index=False,encoding='utf-8-sig'); (mis.groupby(['class_label','pred_label_file']).size().rename('n_files').reset_index() if len(mis) else pd.DataFrame(columns=['class_label','pred_label_file','n_files'])).to_csv(OUT/'misclassification_pairs.csv',index=False,encoding='utf-8-sig')
 lr=[]
 for load,g in f.groupby('load_hp'):lr.append({'load_hp':load,**met(g.class_label,g.pred_label_file)})
 pd.DataFrame(lr).sort_values('load_hp').to_csv(OUT/'file_metrics_by_load.csv',index=False,encoding='utf-8-sig')
 sample='source_domain/cwru_48khz_de/OR007@12_0.mat'; sw=w[w.independent_object_id.map(norm_group)==sample]; sf=f[f.independent_object_id.map(norm_group)==sample].iloc[0]; rp={a:float(sw[f'p_{a}'].mean()) for a in LABELS}; sp={a:float(sf[f'p_{a}']) for a in LABELS}; rec={'manual_macro_f1_from_confusion':manual_macro(cm),'reported_macro_f1':pooled['macro_f1'],'macro_f1_abs_diff':abs(manual_macro(cm)-pooled['macro_f1']),'sample_independent_object_id':sample,'sample_window_count':int(len(sw)),'recalc_mean_probabilities':rp,'stored_file_probabilities':sp,'max_probability_abs_diff':max(abs(rp[a]-sp[a]) for a in LABELS),'recalc_pred_label':max(rp,key=rp.get),'stored_pred_label':str(sf.pred_label_file)}; rec['pass']=bool(rec['macro_f1_abs_diff']<1e-12 and rec['max_probability_abs_diff']<1e-12 and rec['recalc_pred_label']==rec['stored_pred_label']); (OUT/'recalculation_evidence.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
 vals=runm.macro_f1.to_numpy(float); dist={'macro_f1_median':float(np.median(vals)),'macro_f1_min':float(vals.min()),'macro_f1_max':float(vals.max()),'macro_f1_range':float(vals.max()-vals.min()),'macro_f1_mean':float(vals.mean())}; flags={'pooled_macro_f1_below_0.60':pooled['macro_f1']<.60,'any_pooled_class_recall_below_0.50':any(pooled[f'recall_{a}']<.5 for a in LABELS),'fold_macro_f1_range_above_0.20':dist['macro_f1_range']>.20}; summary={'status':'PASS','model':'M1 L2 multinomial LogisticRegression only','file_level_pooled_oof':pooled,'outer_run_distribution':dist,'window_level_auxiliary':wax,'performance_warning_flags_from_step05':flags}; (OUT/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 manifest={'step':'Q2B_STEP06_STEP05_FROZEN','generated_utc':datetime.now(timezone.utc).isoformat(),'git_sha_at_run_start':git_sha(),'python':sys.version,'platform':platform.platform(),'random_seed':SEED,'input_source_csv':str(SRC.relative_to(ROOT)),'input_source_sha256':sha256(SRC),'input_interface_json':str(IFACE.relative_to(ROOT)),'input_interface_sha256':sha256(IFACE),'step05_hashes':STEP05_HASHES,'feature_count':28,'labels':LABELS,'evaluation_runs':RUNS,'candidate_model_family':'M1 only','candidate_configurations_per_run':6,'maximum_validation_candidate_fits':24,'model':'L2 multinomial LogisticRegression','imputer':'median, training-boundary only','scaler':'StandardScaler, training-boundary only','feature_selection':'none','class_imbalance':'file-equal window weights + class_weight validation hyperparameter; no oversampling','aggregation':'mean window probabilities per independent file'}; (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 checks={'smoke_test_pass':sm['status']=='PASS','source_56_files_400_windows':len(fm)==56 and len(df)==400,'four_runs_group_disjoint':True,'all_roles_have_four_classes':True,'only_m1_model_family_run':True,'training_boundary_pipeline':True,'no_feature_name_leakage':not bool(forbidden.intersection(feats)),'no_window_oversampling':True,'oof_files_exactly_56_once':len(f)==56 and f.independent_object_id.map(norm_group).nunique()==56,'oof_windows_exactly_400_once':len(w)==400 and w.window_id.nunique()==400,'independent_recalc_pass':rec['pass'],'model_files_saved':len(list(MODELS.glob('logreg_R*.joblib')))==4}; val={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'selected_config_by_run':runm[['evaluation_run','selected_config_id','selected_C','selected_class_weight']].to_dict('records'),'performance_warning_flags_from_step05':flags}; (OUT/'validation_report.json').write_text(json.dumps(val,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['Q2B / STEP06 minimal viable baseline under STEP05 frozen protocol',f"status={val['status']}",f"smoke={sm['status']}",'selected_configs='+','.join(f"{x.evaluation_run}:{x.selected_config_id}" for _,x in runm.iterrows()),f"file_pooled_macro_f1={pooled['macro_f1']:.6f}",f"file_pooled_balanced_accuracy={pooled['balanced_accuracy']:.6f}",f"file_pooled_accuracy={pooled['accuracy']:.6f}",'class_recalls='+','.join(f"{a}:{pooled[f'recall_{a}']:.6f}" for a in LABELS),f"window_aux_macro_f1={wax['macro_f1']:.6f}",f'misclassified_files={len(mis)}',f"manual_recalc_pass={rec['pass']}",f'performance_warning_flags={flags}']; (OUT/'result_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines));
 if val['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
