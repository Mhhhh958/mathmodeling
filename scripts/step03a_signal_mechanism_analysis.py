#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import welch, hilbert, detrend, firwin, resample_poly
from scipy.stats import kurtosis

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
CFG=ROOT/"protocol"/"03A"/"run_config.json"
BLUEPRINT=ROOT/"protocol"/"03A"/"question1B_argument_blueprint.json"
BASIS=ROOT/"protocol"/"03A"/"problem_statement_mechanism_basis.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
DM=ROOT/"protocol"/"01A"/"data_manifest.json"
FREEZE02=ROOT/"protocol"/"02A"/"latest_freeze_package.json"
RUN02=ROOT/"protocol"/"02A"/"latest_run_id.txt"
SEED=20260919
COMMON_FS=12000
COMMON_FMAX=6000.0
NPERSEG=4096
HARMONICS=3
TREND_R2_THRESHOLD=0.01
AA_TAPS=401
AA_CUTOFF=5500.0
AA_BETA=8.6

PARAMS={
 "SKF6205_DE":{"Nd":9,"d":0.3126,"D":1.537},
 "SKF6203_FE":{"Nd":9,"d":0.2656,"D":1.122},
}
CLASS_CHAR={"OR":"BPFO","IR":"BPFI","B":"BSF"}

def cj(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b):return hashlib.sha256(b).hexdigest()
def hf(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def git_head():return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()

def boolish(v):
 if isinstance(v,bool): return v
 return str(v).strip().lower() in {"true","1","yes"}

def primary_signal(mat,subgroup,stem):
 keys=[k for k in mat if not k.startswith("__")]
 want="FE_time" if subgroup=="source_12khz_fe" else "DE_time"
 for k in keys:
  if k.endswith(want):return np.asarray(mat[k],dtype=float).squeeze(),k
 raise RuntimeError(f"no {want} in {stem} ({subgroup})")

def bearing_for(subgroup):
 if subgroup=="source_12khz_fe": return "SKF6203_FE"
 if subgroup=="source_48khz_normal": return "SKF6205_DE"
 raise RuntimeError(f"03-A MVP subgroup not expected: {subgroup}")

def native_fs(subgroup):
 if subgroup=="source_12khz_fe":return 12000
 if subgroup=="source_48khz_normal":return 48000
 raise RuntimeError(subgroup)

def mechanism(rpm,bearing):
 p=PARAMS[bearing]; fr=float(rpm)/60.0; q=p["d"]/p["D"]
 out={
   "fr":fr,
   "FTF":0.5*fr*(1-q),
   "BPFO":fr*p["Nd"]/2.0*(1-q),
   "BPFI":fr*p["Nd"]/2.0*(1+q),
   "BSF":fr*p["D"]/p["d"]*(1-q*q),
 }
 return out

def order_factors(bearing):
 p=PARAMS[bearing]; q=p["d"]/p["D"]
 return {
   "FTF_order":0.5*(1-q),
   "BPFO_order":p["Nd"]/2.0*(1-q),
   "BPFI_order":p["Nd"]/2.0*(1+q),
   "BSF_order":p["D"]/p["d"]*(1-q*q),
 }

def trend_r2(x):
 y=np.asarray(x,float); n=len(y)
 t=np.linspace(-1.0,1.0,n)
 coef=np.polyfit(t,y,1); pred=np.polyval(coef,t)
 den=np.sum((y-y.mean())**2)
 return float(np.sum((pred-pred.mean())**2)/den) if den>0 else 0.0

def anti_alias_resample_48_to_12(x):
 taps=firwin(AA_TAPS,AA_CUTOFF,window=("kaiser",AA_BETA),fs=48000,pass_zero="lowpass")
 y=resample_poly(x,up=1,down=4,window=taps,padtype="line")
 return np.asarray(y,float),taps

def preprocess(x,fs):
 raw=np.asarray(x,float).ravel()
 demean=raw-np.mean(raw)
 r2=trend_r2(demean)
 detrend_applied=r2>TREND_R2_THRESHOLD
 y=detrend(demean,type="linear") if detrend_applied else demean.copy()
 resampled=False
 if fs==48000:
  y,taps=anti_alias_resample_48_to_12(y); resampled=True
 elif fs==12000:
  taps=None
 else: raise RuntimeError(fs)
 return raw,demean,y,{"trend_r2":r2,"detrend_applied":detrend_applied,"resampled":resampled,"taps":taps}

def psd(x,fs):
 nper=min(NPERSEG,len(x))
 f,p=welch(x,fs=fs,window="hann",nperseg=nper,noverlap=nper//2,detrend=False,scaling="density")
 return f,p,float(fs/nper)

def integ(f,p,lo,hi):
 m=(f>=max(0,lo))&(f<=min(float(f[-1]),hi))
 if m.sum()<2:return 0.0
 return float(np.trapz(p[m],f[m]))

def freq_band(f,p,fc,df):
 tol=max(2.0*df,0.02*fc)
 return integ(f,p,fc-tol,fc+tol),tol

def mechanism_metrics(x,fs,m,class_code):
 f,p,df=psd(x,fs)
 total=integ(f,p,0,min(COMMON_FMAX,float(f[-1])))+1e-30
 char=CLASS_CHAR[class_code]
 f0=m[char]
 h_energy=0.0; sb_energy=0.0; bands=[]; sidebands=[]
 for k in range(1,HARMONICS+1):
  fc=k*f0
  if fc>=min(COMMON_FMAX,float(f[-1])):continue
  e,tol=freq_band(f,p,fc,df);h_energy+=e
  bands.append({"harmonic":k,"center_hz":fc,"tol_hz":tol,"energy":e})
  mod=m["fr"] if class_code=="IR" else (m["FTF"] if class_code=="B" else None)
  if mod is not None:
   for sign in (-1,1):
    sb=fc+sign*mod
    if 0<sb<min(COMMON_FMAX,float(f[-1])):
     ee,tt=freq_band(f,p,sb,df);sb_energy+=ee
     sidebands.append({"harmonic":k,"center_hz":sb,"tol_hz":tt,"energy":ee})
 # nearest peak around fundamental
 _,tol=freq_band(f,p,f0,df)
 mm=(f>=max(0,f0-tol))&(f<=f0+tol)
 if mm.any():
  local_f=f[mm];local_p=p[mm];peak_f=float(local_f[np.argmax(local_p)])
 else: peak_f=float("nan")
 return {
   "mechanism_class":class_code,"char_name":char,"char_hz":f0,"welch_df_hz":df,
   "harmonic_energy_ratio":h_energy/total,
   "sideband_energy_ratio":sb_energy/total,
   "mechanism_score":(h_energy+0.5*sb_energy)/total,
   "local_peak_hz":peak_f,
   "peak_offset_hz":peak_f-f0 if np.isfinite(peak_f) else np.nan,
   "peak_offset_order":(peak_f-f0)/m["fr"] if np.isfinite(peak_f) else np.nan,
   "bands_json":cj(bands),"sidebands_json":cj(sidebands),
 }

def time_spectral_metrics(x,fs):
 x=np.asarray(x,float)
 rms=float(np.sqrt(np.mean(x*x))); peak=float(np.max(np.abs(x))); meanabs=float(np.mean(np.abs(x)))
 f,p,df=psd(x,fs)
 total=integ(f,p,0,float(f[-1]))
 common=integ(f,p,0,min(COMMON_FMAX,float(f[-1])))
 pp=p[p>0]; pn=pp/pp.sum() if len(pp) else np.array([1.0])
 sent=float(-(pn*np.log(pn)).sum()/np.log(len(pn))) if len(pn)>1 else 0.0
 return {
  "rms":rms,"peak":peak,"crest_factor":peak/(rms+1e-30),"impulse_factor":peak/(meanabs+1e-30),
  "kurtosis":float(kurtosis(x,fisher=False,bias=False)),"total_psd_energy":total,"common_0_6k_psd_energy":common,
  "spectral_entropy":sent,"welch_df_hz":df
 }

def hypothetical_spurious(x,fs,m):
 vals={}
 for c in ["OR","IR","B"]:
  mm=mechanism_metrics(x,fs,m,c)
  vals[c]=mm["mechanism_score"]
 return max(vals.values()),vals

def envelope_metrics(x,fs,m,class_code):
 env=np.abs(hilbert(np.asarray(x,float)))
 env=env-np.mean(env)
 q=mechanism_metrics(env,fs,m,class_code)
 q={f"env_{k}":v for k,v in q.items() if k not in {"bands_json","sidebands_json"}}
 return q,env

def spectrum_rows(x,fs,sample_id,stage):
 f,p,df=psd(x,fs);m=f<=COMMON_FMAX
 return pd.DataFrame({"sample_id":sample_id,"stage":stage,"frequency_hz":f[m],"psd":p[m]})

def waveform_rows(x,fs,sample_id,stage,duration=0.5):
 n=min(len(x),int(round(fs*duration)))
 return pd.DataFrame({"sample_id":sample_id,"stage":stage,"time_s":np.arange(n)/fs,"value":np.asarray(x[:n],float)})

def psd_fidelity(x,fs):
 # Match physical frequency resolution across sampling rates:
 # 48k uses 16384 samples when 12k uses 4096, so both have df=2.9296875 Hz.
 target_nper=int(round(NPERSEG*float(fs)/COMMON_FS))
 nper=min(target_nper,len(x))
 f,p=welch(x,fs=fs,window="hann",nperseg=nper,noverlap=nper//2,detrend=False,scaling="density")
 return f,p,float(fs/nper)

def fidelity_spectral_metrics(x,fs):
 x=np.asarray(x,float)
 f,p,df=psd_fidelity(x,fs)
 return {
  "rms":float(np.sqrt(np.mean(x*x))),
  "total_psd_energy":integ(f,p,0,float(f[-1])),
  "common_0_6k_psd_energy":integ(f,p,0,min(COMMON_FMAX,float(f[-1]))),
  "df_hz":df,
 }

def fidelity_row(rel,stage,before,bfs,after,afs,m):
 bm=fidelity_spectral_metrics(before,bfs);am=fidelity_spectral_metrics(after,afs)
 bf,bp,bdf=psd_fidelity(before,bfs); af,ap,adf=psd_fidelity(after,afs)
 common_df=max(bdf,adf)
 row={"relative_path":rel,"stage":stage,"before_fs_hz":bfs,"after_fs_hz":afs,
      "fidelity_df_before_hz":bdf,"fidelity_df_after_hz":adf,
      "rms_before":bm["rms"],"rms_after":am["rms"],"rms_ratio":am["rms"]/(bm["rms"]+1e-30),
      "total_psd_before":bm["total_psd_energy"],"total_psd_after":am["total_psd_energy"],"total_psd_ratio":am["total_psd_energy"]/(bm["total_psd_energy"]+1e-30),
      "common_0_6k_before":bm["common_0_6k_psd_energy"],"common_0_6k_after":am["common_0_6k_psd_energy"],"common_0_6k_ratio":am["common_0_6k_psd_energy"]/(bm["common_0_6k_psd_energy"]+1e-30)}
 for char in ["BPFO","BPFI","BSF"]:
  fc=m[char]; tol=max(2.0*common_df,0.02*fc)
  eb=integ(bf,bp,fc-tol,fc+tol); ea=integ(af,ap,fc-tol,fc+tol)
  row[f"{char}_common_tol_hz"]=tol
  row[f"{char}_fund_energy_before"]=eb
  row[f"{char}_fund_energy_after"]=ea
  row[f"{char}_fund_energy_ratio"]=ea/(eb+1e-30)
 return row

def main():
 t0=time.time();random.seed(SEED);np.random.seed(SEED)
 cfg=json.loads(CFG.read_text(encoding="utf-8"))
 basis=json.loads(BASIS.read_text(encoding="utf-8"))
 dm=json.loads(DM.read_text(encoding="utf-8"))
 fr02=json.loads(FREEZE02.read_text(encoding="utf-8"))
 assert fr02["status"]=="passed"
 assert dm["raw_data_version_id"]==fr02["versions"]["raw_data_version_id"]
 run02=RUN02.read_text(encoding="utf-8").strip()
 sel_path=ROOT/"outputs"/"runs"/run02/"artifacts"/"source_selection_list.csv"
 sel=pd.read_csv(sel_path)
 mvp=sel[sel["MVP_retain"].map(boolish)].copy()
 if len(mvp)!=49: raise RuntimeError(f"MVP file count changed: {len(mvp)}")
 if set(mvp["subgroup"])!={"source_12khz_fe","source_48khz_normal"}: raise RuntimeError("MVP subgroups changed")
 meta=pd.read_csv(ROOT/dm["metadata_artifact"])
 q=mvp.merge(meta,on=["relative_path","independent_object_id","subgroup","class_label","load_hp","fault_size_in","outer_race_position_clock"],how="left",suffixes=("","_meta"))
 if len(q)!=49 or q["rpm"].isna().any():raise RuntimeError("MVP metadata/RPM join failed")

 created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 code=git_head();cfgsha=hf(CFG);envsha=hf(ENV);bpsha=hf(BLUEPRINT);basis_sha=hf(BASIS)
 binding={"code_version_id":"git:"+code,"data_version_id":fr02["freeze_package_id"],"run_config_sha256":cfgsha,"seed":SEED,"environment_sha256":envsha}
 bd=hb(cj(binding).encode())
 run_id=f"run_03-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
 out=ROOT/"outputs"/"runs"/run_id;art=out/"artifacts";logs=out/"logs";mani=out/"manifests"
 for d in [art,logs,mani]:d.mkdir(parents=True,exist_ok=False)

 # Formula and bearing parameter table.
 param_rows=[]
 for b,p in PARAMS.items():
  of=order_factors(b)
  param_rows.append({"bearing":b,"rolling_elements_Nd":p["Nd"],"ball_diameter_in":p["d"],"pitch_diameter_in":p["D"],"d_over_D":p["d"]/p["D"],**of,
    "formula_convention":"problem-statement zero-contact-angle form"})
 pd.DataFrame(param_rows).to_csv(art/"bearing_parameters_and_order_factors.csv",index=False,encoding="utf-8-sig")

 analysis=[];freqrows=[];fidelity=[];cache={};readlog=[]
 for r in q.sort_values("relative_path").itertuples(index=False):
  p=RAW/Path(r.relative_path);mat=loadmat(p);x,var=primary_signal(mat,r.subgroup,p.stem)
  fs=native_fs(r.subgroup);bearing=bearing_for(r.subgroup);m=mechanism(float(r.rpm),bearing)
  raw,demean,proc,pp=preprocess(x,fs)
  pfs=COMMON_FS
  # Stage fidelity: demean always.
  fidelity.append(fidelity_row(r.relative_path,"demean",raw,fs,demean,fs,m))
  if pp["detrend_applied"]:
   detr=detrend(demean,type="linear")
   fidelity.append(fidelity_row(r.relative_path,"linear_detrend",demean,fs,detr,fs,m))
  if pp["resampled"]:
   before=detrend(demean,type="linear") if pp["detrend_applied"] else demean
   fidelity.append(fidelity_row(r.relative_path,"anti_alias_resample_48k_to_12k",before,48000,proc,12000,m))
  tm=time_spectral_metrics(proc,pfs)
  if r.class_label in CLASS_CHAR:
   mm=mechanism_metrics(proc,pfs,m,r.class_label)
   spurious=np.nan
  else:
   spurious,hyp=hypothetical_spurious(proc,pfs,m)
   # use strongest hypothetical mechanism as N hard-negative score.
   fake=max(hyp,key=hyp.get);mm=mechanism_metrics(proc,pfs,m,fake)
  analysis.append({"relative_path":r.relative_path,"independent_object_id":r.independent_object_id,"subgroup":r.subgroup,"class_label":r.class_label,
    "load_hp":r.load_hp,"fault_size_in":r.fault_size_in,"outer_race_position_clock":r.outer_race_position_clock,
    "rpm":r.rpm,"bearing":bearing,"native_fs_hz":fs,"processed_fs_hz":pfs,"signal_var":var,
    "raw_mean":float(np.mean(raw)),"trend_r2":pp["trend_r2"],"detrend_applied":pp["detrend_applied"],"resampled":pp["resampled"],
    **tm,**{k:m[k] for k in ["fr","FTF","BPFO","BPFI","BSF"]},**mm,
    "spurious_mechanism_score":spurious})
  freqrows.append({"relative_path":r.relative_path,"class_label":r.class_label,"rpm":r.rpm,"bearing":bearing,**m,
    "BPFO_order":m["BPFO"]/m["fr"],"BPFI_order":m["BPFI"]/m["fr"],"BSF_order":m["BSF"]/m["fr"],"FTF_order":m["FTF"]/m["fr"],
    "frequency_use":"class mechanism" if r.class_label in CLASS_CHAR else "hypothetical reference only; N has no fault truth"})
  cache[r.relative_path]={"raw":raw,"processed":proc,"fs_native":fs,"fs_processed":pfs,"m":m,"bearing":bearing}
  readlog.append({"relative_path":r.relative_path,"signal_var":var,"native_points":len(raw),"native_fs_hz":fs,"processed_points":len(proc),"processed_fs_hz":pfs,
    "detrend_applied":pp["detrend_applied"],"resampled":pp["resampled"],"status":"OK"})

 adf=pd.DataFrame(analysis).sort_values("relative_path").reset_index(drop=True)
 adf.to_csv(art/"source_signal_mechanism_metrics.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame(freqrows).to_csv(art/"file_mechanism_frequencies.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame(readlog).to_csv(logs/"actual_signal_read_log.csv",index=False,encoding="utf-8-sig")
 fdf=pd.DataFrame(fidelity)
 # Add stage-specific acceptance, with full-spectrum loss on resampling reported but not treated as failure.
 def accept(row):
  if row.stage=="demean":
   return 0.99<=row["rms_ratio"]<=1.01 and all(0.98<=row[f"{c}_fund_energy_ratio"]<=1.02 for c in ["BPFO","BPFI","BSF"])
  if row.stage=="linear_detrend":
   return 0.95<=row["rms_ratio"]<=1.05 and all(0.90<=row[f"{c}_fund_energy_ratio"]<=1.10 for c in ["BPFO","BPFI","BSF"])
  if row.stage=="anti_alias_resample_48k_to_12k":
   return 0.80<=row["common_0_6k_ratio"]<=1.10 and all(0.90<=row[f"{c}_fund_energy_ratio"]<=1.10 for c in ["BPFO","BPFI","BSF"])
  return False
 fdf["information_fidelity_pass"]=fdf.apply(accept,axis=1)
 fdf.to_csv(art/"preprocessing_information_fidelity.csv",index=False,encoding="utf-8-sig")

 # Deterministic representative/hard selection.
 picks=[]
 for c in ["OR","IR","B"]:
  g=adf[adf.class_label==c].sort_values("relative_path").copy()
  med=float(g.mechanism_score.median())
  g["median_gap"]=(g.mechanism_score-med).abs()
  typ=g.sort_values(["median_gap","relative_path"]).iloc[0]
  dif=g.sort_values(["mechanism_score","relative_path"]).iloc[0]
  if typ.relative_path==dif.relative_path and len(g)>1:
   dif=g.sort_values(["mechanism_score","relative_path"]).iloc[1]
  picks.append({"class_label":c,"sample_role":"typical","relative_path":typ.relative_path,"selection_metric":"mechanism_score closest to class median","mechanism_score":typ.mechanism_score,"class_median_score":med})
  picks.append({"class_label":c,"sample_role":"difficult","relative_path":dif.relative_path,"selection_metric":"lowest class mechanism_score","mechanism_score":dif.mechanism_score,"class_median_score":med})
 g=adf[adf.class_label=="N"].sort_values("relative_path").copy()
 med=float(g.spurious_mechanism_score.median())
 g["median_gap"]=(g.spurious_mechanism_score-med).abs()
 typ=g.sort_values(["median_gap","relative_path"]).iloc[0]
 dif=g.sort_values(["spurious_mechanism_score","relative_path"],ascending=[False,True]).iloc[0]
 if typ.relative_path==dif.relative_path and len(g)>1:
  dif=g.sort_values(["spurious_mechanism_score","relative_path"],ascending=[False,True]).iloc[1]
 picks.append({"class_label":"N","sample_role":"typical","relative_path":typ.relative_path,"selection_metric":"spurious mechanism score closest to N median","mechanism_score":typ.spurious_mechanism_score,"class_median_score":med})
 picks.append({"class_label":"N","sample_role":"difficult","relative_path":dif.relative_path,"selection_metric":"highest spurious mechanism score (hard negative)","mechanism_score":dif.spurious_mechanism_score,"class_median_score":med})
 picks=pd.DataFrame(picks)
 rep=picks.merge(adf,on=["class_label","relative_path"],how="left",suffixes=("_selection",""))
 env_rows=[];wave=[];spec=[];envspec=[]
 for rr in rep.itertuples(index=False):
  c=rr.class_label; obj=cache[rr.relative_path]; proc=obj["processed"];m=obj["m"]
  env_class=c if c in CLASS_CHAR else max(["OR","IR","B"],key=lambda cc: mechanism_metrics(proc,COMMON_FS,m,cc)["mechanism_score"])
  em,env=envelope_metrics(proc,COMMON_FS,m,env_class)
  env_rows.append({"relative_path":rr.relative_path,"class_label":c,"sample_role":rr.sample_role,"envelope_reference_class":env_class,**em})
  sid=f"{c}_{rr.sample_role}_{Path(rr.relative_path).stem}"
  wave.append(waveform_rows(obj["raw"],obj["fs_native"],sid,"raw_native"))
  wave.append(waveform_rows(proc,COMMON_FS,sid,"primary_processed"))
  spec.append(spectrum_rows(proc,COMMON_FS,sid,"primary_processed"))
  envspec.append(spectrum_rows(env,COMMON_FS,sid,"hilbert_envelope_no_bandpass"))
 envdf=pd.DataFrame(env_rows)
 rep=rep.merge(envdf,on=["relative_path","class_label","sample_role"],how="left")
 rep.to_csv(art/"typical_difficult_sample_comparison.csv",index=False,encoding="utf-8-sig")
 pd.concat(wave,ignore_index=True).to_csv(art/"representative_waveform_segments.csv",index=False,encoding="utf-8-sig")
 pd.concat(spec,ignore_index=True).to_csv(art/"representative_psd.csv",index=False,encoding="utf-8-sig")
 pd.concat(envspec,ignore_index=True).to_csv(art/"representative_envelope_psd.csv",index=False,encoding="utf-8-sig")

 # Aggregate mechanism/noise/operating-condition summary.
 agg=[]
 for c,g in adf.groupby("class_label"):
  agg.append({"class_label":c,"files":len(g),"rms_median":float(g.rms.median()),"rms_iqr":float(g.rms.quantile(.75)-g.rms.quantile(.25)),
   "kurtosis_median":float(g["kurtosis"].median()),"crest_factor_median":float(g.crest_factor.median()),
   "spectral_entropy_median":float(g.spectral_entropy.median()),
   "mechanism_score_median":float(g.mechanism_score.median()),"mechanism_score_min":float(g.mechanism_score.min()),"mechanism_score_max":float(g.mechanism_score.max()),
   "sideband_ratio_median":float(g.sideband_energy_ratio.median()),"rpm_min":float(g.rpm.min()),"rpm_max":float(g.rpm.max())})
 pd.DataFrame(agg).to_csv(art/"class_signal_mechanism_summary.csv",index=False,encoding="utf-8-sig")

 # Preprocessing decisions based on actual diagnostics.
 ntrend=int(adf.detrend_applied.sum())
 aa=fdf[fdf.stage=="anti_alias_resample_48k_to_12k"]
 decisions=[
  {"operation":"去均值","decision":"必要并应用","reason":"移除DC，避免0Hz分量干扰谱能量；关键机理频带保真由逐文件检查确认","applied_files":49},
  {"operation":"线性去趋势","decision":"仅当trend_R2>0.01时应用","reason":f"固定阈值检查后实际触发{ntrend}/49文件；不对其余文件强制去趋势","applied_files":ntrend},
  {"operation":"一般带通/降噪滤波","decision":"不应用","reason":"没有足够依据设定统一带通；避免人为滤掉冲击谐波和调制侧带","applied_files":0},
  {"operation":"48kHz→12kHz重采样","decision":"仅N类4文件应用","reason":"MVP故障类为12kHz FE而N类为48kHz；统一时基前使用显式抗混叠FIR","applied_files":4},
  {"operation":"包络解调","decision":"仅代表性/困难样本诊断分支","reason":"Hilbert包络不替换主信号；不增加带通滤波，主要检查弱机理成分","applied_files":8},
 ]
 pd.DataFrame(decisions).to_csv(art/"preprocessing_decision_table.csv",index=False,encoding="utf-8-sig")

 filter_spec={
  "resampling_filter":{"type":"symmetric FIR low-pass passed explicitly to scipy.signal.resample_poly","native_fs_hz":48000,"output_fs_hz":12000,
   "downsample_factor":4,"num_taps":AA_TAPS,"fir_order":AA_TAPS-1,"cutoff_hz":AA_CUTOFF,"window":"Kaiser","kaiser_beta":AA_BETA,
   "phase_handling":"linear-phase symmetric FIR; resample_poly centers/compensates delay (zero-phase aligned output)","anti_alias_role":"attenuate content approaching/new Nyquist 6000 Hz before downsampling"},
  "general_bandpass":{"applied":False,"reason":"no evidence-supported single band that can be used without risking mechanism harmonics/sidebands"},
  "envelope":{"extra_bandpass_applied":False,"method":"abs(hilbert(primary_processed)) then remove envelope mean"}
 }
 (art/"filter_resampling_spec.json").write_text(json.dumps(filter_spec,ensure_ascii=False,indent=2),encoding="utf-8")

 candidates=pd.DataFrame([
  ["td_rms","RMS","冲击/振动总体强度","time","all"],
  ["td_kurtosis","峭度","稀疏冲击敏感","time","all"],
  ["td_crest_factor","峰值因子","峰值冲击相对能量","time","all"],
  ["mech_harmonic_energy_ratio","故障频率1-3次谐波邻域能量占比","BPFO/BPFI/BSF机理一致性","frequency/order","OR/IR/B"],
  ["mech_peak_offset_order","理论基频附近实测峰的阶次偏差","弱化RPM差异后的频率偏移","order","OR/IR/B"],
  ["ir_fr_sideband_ratio","BPFI谐波±fr侧带能量占比","内圈受转频调制","frequency/order","IR"],
  ["b_ftf_sideband_ratio","BSF谐波±FTF侧带能量占比","滚动体受公转频率调制","frequency/order","B"],
  ["env_harmonic_energy_ratio","Hilbert包络中的故障谐波能量占比","弱冲击解调候选","envelope/order","OR/IR/B"],
  ["spectral_entropy","谱熵","噪声/分散程度","frequency","all"],
 ],columns=["feature_id","name","mechanism_rationale","domain","applicable_class"])
 candidates["status"]="candidate_for_later_feature_extraction_not_finalized"
 candidates.to_csv(art/"mechanism_feature_candidates.csv",index=False,encoding="utf-8-sig")

 boundary={
  "source_bearing_parameters_used":{"source_12khz_fe":"SKF6203_FE","source_48khz_normal":"SKF6205_DE"},
  "target_domain_geometry_used":False,
  "target_rpm_used":False,
  "statement":"03-A analyzes source signals only. SKF6203/SKF6205 dimensions are not transferred to target A-P because equivalent target geometry is not frozen."
 }
 (art/"source_target_geometry_boundary.json").write_text(json.dumps(boundary,ensure_ascii=False,indent=2),encoding="utf-8")

 # Verification 1: random typical and difficult recomputation from raw.
 rng=random.Random(SEED)
 typ_path=rng.choice(rep[rep.sample_role=="typical"].relative_path.tolist())
 dif_path=rng.choice(rep[rep.sample_role=="difficult"].relative_path.tolist())
 recalc=[]
 for path in [typ_path,dif_path]:
  sr=adf[adf.relative_path==path].iloc[0]
  mr=meta[meta.relative_path==path].iloc[0]
  mat=loadmat(RAW/Path(path));x,var=primary_signal(mat,sr.subgroup,Path(path).stem)
  _,_,proc,pp=preprocess(x,native_fs(sr.subgroup));m=mechanism(float(sr.rpm),sr.bearing)
  tm=time_spectral_metrics(proc,COMMON_FS)
  cls=sr.class_label if sr.class_label in CLASS_CHAR else CLASS_CHAR.keys().__iter__().__next__()
  if sr.class_label=="N":
   _,hyp=hypothetical_spurious(proc,COMMON_FS,m);cls=max(hyp,key=hyp.get)
  mm=mechanism_metrics(proc,COMMON_FS,m,cls)
  recalc.append({"relative_path":path,"stored_rms":float(sr.rms),"recalc_rms":tm["rms"],"abs_diff_rms":abs(float(sr.rms)-tm["rms"]),
    "stored_mechanism_score":float(sr.mechanism_score),"recalc_mechanism_score":mm["mechanism_score"],"abs_diff_mechanism_score":abs(float(sr.mechanism_score)-mm["mechanism_score"]),"reference_class":cls})
 recalc_df=pd.DataFrame(recalc);recalc_df.to_csv(art/"validation_metric_recalculation.csv",index=False,encoding="utf-8-sig")
 recalc_ok=bool((recalc_df.abs_diff_rms<1e-12).all() and (recalc_df.abs_diff_mechanism_score<1e-12).all())

 # Verification 2: independent formula/order calculation consistency.
 formula_checks=[]
 for b,p in PARAMS.items():
  fac=order_factors(b);rpm=1770.0;fr=rpm/60.0;m=mechanism(rpm,b)
  for name in ["FTF","BPFO","BPFI","BSF"]:
   formula_checks.append({"bearing":b,"quantity":name,"rpm_test":rpm,"frequency_hz":m[name],"order_from_frequency":m[name]/fr,"order_direct":fac[name+"_order"],"abs_diff":abs(m[name]/fr-fac[name+"_order"])})
 fchk=pd.DataFrame(formula_checks);fchk.to_csv(art/"validation_formula_unit_check.csv",index=False,encoding="utf-8-sig")
 formula_ok=bool((fchk.abs_diff<1e-12).all())

 # Verification 3: information fidelity all applied operations.
 fidelity_ok=bool(fdf.information_fidelity_pass.all())
 # Key evidence numbers for resampling.
 aa_summary={
  "files":int(len(aa)),
  "common_band_ratio_min":float(aa.common_0_6k_ratio.min()) if len(aa) else None,
  "common_band_ratio_max":float(aa.common_0_6k_ratio.max()) if len(aa) else None,
  "mechanism_band_ratio_min":float(aa[[f"{c}_fund_energy_ratio" for c in ["BPFO","BPFI","BSF"]]].min().min()) if len(aa) else None,
  "mechanism_band_ratio_max":float(aa[[f"{c}_fund_energy_ratio" for c in ["BPFO","BPFI","BSF"]]].max().max()) if len(aa) else None,
  "full_spectrum_total_ratio_min":float(aa.total_psd_ratio.min()) if len(aa) else None,
  "full_spectrum_total_ratio_max":float(aa.total_psd_ratio.max()) if len(aa) else None,
  "note":"Fidelity PSDs use matched physical resolution (48k:16384-point Welch; 12k:4096-point Welch). Full-spectrum ratio may be <1 because >6k content is intentionally rejected; pass uses common 0-6k and matched-tolerance mechanism bands."
 }

 validation={
  "step_id":"03-A","run_id":run_id,
  "checks":{
   "random_typical_difficult_recalculation":{"passed":recalc_ok,"evidence":f"outputs/runs/{run_id}/artifacts/validation_metric_recalculation.csv","files":[typ_path,dif_path]},
   "formula_parameter_rpm_unit_check":{"passed":formula_ok,"evidence":f"outputs/runs/{run_id}/artifacts/validation_formula_unit_check.csv","problem_basis":"protocol/03A/problem_statement_mechanism_basis.json"},
   "preprocessing_information_fidelity":{"passed":fidelity_ok,"evidence":f"outputs/runs/{run_id}/artifacts/preprocessing_information_fidelity.csv","resample_summary":aa_summary},
   "typical_and_difficult_all_classes":{"passed":len(rep)==8 and set(rep.class_label)==set(["OR","IR","B","N"]),"evidence":f"outputs/runs/{run_id}/artifacts/typical_difficult_sample_comparison.csv","sample_count":len(rep)},
   "MVP_only":{"passed":len(adf)==49 and set(adf.subgroup)=={"source_12khz_fe","source_48khz_normal"},"evidence":f"outputs/runs/{run_id}/artifacts/source_signal_mechanism_metrics.csv"},
   "target_geometry_not_reused":{"passed":True,"evidence":f"outputs/runs/{run_id}/artifacts/source_target_geometry_boundary.json"},
   "no_final_figures":{"passed":not any(art.glob("*.png")),"evidence":"artifacts directory contains data/table/json/md results only"},
   "no_classifier_training":{"passed":True,"evidence":"03-A code contains no estimator fit/predict"}
  }
 }
 validation["passed"]=all(v["passed"] for v in validation["checks"].values());validation["status"]="passed" if validation["passed"] else "failed"
 (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"03A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

 # concise analysis report with evidence-grounded statements
 rep_small=rep[["class_label","sample_role","relative_path","rpm","mechanism_score","env_mechanism_score","kurtosis","crest_factor","sideband_energy_ratio"]].copy()
 report={
  "run_id":run_id,
  "MVP_files":49,
  "trend":{"threshold_R2":TREND_R2_THRESHOLD,"applied_files":ntrend,"max_R2":float(adf.trend_r2.max()),"median_R2":float(adf.trend_r2.median())},
  "class_summary":pd.read_csv(art/"class_signal_mechanism_summary.csv").to_dict("records"),
  "representative_samples":rep_small.to_dict("records"),
  "resampling_fidelity":aa_summary,
  "interpretation_guard":"Mechanism-frequency proximity is consistency evidence only; no single peak is treated as proof of class.",
  "envelope_policy":"Hilbert envelope without extra bandpass is a diagnostic side branch for representative/difficult samples, not the primary signal replacement."
 }
 (art/"representative_signal_analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

 freeze={
  "schema_version":"00C-A-freeze-1.0","package_type":"A_freeze_package","step_id":"03-A",
  "freeze_package_id":f"FREEZE-03A-{run_id[-8:]}","status":"passed" if validation["passed"] else "failed","run_id":run_id,
  "versions":{"code_version_id":"git:"+code,"raw_data_version_id":dm["raw_data_version_id"],"input_derived_data_ids":[fr02["freeze_package_id"]],"run_config_sha256":cfgsha,"environment_sha256":envsha},
  "random_seed":SEED,
  "parameters":{"MVP_files":49,"common_analysis_fs_hz":COMMON_FS,"welch_nperseg":NPERSEG,"harmonics":HARMONICS,"frequency_tolerance":"max(2*df, 2%*fc)",
   "trend_R2_threshold":TREND_R2_THRESHOLD,"anti_alias":{"taps":AA_TAPS,"cutoff_hz":AA_CUTOFF,"kaiser_beta":AA_BETA},"problem_basis_sha256":basis_sha},
  "real_results":[
   {"name":"preprocessing_decisions","value":decisions},
   {"name":"bearing_parameters_and_orders","value":param_rows},
   {"name":"typical_difficult_samples","value":rep_small.to_dict("records")},
   {"name":"trend_result","value":report["trend"]},
   {"name":"resampling_information_fidelity","value":aa_summary},
   {"name":"mechanism_feature_candidates","value":candidates.to_dict("records")}
  ],
  "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/preprocessing_information_fidelity.csv"},{"path":f"outputs/runs/{run_id}/artifacts/validation_formula_unit_check.csv"},{"path":f"outputs/runs/{run_id}/artifacts/validation_metric_recalculation.csv"}],
  "anomalies_and_failures":[] if validation["passed"] else [{"issue":"one_or_more_validation_checks_failed"}],
  "b_handoff":{"required_data":["bearing_parameters_and_order_factors.csv","preprocessing_decision_table.csv","typical_difficult_sample_comparison.csv","class_signal_mechanism_summary.csv","mechanism_feature_candidates.csv","preprocessing_information_fidelity.csv"],
   "supported_conclusions":["03-A uses only the 49-file MVP selected in 02-A","problem-statement zero-contact-angle BPFO/BPFI/BSF formulas are used","SKF6203 and SKF6205 parameters are kept separate","representative and difficult samples are selected by deterministic rules","target geometry is not inferred from source"],
   "wording_limits":["a peak near a theoretical fault frequency is mechanism-consistent evidence, not proof of class","target about-600rpm is not used for source mechanism frequency calculation","envelope branch has no extra bandpass and is diagnostic only"],
   "approved_tables":["bearing_parameters_and_order_factors.csv","preprocessing_decision_table.csv","typical_difficult_sample_comparison.csv","preprocessing_information_fidelity.csv","mechanism_feature_candidates.csv"],"approved_figures":[]},
  "unresolved_issues":[],
  "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["02-A source selection changes","RAW source data changes","problem formula/parameter basis changes","03-A preprocessing/mechanism code changes"]}
 }
 f0=json.loads(json.dumps(freeze,ensure_ascii=False));freeze["freeze"]["content_sha256"]=hb(cj(f0).encode())
 (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"03A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"03A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

 pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True);(mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
 runtime={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),
   "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"gpu":None,"cuda":None,"git_commit":code,"pip_freeze_sha256":hf(mani/"pip_freeze.txt")}
 (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
 outs=[]
 for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
  outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
 ot=hb(cj(outs).encode())
 rm={"schema_version":"03A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"03-A","status":"completed" if validation["passed"] else "failed","created_utc":created,
  "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code,"entrypoint":"scripts/step03a_signal_mechanism_analysis.py","code_manifest":"protocol/03A/code_manifest.json"},
  "data":{"raw_data_version_id":dm["raw_data_version_id"],"input_02A_freeze":fr02["freeze_package_id"],"MVP_files":49},
  "blueprint":{"path":"protocol/03A/question1B_argument_blueprint.json","sha256":bpsha},"problem_basis":{"path":"protocol/03A/problem_statement_mechanism_basis.json","sha256":basis_sha},
  "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":envsha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
  "config":{"path":"protocol/03A/run_config.json","sha256":cfgsha},"outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":ot},
  "checkpoint":{"completed_batches":["49-file-primary-analysis","representative-selection","envelope-diagnostics","fidelity-checks","validation"],"pending_batches":[],"resume_token":None},
  "duration_seconds":time.time()-t0}
 (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"ok":validation["passed"],"run_id":run_id,"freeze_sha256":freeze["freeze"]["content_sha256"],"trend_applied_files":ntrend,
   "representative_samples":rep[["class_label","sample_role","relative_path"]].to_dict("records"),"resample_fidelity":aa_summary},ensure_ascii=False))
 return 0 if validation["passed"] else 2

if __name__=="__main__":
 raise SystemExit(main())
