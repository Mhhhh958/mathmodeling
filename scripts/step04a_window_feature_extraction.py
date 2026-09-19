#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, platform, random, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import welch, hilbert, firwin, resample_poly

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
CFG=ROOT/"protocol"/"04A"/"run_config.json"
BLUEPRINT=ROOT/"protocol"/"04A"/"question1C_argument_blueprint.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
DM=ROOT/"protocol"/"01A"/"data_manifest.json"
FREEZE02=ROOT/"protocol"/"02A"/"latest_freeze_package.json"
FREEZE03=ROOT/"protocol"/"03A"/"latest_freeze_package.json"
RUN02=ROOT/"protocol"/"02A"/"latest_run_id.txt"

SEED=20260919
COMMON_FS=12000
FEATURE_FMAX=5500.0
WINDOW_SECONDS=1.0
STRIDE_SECONDS=0.5
WINDOW_N=int(round(WINDOW_SECONDS*COMMON_FS))
STRIDE_N=int(round(STRIDE_SECONDS*COMMON_FS))
EPS=1e-12
WELCH_NPER=4096
FORBIDDEN_FEATURE_TOKENS={"class","label","file","path","group","load","channel","rpm","bearing","domain","subgroup","target","position","size"}

PARAMS={
 "SKF6205_DE":{"Nd":9,"d":0.3126,"D":1.537},
 "SKF6203_FE":{"Nd":9,"d":0.2656,"D":1.122},
}

COMMON_FEATURES=[
 "mean","std","rms","mean_abs","peak_abs","peak_to_peak","skewness","kurtosis",
 "crest_factor","impulse_factor","shape_factor","clearance_factor","zero_cross_rate",
 "spectral_centroid_hz","spectral_rms_hz","spectral_entropy","spectral_flatness",
 "dominant_frequency_hz","rolloff95_hz",
 "band_ratio_0_500","band_ratio_500_1500","band_ratio_1500_3000","band_ratio_3000_5500",
 "envelope_rms","envelope_kurtosis","envelope_spectral_entropy"
]
MECH_FEATURES=[
 "mech_bpfo_h1_h3_energy_ratio","mech_bpfi_h1_h3_energy_ratio","mech_bsf_h1_h3_energy_ratio",
 "mech_bpfi_fr_sideband_energy_ratio","mech_bsf_ftf_sideband_energy_ratio"
]

def cj(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b): return hashlib.sha256(b).hexdigest()
def hf(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def git_head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
def boolish(v): return v if isinstance(v,bool) else str(v).strip().lower() in {"true","1","yes"}

def primary_signal(mat,subgroup,stem,domain):
 keys=[k for k in mat if not k.startswith("__")]
 if domain=="target":
  if stem in mat: return np.asarray(mat[stem],dtype=float).squeeze(),stem
  for k in keys:
   a=np.asarray(mat[k]).squeeze()
   if np.issubdtype(a.dtype,np.number) and a.ndim==1 and a.size>1000:
    return np.asarray(a,dtype=float),k
  raise RuntimeError(f"target signal missing: {stem}")
 want="FE_time" if subgroup=="source_12khz_fe" else "DE_time"
 for k in keys:
  if k.endswith(want): return np.asarray(mat[k],dtype=float).squeeze(),k
 raise RuntimeError(f"source signal missing: {stem} / {subgroup}")

def native_fs(subgroup,domain):
 if domain=="target": return 32000
 if subgroup=="source_12khz_fe": return 12000
 if subgroup=="source_48khz_normal": return 48000
 raise RuntimeError(f"unexpected source subgroup {subgroup}")

def preprocess_common(x,fs,domain,subgroup):
 x=np.asarray(x,dtype=float).ravel()
 x=x-np.mean(x)
 if fs==COMMON_FS:
  return x,{"resampled":False,"up":1,"down":1,"fir_taps":0,"cutoff_hz":None,"filter_design_fs_hz":None}
 if fs==48000:
  taps=firwin(401,5500,window=("kaiser",8.6),fs=48000,pass_zero="lowpass")
  y=resample_poly(x,up=1,down=4,window=taps,padtype="line")
  return np.asarray(y,float),{"resampled":True,"up":1,"down":4,"fir_taps":401,"cutoff_hz":5500,"filter_design_fs_hz":48000}
 if fs==32000:
  up,down=3,8
  design_fs=fs*up
  taps=firwin(801,5500,window=("kaiser",8.6),fs=design_fs,pass_zero="lowpass")
  y=resample_poly(x,up=up,down=down,window=taps,padtype="line")
  return np.asarray(y,float),{"resampled":True,"up":up,"down":down,"fir_taps":801,"cutoff_hz":5500,"filter_design_fs_hz":design_fs}
 raise RuntimeError(fs)

def psd_centered(w):
 z=np.asarray(w,float)-float(np.mean(w))
 nper=min(WELCH_NPER,len(z))
 f,p=welch(z,fs=COMMON_FS,window="hann",nperseg=nper,noverlap=nper//2,detrend=False,scaling="density")
 m=f<=FEATURE_FMAX
 return z,f[m],p[m],float(COMMON_FS/nper)

def trap(f,p,lo,hi):
 m=(f>=max(0.0,lo))&(f<=min(float(f[-1]),hi))
 if m.sum()<2: return 0.0
 return float(np.trapz(p[m],f[m]))

def spectral_entropy(p):
 p=np.asarray(p,float)
 s=float(np.sum(p))
 if s<=EPS or len(p)<=1: return 0.0
 q=p/s
 return float(-(q*np.log(q+EPS)).sum()/math.log(len(q)))

def moment_skew(z):
 s=float(np.sqrt(np.mean(z*z)))
 if s<=EPS:return 0.0
 return float(np.mean(z**3)/(s**3))
def moment_kurt(z):
 v=float(np.mean(z*z))
 if v<=EPS:return 0.0
 return float(np.mean(z**4)/(v*v))

def common_features(w):
 w=np.asarray(w,float)
 mean=float(np.mean(w)); z=w-mean
 std=float(np.sqrt(np.mean(z*z)))
 rms=float(np.sqrt(np.mean(w*w)))
 mean_abs=float(np.mean(np.abs(w)))
 peak=float(np.max(np.abs(w)))
 p2p=float(np.max(w)-np.min(w))
 root_abs=float(np.mean(np.sqrt(np.abs(w))))
 _,f,p,df=psd_centered(w)
 total=float(np.sum(p))
 if total<=EPS:
  centroid=rmsf=sent=sflat=dom=roll=0.0
  bands=[0.0]*4
 else:
  centroid=float(np.sum(f*p)/total)
  rmsf=float(np.sqrt(np.sum((f*f)*p)/total))
  sent=spectral_entropy(p)
  sflat=float(np.exp(np.mean(np.log(p+EPS)))/(np.mean(p)+EPS))
  nonzero=np.where(f>0)[0]
  dom=float(f[nonzero[np.argmax(p[nonzero])]]) if len(nonzero) else 0.0
  cs=np.cumsum(p); idx=min(len(f)-1,int(np.searchsorted(cs,0.95*cs[-1])))
  roll=float(f[idx])
  etot=trap(f,p,0,FEATURE_FMAX)+EPS
  bands=[
    trap(f,p,0,500)/etot,
    trap(f,p,500,1500)/etot,
    trap(f,p,1500,3000)/etot,
    trap(f,p,3000,FEATURE_FMAX)/etot
  ]
 env=np.abs(hilbert(z))
 env_mean=float(np.mean(env)); envz=env-env_mean
 env_rms=float(np.sqrt(np.mean(env*env)))
 env_kurt=moment_kurt(envz)
 nper=min(WELCH_NPER,len(envz))
 ef,ep=welch(envz,fs=COMMON_FS,window="hann",nperseg=nper,noverlap=nper//2,detrend=False,scaling="density")
 em=ef<=FEATURE_FMAX
 env_sent=spectral_entropy(ep[em])
 return {
  "mean":mean,"std":std,"rms":rms,"mean_abs":mean_abs,"peak_abs":peak,"peak_to_peak":p2p,
  "skewness":moment_skew(z),"kurtosis":moment_kurt(z),
  "crest_factor":peak/(rms+EPS),"impulse_factor":peak/(mean_abs+EPS),
  "shape_factor":rms/(mean_abs+EPS),"clearance_factor":peak/(root_abs*root_abs+EPS),
  "zero_cross_rate":float(np.mean(np.signbit(z[1:])!=np.signbit(z[:-1]))) if len(z)>1 else 0.0,
  "spectral_centroid_hz":centroid,"spectral_rms_hz":rmsf,"spectral_entropy":sent,"spectral_flatness":sflat,
  "dominant_frequency_hz":dom,"rolloff95_hz":roll,
  "band_ratio_0_500":bands[0],"band_ratio_500_1500":bands[1],"band_ratio_1500_3000":bands[2],"band_ratio_3000_5500":bands[3],
  "envelope_rms":env_rms,"envelope_kurtosis":env_kurt,"envelope_spectral_entropy":env_sent
 }

def mechanism(rpm,bearing):
 p=PARAMS[bearing]; fr=float(rpm)/60.0; q=p["d"]/p["D"]
 return {"fr":fr,"FTF":0.5*fr*(1-q),"BPFO":fr*p["Nd"]/2*(1-q),"BPFI":fr*p["Nd"]/2*(1+q),"BSF":fr*p["D"]/p["d"]*(1-q*q)}

def mech_features(w,rpm,bearing):
 _,f,p,df=psd_centered(w)
 total=trap(f,p,0,FEATURE_FMAX)+EPS
 m=mechanism(rpm,bearing)
 def harmonics(name):
  e=0.0
  for k in (1,2,3):
   fc=k*m[name]
   if fc>=FEATURE_FMAX: continue
   tol=max(2*df,0.02*fc)
   e+=trap(f,p,fc-tol,fc+tol)
  return e/total
 def sidebands(name,mod):
  e=0.0
  for k in (1,2,3):
   fc=k*m[name]
   if fc>=FEATURE_FMAX:continue
   for sg in (-1,1):
    sb=fc+sg*m[mod]
    if 0<sb<FEATURE_FMAX:
     tol=max(2*df,0.02*sb)
     e+=trap(f,p,sb-tol,sb+tol)
  return e/total
 return {
  "mech_bpfo_h1_h3_energy_ratio":harmonics("BPFO"),
  "mech_bpfi_h1_h3_energy_ratio":harmonics("BPFI"),
  "mech_bsf_h1_h3_energy_ratio":harmonics("BSF"),
  "mech_bpfi_fr_sideband_energy_ratio":sidebands("BPFI","fr"),
  "mech_bsf_ftf_sideband_energy_ratio":sidebands("BSF","FTF")
 }

def window_id(group_id,start,end):
 return "WIN-"+hb(f"{group_id}|{start}|{end}|{COMMON_FS}".encode())[:20]

def feature_dictionary():
 amp="原始振动信号幅值单位（题面/冻结元数据未给出物理标定单位）"
 rows=[
 ("mean","common","time","mean(x)",amp,"none","finite deterministic input","窗口样本"),
 ("std","common","time","sqrt(mean((x-mean(x))^2))",amp,"none","0 for degenerate window","窗口样本"),
 ("rms","common","time","sqrt(mean(x^2))",amp,"none","0 for zero signal","窗口样本"),
 ("mean_abs","common","time","mean(|x|)",amp,"none","0 for zero signal","窗口样本"),
 ("peak_abs","common","time","max(|x|)",amp,"none","0 for zero signal","窗口样本"),
 ("peak_to_peak","common","time","max(x)-min(x)",amp,"none","0 for zero signal","窗口样本"),
 ("skewness","common","time","mean(z^3)/std(z)^3","1","if std<=1e-12 -> 0","finite","窗口内中心化z"),
 ("kurtosis","common","time","mean(z^4)/mean(z^2)^2","1","if variance<=1e-12 -> 0","finite","population kurtosis"),
 ("crest_factor","common","time","peak_abs/(rms+eps)","1","eps=1e-12","finite","窗口样本"),
 ("impulse_factor","common","time","peak_abs/(mean_abs+eps)","1","eps=1e-12","finite","窗口样本"),
 ("shape_factor","common","time","rms/(mean_abs+eps)","1","eps=1e-12","finite","窗口样本"),
 ("clearance_factor","common","time","peak_abs/(mean(sqrt(|x|))^2+eps)","1","eps=1e-12","finite","窗口样本"),
 ("zero_cross_rate","common","time","sign changes of centered z /(N-1)","1","N<2 ->0","finite","窗口内中心化z"),
 ("spectral_centroid_hz","common","frequency","sum(f*P)/sum(P)","Hz","if spectral sum<=eps ->0","finite","Welch 4096, Hann, 50% overlap, 0-5500Hz"),
 ("spectral_rms_hz","common","frequency","sqrt(sum(f^2*P)/sum(P))","Hz","if spectral sum<=eps ->0","finite","同上"),
 ("spectral_entropy","common","frequency","-sum(q log q)/log(K), q=P/sum(P)","1","if zero spectrum ->0","finite","同上"),
 ("spectral_flatness","common","frequency","geometric_mean(P+eps)/(mean(P)+eps)","1","eps=1e-12","finite","同上"),
 ("dominant_frequency_hz","common","frequency","argmax P(f), f>0","Hz","if no nonzero bin ->0","finite","同上"),
 ("rolloff95_hz","common","frequency","min f with cumulative P >=95% total","Hz","if zero spectrum ->0","finite","同上"),
 ("band_ratio_0_500","common","frequency","E[0,500]/E[0,5500]","1","denominator +eps","finite","Welch积分"),
 ("band_ratio_500_1500","common","frequency","E[500,1500]/E[0,5500]","1","denominator +eps","finite","Welch积分"),
 ("band_ratio_1500_3000","common","frequency","E[1500,3000]/E[0,5500]","1","denominator +eps","finite","Welch积分"),
 ("band_ratio_3000_5500","common","frequency","E[3000,5500]/E[0,5500]","1","denominator +eps","finite","Welch积分"),
 ("envelope_rms","common","envelope","RMS(|Hilbert(z)|)",amp,"none","finite","无额外带通；z为窗口中心化信号"),
 ("envelope_kurtosis","common","envelope","population kurtosis(|Hilbert(z)|-mean)","1","if variance<=eps ->0","finite","无额外带通"),
 ("envelope_spectral_entropy","common","envelope","spectral entropy of demeaned Hilbert envelope","1","if zero spectrum ->0","finite","Welch同公共频域参数"),
 ("mech_bpfo_h1_h3_energy_ratio","source_mechanism_aux","mechanism","sum energy near 1-3*BPFO / E[0,5500]","1","denominator +eps","source only; target unavailable","03-A题面公式+文件RPM+轴承参数；tol=max(2df,2%fc)"),
 ("mech_bpfi_h1_h3_energy_ratio","source_mechanism_aux","mechanism","sum energy near 1-3*BPFI / E[0,5500]","1","denominator +eps","source only; target unavailable","同上"),
 ("mech_bsf_h1_h3_energy_ratio","source_mechanism_aux","mechanism","sum energy near 1-3*BSF / E[0,5500]","1","denominator +eps","source only; target unavailable","同上"),
 ("mech_bpfi_fr_sideband_energy_ratio","source_mechanism_aux","mechanism","sum energy at BPFI harmonics ±fr / E[0,5500]","1","denominator +eps","source only; target unavailable","03-A内圈调制口径"),
 ("mech_bsf_ftf_sideband_energy_ratio","source_mechanism_aux","mechanism","sum energy at BSF harmonics ±FTF / E[0,5500]","1","denominator +eps","source only; target unavailable","03-A滚动体调制口径")
 ]
 return pd.DataFrame(rows,columns=["feature_name","tier","domain","definition_formula","unit","zero_denominator_policy","missing_policy","parameter_source"])

def robust_audit(df,features,tier):
 rows=[]
 for c in features:
  x=df[c].to_numpy(float)
  finite=np.isfinite(x)
  vals=x[finite]
  med=float(np.median(vals)) if len(vals) else np.nan
  mad=float(np.median(np.abs(vals-med))) if len(vals) else np.nan
  scale=1.4826*mad if np.isfinite(mad) and mad>EPS else (float(np.std(vals)) if len(vals) else np.nan)
  extreme=0 if not np.isfinite(scale) or scale<=EPS else int(np.sum(np.abs(vals-med)/scale>12))
  rows.append({"tier":tier,"feature":c,"rows":len(x),"nan_count":int(np.isnan(x).sum()),"inf_count":int(np.isinf(x).sum()),
    "finite_count":int(finite.sum()),"constant":bool(len(vals)>0 and np.max(vals)-np.min(vals)<=EPS),
    "min":float(np.min(vals)) if len(vals) else np.nan,"median":med,"max":float(np.max(vals)) if len(vals) else np.nan,
    "robust_scale":scale,"extreme_robust_z_gt12_count":extreme})
 return pd.DataFrame(rows)

def manual_recalc_three(seg):
 # independent formulas for validation, not calling common_features
 x=np.asarray(seg,float); mean=float(np.mean(x)); z=x-mean
 rms=float(np.sqrt(np.sum(x*x)/len(x)))
 peak=float(np.max(np.abs(x))); crest=peak/(rms+EPS)
 nper=min(WELCH_NPER,len(z))
 f,p=welch(z,fs=COMMON_FS,window="hann",nperseg=nper,noverlap=nper//2,detrend=False,scaling="density")
 m=f<=FEATURE_FMAX; f=f[m];p=p[m]
 psum=float(np.sum(p)); centroid=float(np.sum(f*p)/psum) if psum>EPS else 0.0
 return {"rms":rms,"crest_factor":crest,"spectral_centroid_hz":centroid}

def main():
 t0=time.time(); random.seed(SEED); np.random.seed(SEED)
 cfg=json.loads(CFG.read_text(encoding="utf-8"))
 dm=json.loads(DM.read_text(encoding="utf-8"))
 fr02=json.loads(FREEZE02.read_text(encoding="utf-8")); fr03=json.loads(FREEZE03.read_text(encoding="utf-8"))
 assert fr02["status"]=="passed" and fr03["status"]=="passed"
 assert fr03["freeze_package_id"]=="FREEZE-03A-f5307f8f"
 assert dm["raw_data_version_id"]=="RAW-5a5dd129c91bfc64"
 run02=RUN02.read_text(encoding="utf-8").strip()
 sel=pd.read_csv(ROOT/"outputs"/"runs"/run02/"artifacts"/"source_selection_list.csv")
 mvp=sel[sel["MVP_retain"].map(boolish)].copy()
 assert len(mvp)==49
 meta=pd.read_csv(ROOT/dm["metadata_artifact"])
 src=mvp.merge(meta,on=["relative_path","independent_object_id","subgroup","class_label","load_hp","fault_size_in","outer_race_position_clock"],how="left",suffixes=("","_meta"))
 tgt=meta[meta["domain"]=="target"].copy().sort_values("relative_path")
 assert len(src)==49 and len(tgt)==16
 assert set(src["subgroup"])=={"source_12khz_fe","source_48khz_normal"}
 assert (tgt["target_truth_status"]=="unknown").all()

 created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 code=git_head(); cfgsha=hf(CFG); envsha=hf(ENV); bpsha=hf(BLUEPRINT)
 binding={"code_version_id":"git:"+code,"data_version_id":fr03["freeze_package_id"],"run_config_sha256":cfgsha,"seed":SEED,"environment_sha256":envsha}
 bd=hb(cj(binding).encode())
 run_id=f"run_04-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
 out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; logs=out/"logs"; mani=out/"manifests"; q2=art/"q2_interface"; ti=art/"target_interface"
 for d in [art,logs,mani,q2,ti]:d.mkdir(parents=True,exist_ok=False)

 window_rows=[]; common_rows=[]; mech_rows=[]; file_rows=[]; readlog=[]; cache={}
 file_counter=0
 all_records=[]
 for _,r in src.sort_values("relative_path").iterrows():
  rr=r.to_dict(); rr["domain"]="source"; all_records.append(rr)
 for _,r in tgt.iterrows(): all_records.append(r.to_dict())

 for r in all_records:
  file_counter+=1
  rel=str(r["relative_path"]); domain=str(r["domain"]); subgroup=str(r["subgroup"]); p=RAW/Path(rel)
  mat=loadmat(p); x,var=primary_signal(mat,subgroup,p.stem,domain); fs=native_fs(subgroup,domain)
  y,prep=preprocess_common(x,fs,domain,subgroup)
  group_id=str(r["independent_object_id"])
  nwin=0
  max_end=0
  bearing=None
  rpm=None
  if domain=="source":
   rpm=float(r["rpm"])
   bearing="SKF6203_FE" if subgroup=="source_12khz_fe" else "SKF6205_DE"
  for start in range(0,max(0,len(y)-WINDOW_N+1),STRIDE_N):
   end=start+WINDOW_N
   if end>len(y):break
   nwin+=1; max_end=end
   wid=window_id(group_id,start,end)
   start_s=start/COMMON_FS; end_s=end/COMMON_FS
   ns=int(round(start_s*fs)); ne=int(round(end_s*fs))
   ns=max(0,min(ns,len(x))); ne=max(ns,min(ne,len(x)))
   meta_row={
    "window_id":wid,"domain":domain,"file_id":str(r["file_id"]),"independent_object_id":group_id,"group_id":group_id,
    "relative_path":rel,"subgroup":subgroup,"signal_var":var,"common_fs_hz":COMMON_FS,
    "window_start_common_idx":start,"window_end_common_idx_exclusive":end,"window_start_s":start_s,"window_end_s":end_s,
    "native_fs_hz":fs,"native_start_idx":ns,"native_end_idx_exclusive":ne,
    "class_label":str(r["class_label"]) if domain=="source" else "UNKNOWN_TRUTH",
    "label_status":"known_source" if domain=="source" else "unknown_truth",
    "load_hp":r.get("load_hp",np.nan),"fault_size_in":r.get("fault_size_in",np.nan),
    "outer_race_position_clock":r.get("outer_race_position_clock",np.nan),
    "fault_bearing_location":str(r.get("fault_bearing_location","UNKNOWN")),
    "analysis_channel":"FE" if subgroup=="source_12khz_fe" else ("DE" if subgroup=="source_48khz_normal" else "TARGET_UNKNOWN_SENSOR"),
    "rpm":rpm if domain=="source" else np.nan,"bearing_model_for_mechanism":bearing if domain=="source" else ""
   }
   seg=y[start:end]
   cf=common_features(seg)
   window_rows.append(meta_row)
   common_rows.append({"window_id":wid,**cf})
   if domain=="source":
    mf=mech_features(seg,rpm,bearing)
    mech_rows.append({"window_id":wid,**mf})
   cache[wid]=(rel,domain,subgroup,np.asarray(y),start,end,fs)
  covered=max_end/COMMON_FS
  file_rows.append({"domain":domain,"relative_path":rel,"file_id":str(r["file_id"]),"independent_object_id":group_id,"subgroup":subgroup,
    "native_points":int(len(x)),"native_fs_hz":fs,"native_duration_s":len(x)/fs,"common_points":int(len(y)),"common_fs_hz":COMMON_FS,
    "common_duration_s":len(y)/COMMON_FS,"window_count":nwin,"covered_until_s":covered,"tail_uncovered_s":max(0.0,len(y)/COMMON_FS-covered),
    **prep})
  readlog.append({"relative_path":rel,"domain":domain,"signal_var":var,"native_points":len(x),"native_fs_hz":fs,"common_points":len(y),"common_fs_hz":COMMON_FS,"window_count":nwin,"status":"OK"})

 win=pd.DataFrame(window_rows)
 common=pd.DataFrame(common_rows)
 mech=pd.DataFrame(mech_rows)
 files=pd.DataFrame(file_rows)
 assert len(win)==len(common)
 assert win["window_id"].is_unique and common["window_id"].is_unique
 assert set(mech["window_id"]).issubset(set(win["window_id"]))

 full=win.merge(common,on="window_id",how="left").merge(mech,on="window_id",how="left")
 win.to_csv(art/"window_index.csv",index=False,encoding="utf-8-sig")
 common.to_csv(art/"common_feature_table.csv",index=False,encoding="utf-8-sig")
 mech.to_csv(art/"source_mechanism_feature_table.csv",index=False,encoding="utf-8-sig")
 full.to_csv(art/"full_window_feature_table.csv",index=False,encoding="utf-8-sig")
 files.to_csv(art/"file_processing_window_counts.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame(readlog).to_csv(logs/"actual_processing_read_log.csv",index=False,encoding="utf-8-sig")

 # Group boundary registry.
 grp=win.groupby(["domain","independent_object_id","group_id","relative_path"],dropna=False).agg(window_count=("window_id","count"),first_start_s=("window_start_s","min"),last_end_s=("window_end_s","max")).reset_index()
 grp["future_split_rule"]="all windows with same group_id must stay in one partition"
 grp.to_csv(art/"group_boundary_registry.csv",index=False,encoding="utf-8-sig")

 # Feature dictionary and window rationale.
 fd=feature_dictionary()
 fd["default_q2_use"]=fd["tier"].eq("common")
 fd["target_compatible"]=fd["tier"].eq("common")
 fd.to_csv(art/"feature_dictionary.csv",index=False,encoding="utf-8-sig")

 src_rpm=src["rpm"].astype(float)
 fr_min=float(src_rpm.min()/60.0); fr_max=float(src_rpm.max()/60.0)
 # FTF range across both source bearings.
 ftf_vals=[]
 for _,r in src.iterrows():
  b="SKF6203_FE" if r["subgroup"]=="source_12khz_fe" else "SKF6205_DE"
  ftf_vals.append(mechanism(float(r["rpm"]),b)["FTF"])
 rationale=pd.DataFrame([
  {"candidate_window_s":0.5,"candidate_overlap":0.0,"role":"step12 sensitivity candidate","physical_note":"target ~600rpm gives ~5 shaft rotations; weakest for stable low-frequency statistics"},
  {"candidate_window_s":1.0,"candidate_overlap":0.5,"role":"04-A frozen primary candidate","physical_note":f"source fr={fr_min:.2f}-{fr_max:.2f}Hz; source min FTF={min(ftf_vals):.2f}Hz => >= {min(ftf_vals):.1f} FTF cycles/window; target ~600rpm => ~10 shaft rotations"},
  {"candidate_window_s":2.0,"candidate_overlap":0.5,"role":"step12 sensitivity candidate","physical_note":"more cycles per window but fewer windows per file and higher compute per segment"}
 ])
 rationale.to_csv(art/"window_parameter_candidates.csv",index=False,encoding="utf-8-sig")

 # Quality audit.
 qa_common=robust_audit(common,COMMON_FEATURES,"common")
 qa_mech=robust_audit(mech,MECH_FEATURES,"source_mechanism_aux")
 qa=pd.concat([qa_common,qa_mech],ignore_index=True)
 qa.to_csv(art/"feature_quality_audit.csv",index=False,encoding="utf-8-sig")
 common_bad=int(qa_common["nan_count"].sum()+qa_common["inf_count"].sum())
 common_const=qa_common[qa_common["constant"]]
 mech_bad=int(qa_mech["nan_count"].sum()+qa_mech["inf_count"].sum())

 # Q2 interfaces: feature matrix contains no labels or metadata except window_id.
 source_ids=win.loc[win["domain"]=="source","window_id"].tolist()
 target_ids=win.loc[win["domain"]=="target","window_id"].tolist()
 Xs=common[common["window_id"].isin(source_ids)].set_index("window_id").loc[source_ids].reset_index()
 Xt=common[common["window_id"].isin(target_ids)].set_index("window_id").loc[target_ids].reset_index()
 ys=win.loc[win["domain"]=="source",["window_id","class_label"]].copy()
 gs=win.loc[win["domain"]=="source",["window_id","independent_object_id"]].rename(columns={"independent_object_id":"group_id"})
 smeta=win.loc[win["domain"]=="source"].copy()
 tmeta=win.loc[win["domain"]=="target"].copy()
 Xs.to_csv(q2/"X_source_common.csv",index=False,encoding="utf-8-sig")
 ys.to_csv(q2/"y_source_labels.csv",index=False,encoding="utf-8-sig")
 gs.to_csv(q2/"groups_source.csv",index=False,encoding="utf-8-sig")
 smeta.to_csv(q2/"source_window_metadata.csv",index=False,encoding="utf-8-sig")
 mech.to_csv(q2/"source_mechanism_aux_optional.csv",index=False,encoding="utf-8-sig")
 Xt.to_csv(ti/"X_target_common.csv",index=False,encoding="utf-8-sig")
 tmeta.to_csv(ti/"target_window_metadata.csv",index=False,encoding="utf-8-sig")

 # Leakage checks.
 q2_feature_cols=[c for c in Xs.columns if c!="window_id"]
 leakage_cols=[c for c in q2_feature_cols if any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
 nonnumeric=[c for c in q2_feature_cols if not np.issubdtype(Xs[c].dtype,np.number)]
 learning_contract={
  "status":"raw_deterministic_features_only",
  "imputer_fitted":False,"scaler_fitted":False,"feature_selector_fitted":False,"sklearn_used":False,
  "future_rule":"fit any imputer/scaler/selector on training groups only; transform validation/test/target using that fitted object",
  "default_q2_X":"q2_interface/X_source_common.csv",
  "labels":"q2_interface/y_source_labels.csv",
  "groups":"q2_interface/groups_source.csv",
  "optional_source_only_mechanism_features":"q2_interface/source_mechanism_aux_optional.csv",
  "mechanism_default_use":False,
  "mechanism_exclusion_reason":"target geometry/exact RPM unavailable and source fault-vs-normal acquisition settings differ; avoid non-transferable/confounded default feature use"
 }
 (art/"learning_transform_contract.json").write_text(json.dumps(learning_contract,ensure_ascii=False,indent=2),encoding="utf-8")

 interface={
  "schema_version":"04A-interface-1.0","run_id":run_id,"window_seconds":WINDOW_SECONDS,"overlap_fraction":0.5,"group_key":"independent_object_id",
  "source":{"X":"q2_interface/X_source_common.csv","y":"q2_interface/y_source_labels.csv","groups":"q2_interface/groups_source.csv","metadata":"q2_interface/source_window_metadata.csv","rows":len(Xs),"features":q2_feature_cols},
  "source_optional":{"mechanism_aux":"q2_interface/source_mechanism_aux_optional.csv","default_use":False},
  "target":{"X":"target_interface/X_target_common.csv","metadata":"target_interface/target_window_metadata.csv","rows":len(Xt),"truth":"unknown"},
  "fit_boundary":"no learned preprocessing has been fit; Q2 must split by groups before any fit"
 }
 (art/"data_interface_manifest.json").write_text(json.dumps(interface,ensure_ascii=False,indent=2),encoding="utf-8")

 # Quantity conservation.
 srcw=win[win["domain"]=="source"]; tgtw=win[win["domain"]=="target"]
 source_class_windows={k:int(v) for k,v in srcw["class_label"].value_counts().to_dict().items()}
 quantity={
  "source_selected_files":49,"target_files":16,"processed_files":len(files),
  "source_windows":len(srcw),"target_windows":len(tgtw),"total_windows":len(win),
  "source_class_windows":source_class_windows,
  "source_groups":int(srcw["independent_object_id"].nunique()),"target_groups":int(tgtw["independent_object_id"].nunique()),
  "window_samples":WINDOW_N,"stride_samples":STRIDE_N,"window_seconds":WINDOW_SECONDS,"stride_seconds":STRIDE_SECONDS,
  "files_with_dropped_tail":int((files["tail_uncovered_s"]>1e-12).sum()),
  "max_tail_uncovered_s":float(files["tail_uncovered_s"].max()),
  "no_padding":True
 }
 (art/"quantity_conservation.json").write_text(json.dumps(quantity,ensure_ascii=False,indent=2),encoding="utf-8")

 # Q1 answer material.
 q1mat={
  "window_choice":{"duration_s":1.0,"overlap":0.5,"common_fs_hz":12000,"samples":WINDOW_N,"stride_samples":STRIDE_N,
    "reason":"physical-time choice: source window contains ~29 shaft rotations and at least ~11 FTF cycles; target ~600rpm contains ~10 shaft rotations; candidate sensitivity 0.5/1.0/2.0s and 0/50% overlap deferred to STEP12"},
  "counts":quantity,
  "features":{"common_count":len(COMMON_FEATURES),"source_mechanism_aux_count":len(MECH_FEATURES),"default_q2_use":"common only"},
  "leakage_guard":["labels separate from X","file/group/load/channel/RPM/bearing are metadata only","source mechanism auxiliary excluded by default"],
  "learning_guard":"no imputer/scaler/selector fit in 04-A; all such fit operations must occur after group split on training groups only"
 }
 (art/"question1_answer_material.json").write_text(json.dumps(q1mat,ensure_ascii=False,indent=2),encoding="utf-8")

 # Validation 1: random window traceability.
 rng=random.Random(SEED)
 sample_wids=rng.sample(win["window_id"].tolist(),8)
 trace=win[win["window_id"].isin(sample_wids)].copy().sort_values("window_id")
 trace["duration_check_s"]=trace["window_end_s"]-trace["window_start_s"]
 trace["native_duration_from_idx_s"]=(trace["native_end_idx_exclusive"]-trace["native_start_idx"])/trace["native_fs_hz"]
 trace["group_equals_independent"]=trace["group_id"]==trace["independent_object_id"]
 trace["bounds_valid"]=(trace["window_start_common_idx"]>=0)&(trace["window_end_common_idx_exclusive"]>trace["window_start_common_idx"])
 trace.to_csv(art/"validation_window_traceability.csv",index=False,encoding="utf-8-sig")
 group_unique=bool((grp["independent_object_id"]==grp["group_id"]).all() and grp["independent_object_id"].nunique()==len(grp))
 trace_ok=bool(trace["group_equals_independent"].all() and trace["bounds_valid"].all() and np.allclose(trace["duration_check_s"],1.0,atol=1e-12))

 # Validation 2: independently recalc 3 features across 3 random windows.
 recalc_wids=rng.sample(win["window_id"].tolist(),3)
 recalc=[]
 for wid,feature in zip(recalc_wids,["rms","crest_factor","spectral_centroid_hz"]):
  rel,domain,subgroup,y,start,end,fs=cache[wid]
  vals=manual_recalc_three(y[start:end])
  stored=float(common.loc[common["window_id"]==wid,feature].iloc[0])
  rr=float(vals[feature])
  recalc.append({"window_id":wid,"relative_path":rel,"feature":feature,"stored":stored,"recalculated":rr,"abs_diff":abs(stored-rr)})
 recalc_df=pd.DataFrame(recalc)
 recalc_df.to_csv(art/"validation_feature_recalculation.csv",index=False,encoding="utf-8-sig")
 recalc_ok=bool((recalc_df["abs_diff"]<1e-10).all())

 # Validation 3: learned transforms and leakage absent.
 leakage_ok=(len(leakage_cols)==0 and len(nonnumeric)==0)
 train_boundary_ok=True # no fitting was performed at all.
 validation={
  "step_id":"04-A","run_id":run_id,
  "checks":{
   "window_traceability_and_group_boundary":{"passed":trace_ok and group_unique,"evidence":f"outputs/runs/{run_id}/artifacts/validation_window_traceability.csv","sampled_windows":len(trace),"groups":len(grp)},
   "three_feature_independent_recalculation":{"passed":recalc_ok,"evidence":f"outputs/runs/{run_id}/artifacts/validation_feature_recalculation.csv","features":["rms","crest_factor","spectral_centroid_hz"],"max_abs_diff":float(recalc_df["abs_diff"].max())},
   "common_feature_finite":{"passed":common_bad==0,"evidence":f"outputs/runs/{run_id}/artifacts/feature_quality_audit.csv","nan_inf_count":common_bad,"constant_features":common_const["feature"].tolist()},
   "mechanism_aux_finite_source":{"passed":mech_bad==0,"evidence":f"outputs/runs/{run_id}/artifacts/feature_quality_audit.csv","nan_inf_count":mech_bad},
   "no_label_or_metadata_in_default_X":{"passed":leakage_ok,"evidence":f"outputs/runs/{run_id}/artifacts/data_interface_manifest.json","leakage_columns":leakage_cols,"nonnumeric_columns":nonnumeric},
   "learning_transforms_not_fit_globally":{"passed":train_boundary_ok,"evidence":f"outputs/runs/{run_id}/artifacts/learning_transform_contract.json","imputer_fitted":False,"scaler_fitted":False,"feature_selector_fitted":False},
   "quantity_conservation":{"passed":len(files)==65 and len(srcw)>0 and len(tgtw)>0,"evidence":f"outputs/runs/{run_id}/artifacts/quantity_conservation.json","key_numbers":quantity},
   "target_truth_unknown_preserved":{"passed":bool((tgtw["label_status"]=="unknown_truth").all()),"evidence":f"outputs/runs/{run_id}/artifacts/window_index.csv"}
  }
 }
 validation["passed"]=all(v["passed"] for v in validation["checks"].values())
 validation["status"]="passed" if validation["passed"] else "failed"
 (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"04A"/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

 # Feature data version computed from stable primary artifacts.
 key_paths=[art/"window_index.csv",art/"common_feature_table.csv",art/"source_mechanism_feature_table.csv",art/"feature_dictionary.csv",art/"data_interface_manifest.json"]
 feature_manifest=[{"name":p.name,"sha256":hf(p),"size_bytes":p.stat().st_size} for p in key_paths]
 feature_data_version_id="FEAT-"+hb(cj(feature_manifest).encode())[:16]
 (art/"feature_data_manifest.json").write_text(json.dumps({"feature_data_version_id":feature_data_version_id,"files":feature_manifest,"run_id":run_id,"raw_data_version_id":dm["raw_data_version_id"],"input_03A_freeze":fr03["freeze_package_id"]},ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"04A"/"feature_data_version_id.txt").write_text(feature_data_version_id+"\n",encoding="utf-8")

 freeze={
  "schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":"04-A",
  "freeze_package_id":f"FREEZE-04A-{run_id[-8:]}","status":"passed" if validation["passed"] else "failed","run_id":run_id,
  "versions":{"code_version_id":"git:"+code,"raw_data_version_id":dm["raw_data_version_id"],"input_derived_data_ids":[fr02["freeze_package_id"],fr03["freeze_package_id"],feature_data_version_id],"run_config_sha256":cfgsha,"environment_sha256":envsha},
  "random_seed":SEED,
  "parameters":{"window_seconds":WINDOW_SECONDS,"overlap_fraction":0.5,"stride_seconds":STRIDE_SECONDS,"common_fs_hz":COMMON_FS,"feature_fmax_hz":FEATURE_FMAX,
    "window_candidates_step12":{"duration_s":[0.5,1.0,2.0],"overlap":[0.0,0.5]},"common_feature_count":len(COMMON_FEATURES),"source_mechanism_aux_count":len(MECH_FEATURES)},
  "real_results":[
   {"name":"quantity_conservation","value":quantity},
   {"name":"feature_data_version_id","value":feature_data_version_id},
   {"name":"common_feature_quality","value":{"nan_inf":common_bad,"constant_features":common_const["feature"].tolist(),"feature_count":len(COMMON_FEATURES)}},
   {"name":"group_boundary","value":{"source_groups":int(srcw["independent_object_id"].nunique()),"target_groups":int(tgtw["independent_object_id"].nunique()),"rule":"same independent_object_id cannot cross future partitions"}},
   {"name":"q2_interface","value":interface}
  ],
  "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/artifacts/validation_window_traceability.csv"},{"path":f"outputs/runs/{run_id}/artifacts/validation_feature_recalculation.csv"},{"path":f"outputs/runs/{run_id}/artifacts/feature_quality_audit.csv"}],
  "anomalies_and_failures":[],
  "b_handoff":{
    "required_data":["window_parameter_candidates.csv","window_index.csv","feature_dictionary.csv","quantity_conservation.json","feature_quality_audit.csv","question1_answer_material.json","data_interface_manifest.json"],
    "supported_conclusions":["windowing is defined by physical time not copied point count","all windows retain original file/group identity","default Q2 interface contains only deterministic target-compatible common features","no learned preprocessing was fit globally"],
    "wording_limits":["1s/50% is the frozen primary candidate, not a proven global optimum; sensitivity is deferred to STEP12","overlapping windows are not independent samples","source mechanism auxiliary features are not target-compatible default features"],
    "approved_tables":["window_parameter_candidates.csv","feature_dictionary.csv","quantity_conservation.json","feature_quality_audit.csv"],"approved_figures":[]
  },
  "unresolved_issues":[],
  "freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["02-A source selection changes","03-A preprocessing/mechanism convention changes","raw data changes","window/feature code or parameter changes"]}
 }
 f0=json.loads(json.dumps(freeze,ensure_ascii=False)); freeze["freeze"]["content_sha256"]=hb(cj(f0).encode())
 (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"04A"/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"protocol"/"04A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

 pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
 (mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
 runtime={"captured_utc":created,"platform":platform.platform(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),
  "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"gpu":None,"cuda":None,"git_commit":code,"pip_freeze_sha256":hf(mani/"pip_freeze.txt")}
 (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
 outs=[]
 for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
  outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
 rm={"schema_version":"04A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"04-A","status":"completed" if validation["passed"] else "failed","created_utc":created,
  "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code,"entrypoint":"scripts/step04a_window_feature_extraction.py","code_manifest":"protocol/04A/code_manifest.json"},
  "data":{"raw_data_version_id":dm["raw_data_version_id"],"input_02A_freeze":fr02["freeze_package_id"],"input_03A_freeze":fr03["freeze_package_id"],"feature_data_version_id":feature_data_version_id},
  "blueprint":{"path":"protocol/04A/question1C_argument_blueprint.json","sha256":bpsha},
  "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":envsha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
  "config":{"path":"protocol/04A/run_config.json","sha256":cfgsha},"outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":hb(cj(outs).encode())},
  "checkpoint":{"completed_batches":["65-file-preprocess","window-index","common-features","source-mechanism-aux","quality-audit","q2-interface","validation"],"pending_batches":[],"resume_token":None},
  "duration_seconds":time.time()-t0}
 (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"ok":validation["passed"],"run_id":run_id,"feature_data_version_id":feature_data_version_id,"source_windows":len(srcw),"target_windows":len(tgtw),"source_class_windows":source_class_windows,"common_features":len(COMMON_FEATURES),"mechanism_aux":len(MECH_FEATURES),"common_nan_inf":common_bad,"constant_common_features":common_const["feature"].tolist(),"max_feature_recalc_abs_diff":float(recalc_df["abs_diff"].max()),"freeze_sha256":freeze["freeze"]["content_sha256"]},ensure_ascii=False))
 return 0 if validation["passed"] else 2

if __name__=="__main__":
 raise SystemExit(main())
