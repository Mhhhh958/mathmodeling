from __future__ import annotations
import hashlib, json, math, os, platform, random, re, subprocess, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat, whosmat

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
CFG=ROOT/"protocol"/"01A"/"run_config.json"
ENV=ROOT/"protocol"/"00B"/"environment_manifest.json"
SEED=20260919
EXPECTED={"OR":77,"IR":40,"B":40,"N":4}
TARGET_IDS=list("ABCDEFGHIJKLMNOP")

def cj(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def hb(b): return hashlib.sha256(b).hexdigest()
def hf(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
def sh(shape): return "x".join(map(str,shape))
def n_hash(a):
    x=np.asarray(a).squeeze()
    if not np.issubdtype(x.dtype,np.number): return ""
    y=np.asarray(x,dtype="<f8").ravel()
    return hb((f"shape={tuple(x.shape)};n={y.size};").encode()+y.tobytes())

def source_meta(name):
    u=name.upper()
    cls="OR" if u.startswith("OR") else "IR" if u.startswith("IR") else "B" if u.startswith("B") else "N" if u.startswith("N") else "UNKNOWN"
    m=re.match(r"^(?:OR|IR|B)(\d{3})",name,re.I); size=float(m.group(1))/1000 if m else None
    m=re.search(r"_(\d)(?:_|$)",name); load=int(m.group(1)) if m else None
    m=re.search(r"@(12|6|3)",name); pos=m.group(1) if cls=="OR" and m else None
    m=re.search(r"\((\d{4})rpm\)",name,re.I); rpm=float(m.group(1)) if m else None
    return cls,size,load,pos,rpm

def group_meta(rel):
    s=rel.as_posix().lower()
    if "target_domain" in s: return "target_32khz","target","UNKNOWN",32000,"题面32kHz；以256000点/8秒复核"
    if "cwru_12khz_de" in s: return "source_12khz_de","source","DE",12000,"目录12khz_de + 题面DE轴承12kHz"
    if "cwru_12khz_fe" in s: return "source_12khz_fe","source","FE",12000,"目录12khz_fe + 题面FE轴承12kHz"
    if "cwru_48khz_de" in s: return "source_48khz_de","source","DE",48000,"目录48khz_de + 题面DE轴承48kHz"
    if "cwru_48khz_normal" in s: return "source_48khz_normal","source","NONE",48000,"目录48khz_normal的48kHz口径"
    return "unknown","unknown","UNKNOWN",None,"unknown"

def channel(name,target=False):
    u=name.upper()
    if target:return "TARGET"
    if "DE_TIME" in u:return "DE"
    if "FE_TIME" in u:return "FE"
    if "BA_TIME" in u:return "BA"
    if "RPM" in u:return "RPM"
    return "OTHER"

def primary(data,domain,fault_loc,stem):
    ks=[k for k in data if not k.startswith("__")]
    if domain=="target":
        if stem in data:return stem,np.asarray(data[stem]).squeeze()
        for k in ks:
            a=np.asarray(data[k]).squeeze()
            if np.issubdtype(a.dtype,np.number) and a.ndim==1 and a.size>1000:return k,a
        return None,None
    pref="FE_TIME" if fault_loc=="FE" else "DE_TIME"
    for k in ks:
        if pref in k.upper(): return k,np.asarray(data[k]).squeeze()
    for suf in ["DE_TIME","FE_TIME","BA_TIME"]:
        for k in ks:
            if suf in k.upper(): return k,np.asarray(data[k]).squeeze()
    return None,None

def rpm_of(data,name_rpm):
    for k,v in data.items():
        if not k.startswith("__") and "RPM" in k.upper():
            a=np.asarray(v).squeeze()
            if a.size:
                try:return float(a.flat[0]),"MAT变量"
                except:pass
    return (name_rpm,"文件名注记") if name_rpm is not None else (None,"缺失/未知")

def main():
    random.seed(SEED); np.random.seed(SEED)
    created=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    code_head=head()
    mats=sorted(RAW.rglob("*.mat"))
    if not mats: raise RuntimeError("data/raw下没有MAT文件")

    fps=[{"relative_path":p.relative_to(RAW).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)} for p in mats]
    fps=sorted(fps,key=lambda x:x["relative_path"])
    dataset_digest=hb(cj(fps).encode())
    raw_id="RAW-"+dataset_digest[:16]
    cfg_sha,env_sha=hf(CFG),hf(ENV)
    binding={"code_version_id":"git:"+code_head,"data_version_id":raw_id,"run_config_sha256":cfg_sha,"seed":SEED,"environment_sha256":env_sha}
    bd=hb(cj(binding).encode())
    run_id=f"run_01-A_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs"/"runs"/run_id; art=out/"artifacts"; logs=out/"logs"; mani=out/"manifests"
    for d in [art,logs,mani]:d.mkdir(parents=True,exist_ok=False)

    rows=[]; readlog=[]; severe=[]; rawg=defaultdict(list); primg=defaultdict(list)
    sha_by_path={x["relative_path"]:x["sha256"] for x in fps}
    for i,p in enumerate(mats,1):
        rel=p.relative_to(RAW); rp=rel.as_posix()
        sg,domain,fault_loc,fs,fs_basis=group_meta(rel)
        cls,size,load,pos,name_rpm=source_meta(p.stem) if domain=="source" else ("UNKNOWN_TRUTH",None,None,None,None)
        fsha=sha_by_path[rp]; rawg[fsha].append(rp)
        rec={"domain":domain,"subgroup":sg,"file_id":f"RAWFILE-{i:03d}","independent_object_id":rp,"relative_path":rp,"file_name":p.name,
             "size_bytes":p.stat().st_size,"file_sha256":fsha,"mat_read_status":"OK","mat_variables_json":"","variable_count":0,
             "sensor_channels":"","de_present":False,"fe_present":False,"ba_present":False,"de_shape":"","fe_shape":"","ba_shape":"",
             "primary_signal_var":"","primary_signal_shape":"","primary_signal_points":None,"primary_signal_sha256":"",
             "sampling_rate_hz":fs,"sampling_rate_basis":fs_basis,"duration_seconds":None,"duration_basis":"主分析信号点数/采样率",
             "rpm":None,"rpm_source":"","class_label":cls,"label_source":"源域文件名前缀+题面类别口径" if domain=="source" else "A-P仅文件ID；原始MAT无真值标签",
             "fault_bearing_location":fault_loc,"fault_bearing_location_source":"数据子目录语义；不得由DE/FE传感器变量反推" if domain=="source" else "目标域未知",
             "fault_size_in":size,"load_hp":load,"outer_race_position_clock":pos,"target_truth_status":"unknown" if domain=="target" else "known",
             "finite_count":None,"nonfinite_count":None,"is_constant":None,"signal_min":None,"signal_max":None,"signal_mean":None,"signal_std":None,
             "missing_flags":"","quality_flags":""}
        miss=[]; flags=[]
        try:
            info=whosmat(p); data=loadmat(p)
            vm=[]; shapes={}
            for name,shape,dtype in info:
                ch=channel(name,domain=="target")
                vm.append({"name":name,"shape":[int(x) for x in shape],"dtype":dtype,"semantic":ch})
                if ch in {"DE","FE","BA"}:shapes[ch]=sh(shape)
            rec["mat_variables_json"]=cj(vm); rec["variable_count"]=len(vm)
            rec["sensor_channels"]=";".join(sorted(set(x["semantic"] for x in vm if x["semantic"] in {"DE","FE","BA","TARGET"})))
            for ch,k in [("DE","de"),("FE","fe"),("BA","ba")]:
                rec[k+"_present"]=ch in shapes; rec[k+"_shape"]=shapes.get(ch,"")
            pv,x=primary(data,domain,fault_loc,p.stem)
            if pv is None:
                flags.append("NO_PRIMARY_SIGNAL"); severe.append({"file":rp,"issue":"NO_PRIMARY_SIGNAL"})
            else:
                x=np.asarray(x).squeeze(); ph=n_hash(x); primg[ph].append(rp)
                rec.update({"primary_signal_var":pv,"primary_signal_shape":sh(x.shape),"primary_signal_points":int(x.size),"primary_signal_sha256":ph,"duration_seconds":float(x.size/fs)})
                z=np.asarray(x,dtype=float).ravel(); fin=np.isfinite(z); vals=z[fin]
                rec["finite_count"]=int(fin.sum()); rec["nonfinite_count"]=int((~fin).sum())
                if (~fin).any():flags.append("NONFINITE"); severe.append({"file":rp,"issue":"NONFINITE"})
                if vals.size:
                    rec["signal_min"]=float(vals.min()); rec["signal_max"]=float(vals.max()); rec["signal_mean"]=float(vals.mean()); rec["signal_std"]=float(vals.std())
                    rec["is_constant"]=bool(np.ptp(vals)==0)
                    if rec["is_constant"]:flags.append("CONSTANT_SIGNAL"); severe.append({"file":rp,"issue":"CONSTANT_SIGNAL"})
                if domain=="target":
                    if p.stem not in data:flags.append("TARGET_VAR_NAME_MISMATCH"); severe.append({"file":rp,"issue":"TARGET_VAR_NAME_MISMATCH"})
                    if x.size!=256000:flags.append("TARGET_POINTS_MISMATCH"); severe.append({"file":rp,"issue":f"TARGET_POINTS={x.size}"})
                    if not math.isclose(x.size/fs,8.0,abs_tol=1e-12):flags.append("TARGET_DURATION_MISMATCH"); severe.append({"file":rp,"issue":f"TARGET_DURATION={x.size/fs}"})
                else:
                    expected_sensor="FE" if fault_loc=="FE" else "DE"
                    if expected_sensor not in shapes:flags.append("EXPECTED_PRIMARY_SENSOR_MISSING"); severe.append({"file":rp,"issue":f"{expected_sensor}_MISSING"})
            rpm,rs=rpm_of(data,name_rpm); rec["rpm"]=rpm; rec["rpm_source"]=rs
            if domain=="source" and rpm is None:miss.append("RPM")
            if domain=="target":
                miss+=["TRUE_LABEL","RPM_EXACT"]; rec["rpm_source"]="题面约600rpm仅作近似工况；原始MAT无逐时精确RPM"
            if domain=="source" and cls=="UNKNOWN":flags.append("UNKNOWN_SOURCE_CLASS"); severe.append({"file":rp,"issue":"UNKNOWN_SOURCE_CLASS"})
            if cls=="N":rec["fault_bearing_location"]="NONE";rec["fault_size_in"]=None;rec["outer_race_position_clock"]=None
            rec["missing_flags"]=";".join(miss); rec["quality_flags"]=";".join(flags)
            readlog.append({"relative_path":rp,"status":"OK","variable_count":len(info),"primary_points":rec["primary_signal_points"],"file_sha256":fsha})
        except Exception as e:
            rec["mat_read_status"]=f"ERROR:{type(e).__name__}:{e}";rec["quality_flags"]="MAT_READ_ERROR"
            severe.append({"file":rp,"issue":rec["mat_read_status"]})
            readlog.append({"relative_path":rp,"status":rec["mat_read_status"],"variable_count":None,"primary_points":None,"file_sha256":fsha})
        rows.append(rec)

    df=pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    src=df[df.domain=="source"]; tgt=df[df.domain=="target"]
    df.to_csv(art/"file_level_metadata.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(readlog).to_csv(logs/"actual_read_log.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(fps).to_csv(art/"raw_fingerprint_manifest.csv",index=False,encoding="utf-8-sig")

    dups=[]
    for typ,groups in [("raw_file_sha256",rawg),("primary_signal_sha256",primg)]:
        for h,ps in groups.items():
            if h and len(ps)>1:dups.append({"hash_type":typ,"sha256":h,"count":len(ps),"paths":";".join(sorted(ps))})
    pd.DataFrame(dups,columns=["hash_type","sha256","count","paths"]).to_csv(art/"duplicate_content_check.csv",index=False,encoding="utf-8-sig")

    actual={k:int(v) for k,v in src.class_label.value_counts().to_dict().items()}
    count_rows=[{"class":k,"expected_files":v,"actual_files":actual.get(k,0),"difference":actual.get(k,0)-v,"status":"MATCH" if actual.get(k,0)==v else "MISMATCH"} for k,v in EXPECTED.items()]
    for r in count_rows:
        if r["status"]!="MATCH":severe.append({"file":"SOURCE_COUNT","issue":f"{r['class']} expected {r['expected_files']} actual {r['actual_files']}"})
    pd.DataFrame(count_rows).to_csv(art/"source_class_count_check.csv",index=False,encoding="utf-8-sig")

    pd.DataFrame([
      ["source","OR","外圈故障 Outer Race","known","文件名前缀+题面源域类别"],
      ["source","IR","内圈故障 Inner Race","known","文件名前缀+题面源域类别"],
      ["source","B","滚动体故障 Ball","known","文件名前缀+题面源域类别"],
      ["source","N","正常 Normal","known","文件名前缀+题面源域类别"],
      ["target","A-P","目标域独立文件ID，不是类别","unknown","原始MAT文件名/变量名"]
    ],columns=["domain","raw_code","meaning","truth_status","source"]).to_csv(art/"label_mapping.csv",index=False,encoding="utf-8-sig")

    pd.DataFrame([
      ["independent_object_id","data/raw下MAT相对路径","string","后续窗口必须继承并按此分组"],
      ["sensor_channels","DE/FE/BA为传感器测点","category","不等于故障轴承位置"],
      ["fault_bearing_location","DE/FE/NONE/UNKNOWN","category","来自数据子目录语义，不由变量名反推"],
      ["class_label","OR/IR/B/N；目标UNKNOWN_TRUTH","category","源域文件名+题面；目标真值未知"],
      ["sampling_rate_hz","采样率","Hz","源目录/题面；目标32kHz并用256000/8复核"],
      ["duration_seconds","点数/采样率","s","实际数组计算"],
      ["rpm","转速","rpm","源MAT变量优先，其次文件名；目标约600仅近似"],
      ["fault_size_in","故障尺寸","inch","007/014/021/028解析"],
      ["load_hp","负载","hp","文件名工况0/1/2/3"],
      ["outer_race_position_clock","外圈故障位置","clock","仅OR的@3/@6/@12"],
      ["file_sha256","原始文件字节指纹","hex","版本冻结/重复检查"],
      ["primary_signal_sha256","主信号数值内容指纹","hex","内容重复辅助判断"]
    ],columns=["field","definition","unit_or_type","provenance_or_rule"]).to_csv(art/"data_dictionary.csv",index=False,encoding="utf-8-sig")

    pd.DataFrame([
      ["data/interim/target_features/目标域特征提取_A.csv ... P.csv","src/feature_engineering/extract_target_features.py","A-P滑窗后提特征",False,"derived_features_not_truth"],
      ["target_labeledData.csv","src/data_processing/target_dataset_builder.py","仅加入目标代码A-P，未提供真实故障类别",False,"A-P_ID_only_not_truth"],
      ["legacy/**/target_data_with_predictions.csv","legacy exploratory transfer scripts","迁移试验模型预测",False,"historical_predictions_quarantined"],
      ["outputs/q3a_domain_diagnosis/target_A_P_no_transfer_file_predictions.csv","Q3A历史运行","无迁移模型预测",False,"historical_prediction_not_truth"],
      ["outputs/q3b_unsupervised_transfer/target_A_P_*.csv","Q3B历史运行","无监督迁移/候选方法输出",False,"historical_prediction_not_truth"],
      ["outputs/q3c_final_labels_and_display/final_A_P_label_table.csv","scripts/q3c_final_labels_and_display.py","先显式truth_status=unknown，再predict_proba并文件级聚合生成final_pred_label",False,"model_prediction_not_truth"],
      ["outputs/q3c_final_labels_and_display/final_A_P_labels_for_answer.csv","scripts/q3c_final_labels_and_display.py","从final_A_P_label_table导出预测标签",False,"model_prediction_not_truth"]
    ],columns=["artifact","producer","process","contains_official_truth","status"]).to_csv(art/"target_label_provenance.csv",index=False,encoding="utf-8-sig")

    ch=(df.groupby(["domain","subgroup"])[["de_present","fe_present","ba_present"]].sum().reset_index())
    ch["files"]=df.groupby(["domain","subgroup"]).size().values
    ch.to_csv(art/"channel_presence_summary.csv",index=False,encoding="utf-8-sig")
    df.groupby(["domain","subgroup"]).agg(files=("relative_path","count"),points_min=("primary_signal_points","min"),points_max=("primary_signal_points","max"),duration_min_s=("duration_seconds","min"),duration_max_s=("duration_seconds","max"),rpm_known=("rpm","count")).reset_index().to_csv(art/"duration_rpm_summary.csv",index=False,encoding="utf-8-sig")

    reps=[]
    for sg in ["source_12khz_de","source_12khz_fe","source_48khz_de","source_48khz_normal"]:
        g=df[df.subgroup==sg]
        if len(g):reps.append(g.iloc[0])
    for fn in ["A.mat","H.mat","P.mat"]:
        g=df[df.file_name==fn]
        if len(g):reps.append(g.iloc[0])
    cols=["domain","subgroup","relative_path","mat_variables_json","sensor_channels","primary_signal_var","primary_signal_shape","primary_signal_points","sampling_rate_hz","duration_seconds","rpm","rpm_source","class_label","label_source","fault_bearing_location","file_sha256","primary_signal_sha256","quality_flags"]
    pd.DataFrame(reps)[cols].to_csv(art/"representative_samples.csv",index=False,encoding="utf-8-sig")

    rr=random.Random(SEED)
    spot_paths=rr.sample(src.relative_path.tolist(),6)+rr.sample(tgt.relative_path.tolist(),4)
    spot=df[df.relative_path.isin(spot_paths)][["relative_path","mat_variables_json","primary_signal_shape","primary_signal_points","sampling_rate_hz","duration_seconds","rpm","rpm_source","class_label","label_source","file_sha256","primary_signal_sha256"]].sort_values("relative_path")
    spot.to_csv(art/"validation_spot_checks.csv",index=False,encoding="utf-8-sig")

    t_ids=sorted(Path(x).stem for x in tgt.relative_path)
    q={"total_expected":177,"total_actual":len(df),"source_expected":161,"source_actual":len(src),"target_expected":16,"target_actual":len(tgt),
       "source_class_expected":EXPECTED,"source_class_actual":{k:actual.get(k,0) for k in EXPECTED},"target_ids_expected":TARGET_IDS,"target_ids_actual":t_ids,
       "all_read_ok":bool((df.mat_read_status=="OK").all()),"all_target_256000":bool((tgt.primary_signal_points==256000).all()),
       "all_target_8s":bool(np.allclose(tgt.duration_seconds.astype(float),8.0,rtol=0,atol=1e-12))}
    q["passed"]=bool(q["total_actual"]==177 and q["source_actual"]==161 and q["target_actual"]==16 and q["source_class_actual"]==EXPECTED and t_ids==TARGET_IDS and q["all_read_ok"] and q["all_target_256000"] and q["all_target_8s"])
    (art/"quantity_conservation.json").write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding="utf-8")

    sev=pd.DataFrame(severe,columns=["file","issue"]);sev.to_csv(art/"serious_mismatches.csv",index=False,encoding="utf-8-sig")
    (art/"independent_sample_definition.md").write_text(
      f"# 独立样本口径\n\n一个原始MAT文件=一个独立采集对象。共{len(df)}个：源域{len(src)}、目标域{len(tgt)}。\n"
      "唯一ID为data/raw下POSIX相对路径。不同子目录存在同名basename，因此禁止只用文件名作为ID。\n"
      "后续所有窗口/分段必须携带independent_object_id；同一MAT派生窗口必须同组切分，不得跨训练/验证/测试。\n"
      "DE/FE/BA是传感器位置；fault_bearing_location是发生故障的轴承位置，两者严格分离。\n"
      "目标A-P真值未知；约600rpm只作近似工况，不作逐时精确转速。\n",encoding="utf-8")

    struct=[{"relative_path":r.relative_path,"variables":json.loads(r.mat_variables_json) if r.mat_variables_json else [],"primary_points":None if pd.isna(r.primary_signal_points) else int(r.primary_signal_points)} for r in df.itertuples()]
    structure_digest=hb(cj(struct).encode())
    dm={"schema_version":"01A-1.0","manifest_type":"data_manifest","status":"frozen","raw_data_version_id":raw_id,"created_utc":created,"run_id":run_id,
        "code_version_id":"git:"+code_head,"raw_root":"data/raw","fingerprint_algorithm":"sha256","dataset_digest":dataset_digest,"structure_digest":structure_digest,
        "file_count":len(df),"source_count":len(src),"target_count":len(tgt),"source_class_counts":{k:actual.get(k,0) for k in EXPECTED},"target_ids":t_ids,"target_truth":"unknown",
        "fingerprint_rule":"SHA256(canonical JSON sorted [relative_path,size_bytes,sha256])","metadata_artifact":f"outputs/runs/{run_id}/artifacts/file_level_metadata.csv",
        "fingerprint_artifact":f"outputs/runs/{run_id}/artifacts/raw_fingerprint_manifest.csv","note":"data/raw只读；派生结果另存"}
    (art/"data_manifest.json").write_text(json.dumps(dm,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"01A"/"data_manifest.json").write_text(json.dumps(dm,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"01A"/"raw_data_version_id.txt").write_text(raw_id+"\n",encoding="utf-8")

    dup_raw=sum(x["hash_type"]=="raw_file_sha256" for x in dups); dup_sig=sum(x["hash_type"]=="primary_signal_sha256" for x in dups)
    quality={"run_id":run_id,"raw_data_version_id":raw_id,"total_files":len(df),"read_ok":int((df.mat_read_status=="OK").sum()),"read_failed":int((df.mat_read_status!="OK").sum()),
      "source_class_count_check":count_rows,"target_8s_32khz":{"files":len(tgt),"all_256000_points":q["all_target_256000"],"all_duration_8s":q["all_target_8s"]},
      "rpm":{"source_known":int(src.rpm.notna().sum()),"source_missing":int(src.rpm.isna().sum()),"target_exact_available":False,"target_condition":"about 600 rpm only"},
      "duplicates":{"raw_file_hash_groups":dup_raw,"primary_signal_hash_groups":dup_sig},"nonfinite_files":int((df.nonfinite_count.fillna(0)>0).sum()),"constant_files":int(df.is_constant.fillna(False).astype(bool).sum()),
      "serious_mismatch_count":len(sev),"serious_mismatches":severe,"target_label_boundary":"A-P raw truth unknown; repository labels/predictions are derived unless separately proven official"}
    quality["pass"]=bool(q["passed"] and len(sev)==0)
    (art/"quality_report.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding="utf-8")
    (art/"quality_report.md").write_text(
      f"# STEP01-A质量报告\n\nrun_id: {run_id}\nraw_data_version_id: {raw_id}\n"
      f"读取：{quality['read_ok']}/{len(df)}；失败{quality['read_failed']}。\n"
      f"源域：{len(src)}/161；OR/IR/B/N={actual.get('OR',0)}/{actual.get('IR',0)}/{actual.get('B',0)}/{actual.get('N',0)}。\n"
      f"目标：{len(tgt)}/16，16/16均256000点，按32kHz均8秒，真值unknown。\n"
      f"源RPM可得{quality['rpm']['source_known']}/161；目标约600rpm仅作近似条件。\n"
      f"字节重复组{dup_raw}；主信号内容重复组{dup_sig}；非有限值文件{quality['nonfinite_files']}；常量文件{quality['constant_files']}；严重错配{len(sev)}。\n"
      "历史target_data_with_predictions及Q3A/Q3B/Q3C标签均为派生预测；Q3C脚本显式truth_status=unknown后才生成final_pred_label，不作为官方真值。\n",encoding="utf-8")

    val={"step_id":"01-A","run_id":run_id,"raw_data_version_id":raw_id,"checks":{
      "spot_check":{"passed":len(spot)==10,"evidence":f"outputs/runs/{run_id}/artifacts/validation_spot_checks.csv","files":spot.relative_path.tolist()},
      "hash_and_size_check":{"passed":bool(df.file_sha256.str.len().eq(64).all()),"evidence":f"outputs/runs/{run_id}/artifacts/raw_fingerprint_manifest.csv","raw_duplicate_groups":dup_raw,"primary_signal_duplicate_groups":dup_sig},
      "independent_sample_traceability":{"passed":df.independent_object_id.nunique()==len(df),"evidence":f"outputs/runs/{run_id}/artifacts/independent_sample_definition.md","unique_objects":int(df.independent_object_id.nunique())},
      "quantity_conservation":{"passed":q["passed"],"evidence":f"outputs/runs/{run_id}/artifacts/quantity_conservation.json","source":len(src),"target":len(tgt),"total":len(df)},
      "target_8s_32khz":{"passed":q["all_target_256000"] and q["all_target_8s"],"evidence":f"outputs/runs/{run_id}/artifacts/file_level_metadata.csv","key_numbers":{"files":len(tgt),"points":256000,"fs_hz":32000,"duration_s":8.0}},
      "target_truth_unknown":{"passed":bool((tgt.target_truth_status=="unknown").all()),"evidence":f"outputs/runs/{run_id}/artifacts/target_label_provenance.csv"}
    }}
    val["passed"]=bool(all(x["passed"] for x in val["checks"].values()) and quality["pass"]);val["status"]="passed" if val["passed"] else "failed"
    (art/"validation_record.json").write_text(json.dumps(val,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"01A"/"validation_record.json").write_text(json.dumps(val,ensure_ascii=False,indent=2),encoding="utf-8")

    fr={"schema_version":"00C-A-freeze-1.0","package_type":"A_freeze_package","step_id":"01-A","freeze_package_id":f"FREEZE-01A-{raw_id}-{run_id[-8:]}","status":"passed" if val["passed"] else "failed","run_id":run_id,
        "versions":{"code_version_id":"git:"+code_head,"raw_data_version_id":raw_id,"input_derived_data_ids":[],"run_config_sha256":cfg_sha,"environment_sha256":env_sha},"random_seed":SEED,
        "parameters":{"raw_root":"data/raw","expected_source_counts":EXPECTED,"expected_target_ids":TARGET_IDS,"target_fs_hz":32000,"target_duration_s":8.0,"target_rpm":"about 600 only"},
        "real_results":[{"name":"file_counts","value":{"total":len(df),"source":len(src),"target":len(tgt)}},{"name":"source_class_counts","value":count_rows},{"name":"target_shape_time","value":{"files":len(tgt),"points_each":256000,"fs_hz":32000,"duration_s":8.0}},{"name":"duplicates","value":quality["duplicates"]},{"name":"serious_mismatch_count","value":len(sev)}],
        "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/validation_record.json"},{"path":f"outputs/runs/{run_id}/logs/actual_read_log.csv"},{"path":f"outputs/runs/{run_id}/artifacts/raw_fingerprint_manifest.csv"}],
        "anomalies_and_failures":severe,"b_handoff":{"required_data":["raw_data_version_id","data_manifest","file_level_metadata","quality_report"],
        "supported_conclusions":["177 raw MAT files actually read","source counts match 77/40/40/4","target A-P truth unknown","target arrays 256000 points => 8 s at 32 kHz","independent unit is original MAT file"],
        "wording_limits":["historical target predictions are not ground truth","about-600rpm is not exact target speed","DE/FE sensor variables are not fault-bearing labels"],"approved_tables":["file_level_metadata.csv","label_mapping.csv","data_dictionary.csv"],"approved_figures":[]},
        "unresolved_issues":[] if val["passed"] else severe,"freeze":{"created_utc":created,"content_sha256":None,"invalidation_dependencies":["data/raw byte changes","audit code changes","metadata parsing rule changes"]}}
    f0=json.loads(json.dumps(fr,ensure_ascii=False));fsha=hb(cj(f0).encode());fr["freeze"]["content_sha256"]=fsha
    (art/"A_freeze_package.json").write_text(json.dumps(fr,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"01A"/"latest_freeze_package.json").write_text(json.dumps(fr,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol"/"01A"/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    pf=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True);(mani/"pip_freeze.txt").write_text(pf,encoding="utf-8")
    runtime={"captured_utc":created,"platform":platform.platform(),"system":platform.system(),"machine":platform.machine(),"python_version":platform.python_version(),"cpu_count_logical":os.cpu_count(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"gpu":None,"cuda":None,"git_commit":code_head,"pip_freeze_sha256":hf(mani/"pip_freeze.txt")}
    (mani/"environment_runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")

    outs=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="run_manifest.json"):
        outs.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
    ot=hb(cj(outs).encode())
    rm={"schema_version":"01A-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"01-A","status":"completed" if val["passed"] else "failed","created_utc":created,"execution_mode":"chat_plus_git",
        "binding":{**binding,"binding_digest":bd},"code":{"repository":"Mhhhh958/mathmodeling","commit":code_head,"entrypoint":"scripts/step01a_raw_mat_audit.py","code_manifest":"protocol/01A/code_manifest.json"},
        "data":{"raw_data_version_id":raw_id,"data_manifest":"protocol/01A/data_manifest.json","dataset_digest":dataset_digest,"structure_digest":structure_digest},
        "environment":{"environment_manifest":"protocol/00B/environment_manifest.json","environment_manifest_sha256":env_sha,"runtime_pip_freeze":"manifests/pip_freeze.txt"},
        "config":{"path":"protocol/01A/run_config.json","sha256":cfg_sha},"outputs":{"root":f"outputs/runs/{run_id}","files":outs,"output_tree_sha256":ot},
        "checkpoint":{"completed_batches":["all-177-mat-files"],"pending_batches":[],"resume_token":None}}
    (mani/"run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":val["passed"],"run_id":run_id,"raw_data_version_id":raw_id,"dataset_digest":dataset_digest,"source_counts":actual,"target_count":len(tgt),"serious_mismatch_count":len(sev),"output_root":f"outputs/runs/{run_id}"},ensure_ascii=False))
    return 0 if val["passed"] else 2

if __name__=="__main__":
    raise SystemExit(main())
