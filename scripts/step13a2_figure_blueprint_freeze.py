#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP13-A2: freeze final figure/table source snapshots and figure blueprints.
No paper Word edit and no final figure rendering are performed here.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "protocol/13A2/run_config.json"
ENV = ROOT / "protocol/00B/environment_manifest.json"
STATE = ROOT / "protocol/00C/process_state_card.json"
FMT = ROOT / "protocol/00D/format_master.json"
SRC_FREEZE = ROOT / "protocol/13A1/latest_freeze_package.json"
SRC_VALID = ROOT / "protocol/13A1/validation_record.json"
SRC_CODE = ROOT / "protocol/13A1/code_manifest.json"
RUN13 = "run_13-A1_20260920T171911944243Z_23a7d455_0eedc9bc"
ART13 = ROOT / "outputs/runs" / RUN13 / "artifacts"
CORE = ART13 / "core_numbers_master.csv"
AP = ART13 / "A_P_final_answer_table.csv"
RUNLIST = ART13 / "final_run_id_list.csv"
BUNDLE = ART13 / "final_code_bundle.json"
TRACE = ART13 / "traceability_index.csv"

SRC = {
 "q1_wave": ROOT/"outputs/runs/run_03-A_20260919T164527811276Z_71ebe710_f5307f8f/artifacts/representative_waveform_segments.csv",
 "q1_psd": ROOT/"outputs/runs/run_03-A_20260919T164527811276Z_71ebe710_f5307f8f/artifacts/representative_psd.csv",
 "q1_envpsd": ROOT/"outputs/runs/run_03-A_20260919T164527811276Z_71ebe710_f5307f8f/artifacts/representative_envelope_psd.csv",
 "q1_freq": ROOT/"outputs/runs/run_03-A_20260919T164527811276Z_71ebe710_f5307f8f/artifacts/file_mechanism_frequencies.csv",
 "q2_conf": ROOT/"outputs/runs/run_06-A_20260920T035742210347Z_b83cb3a6_01576948/artifacts/pooled_file_confusion_matrix.csv",
 "q2_winpred": ROOT/"outputs/runs/run_06-A_20260920T035742210347Z_b83cb3a6_01576948/artifacts/oof_window_predictions.csv",
 "q2_pair": ROOT/"outputs/runs/run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb/artifacts/Base_vs_H1_paired_outer.csv",
 "window_index": ROOT/"outputs/runs/run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0/artifacts/window_index.csv",
 "q3_shift": ROOT/"outputs/runs/run_08-A_20260920T060936354760Z_049c8ddb_8cd7401f/artifacts/feature_shift_file_level.csv",
 "q3_tune": ROOT/"outputs/runs/run_09-A_20260920T070725126712Z_6ef81c21_26f99878/artifacts/tuning_metrics_loads0_1_2.csv",
 "q3_hold": ROOT/"outputs/runs/run_09-A_20260920T070725126712Z_6ef81c21_26f99878/artifacts/held_load3_primary_metrics.csv",
 "q3_boot": ROOT/"outputs/runs/run_12-A2_20260920T162707032450Z_88add8e8_f107c2fa/artifacts/A2_05_source_bootstrap_summary.csv",
 "q3_loto": ROOT/"outputs/runs/run_12-A2_20260920T162707032450Z_88add8e8_f107c2fa/artifacts/A2_04_target_LOTO_stability.csv",
 "q4_contrib": ROOT/"outputs/runs/run_11-A_20260920T151947188398Z_54a3c266_1f812c22/artifacts/file_feature_contributions.csv",
 "q4_faith": ROOT/"outputs/runs/run_11-A_20260920T151947188398Z_54a3c266_1f812c22/artifacts/faithfulness_ablation.csv",
}

FEATURE_CN = {
 "mean":"均值","std":"标准差","rms":"均方根","mean_abs":"绝对值均值","peak_abs":"绝对峰值",
 "peak_to_peak":"峰峰值","skewness":"偏度","kurtosis":"峭度","crest_factor":"峰值因子",
 "impulse_factor":"脉冲因子","shape_factor":"波形因子","clearance_factor":"裕度因子",
 "zero_cross_rate":"过零率","spectral_centroid_hz":"谱质心","spectral_rms_hz":"谱均方根频率",
 "spectral_entropy":"谱熵","spectral_flatness":"谱平坦度","dominant_frequency_hz":"主频",
 "rolloff95_hz":"95%滚降频率","band_ratio_0_500":"0–500 Hz能量占比",
 "band_ratio_500_1500":"500–1500 Hz能量占比","band_ratio_1500_3000":"1500–3000 Hz能量占比",
 "band_ratio_3000_5500":"3000–5500 Hz能量占比","envelope_rms":"包络均方根",
 "envelope_kurtosis":"包络峭度","envelope_spectral_entropy":"包络谱熵"
}
CASE_CN = {
 "SRC_OR_REP":"源域OR代表","SRC_IR_REP":"源域IR代表","SRC_B_REP":"源域B代表","SRC_N_REP":"源域N代表",
 "SRC_OOF_DIFFICULT":"历史OOF困难OR","TGT_STABLE_OR":"目标H（预测OR）","TGT_STABLE_B":"目标O（预测B）",
 "TGT_LOW_BOOT":"目标D（预测OR）","TGT_LOW_SETTING":"目标N（预测OR）"
}

def readj(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def shaf(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def canon(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()

def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()

def require(cond, msg):
    if not cond: raise RuntimeError(msg)

def rdf(p):
    return pd.read_csv(p,encoding="utf-8-sig")

def write_df(df, p):
    p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding="utf-8-sig")

def cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s))

def core_value(core, nid):
    r=core.loc[core["number_id"]==nid]
    require(len(r)==1, f"missing core number {nid}")
    return float(r.iloc[0]["exact_value"])

def source_snapshot(outsrc: Path, figid: str, core: pd.DataFrame, ap: pd.DataFrame):
    paths=[]
    if figid=="FIG-Q1-01":
        obj={"figure_id":figid,"nodes":[
            {"id":"s","text":"源域49文件 + 目标域16文件","evidence":{"Q1-N-SRC":core_value(core,"Q1-N-SRC"),"Q1-N-TGT":core_value(core,"Q1-N-TGT")}},
            {"id":"r","text":"统一12 kHz共同分析时基","evidence":"03-A/04-A冻结处理口径"},
            {"id":"w","text":"1 s物理时间窗 + 50%重叠","evidence":{"source_windows":core_value(core,"Q1-W-SRC"),"target_windows":core_value(core,"Q1-W-TGT")}},
            {"id":"f","text":"26维源—目标兼容公共特征","evidence":{"feature_count":core_value(core,"Q1-FEAT")}},
            {"id":"o","text":"问题二接口：X + y + group_id","evidence":"同源窗口保持同一独立文件分组"}],
            "edges":[["s","r"],["r","w"],["w","f"],["f","o"]]}
        p=outsrc/f"{figid}.json"; p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8"); paths=[p]
    elif figid=="FIG-Q1-02":
        df=rdf(SRC["q1_wave"])
        m=df["sample_id"].astype(str).str.contains(r"^B_(?:typical|difficult)_",regex=True)
        m &= df["stage"].astype(str).eq("primary_processed")
        m &= pd.to_numeric(df["time_s"],errors="coerce").le(0.12)
        s=df.loc[m,["sample_id","stage","time_s","value"]].copy()
        require(s["sample_id"].nunique()==2 and len(s)>100, "Q1 waveform snapshot incomplete")
        s["sample_role_cn"]=np.where(s["sample_id"].str.contains("typical"),"典型样本","困难样本")
        p=outsrc/f"{figid}.csv"; write_df(s,p); paths=[p]
    elif figid=="FIG-Q1-03":
        parts=[]
        for key,view in [("q1_psd","主信号功率谱"),("q1_envpsd","Hilbert包络谱")]:
            d=rdf(SRC[key]); m=d["sample_id"].astype(str).str.contains("B_difficult_B014_2")
            m &= pd.to_numeric(d["frequency_hz"],errors="coerce").le(500)
            z=d.loc[m,["sample_id","stage","frequency_hz","psd"]].copy(); z["view_cn"]=view; parts.append(z)
        s=pd.concat(parts,ignore_index=True)
        require(set(s["view_cn"])=={"主信号功率谱","Hilbert包络谱"} and len(s)>20,"Q1 spectrum snapshot incomplete")
        p=outsrc/f"{figid}.csv"; write_df(s,p); paths=[p]
        fr=rdf(SRC["q1_freq"]); col="relative_path" if "relative_path" in fr.columns else fr.columns[0]
        fm=fr[col].astype(str).str.endswith("B014_2.mat")
        mr=fr.loc[fm].copy()
        require(len(mr)>=1,"Q1 spectrum mechanism markers missing")
        mp=outsrc/f"{figid}_markers.csv"; write_df(mr,mp); paths.append(mp)
    elif figid=="FIG-Q2-01":
        obj={"figure_id":figid,"nodes":[
          {"id":"i","text":"49个独立源文件 + 26维公共特征","evidence":{"files":core_value(core,"Q1-N-SRC"),"features":core_value(core,"Q1-FEAT")}},
          {"id":"g","text":"按原始文件分组的嵌套验证","evidence":"05-A冻结外层/内层分组"},
          {"id":"b","text":"Base逻辑回归基线","evidence":{"Macro-F1":core_value(core,"Q2-BASE")}},
          {"id":"h","text":"H1_RF单因素改进并冻结","evidence":{"development_Macro-F1":core_value(core,"Q2-DEV"),"trees":core_value(core,"Q2-TREES"),"depth":core_value(core,"Q2-DEPTH")}},
          {"id":"v","text":"固定模型跨载荷正交压力测试","evidence":{"Macro-F1":core_value(core,"Q2-ORTHO"),"min_recall":core_value(core,"Q2-MINREC")}},
          {"id":"o","text":"问题三接口：冻结RF + 26维特征","evidence":"MODEL-SOURCE-4ee408ac96054e3b"}],
          "edges":[["i","g"],["g","b"],["b","h"],["h","v"],["v","o"]]}
        p=outsrc/f"{figid}.json"; p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8"); paths=[p]
    elif figid=="FIG-Q2-02":
        d=rdf(SRC["q2_conf"]); d=d.rename(columns={d.columns[0]:"true_label"})
        require(d.shape==(4,5),"Q2 confusion matrix shape mismatch")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q2-03":
        d=rdf(SRC["q2_pair"])
        require(len(d)==4 and (d["delta_macro_f1"]<0).any(),"Q2 paired result missing negative fold")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q2-04":
        pwin=rdf(SRC["q2_winpred"]); idx=rdf(SRC["window_index"])[["window_id","window_start_s","window_end_s"]]
        d=pwin[pwin["group_id"].astype(str).str.endswith(("IR007_0.mat","N_0.mat"))].copy().merge(idx,on="window_id",how="left",validate="one_to_one")
        require(d["window_start_s"].notna().all() and d["group_id"].nunique()==2,"Q2 failure timing join failed")
        keep=["group_id","class_label","window_start_s","window_end_s","window_pred_label","score_OR","score_IR","score_B","score_N"]
        d=d[keep].sort_values(["group_id","window_start_s"])
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q3-01":
        obj={"figure_id":figid,"nodes":[
          {"id":"i","text":"冻结RF + 源/目标26维公共特征","evidence":{"source_files":core_value(core,"Q1-N-SRC"),"target_files":core_value(core,"Q1-N-TGT")}},
          {"id":"d","text":"无标签域差异诊断","evidence":{"MMD2":core_value(core,"Q3-MMD")}},
          {"id":"p","text":"载荷0/1/2模拟目标域选模","evidence":"真实A—P真值不参与选模"},
          {"id":"t","text":"冻结T1位置—尺度收缩","evidence":{"alpha":core_value(core,"Q3-ALPHA")}},
          {"id":"a","text":"真实A—P传导式无监督适配","evidence":"16文件×15窗口"},
          {"id":"r","text":"逐文件稳健性复核与正式标签","evidence":{"LOTO_agreement":core_value(core,"Q3-LOTO"),"bootstrap_min":core_value(core,"Q3-BOOT")}}],
          "edges":[["i","d"],["d","p"],["p","t"],["t","a"],["a","r"]]}
        p=outsrc/f"{figid}.json"; p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8"); paths=[p]
    elif figid=="FIG-Q3-02":
        d=rdf(SRC["q3_shift"]).sort_values("abs_standardized_mean_shift",ascending=False).head(8).copy()
        d["feature_cn"]=d["feature"].map(FEATURE_CN).fillna(d["feature"])
        d["family_cn"]=d["family"].map({"amplitude_or_scale":"幅值/尺度","geometry_free_mechanism_proxy":"几何无关机理代理","shape_statistics":"形态统计"}).fillna(d["family"])
        require(len(d)==8,"Q3 shift top8 missing")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q3-03":
        t=rdf(SRC["q3_tune"])
        sel=((t["seed"]==20260919)&(
             ((t["method"]=="A0")&(pd.to_numeric(t["setting"],errors="coerce")==0))|
             ((t["method"]=="T1")&(np.isclose(pd.to_numeric(t["setting"],errors="coerce"),0.25)))|
             ((t["method"]=="T2")&(np.isclose(pd.to_numeric(t["setting"],errors="coerce"),1.0)))|
             ((t["method"]=="T3")&(np.isclose(pd.to_numeric(t["setting"],errors="coerce"),2.0)))))
        a=t.loc[sel,["load","seed","method","setting","macro_f1","accuracy","min_class_recall"]].copy()
        h=rdf(SRC["q3_hold"])
        h=h[(h["seed"]==20260919)&h["method"].isin(["A0","T1"])][["load","seed","method","setting","macro_f1","accuracy","min_class_recall"]]
        d=pd.concat([a,h],ignore_index=True).sort_values(["method","load"])
        require(set(d["load"].unique())=={0,1,2,3} and set(d[d["load"]==3]["method"])=={"A0","T1"},"Q3 transfer scope mismatch")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q3-04":
        b=rdf(SRC["q3_boot"]); l=rdf(SRC["q3_loto"])[["target_id","loto_label","same_label","top_score_delta"]]
        d=b.merge(l,on="target_id",how="left",validate="one_to_one").sort_values("target_id")
        require(len(d)==16 and (~d["same_label"]).sum()==1 and d.loc[~d["same_label"],"target_id"].iloc[0]=="D","Q3 robustness expected D-only LOTO flip")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q4-01":
        obj={"figure_id":figid,"nodes":[
          {"id":"i","text":"T1适配后的26维目标输入","evidence":{"alpha":core_value(core,"Q3-ALPHA")}},
          {"id":"m","text":"冻结H1_RF文件级预测","evidence":"MODEL-TRANSFER-82c6b6ba4d2dd557"},
          {"id":"e","text":"逐特征遮蔽贡献","evidence":"源域独立文件均值的中位数作为参考"},
          {"id":"c","text":"源域已知标签/目标预测案例","evidence":"9个冻结解释案例"},
          {"id":"v","text":"忠实性 + 相邻窗口稳定性 + 机理旁证","evidence":{"top5_drop_median":core_value(core,"Q4-FAITH"),"spearman_median":core_value(core,"Q4-SPEAR"),"jaccard_median":core_value(core,"Q4-JACC")}}],
          "edges":[["i","m"],["m","e"],["e","c"],["c","v"]]}
        p=outsrc/f"{figid}.json"; p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8"); paths=[p]
    elif figid in ("FIG-Q4-02","FIG-Q4-03"):
        d=rdf(SRC["q4_contrib"])
        target = figid=="FIG-Q4-03"
        d=d[(d["domain"]==("target" if target else "source")) & (d["rank"]<=5)].copy()
        d["feature_cn"]=d["feature"].map(FEATURE_CN).fillna(d["feature"])
        d["case_cn"]=d["case_id"].map(CASE_CN).fillna(d["case_id"])
        expect=20 if target else 25
        require(len(d)==expect,"Q4 contribution row count mismatch")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    elif figid=="FIG-Q4-04":
        d=rdf(SRC["q4_faith"]).copy()
        d["case_cn"]=d["case_id"].map(CASE_CN).fillna(d["case_id"])
        require(len(d)==9 and d["faithfulness_case_pass"].astype(str).str.lower().isin(["true","1"]).all(),"Q4 faithfulness cases incomplete")
        p=outsrc/f"{figid}.csv"; write_df(d,p); paths=[p]
    else:
        raise KeyError(figid)
    return paths

def coordinate_limits(figid, paths):
    if figid in {"FIG-Q1-01","FIG-Q2-01","FIG-Q3-01","FIG-Q4-01"}:
        return {"canvas_x":[0,1],"canvas_y":[0,1],"scale":"normalized"}
    p=paths[0]
    d=rdf(p) if p.suffix==".csv" else None
    if figid=="FIG-Q1-02":
        m=float(np.nanmax(np.abs(pd.to_numeric(d["value"],errors="coerce"))))
        return {"x":[0.0,0.12],"y":[-1.05*m,1.05*m],"scale":"linear"}
    if figid=="FIG-Q1-03":
        res={"x":[0.0,500.0],"scale":"log_y","y_by_view":{}}
        for v,g in d.groupby("view_cn"):
            y=pd.to_numeric(g["psd"],errors="coerce"); y=y[y>0]
            res["y_by_view"][v]=[float(y.min()*0.8),float(y.max()*1.2)]
        return res
    if figid=="FIG-Q2-02":
        vals=d.drop(columns=["true_label"]).apply(pd.to_numeric,errors="coerce").values
        return {"color":[0,float(np.nanmax(vals))],"scale":"integer_counts"}
    if figid=="FIG-Q2-03": return {"x":[-0.3,3.3],"y":[0.0,1.0],"scale":"linear"}
    if figid=="FIG-Q2-04":
        x=pd.to_numeric(d["window_start_s"],errors="coerce")
        return {"x":[float(x.min()),float(x.max())],"y":[0.0,1.0],"scale":"linear"}
    if figid=="FIG-Q3-02":
        x=pd.to_numeric(d["abs_standardized_mean_shift"],errors="coerce")
        return {"x":[0.0,float(x.max()*1.05)],"scale":"linear"}
    if figid=="FIG-Q3-03": return {"x":[-0.2,3.2],"y":[0.0,1.0],"scale":"linear"}
    if figid=="FIG-Q3-04": return {"x_categories":"A-P","y":[0.0,1.0],"scale":"linear"}
    if figid in {"FIG-Q4-02","FIG-Q4-03"}:
        x=pd.to_numeric(d["file_score_drop_when_occluded"],errors="coerce")
        lo=min(0.0,float(x.min()*1.05)); hi=float(x.max()*1.05)
        return {"x":[lo,hi],"scale":"linear"}
    if figid=="FIG-Q4-04":
        cols=["top5_drop","low5_drop","random_null_median"]
        vals=pd.concat([pd.to_numeric(d[c],errors="coerce") for c in cols])
        lo=min(0.0,float(vals.min()*1.08)); hi=float(vals.max()*1.05)
        return {"x":[lo,hi],"scale":"linear"}
    return {}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--validate-only",action="store_true"); args=ap.parse_args()
    cfg,env,state,fmt,freeze,val,code = map(readj,[CFG,ENV,STATE,FMT,SRC_FREEZE,SRC_VALID,SRC_CODE])
    require(state["CURRENT_ALLOWED_STEP"]=="13-A2" and state["NEXT_ALLOWED"]=="13-A2","gate CURRENT/NEXT not 13-A2")
    require(state["LAST_PASS_TOKEN"]=="13-A1-V2026.09.20-24e90683-通过","invalid previous pass token")
    require(state.get("OPEN_P0")==[],"OPEN_P0 not empty")
    require(freeze["status"]=="passed" and val["passed"] is True,"13-A1 not passed")
    require(freeze["freeze_package_id"]==cfg["source_freeze"]["freeze_package_id"],"13-A1 freeze identity mismatch")
    require(env["environment_version_id"]==cfg["environment_version_id"],"00-B environment mismatch")
    for p in [CORE,AP,RUNLIST,BUNDLE,TRACE,*SRC.values()]: require(p.exists(),f"required source missing: {p}")
    runs=rdf(RUNLIST); auth=set(runs.loc[runs["status"]=="passed","run_id"].astype(str))
    for p in SRC.values():
        s=str(p)
        if "/outputs/runs/" in s:
            rid=s.split("/outputs/runs/",1)[1].split("/",1)[0]
            require(rid in auth,f"source run not in 13-A1 passed run list: {rid}")
    core,apdf=rdf(CORE),rdf(AP)
    body=fmt["page_setup"]["width_cm"]-fmt["page_setup"]["margins_cm"]["left"]-fmt["page_setup"]["margins_cm"]["right"]
    require(abs(body-cfg["width_policy"]["expected_body_width_cm"])<1e-9,f"00-D body width mismatch {body}")
    bind_obj={"step":"13-A2","head":git("rev-parse","HEAD"),"cfg":shaf(CFG),"env":shaf(ENV),"src_freeze":freeze["freeze"]["content_sha256"],"seed":cfg["random_seed"]}
    bind=canon(bind_obj)
    created=datetime.now(timezone.utc).isoformat()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id=f"run_13-A2_{stamp}_{bind[:8]}_{freeze['freeze_package_id'][-8:]}"
    out=ROOT/"outputs/runs"/run_id; art=out/"artifacts"; srcdir=art/"figure_sources"; mani=out/"manifests"; logs=out/"logs"
    for d in [srcdir,mani,logs]: d.mkdir(parents=True,exist_ok=True)

    width_map={k:body*float(v) for k,v in cfg["width_policy"]["classes"].items()}
    visual={
      "schema_version":"13A2-visual-1.0","body_width_cm":body,"width_classes_cm":width_map,
      "font":{"primary_cn":"Noto Sans CJK SC","fallback":["Source Han Sans SC","Microsoft YaHei","SimHei"],"math":"DejaVu Sans",
              "sizes_pt":{"axis_label":10,"tick":9,"legend":9,"annotation":9,"panel_mark":10,"internal_title":0}},
      "lines_markers":{"main_line_width_pt":1.2,"axis_line_width_pt":0.8,"marker_size_pt":4.5,"error_line_width_pt":1.0},
      "colors":{"class":{"OR":"#4477AA","IR":"#CC6677","B":"#228833","N":"#CCBB44"},
                "model":{"Base/A0":"#6B7280","H1_RF":"#4477AA","T1":"#228833","T2":"#CC6677","T3":"#AA3377"},
                "neutral_text":"#222222","grid":"#B8BDC5"},
      "legend":{"default":"best without covering data","outside_when_dense":True,"frame":False},
      "grid":{"quantitative":"y-only light dashed where helpful","heatmap_framework":"off"},
      "background":"white","forbidden":["gradient","shadow","rounded_card","status_icon","decorative_block","dashboard"],
      "export":{"primary":"SVG","fallback":"PNG","png_dpi":300,"transparent":False,"tight_bbox":True,"embed_font_or_convert_text":False},
      "framework":{"tool":"matplotlib vector objects","box":"white rectangle + 0.8pt border","arrow":"simple arrow","max_accent_colors":2,"no_generated_image":True},
      "language":"Chinese titles/axes/legends/colorbars/annotations; necessary abbreviations/math symbols allowed"
    }
    (art/"research_figure_visual_spec.json").write_text(json.dumps(visual,ensure_ascii=False,indent=2),encoding="utf-8")

    inv=[]; locator=[]; captions=[]; trace_rows=[]; srcmanifest=[]; coords={}
    for f in cfg["figures"]:
        paths=source_snapshot(srcdir,f["figure_id"],core,apdf)
        coords[f["figure_id"]]=coordinate_limits(f["figure_id"],paths)
        width=width_map[f["width_class"]]
        entry=f"python scripts/step13a2_figure_renderer.py --run-id {run_id} --figure-id {f['figure_id']}"
        inv.append({**f,"insert_width_cm":round(width,6),"source_snapshot_files":";".join(p.relative_to(ROOT).as_posix() for p in paths),
                    "renderer_entry":entry,"coordinate_limits_json":json.dumps(coords[f["figure_id"]],ensure_ascii=False)})
        locator.append({"figure_id":f["figure_id"],"question":f["question"],"section":f["section"],"title_cn":f["title_cn"],
                        "purpose":f["purpose"],"pre_sentence":f["pre_sentence"],"post_conclusion":f["post_conclusion"],
                        "required":f["required"],"insert_width_cm":round(width,6)})
        captions.append({"figure_id":f["figure_id"],"title_cn":f["title_cn"],
          "caption_draft_cn":f"{f['title_cn']}。样本口径：{f['sample_scope']}；单位：{f['units']}。{f['caption_key']}"})
        orig=[]
        if f["figure_id"].startswith("FIG-Q1-02"): orig=[SRC["q1_wave"]]
        elif f["figure_id"].startswith("FIG-Q1-03"): orig=[SRC["q1_psd"],SRC["q1_envpsd"],SRC["q1_freq"]]
        elif f["figure_id"]=="FIG-Q2-02": orig=[SRC["q2_conf"]]
        elif f["figure_id"]=="FIG-Q2-03": orig=[SRC["q2_pair"]]
        elif f["figure_id"]=="FIG-Q2-04": orig=[SRC["q2_winpred"],SRC["window_index"]]
        elif f["figure_id"]=="FIG-Q3-02": orig=[SRC["q3_shift"]]
        elif f["figure_id"]=="FIG-Q3-03": orig=[SRC["q3_tune"],SRC["q3_hold"]]
        elif f["figure_id"]=="FIG-Q3-04": orig=[SRC["q3_boot"],SRC["q3_loto"]]
        elif f["figure_id"] in {"FIG-Q4-02","FIG-Q4-03"}: orig=[SRC["q4_contrib"]]
        elif f["figure_id"]=="FIG-Q4-04": orig=[SRC["q4_faith"]]
        else: orig=[CORE]
        orig_rel=[p.relative_to(ROOT).as_posix() for p in orig]
        run_ids=[]
        for p in orig_rel:
            if p.startswith("outputs/runs/"): run_ids.append(p.split("/")[2])
            else: run_ids.append(RUN13)
        trace_rows.append({"figure_id":f["figure_id"],"section":f["section"],"purpose":f["purpose"],
          "original_sources":";".join(orig_rel),"source_run_ids":";".join(sorted(set(run_ids))),
          "snapshot_files":";".join(p.relative_to(ROOT).as_posix() for p in paths),
          "snapshot_sha256":";".join(shaf(p) for p in paths),"renderer_entry":entry,
          "pre_sentence":f["pre_sentence"],"post_conclusion":f["post_conclusion"]})
        for p in paths:
            srcmanifest.append({"figure_id":f["figure_id"],"snapshot_path":p.relative_to(ROOT).as_posix(),
              "sha256":shaf(p),"size_bytes":p.stat().st_size,"original_sources":";".join(orig_rel),"source_run_ids":";".join(sorted(set(run_ids)))})
    write_df(pd.DataFrame(inv),art/"final_figure_inventory.csv")
    write_df(pd.DataFrame(locator),art/"figure_section_locator.csv")
    write_df(pd.DataFrame(captions),art/"caption_drafts.csv")
    write_df(pd.DataFrame(trace_rows),art/"figure_traceability_index.csv")
    write_df(pd.DataFrame(srcmanifest),art/"figure_source_manifest.csv")
    write_df(pd.DataFrame(cfg["table_preferred"]),art/"table_preference_decisions.csv")
    (art/"coordinate_limits.json").write_text(json.dumps(coords,ensure_ascii=False,indent=2),encoding="utf-8")

    framework=[
      {"question":"Q1","need_framework":True,"criteria":"≥2连续处理阶段+明确Q1→Q2数据接口","location":"5.1标题下、5.1.1前","modules":"输入文件→统一时基→切片→公共特征→Q2接口"},
      {"question":"Q2","need_framework":True,"criteria":"分组协议+基线+改进+冻结+正交验证，多模块连续流程","location":"5.2标题下、5.2.1前","modules":"26维输入→分组验证→Base→H1_RF→正交验证→Q3接口"},
      {"question":"Q3","need_framework":True,"criteria":"域诊断+模拟域选模+真实目标适配+稳健性复核，且承接Q2接口","location":"5.3标题下、5.3.1前","modules":"冻结RF→域诊断→模拟域→T1→A-P适配→稳健性"},
      {"question":"Q4","need_framework":True,"criteria":"最终链路+遮蔽解释+多验证模块，且承接Q3最终预测","location":"5.4标题下、5.4.1前","modules":"T1输入→RF→遮蔽→案例→忠实性/稳定性/机理旁证"}
    ]
    write_df(pd.DataFrame(framework),art/"question_framework_decision.csv")

    type_plan=[
      {"type":"总体流程图","selected":False,"reason":"逐问框架已覆盖接口；再画全局流程会重复"},
      {"type":"数据特征图","selected":True,"reason":"Q1时域/频域与Q3域偏移具有直接解释价值"},
      {"type":"模型结构图","selected":False,"reason":"RF结构参数表格更清楚；逐问框架已表达模型接口"},
      {"type":"训练/收敛图","selected":False,"reason":"最终RF/T1没有可解释为性能的迭代训练损失曲线，禁止人为造损失图"},
      {"type":"结果对比图","selected":True,"reason":"Q2配对外层结果、Q3迁移候选可在同口径下比较"},
      {"type":"失败案例图","selected":True,"reason":"Q2真实误判文件必须保留"},
      {"type":"稳健性/敏感性图","selected":True,"reason":"Q3逐文件Bootstrap/LOTO与Q4忠实性可直接展示失效边界"}
    ]
    write_df(pd.DataFrame(type_plan),art/"figure_type_plan.csv")

    # Validation
    invdf=pd.DataFrame(inv)
    snapshot_ok=(pd.DataFrame(srcmanifest)["size_bytes"]>0).all()
    narrative_ok=invdf["section"].astype(str).str.len().gt(0).all() and invdf["pre_sentence"].astype(str).str.len().gt(0).all() and invdf["post_conclusion"].astype(str).str.len().gt(0).all()
    purpose_ok=invdf["purpose"].astype(str).str.len().gt(0).all()
    renderer_ok=all("step13a2_figure_renderer.py" in x for x in invdf["renderer_entry"])
    zh_ok=all(cjk(x) for x in invdf["title_cn"]) and all(cjk(x) or x=="" for x in invdf["x_label_cn"]) and all(cjk(x) or x=="" for x in invdf["y_label_cn"])
    units_ok=invdf["units"].astype(str).str.len().gt(0).all()
    coord_ok=all(bool(coords[x]) for x in invdf["figure_id"])
    width_ok=all(0 < float(x) <= body+1e-9 for x in invdf["insert_width_cm"])
    forbidden_ok=not invdf["layout"].astype(str).str.contains("2x2|2x3|dashboard|card|gradient|shadow",case=False,regex=True).any()
    subplots_ok=all(
      ("subplots" not in str(r.layout)) or r.figure_id in {"FIG-Q1-03","FIG-Q2-04"}
      for r in invdf.itertuples(index=False)
    )
    no_figure_files=not any(p.suffix.lower() in {".png",".svg",".pdf",".jpg",".jpeg"} for p in out.rglob("*") if p.is_file())
    checks={
      "gate_valid":True,"13A1_source_freeze_passed":True,"authoritative_source_runs_only":True,
      "all_planned_figures_have_nonempty_source_snapshots":bool(snapshot_ok),
      "all_figures_have_section_pre_and_post_narrative":bool(narrative_ok),
      "all_figures_have_single_primary_purpose":bool(purpose_ok),
      "renderer_entry_present_for_every_figure":bool(renderer_ok),
      "chinese_title_axis_legend_plan_passed":bool(zh_ok),
      "units_and_sample_scope_present":bool(units_ok),
      "coordinate_limits_frozen":bool(coord_ok),
      "width_derived_from_00D_body_width":bool(width_ok and abs(body-15.999)<1e-9),
      "no_unrelated_2x2_2x3_dashboard_card_layout":bool(forbidden_ok and subplots_ok),
      "difficulty_and_failure_samples_retained":bool("FIG-Q2-04" in set(invdf.figure_id) and "FIG-Q3-04" in set(invdf.figure_id)),
      "no_word_edit_performed":True,"no_formal_paper_figures_generated":True,"no_final_figure_files_created":bool(no_figure_files)
    }
    passed=all(v is True for v in checks.values())
    evidence=[
      {"check":"源数据","evidence":f"{len(invdf)} figures; {len(srcmanifest)} nonempty snapshots","file":"figure_source_manifest.csv","run_id":run_id},
      {"check":"关键数字-Q2","evidence":f"orthogonal Macro-F1={core_value(core,'Q2-ORTHO'):.4f}","file":"core_numbers_master.csv","run_id":RUN13},
      {"check":"关键数字-Q3","evidence":f"LOTO={core_value(core,'Q3-LOTO'):.4f}; min bootstrap retention={core_value(core,'Q3-BOOT'):.4f}","file":"FIG-Q3-04.csv","run_id":"run_12-A2_20260920T162707032450Z_88add8e8_f107c2fa"},
      {"check":"关键数字-Q4","evidence":f"median Top-5 drop={core_value(core,'Q4-FAITH'):.4f}; median Spearman={core_value(core,'Q4-SPEAR'):.4f}","file":"FIG-Q4-04.csv","run_id":"run_11-A_20260920T151947188398Z_54a3c266_1f812c22"},
      {"check":"正文宽度","evidence":f"21.001-2.501-2.501={body:.3f} cm","file":"protocol/00D/format_master.json","run_id":"run_00-D_20260919T154254724640Z_201f7b70_2bed85de"}
    ]
    validation={"schema_version":"13A2-validation-1.0","step_id":"13-A2","run_id":run_id,"status":"passed" if passed else "failed","passed":passed,
      "checks":checks,"evidence":evidence,"planned_figure_count":len(invdf),"source_snapshot_count":len(srcmanifest),
      "word_edit_performed":False,"formal_paper_figures_generated":False}
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    # code manifest records exact blobs from this execution commit.
    tracked=["scripts/step13a2_figure_blueprint_freeze.py","scripts/step13a2_figure_renderer.py",".github/workflows/step13a2_figure_blueprint_freeze.yml"]
    cm={"schema_version":"13A2-1.0","manifest_type":"code_manifest","mode":"git","repository":"Mhhhh958/mathmodeling",
        "tracked_code":[{"path":p,"git_blob_sha":git("rev-parse",f"HEAD:{p}")} for p in tracked],
        "frozen_inputs":[
          {"path":"protocol/13A2/run_config.json","git_blob_sha":git("rev-parse","HEAD:protocol/13A2/run_config.json")},
          {"path":"protocol/13A1/latest_freeze_package.json","freeze_id":freeze["freeze_package_id"],"content_sha256":freeze["freeze"]["content_sha256"]},
          {"path":"protocol/00D/format_master.json","body_width_cm":body},
          {"path":"protocol/00B/environment_manifest.json","environment_version_id":env["environment_version_id"]}],
        "constraints":["source/blueprint freeze only; no retraining/tuning/model selection","no Word edit","no formal figure rendering in 13-A2",
                       "renderer may only use 13-A2 frozen source snapshots in 13-B1","all final plot text planned in Chinese except necessary abbreviations/math symbols"]}
    (ROOT/"protocol/13A2").mkdir(parents=True,exist_ok=True)
    (ROOT/"protocol/13A2/code_manifest.json").write_text(json.dumps(cm,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze13={"schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":"13-A2",
      "freeze_package_id":f"FREEZE-13A2-{bind[:8]}","status":"passed" if passed else "failed","run_id":run_id,
      "versions":{"code_version_id":"git:"+git("rev-parse","HEAD"),"raw_data_version_id":freeze["versions"]["raw_data_version_id"],
                  "input_derived_data_ids":[freeze["freeze_package_id"]],"run_config_sha256":shaf(CFG),"environment_sha256":shaf(ENV)},
      "random_seed":cfg["random_seed"],
      "parameters":{"figure_count":len(invdf),"body_width_cm":body,"width_classes_cm":width_map,"final_rendering_performed":False,"word_edit_performed":False},
      "real_results":[
        {"name":"final_figure_count","value":len(invdf)},
        {"name":"framework_figure_count","value":int((invdf["kind"]=="技术框架图").sum())},
        {"name":"empirical_figure_count","value":int((invdf["kind"]!="技术框架图").sum())},
        {"name":"source_snapshot_count","value":len(srcmanifest)},
        {"name":"body_width_cm","value":body},
        {"name":"table_preferred_count","value":len(cfg["table_preferred"])}],
      "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/{x}"} for x in [
        "final_figure_inventory.csv","figure_section_locator.csv","question_framework_decision.csv","research_figure_visual_spec.json",
        "caption_drafts.csv","figure_traceability_index.csv","figure_source_manifest.csv","coordinate_limits.json","figure_type_plan.csv",
        "table_preference_decisions.csv","validation_record.json"]],
      "b_handoff":{"next_step":"13-B1","required_data":["final_figure_inventory.csv","figure_section_locator.csv","research_figure_visual_spec.json",
          "caption_drafts.csv","figure_traceability_index.csv","figure_source_manifest.csv","coordinate_limits.json","figure_sources/"],
        "renderer":"scripts/step13a2_figure_renderer.py",
        "rules":["render only from this 13-A2 source snapshot package","do not retrain/retune/reselect models","keep each figure in its frozen section",
                 "do not combine unrelated evidence","preserve difficult/failure cases","use Chinese figure text and 00-D-derived widths"]},
      "unresolved_issues":[],
      "freeze":{"created_utc":created,"content_sha256":None,
        "invalidation_dependencies":["13-A1 freeze identity changes","authoritative source run/result changes","00-D page width/margins change","figure list/source/renderer changes"]}}
    tmp=json.loads(json.dumps(freeze13)); freeze13["freeze"]["content_sha256"]=canon(tmp)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze13,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol/13A2/latest_freeze_package.json").write_text(json.dumps(freeze13,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol/13A2/validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"protocol/13A2/latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":shaf(p)})
    manifest={"schema_version":"13A2-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"13-A2",
      "status":"completed" if passed else "failed","created_utc":created,"binding":bind_obj,
      "code":{"repository":"Mhhhh958/mathmodeling","commit":git("rev-parse","HEAD"),"entrypoint":"scripts/step13a2_figure_blueprint_freeze.py","code_manifest":"protocol/13A2/code_manifest.json"},
      "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canon(files)}}
    (mani/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":passed,"run_id":run_id,"freeze_package_id":freeze13["freeze_package_id"],
      "freeze_content_sha256":freeze13["freeze"]["content_sha256"],"figure_count":len(invdf),"source_snapshots":len(srcmanifest),"body_width_cm":body},ensure_ascii=False))
    if not passed: raise SystemExit(2)

if __name__=="__main__":
    main()
