#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderer for STEP13-A2 frozen figure blueprints.

STEP13-A2 itself invokes this script only with --validate-only. Formal SVG/PNG
rendering is reserved for STEP13-B1 and may only use the frozen snapshots
inside the selected STEP13-A2 run.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]

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

def rdf(p: Path):
    return pd.read_csv(p, encoding="utf-8-sig")

def configure_font(spec):
    prefs=[spec["font"]["primary_cn"], *spec["font"].get("fallback",[])]
    names={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((x for x in prefs if x in names), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"]=[chosen,"DejaVu Sans"]
    else:
        matplotlib.rcParams["font.sans-serif"]=prefs+["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"]=False
    return chosen

def parse_sources(row):
    return [ROOT / s for s in str(row["source_snapshot_files"]).split(";") if s]

def set_common(ax, xlabel="", ylabel="", grid=False):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def framework(fig, ax, source, spec):
    obj=json.loads(source.read_text(encoding="utf-8"))
    nodes=obj["nodes"]; edges=obj["edges"]
    n=len(nodes); xs=np.linspace(0.08,0.92,n); y=0.5
    pos={}
    for x,node in zip(xs,nodes):
        pos[node["id"]]=(x,y)
        ax.text(x,y,node["text"],ha="center",va="center",fontsize=9,
                bbox=dict(boxstyle="square,pad=0.35",fc="white",ec="#444444",lw=0.8))
    for a,b in edges:
        xa,ya=pos[a]; xb,yb=pos[b]
        ax.annotate("",xy=(xb-0.055,yb),xytext=(xa+0.055,ya),
                    arrowprops=dict(arrowstyle="->",lw=1.0,color="#555555"))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

def q1_wave(ax, d):
    d=d.sort_values(["sample_role_cn","time_s"])
    for role,g in d.groupby("sample_role_cn",sort=False):
        ax.plot(g["time_s"],g["value"],label=role,lw=1.0)
    set_common(ax,"时间 / s","振动幅值")
    ax.legend(frameon=False)

def q1_psd(fig, axes, d, marker_path):
    markers=rdf(marker_path)
    for ax,(view,g) in zip(axes,d.groupby("view_cn",sort=False)):
        g=g.sort_values("frequency_hz")
        ax.plot(g["frequency_hz"],g["psd"],lw=1.0,label=view)
        ax.set_yscale("log")
        set_common(ax,"频率 / Hz","功率谱密度")
        ax.legend(frameon=False,loc="upper right")
        row=markers.iloc[0]
        cols={c.lower():c for c in markers.columns}
        # Prefer explicit BSF/FTF frequency columns if present.
        bsf=None; ftf=None
        for k,c in cols.items():
            if "bsf" in k and ("hz" in k or "frequency" in k) and bsf is None:
                try: bsf=float(row[c])
                except Exception: pass
            if "ftf" in k and ("hz" in k or "frequency" in k) and ftf is None:
                try: ftf=float(row[c])
                except Exception: pass
        if bsf and np.isfinite(bsf):
            for mult in [1,2,3]:
                x=mult*bsf
                if x<=500: ax.axvline(x,ls="--",lw=0.8,color="#666666")
            if ftf and np.isfinite(ftf):
                for x in [bsf-ftf,bsf+ftf]:
                    if 0<=x<=500: ax.axvline(x,ls=":",lw=0.8,color="#888888")

def confusion(ax,d):
    labels=["OR","IR","B","N"]; arr=d[[f"pred_{x}" for x in labels]].to_numpy(float)
    im=ax.imshow(arr,cmap="Blues",vmin=0,vmax=max(1,float(np.max(arr))))
    ax.set_xticks(range(4),labels); ax.set_yticks(range(4),labels)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(int(arr[i,j])),ha="center",va="center",fontsize=9)
    return im

def paired(ax,d):
    x=d["outer_fold"].to_numpy()
    for _,r in d.iterrows():
        ax.plot([r["outer_fold"]-0.12,r["outer_fold"]+0.12],[r["base_macro_f1"],r["H1_macro_f1"]],
                color="#999999",lw=0.9,zorder=1)
    ax.scatter(x-0.12,d["base_macro_f1"],label="Base逻辑回归",s=28,zorder=2)
    ax.scatter(x+0.12,d["H1_macro_f1"],label="最终随机森林",s=28,zorder=2)
    set_common(ax,"外层折","文件级 Macro-F1",grid=True)
    ax.set_ylim(0,1.04); ax.set_xticks(x); ax.legend(frameon=False)

def failure_two(axes,d):
    for ax,(gid,g) in zip(axes,d.groupby("group_id",sort=False)):
        g=g.sort_values("window_start_s")
        for lab in ["OR","IR","B","N"]:
            ax.plot(g["window_start_s"],g[f"score_{lab}"],marker="o",ms=2.8,lw=0.9,label=lab)
        name=Path(gid).name
        ax.text(0.01,0.96,name,transform=ax.transAxes,va="top",ha="left",fontsize=9)
        set_common(ax,"窗口起始时间 / s","模型分数",grid=True)
        ax.set_ylim(0,1.02)
    axes[0].legend(frameon=False,ncol=4,loc="upper right")

def shift_bar(ax,d):
    d=d.sort_values("abs_standardized_mean_shift")
    y=np.arange(len(d))
    fams=d["family_cn"].astype(str)
    colors=[("#4477AA" if "幅值" in f else "#228833") for f in fams]
    ax.barh(y,d["abs_standardized_mean_shift"],color=colors)
    ax.set_yticks(y,d["feature_cn"])
    set_common(ax,"标准化绝对均值偏移","公共特征",grid=False)

def transfer(ax,d):
    label={"A0":"无迁移A0","T1":"T1位置—尺度对齐","T2":"T2 CORAL","T3":"T3源实例加权"}
    for method,g in d.groupby("method",sort=False):
        g=g.sort_values("load")
        ax.plot(g["load"],g["macro_f1"],marker="o",ms=4,lw=1.1,label=label.get(method,method))
    set_common(ax,"模拟目标载荷 / hp","文件级 Macro-F1",grid=True)
    ax.set_xticks([0,1,2,3]); ax.set_ylim(0,1.04); ax.legend(frameon=False,ncol=2)

def robustness(ax,d):
    d=d.sort_values("target_id"); x=np.arange(len(d))
    ax.bar(x,d["label_retention_rate"])
    flip=~d["same_label"].astype(bool)
    if flip.any():
        ax.scatter(x[flip],d.loc[flip,"label_retention_rate"],marker="x",s=50,label="目标留一翻转",zorder=3)
    ax.set_xticks(x,d["target_id"])
    set_common(ax,"目标文件","正式标签保持率",grid=True)
    ax.set_ylim(0,1.04)
    if flip.any(): ax.legend(frameon=False)

def contrib(ax,d):
    # One continuous ranked plot, grouped by case; not card panels.
    d=d.sort_values(["case_cn","rank"],ascending=[True,False]).copy()
    labels=[f"{r.case_cn}｜{r.feature_cn}" for r in d.itertuples()]
    y=np.arange(len(d))
    vals=d["file_score_drop_when_occluded"].to_numpy(float)
    ax.hlines(y,0,vals,lw=0.8,color="#BBBBBB")
    ax.scatter(vals,y,s=20)
    ax.axvline(0,color="#666666",lw=0.7)
    ax.set_yticks(y,labels,fontsize=7)
    set_common(ax,"遮蔽后模型分数下降","案例与特征",grid=False)

def faithfulness(ax,d):
    d=d.sort_values("case_cn").copy()
    y=np.arange(len(d))
    ax.scatter(d["top5_drop"],y,label="Top-5",s=26)
    ax.scatter(d["low5_drop"],y,label="Low-5",s=26)
    ax.scatter(d["random_null_median"],y,label="随机5特征中位数",s=26)
    ax.axvline(0,color="#666666",lw=0.7)
    ax.set_yticks(y,d["case_cn"],fontsize=8)
    set_common(ax,"联合遮蔽后的模型分数下降","解释案例",grid=False)
    ax.legend(frameon=False)

def build(row,spec,coords):
    fid=row["figure_id"]; paths=parse_sources(row)
    width=float(row["insert_width_cm"])
    base_h=width*0.55
    if fid in {"FIG-Q1-03","FIG-Q2-04"}:
        fig,axes=plt.subplots(2,1,figsize=(width/2.54,base_h*1.15/2.54),sharex=False)
    else:
        h=base_h
        if fid in {"FIG-Q4-02","FIG-Q4-03"}: h=width*0.9
        fig,ax=plt.subplots(figsize=(width/2.54,h/2.54)); axes=[ax]
    if fid in {"FIG-Q1-01","FIG-Q2-01","FIG-Q3-01","FIG-Q4-01"}:
        framework(fig,axes[0],paths[0],spec)
    elif fid=="FIG-Q1-02": q1_wave(axes[0],rdf(paths[0]))
    elif fid=="FIG-Q1-03": q1_psd(fig,axes,rdf(paths[0]),paths[1])
    elif fid=="FIG-Q2-02": im=confusion(axes[0],rdf(paths[0])); fig.colorbar(im,ax=axes[0],label="文件数",fraction=0.045,pad=0.04)
    elif fid=="FIG-Q2-03": paired(axes[0],rdf(paths[0]))
    elif fid=="FIG-Q2-04": failure_two(axes,rdf(paths[0]))
    elif fid=="FIG-Q3-02": shift_bar(axes[0],rdf(paths[0]))
    elif fid=="FIG-Q3-03": transfer(axes[0],rdf(paths[0]))
    elif fid=="FIG-Q3-04": robustness(axes[0],rdf(paths[0]))
    elif fid in {"FIG-Q4-02","FIG-Q4-03"}: contrib(axes[0],rdf(paths[0]))
    elif fid=="FIG-Q4-04": faithfulness(axes[0],rdf(paths[0]))
    else: raise KeyError(fid)
    # No oversized in-figure title; Word caption carries title.
    fig.tight_layout()
    return fig

def validate(run_dir, inv, spec, coords):
    errs=[]
    required={"figure_id","source_snapshot_files","insert_width_cm","renderer_entry","title_cn","section"}
    if not required.issubset(inv.columns): errs.append(f"inventory columns missing: {required-set(inv.columns)}")
    for r in inv.to_dict("records"):
        ps=parse_sources(r)
        if not ps or not all(p.exists() and p.stat().st_size>0 for p in ps):
            errs.append(f"{r.get('figure_id')}: source snapshot missing/empty")
        if r.get("figure_id") not in coords: errs.append(f"{r.get('figure_id')}: coordinate limit missing")
        if float(r.get("insert_width_cm",0))<=0 or float(r.get("insert_width_cm",0))>float(spec["body_width_cm"])+1e-9:
            errs.append(f"{r.get('figure_id')}: width invalid")
    expected={
      "FIG-Q1-01","FIG-Q1-02","FIG-Q1-03","FIG-Q2-01","FIG-Q2-02","FIG-Q2-03","FIG-Q2-04",
      "FIG-Q3-01","FIG-Q3-02","FIG-Q3-03","FIG-Q3-04","FIG-Q4-01","FIG-Q4-02","FIG-Q4-03","FIG-Q4-04"}
    if set(inv["figure_id"])!=expected: errs.append("figure id set mismatch")
    if any(run_dir.rglob("*.png")) or any(run_dir.rglob("*.svg")) or any(run_dir.rglob("*.pdf")):
        errs.append("formal figure file already exists inside 13-A2 run")
    if errs: raise RuntimeError("; ".join(errs))
    return {"ok":True,"figure_count":len(inv),"body_width_cm":spec["body_width_cm"],"font_available":configure_font(spec)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--figure-id")
    ap.add_argument("--output-dir")
    ap.add_argument("--validate-only",action="store_true")
    args=ap.parse_args()
    run_dir=ROOT/"outputs/runs"/args.run_id
    art=run_dir/"artifacts"
    inv=rdf(art/"final_figure_inventory.csv")
    spec=json.loads((art/"research_figure_visual_spec.json").read_text(encoding="utf-8"))
    coords=json.loads((art/"coordinate_limits.json").read_text(encoding="utf-8"))
    result=validate(run_dir,inv,spec,coords)
    if args.validate_only:
        print(json.dumps(result,ensure_ascii=False)); return
    if not args.figure_id:
        raise SystemExit("--figure-id is required unless --validate-only")
    rows=inv[inv["figure_id"]==args.figure_id]
    if len(rows)!=1: raise SystemExit(f"unknown/duplicate figure id: {args.figure_id}")
    configure_font(spec)
    row=rows.iloc[0].to_dict()
    fig=build(row,spec,coords)
    out=Path(args.output_dir) if args.output_dir else (run_dir/"rendered")
    out.mkdir(parents=True,exist_ok=True)
    for ext in ["svg","png"]:
        p=out/f"{args.figure_id}.{ext}"
        fig.savefig(p,dpi=300 if ext=="png" else None,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(json.dumps({"ok":True,"figure_id":args.figure_id,"output_dir":str(out)},ensure_ascii=False))

if __name__=="__main__":
    main()
