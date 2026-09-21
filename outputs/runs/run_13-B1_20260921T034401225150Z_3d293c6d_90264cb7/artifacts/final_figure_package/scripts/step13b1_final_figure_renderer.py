#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP13-B1 final scientific figure renderer.
Reads only STEP13-A2 frozen blueprints/source snapshots; no model calculation.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
A2_RUN="run_13-A2_20260920T174314251414Z_90264cb7_0eedc9bc"
A2_ART=ROOT/"outputs/runs"/A2_RUN/"artifacts"

def rdf(p): return pd.read_csv(p,encoding="utf-8-sig")

def configure(spec):
    prefs=[spec["font"]["primary_cn"],*spec["font"].get("fallback",[]),"Noto Sans CJK JP"]
    names={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((x for x in prefs if x in names),None)
    if not chosen:
        raise RuntimeError("No CJK font from frozen visual specification/family fallback is available.")
    matplotlib.rcParams.update({
      "font.sans-serif":[chosen,"DejaVu Sans"],"axes.unicode_minus":False,
      "font.size":9,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
      "legend.fontsize":9,"axes.linewidth":0.8,"lines.linewidth":1.2,
      "svg.fonttype":"none","pdf.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white",
      "savefig.facecolor":"white"
    })
    return chosen

def inputs(blueprint_dir=None):
    b=Path(blueprint_dir) if blueprint_dir else A2_ART
    return rdf(b/"final_figure_inventory.csv"), json.loads((b/"research_figure_visual_spec.json").read_text(encoding="utf-8")), json.loads((b/"coordinate_limits.json").read_text(encoding="utf-8"))

def source_paths(row,source_dir=None):
    raw=[x for x in str(row["source_snapshot_files"]).split(";") if x]
    if source_dir:
        s=Path(source_dir); return [s/Path(x).name for x in raw]
    return [ROOT/x for x in raw]

def common(ax,xlabel="",ylabel="",grid=False):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis="y",linestyle="--",linewidth=0.6,color="#B8BDC5",alpha=0.55,zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8,length=3)

def framework(ax,obj,spec):
    nodes=obj["nodes"]; edges=obj["edges"]; n=len(nodes)
    xs=np.linspace(0.08,0.92,n); y=0.52; pos={}
    for i,(x,node) in enumerate(zip(xs,nodes)):
        pos[node["id"]]=(x,y)
        txt=str(node["text"])
        # deterministic CJK wrapping without adding decorative cards
        width=9 if n<=5 else 8
        lines=[txt[j:j+width] for j in range(0,len(txt),width)]
        txt="\n".join(lines)
        w=0.135 if n<=5 else 0.12; h=0.23 if len(lines)<=2 else 0.30
        rect=Rectangle((x-w/2,y-h/2),w,h,facecolor="white",edgecolor="#555555",linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x,y,txt,ha="center",va="center",fontsize=8.2,color="#222222")
    for a,b in edges:
        xa,ya=pos[a]; xb,yb=pos[b]
        ax.annotate("",xy=(xb-0.065,yb),xytext=(xa+0.065,ya),
                    arrowprops=dict(arrowstyle="->",lw=1.0,color="#555555",shrinkA=0,shrinkB=0))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

def q1_wave(ax,d,spec):
    colors={"典型样本":"#4477AA","困难样本":"#CC6677"}
    for role,g in d.sort_values(["sample_role_cn","time_s"]).groupby("sample_role_cn",sort=False):
        ax.plot(g["time_s"],g["value"],label=role,color=colors.get(role,"#555555"),lw=1.0)
    common(ax,"时间 / s","振动幅值"); ax.legend(frameon=False,loc="upper right")

def q1_psd(axes,d,markers,spec):
    views=["主信号功率谱","Hilbert包络谱"]; colors={"主信号功率谱":"#4477AA","Hilbert包络谱":"#228833"}
    row=markers.iloc[0]; cols={c.lower():c for c in markers.columns}; bsf=ftf=None
    for k,c in cols.items():
        if bsf is None and "bsf" in k and ("hz" in k or "frequency" in k):
            try: bsf=float(row[c])
            except: pass
        if ftf is None and "ftf" in k and ("hz" in k or "frequency" in k):
            try: ftf=float(row[c])
            except: pass
    for idx,(ax,view) in enumerate(zip(axes,views)):
        g=d[d["view_cn"]==view].sort_values("frequency_hz")
        ax.plot(g["frequency_hz"],g["psd"],color=colors[view],lw=1.0,label=view)
        ax.set_yscale("log"); common(ax,"频率 / Hz","功率谱密度")
        ax.text(0.01,0.96,f"（{'a' if idx==0 else 'b'}）{view}",transform=ax.transAxes,ha="left",va="top",fontsize=9)
        if bsf and np.isfinite(bsf):
            for mult in [1,2,3]:
                x=mult*bsf
                if 0<=x<=500: ax.axvline(x,ls="--",lw=0.8,color="#666666",alpha=0.85)
            if ftf and np.isfinite(ftf):
                for x in [bsf-ftf,bsf+ftf]:
                    if 0<=x<=500: ax.axvline(x,ls=":",lw=0.8,color="#888888",alpha=0.85)
        ax.set_xlim(0,500)
    handles=[Line2D([0],[0],color="#666666",ls="--",lw=0.8,label="BSF及谐波"),
             Line2D([0],[0],color="#888888",ls=":",lw=0.8,label="BSF邻近侧带")]
    axes[0].legend(handles=[Line2D([0],[0],color=colors["主信号功率谱"],lw=1.0,label="主信号功率谱"),*handles],
                   frameon=False,loc="upper right")
    axes[1].legend(handles=[Line2D([0],[0],color=colors["Hilbert包络谱"],lw=1.0,label="Hilbert包络谱"),*handles],
                   frameon=False,loc="upper right")

def q2_confusion(ax,d):
    labels=["OR","IR","B","N"]; arr=d[[f"pred_{x}" for x in labels]].to_numpy(float)
    vmax=max(1,float(np.nanmax(arr))); cmap=plt.get_cmap("Blues")
    # Fully editable vector heatmap: every cell is an SVG/PDF rectangle.
    # We deliberately avoid imshow/pcolormesh+colorbar because some backends
    # serialize the continuous colorbar as a raster <image> layer.
    for i in range(4):
        for j in range(4):
            v=float(arr[i,j]); color=cmap(v/vmax)
            ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,facecolor=color,edgecolor="white",linewidth=0.8))
    ax.set_xlim(-0.5,3.5); ax.set_ylim(3.5,-0.5); ax.set_aspect("equal")
    ax.set_xticks(range(4),labels); ax.set_yticks(range(4),labels)
    common(ax,"预测类别","真实类别")
    threshold=np.nanmax(arr)*0.55
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(int(arr[i,j])),ha="center",va="center",fontsize=9,
                    color="white" if arr[i,j]>=threshold and arr[i,j]>0 else "#222222")
    ax.text(1.03,0.5,"方格数字为文件数",transform=ax.transAxes,rotation=90,
            ha="left",va="center",fontsize=8,color="#444444")
    return None

def q2_paired(ax,d,spec):
    c=spec["colors"]["model"]; x=d["outer_fold"].to_numpy()
    for _,r in d.iterrows():
        ax.plot([r["outer_fold"]-0.12,r["outer_fold"]+0.12],[r["base_macro_f1"],r["H1_macro_f1"]],
                color="#A0A0A0",lw=0.9,zorder=1)
    ax.scatter(x-0.12,d["base_macro_f1"],label="Base逻辑回归",s=30,color=c["Base/A0"],zorder=2)
    ax.scatter(x+0.12,d["H1_macro_f1"],label="最终随机森林",s=30,color=c["H1_RF"],zorder=2)
    common(ax,"外层折","文件级 Macro-F1",True); ax.set_ylim(0,1.04); ax.set_xticks(x)
    ax.legend(frameon=False,loc="lower right")

def q2_failure(axes,d,spec):
    cls=spec["colors"]["class"]
    groups=sorted(d["group_id"].unique(),key=lambda x:("IR007_0" not in x,x))
    for idx,(ax,gid) in enumerate(zip(axes,groups)):
        g=d[d["group_id"]==gid].sort_values("window_start_s")
        for lab in ["OR","IR","B","N"]:
            ax.plot(g["window_start_s"],g[f"score_{lab}"],marker="o",ms=2.8,lw=0.9,color=cls[lab],label=lab)
        ax.text(0.01,0.96,f"（{'a' if idx==0 else 'b'}）{Path(gid).name}",transform=ax.transAxes,va="top",ha="left",fontsize=9)
        common(ax,"窗口起始时间 / s","模型分数",True); ax.set_ylim(0,1.02)
    axes[0].legend(frameon=False,ncol=4,loc="upper right")

def q3_shift(ax,d,spec):
    d=d.sort_values("abs_standardized_mean_shift"); y=np.arange(len(d))
    cmap={"幅值/尺度":"#4477AA","几何无关机理代理":"#228833","形态统计":"#CC6677"}
    colors=[cmap.get(x,"#777777") for x in d["family_cn"].astype(str)]
    ax.barh(y,d["abs_standardized_mean_shift"],color=colors,height=0.68)
    ax.set_yticks(y,d["feature_cn"]); common(ax,"标准化绝对均值偏移","公共特征",True)
    present=[]
    for fam in d["family_cn"].astype(str):
        if fam not in present: present.append(fam)
    ax.legend(handles=[Patch(facecolor=cmap.get(x,"#777777"),edgecolor="none",label=x) for x in present],
              frameon=False,loc="lower right")

def q3_transfer(ax,d,spec):
    c=spec["colors"]["model"]
    cmap={"A0":c["Base/A0"],"T1":c["T1"],"T2":c["T2"],"T3":c["T3"]}
    lab={"A0":"无迁移A0","T1":"T1位置—尺度对齐","T2":"T2 CORAL","T3":"T3源实例加权"}
    order=["A0","T1","T2","T3"]
    for method in order:
        g=d[d["method"]==method].sort_values("load")
        if len(g): ax.plot(g["load"],g["macro_f1"],marker="o",ms=4,lw=1.1,color=cmap[method],label=lab[method])
    common(ax,"模拟目标载荷 / hp","文件级 Macro-F1",True); ax.set_xticks([0,1,2,3]); ax.set_ylim(0,1.04)
    ax.legend(frameon=False,ncol=2,loc="lower left")

def q3_robust(ax,d,spec):
    d=d.sort_values("target_id"); x=np.arange(len(d)); cls=spec["colors"]["class"]
    colors=[cls.get(str(v),"#888888") for v in d["official_label"]]
    ax.bar(x,d["label_retention_rate"],color=colors,width=0.72)
    flip=~d["same_label"].astype(bool)
    if flip.any(): ax.scatter(x[flip],d.loc[flip,"label_retention_rate"],marker="x",s=55,color="#111111",linewidths=1.2,zorder=3)
    ax.set_xticks(x,d["target_id"]); common(ax,"目标文件","正式标签保持率",True); ax.set_ylim(0,1.04)
    labs=[]
    for k in ["OR","IR","B","N"]:
        if k in set(d["official_label"]): labs.append(Patch(facecolor=cls[k],label=f"正式预测{k}"))
    if flip.any(): labs.append(Line2D([0],[0],marker="x",linestyle="None",color="#111111",label="目标留一翻转"))
    ax.legend(handles=labs,frameon=False,ncol=min(3,len(labs)),loc="lower left")

def q4_contrib(ax,d,spec):
    d=d.sort_values(["case_cn","rank"],ascending=[True,False]).copy()
    y=np.arange(len(d)); vals=d["file_score_drop_when_occluded"].to_numpy(float)
    cls=spec["colors"]["class"]; colors=[cls.get(str(v),"#777777") for v in d["explained_label"]]
    ax.hlines(y,0,vals,lw=0.8,color="#B8BDC5",zorder=1); ax.scatter(vals,y,s=24,color=colors,zorder=2)
    ax.axvline(0,color="#666666",lw=0.7); labels=[f"{r.case_cn}｜{r.feature_cn}" for r in d.itertuples()]
    ax.set_yticks(y,labels,fontsize=7); common(ax,"遮蔽后模型分数下降","案例与特征",False)
    present=[]
    for k in d["explained_label"].astype(str):
        if k not in present: present.append(k)
    ax.legend(handles=[Line2D([0],[0],marker="o",linestyle="None",color=cls.get(k,"#777777"),label=f"解释类别{k}") for k in present],
              frameon=False,loc="lower right")

def q4_faith(ax,d,spec):
    d=d.sort_values("case_cn"); y=np.arange(len(d))
    ax.scatter(d["top5_drop"],y,label="Top-5",s=28,color="#4477AA")
    ax.scatter(d["low5_drop"],y,label="Low-5",s=28,color="#6B7280")
    ax.scatter(d["random_null_median"],y,label="随机5特征中位数",s=28,color="#228833")
    ax.axvline(0,color="#666666",lw=0.7); ax.set_yticks(y,d["case_cn"],fontsize=8)
    common(ax,"联合遮蔽后的模型分数下降","解释案例",False); ax.legend(frameon=False,loc="lower right")

def build(row,spec,coords,source_dir=None):
    fid=row["figure_id"]; ps=source_paths(row,source_dir); width=float(row["insert_width_cm"])
    if fid in {"FIG-Q1-01","FIG-Q2-01","FIG-Q3-01","FIG-Q4-01"}:
        h=5.1
    elif fid=="FIG-Q2-02":
        # Square confusion matrix must remain readable at the frozen 13.59915 cm insertion width.
        h=12.8
    elif fid in {"FIG-Q1-03","FIG-Q2-04"}:
        h=10.0
    elif fid in {"FIG-Q4-02","FIG-Q4-03"}:
        h=14.2
    else:
        h=max(6.3,width*0.48)
    if fid in {"FIG-Q1-03","FIG-Q2-04"}:
        fig,axes=plt.subplots(2,1,figsize=(width/2.54,h/2.54))
    else:
        fig,ax=plt.subplots(figsize=(width/2.54,h/2.54)); axes=[ax]
    if fid in {"FIG-Q1-01","FIG-Q2-01","FIG-Q3-01","FIG-Q4-01"}:
        framework(axes[0],json.loads(ps[0].read_text(encoding="utf-8")),spec)
    elif fid=="FIG-Q1-02": q1_wave(axes[0],rdf(ps[0]),spec)
    elif fid=="FIG-Q1-03": q1_psd(axes,rdf(ps[0]),rdf(ps[1]),spec)
    elif fid=="FIG-Q2-02":
        q2_confusion(axes[0],rdf(ps[0]))
    elif fid=="FIG-Q2-03": q2_paired(axes[0],rdf(ps[0]),spec)
    elif fid=="FIG-Q2-04": q2_failure(axes,rdf(ps[0]),spec)
    elif fid=="FIG-Q3-02": q3_shift(axes[0],rdf(ps[0]),spec)
    elif fid=="FIG-Q3-03": q3_transfer(axes[0],rdf(ps[0]),spec)
    elif fid=="FIG-Q3-04": q3_robust(axes[0],rdf(ps[0]),spec)
    elif fid in {"FIG-Q4-02","FIG-Q4-03"}: q4_contrib(axes[0],rdf(ps[0]),spec)
    elif fid=="FIG-Q4-04": q4_faith(axes[0],rdf(ps[0]),spec)
    else: raise KeyError(fid)
    # no figure/axes title: captions stay in Word in 13-B2
    if fid in {"FIG-Q4-02","FIG-Q4-03"}:
        fig.subplots_adjust(left=0.42,right=0.98,bottom=0.08,top=0.99)
    else:
        fig.tight_layout(pad=0.8)
    return fig

def visible_texts(fig):
    fig.canvas.draw(); out=[]
    for ax in fig.axes:
        for t in ax.texts+[ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
            if t.get_visible() and str(t.get_text()).strip(): out.append((str(t.get_text()),float(t.get_fontsize())))
        leg=ax.get_legend()
        if leg:
            for t in leg.get_texts():
                if t.get_visible() and str(t.get_text()).strip(): out.append((str(t.get_text()),float(t.get_fontsize())))
    return out

def save(fig,outbase,dpi=450,pad=0.03):
    meta_common={"Creator":"STEP13-B1","Title":outbase.name}
    fig.savefig(outbase.with_suffix(".svg"),format="svg",bbox_inches="tight",pad_inches=pad,metadata={"Creator":"STEP13-B1","Date":None})
    fig.savefig(outbase.with_suffix(".pdf"),format="pdf",bbox_inches="tight",pad_inches=pad,metadata={"Creator":"STEP13-B1","CreationDate":None,"ModDate":None})
    fig.savefig(outbase.with_suffix(".png"),format="png",dpi=dpi,bbox_inches="tight",pad_inches=pad,metadata={"Software":"STEP13-B1"})
    return [outbase.with_suffix(x) for x in [".svg",".pdf",".png"]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--figure-id",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--blueprint-dir"); ap.add_argument("--source-dir"); ap.add_argument("--dpi",type=int,default=450)
    args=ap.parse_args()
    inv,spec,coords=inputs(args.blueprint_dir); chosen=configure(spec)
    rows=inv[inv["figure_id"]==args.figure_id]
    if len(rows)!=1: raise SystemExit(f"unknown/duplicate figure: {args.figure_id}")
    row=rows.iloc[0].to_dict(); fig=build(row,spec,coords,args.source_dir)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); base=out/args.figure_id
    texts=visible_texts(fig); files=save(fig,base,args.dpi); plt.close(fig)
    print(json.dumps({"ok":True,"figure_id":args.figure_id,"font":chosen,"files":[str(p) for p in files],"visible_texts":texts},ensure_ascii=False))

if __name__=="__main__": main()
