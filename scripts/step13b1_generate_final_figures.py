#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP13-B1: render and validate final scientific figures from STEP13-A2 frozen snapshots."""
from __future__ import annotations
import csv, hashlib, json, math, os, re, shutil, subprocess, sys, warnings, zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import font_manager, colors as mcolors
from matplotlib.ft2font import FT2Font
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import step13b1_final_figure_renderer as R

CFG=ROOT/"protocol/13B1/run_config.json"
STATE=ROOT/"protocol/00C/process_state_card.json"
A1=ROOT/"protocol/13A1/latest_freeze_package.json"
A2=ROOT/"protocol/13A2/latest_freeze_package.json"
A2_RUN="run_13-A2_20260920T174314251414Z_90264cb7_0eedc9bc"
A2_ART=ROOT/"outputs/runs"/A2_RUN/"artifacts"
INV=A2_ART/"final_figure_inventory.csv"
SPEC=A2_ART/"research_figure_visual_spec.json"
COORD=A2_ART/"coordinate_limits.json"
SRC_MAN=A2_ART/"figure_source_manifest.csv"
TRACE=A2_ART/"figure_traceability_index.csv"
LOC=A2_ART/"figure_section_locator.csv"
CAP=A2_ART/"caption_drafts.csv"
TABLE_PREF=A2_ART/"table_preference_decisions.csv"
BLUEPRINT_FILES=[INV,SPEC,COORD,SRC_MAN,TRACE,LOC,CAP,TABLE_PREF,A2_ART/"question_framework_decision.csv",A2_ART/"figure_type_plan.csv"]

def j(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rdf(p): return pd.read_csv(p,encoding="utf-8-sig")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canon(o): return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def require(c,m):
    if not c: raise RuntimeError(m)
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def write_df(df,p):
    Path(p).parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False,encoding="utf-8-sig")
def hexnorm(c):
    try: return mcolors.to_hex(c,keep_alpha=False).lower()
    except: return str(c).lower()

def font_glyph_check(font_name):
    path=font_manager.findfont(font_name,fallback_to_default=False)
    ft=FT2Font(path); chars="问题数据模型故障预测频率类别稳定性"
    missing=[ch for ch in chars if ft.get_char_index(ord(ch))==0]
    return path,missing

def visible_text_policy(texts,forbidden):
    bad=[]
    for txt,_ in texts:
        lo=txt.lower()
        for w in forbidden:
            if re.search(rf"(?<![a-z]){re.escape(w.lower())}(?![a-z])",lo):
                bad.append((txt,w))
    return bad

def color_contract(fig,fid,spec):
    cls={k:v.lower() for k,v in spec["colors"]["class"].items()}
    mod={k:v.lower() for k,v in spec["colors"]["model"].items()}
    errs=[]
    # Line labels that directly encode class.
    for ax in fig.axes:
        for ln in ax.get_lines():
            lab=str(ln.get_label())
            if lab in cls and hexnorm(ln.get_color())!=cls[lab]:
                errs.append(f"{fid}: class {lab} line color {hexnorm(ln.get_color())}!={cls[lab]}")
            expect={"无迁移A0":mod["Base/A0"],"T1位置—尺度对齐":mod["T1"],"T2 CORAL":mod["T2"],"T3源实例加权":mod["T3"]}
            if lab in expect and hexnorm(ln.get_color())!=expect[lab]:
                errs.append(f"{fid}: model {lab} color mismatch")
        for coll in ax.collections:
            lab=str(coll.get_label())
            if lab=="Base逻辑回归":
                fc=hexnorm(coll.get_facecolor()[0]); 
                if fc!=mod["Base/A0"]: errs.append(f"{fid}: Base color mismatch")
            if lab=="最终随机森林":
                fc=hexnorm(coll.get_facecolor()[0])
                if fc!=mod["H1_RF"]: errs.append(f"{fid}: H1_RF color mismatch")
    return errs

def deterministic_zip(src_dir,zip_path):
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(x for x in Path(src_dir).rglob("*") if x.is_file()):
            info=zipfile.ZipInfo(p.relative_to(src_dir).as_posix(),date_time=(2026,9,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o644 & 0xFFFF)<<16
            z.writestr(info,p.read_bytes())

def render_one(row,spec,coords,outdir,source_dir=None):
    fid=row["figure_id"]; chosen=R.configure(spec)
    with warnings.catch_warnings(record=True) as wr:
        warnings.simplefilter("always")
        fig=R.build(row,spec,coords,source_dir)
        fig.canvas.draw()
        texts=R.visible_texts(fig)
        renderer=fig.canvas.get_renderer()
        bbox=fig.get_tightbbox(renderer)
        export_width_in=float(bbox.width)+0.06
        target_width_in=float(row["insert_width_cm"])/2.54
        scale=target_width_in/export_width_in if export_width_in>0 else 0
        min_nominal=min((fs for _,fs in texts),default=9.0)
        effective_font=min_nominal*scale
        color_err=color_contract(fig,fid,spec)
        title_ok=(fig._suptitle is None or not str(fig._suptitle.get_text()).strip()) and all(not ax.get_title().strip() for ax in fig.axes)
        base=Path(outdir)/fid
        files=R.save(fig,base,450)
        glyph_warn=[str(x.message) for x in wr if "Glyph" in str(x.message) and "missing" in str(x.message)]
        plt.close(fig)
    png=base.with_suffix(".png")
    with Image.open(png) as im:
        wpx,hpx=im.size
    eff_dpi=wpx/target_width_in
    svg=base.with_suffix(".svg").read_text(encoding="utf-8")
    vector_editable=("<text" in svg and "<image" not in svg)
    return {
      "font":chosen,"texts":texts,"effective_font_pt":effective_font,"export_width_in":export_width_in,
      "scale_to_insert":scale,"png_width_px":wpx,"png_height_px":hpx,"effective_png_dpi":eff_dpi,
      "glyph_warnings":glyph_warn,"color_errors":color_err,"no_internal_title":title_ok,
      "svg_editable":vector_editable,"files":files
    }

def main():
    cfg,state,a1,a2=map(j,[CFG,STATE,A1,A2])
    require(state["CURRENT_ALLOWED_STEP"]=="13-B1" and state["NEXT_ALLOWED"]=="13-B1","gate not 13-B1")
    require(state["LAST_PASS_TOKEN"]=="13-A2-V2026.09.21-002855b0-通过","invalid 13-A2 pass token")
    require(state.get("OPEN_P0")==[],"OPEN_P0 not empty")
    require(a1["status"]=="passed" and a2["status"]=="passed","A1/A2 not passed")
    require(a2["freeze_package_id"]=="FREEZE-13A2-90264cb7" and a2["freeze"]["content_sha256"]==cfg["source_13A2_content_sha256"],"A2 freeze mismatch")
    require(state["LATEST_WORD"]["artifact_id"]==cfg["latest_word_artifact_id"],"Word identity drifted before B1")
    for p in BLUEPRINT_FILES: require(p.exists(),f"missing frozen blueprint {p}")
    inv=R.rdf(INV); spec=j(SPEC); coords=j(COORD); sm=rdf(SRC_MAN)
    require(len(inv)==15 and inv["required"].astype(str).str.lower().eq("true").all(),"final necessary figure set != 15")
    require(int(a2["parameters"]["figure_count"])==15,"A2 figure count mismatch")
    # Verify every frozen source byte against 13-A2 manifest.
    source_checks=[]
    for r in sm.itertuples(index=False):
        p=ROOT/r.snapshot_path
        ok=p.exists() and sha(p)==r.sha256 and p.stat().st_size==int(r.size_bytes)
        source_checks.append(ok)
        require(ok,f"frozen source snapshot drift: {r.snapshot_path}")
    bind={"step":"13-B1","head":git("rev-parse","HEAD"),"cfg_sha256":sha(CFG),"a2_freeze_sha256":a2["freeze"]["content_sha256"],"word":state["LATEST_WORD"]["artifact_id"]}
    digest=canon(bind); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id=f"run_13-B1_{stamp}_{digest[:8]}_{a2['freeze_package_id'][-8:]}"
    out=ROOT/"outputs/runs"/run_id; art=out/"artifacts"; pkg=art/"final_figure_package"
    figdir=pkg/"figures"; srcdir=pkg/"source_data"; scriptdir=pkg/"scripts"; bpdir=pkg/"blueprint"; repdir=art/"repro_check"
    for d in [figdir,srcdir,scriptdir,bpdir,repdir,out/"manifests",out/"logs"]: d.mkdir(parents=True,exist_ok=True)

    # Portable frozen package inputs.
    for p in BLUEPRINT_FILES: shutil.copy2(p,bpdir/p.name)
    for p in sorted({ROOT/x for s in inv["source_snapshot_files"] for x in str(s).split(";") if x}):
        shutil.copy2(p,srcdir/p.name)
    shutil.copy2(ROOT/"scripts/step13b1_final_figure_renderer.py",scriptdir/"step13b1_final_figure_renderer.py")
    shutil.copy2(ROOT/"scripts/step13a2_figure_renderer.py",scriptdir/"step13a2_figure_renderer.py")

    chosen=R.configure(spec); font_path,missing=font_glyph_check(chosen)
    require(not missing,f"CJK font glyph missing: {missing}")
    per=[]; failed=[]
    forbidden=cfg["forbidden_visible_english_words"]
    for row in inv.to_dict("records"):
        fid=row["figure_id"]; reasons=[]
        try:
            rr=render_one(row,spec,coords,figdir)
            bad_english=visible_text_policy(rr["texts"],forbidden)
            if rr["glyph_warnings"]: reasons.append("missing_glyph_warning")
            if bad_english: reasons.append("visible_english_not_whitelisted:"+str(bad_english[:3]))
            if rr["color_errors"]: reasons.extend(rr["color_errors"])
            if not rr["no_internal_title"]: reasons.append("internal_big_title_present")
            if not rr["svg_editable"]: reasons.append("svg_not_text_editable_or_embeds_raster")
            if rr["effective_font_pt"]<cfg["readability"]["min_effective_font_pt"]: reasons.append(f"effective_font={rr['effective_font_pt']:.2f}pt")
            if not(cfg["readability"]["min_effective_png_dpi"]<=rr["effective_png_dpi"]<=cfg["readability"]["max_effective_png_dpi"]):
                reasons.append(f"effective_png_dpi={rr['effective_png_dpi']:.1f}")
            expected=[figdir/f"{fid}.png",figdir/f"{fid}.svg",figdir/f"{fid}.pdf"]
            if not all(p.exists() and p.stat().st_size>0 for p in expected): reasons.append("missing_output_format")
            # section/width/source contract inherited from A2
            if not str(row["section"]).strip(): reasons.append("missing_section")
            srcs=[ROOT/x for x in str(row["source_snapshot_files"]).split(";") if x]
            if not srcs or not all(p.exists() for p in srcs): reasons.append("missing_source")
            status="passed" if not reasons else "failed"
            rec={
              "figure_id":fid,"figure_number_placeholder":"图"+fid.replace("FIG-",""),"question":row["question"],"section":row["section"],
              "title_cn":row["title_cn"],"kind":row["kind"],"layout":row["layout"],"insert_width_cm":row["insert_width_cm"],
              "png_dpi_nominal":cfg["render"]["png_dpi"],"effective_png_dpi":round(rr["effective_png_dpi"],2),
              "effective_min_font_pt":round(rr["effective_font_pt"],2),"font":rr["font"],
              "png_path":rel(figdir/f"{fid}.png"),"svg_path":rel(figdir/f"{fid}.svg"),"pdf_path":rel(figdir/f"{fid}.pdf"),
              "source_snapshot_files":row["source_snapshot_files"],"renderer_script":"scripts/step13b1_final_figure_renderer.py",
              "chinese_plan_pass":"true" if not bad_english and not rr["glyph_warnings"] else "false",
              "visual_encoding_pass":"true" if not rr["color_errors"] else "false",
              "readability_pass":"true" if rr["effective_font_pt"]>=cfg["readability"]["min_effective_font_pt"] and cfg["readability"]["min_effective_png_dpi"]<=rr["effective_png_dpi"]<=cfg["readability"]["max_effective_png_dpi"] else "false",
              "editable_vector_pass":"true" if rr["svg_editable"] else "false","status":status,"failure_reason":";".join(reasons)
            }
        except Exception as e:
            status="failed"; reasons=[repr(e)]
            rec={"figure_id":fid,"figure_number_placeholder":"图"+fid.replace("FIG-",""),"question":row["question"],"section":row["section"],"title_cn":row["title_cn"],
                 "kind":row["kind"],"layout":row["layout"],"insert_width_cm":row["insert_width_cm"],"png_dpi_nominal":cfg["render"]["png_dpi"],
                 "effective_png_dpi":"","effective_min_font_pt":"","font":chosen,"png_path":"","svg_path":"","pdf_path":"",
                 "source_snapshot_files":row["source_snapshot_files"],"renderer_script":"scripts/step13b1_final_figure_renderer.py",
                 "chinese_plan_pass":"false","visual_encoding_pass":"false","readability_pass":"false","editable_vector_pass":"false","status":"failed","failure_reason":";".join(reasons)}
        per.append(rec)
        if status!="passed": failed.append(rec)
    manifest=pd.DataFrame(per)
    write_df(manifest,art/"figure_manifest.csv"); write_df(manifest,pkg/"figure_manifest.csv")
    failcols=["figure_id","section","title_cn","failure_reason"]
    write_df(pd.DataFrame(failed,columns=failcols),art/"unpassed_figures.csv"); write_df(pd.DataFrame(failed,columns=failcols),pkg/"unpassed_figures.csv")

    # Source index with immutable hashes and portable copies.
    srcidx=[]
    for r in sm.itertuples(index=False):
        cp=srcdir/Path(r.snapshot_path).name
        srcidx.append({"figure_id":r.figure_id,"frozen_source_path":r.snapshot_path,"package_source_path":cp.relative_to(pkg).as_posix(),
                       "sha256":r.sha256,"size_bytes":r.size_bytes,"source_run_ids":r.source_run_ids,"original_sources":r.original_sources})
    write_df(pd.DataFrame(srcidx),art/"source_data_index.csv"); write_df(pd.DataFrame(srcidx),pkg/"source_data_index.csv")

    # Key-figure clean reruns: exact PNG byte equality.
    repro=[]
    for fid in cfg["reproducibility_spot_check"]:
        row=inv[inv["figure_id"]==fid].iloc[0].to_dict()
        rr=render_one(row,spec,coords,repdir)
        a=sha(figdir/f"{fid}.png"); b=sha(repdir/f"{fid}.png")
        repro.append({"figure_id":fid,"primary_png_sha256":a,"rerun_png_sha256":b,"exact_match":a==b})
    write_df(pd.DataFrame(repro),art/"reproducibility_spot_check.csv")
    repro_ok=all(x["exact_match"] for x in repro)

    # Portable reproduce helper.
    fids=" ".join(inv["figure_id"].tolist())
    (pkg/"reproduce_all.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
mkdir -p reproduced
for f in """+fids+"""; do
  python scripts/step13b1_final_figure_renderer.py --figure-id "$f" --output-dir reproduced --blueprint-dir blueprint --source-dir source_data --dpi 450
done
""",encoding="utf-8")
    (pkg/"README.txt").write_text(
      "STEP13-B1 最终科研图文件包\n"
      "来源：FREEZE-13A1-0eedc9bc + FREEZE-13A2-90264cb7。\n"
      "本包包含15张最终图的PNG(450dpi)、SVG、PDF，13-A2冻结源数据快照、蓝图、绘图脚本和追溯索引。\n"
      "未重新训练、调参或计算模型结果；未编辑Word。\n"
      "在仓库00-B Python环境并安装Noto CJK字体后，可执行 reproduce_all.sh 复现。\n",
      encoding="utf-8")

    # Copy manifests/validation assets after first creation.
    all_pass=bool(len(failed)==0 and repro_ok and len(manifest)==15 and bool(manifest["status"].eq("passed").all()))
    checks={
      "gate_13B1_valid":True,"A1_A2_frozen_sources_only":True,"source_snapshot_sha256_all_match":bool(all(source_checks)),
      "figure_count_15":bool(len(manifest)==15),"all_three_formats_present":bool(all(bool(x) for x in manifest["png_path"]) and all(bool(x) for x in manifest["svg_path"]) and all(bool(x) for x in manifest["pdf_path"])),
      "all_figures_chinese_and_no_missing_glyphs":bool(manifest["chinese_plan_pass"].eq("true").all()),
      "all_figures_visual_encoding_consistent":bool(manifest["visual_encoding_pass"].eq("true").all()),
      "all_figures_final_size_readable":bool(manifest["readability_pass"].eq("true").all()),
      "all_svg_vector_text_editable":bool(manifest["editable_vector_pass"].eq("true").all()),
      "key_figure_rerun_exact_png_reproducible":repro_ok,"difficulty_failure_figures_retained":set(["FIG-Q2-04","FIG-Q3-04"]).issubset(set(manifest["figure_id"])),
      "table_preferred_items_not_rendered":len(inv)==15,"word_edit_performed":False,"training_performed":False,"tuning_performed":False,"model_selection_performed":False
    }
    passed=bool(all(v is True for v in checks.values()) and not failed)
    validation={"schema_version":"13B1-validation-1.0","step_id":"13-B1","run_id":run_id,"status":"passed" if passed else "failed","passed":passed,
      "source_freezes":["FREEZE-13A1-0eedc9bc","FREEZE-13A2-90264cb7"],"font":{"selected":chosen,"path":font_path,"glyph_probe_missing":missing},
      "checks":checks,"figure_count":len(manifest),"unpassed_figure_count":len(failed),"reproducibility_spot_check_count":len(repro),
      "word_artifact_before":state["LATEST_WORD"]["artifact_id"],"word_artifact_after":state["LATEST_WORD"]["artifact_id"],
      "word_edit_performed":False,"training_performed":False,"tuning_performed":False,"model_selection_performed":False}
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copy2(art/"validation_record.json",pkg/"validation_record.json")
    shutil.copy2(art/"reproducibility_spot_check.csv",pkg/"reproducibility_spot_check.csv")
    shutil.copy2(TABLE_PREF,pkg/"table_preference_decisions.csv")

    # Now the package is final; create deterministic zip.
    zip_path=art/"final_figure_package.zip"; deterministic_zip(pkg,zip_path); zip_sha=sha(zip_path)
    pass_token=f"13-B1-V2026.09.21-{zip_sha[:8]}-通过" if passed else None

    delivery={"schema_version":"00C-B-delivery-1.0","package_type":"B_delivery","step_id":"13-B1","run_id":run_id,
      "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"status":"passed" if passed else "failed",
      "source_A_freeze_package_id":"FREEZE-13A2-90264cb7","source_A_freeze_package_ids":["FREEZE-13A1-0eedc9bc","FREEZE-13A2-90264cb7"],
      "latest_word_before":state["LATEST_WORD"],"new_word_after":state["LATEST_WORD"],"mother_draft_incremental_edit_only":True,
      "training_performed":False,"tuning_performed":False,"model_selection_performed":False,
      "figures_and_captions":[{"figure_id":r.figure_id,"placeholder":r.figure_number_placeholder,"title":r.title_cn,"section":r.section} for r in manifest.itertuples()],
      "figure_manifest":f"outputs/runs/{run_id}/artifacts/figure_manifest.csv","source_data_index":f"outputs/runs/{run_id}/artifacts/source_data_index.csv",
      "final_figure_package":f"outputs/runs/{run_id}/artifacts/final_figure_package.zip","final_figure_package_sha256":zip_sha,
      "unpassed_figures":f"outputs/runs/{run_id}/artifacts/unpassed_figures.csv","validation_record":"protocol/13B1/validation_record.json",
      "validation":{"only_frozen_A_results_used":True,"figures_traceable":True,"final_size_readability_checked":True,
        "chinese_and_visual_spec_checked":True,"reproducibility_spot_check_passed":repro_ok,"word_unchanged":True},
      "pass_token":pass_token,"next_allowed":"13-B2" if passed else "13-B1"}
    pdir=ROOT/"protocol/13B1"; pdir.mkdir(parents=True,exist_ok=True)
    (pdir/"B_delivery.json").write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    tracked=["scripts/step13b1_generate_final_figures.py","scripts/step13b1_final_figure_renderer.py",".github/workflows/step13b1_final_figures.yml"]
    cm={"schema_version":"13B1-1.0","manifest_type":"code_manifest","repository":"Mhhhh958/mathmodeling","execution_commit":git("rev-parse","HEAD"),
        "tracked_code":[{"path":p,"git_blob_sha":git("rev-parse",f"HEAD:{p}")} for p in tracked],
        "frozen_renderer_reference":{"path":"scripts/step13a2_figure_renderer.py","git_blob_sha":git("rev-parse","HEAD:scripts/step13a2_figure_renderer.py")},
        "source_13A2_freeze_sha256":a2["freeze"]["content_sha256"],"constraints":cfg["constraints"]}
    (pdir/"code_manifest.json").write_text(json.dumps(cm,ensure_ascii=False,indent=2),encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha(p)})
    rm={"schema_version":"13B1-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":"13-B1",
        "status":"completed" if passed else "failed","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "binding":bind,"outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canon(files)},
        "package":{"path":rel(zip_path),"sha256":zip_sha},"pass_token":pass_token}
    (out/"manifests/run_manifest.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")

    print(json.dumps({"ok":passed,"run_id":run_id,"pass_token":pass_token,"next_allowed":"13-B2" if passed else "13-B1",
                      "figure_count":len(manifest),"unpassed":len(failed),"package":rel(zip_path),"package_sha256":zip_sha,
                      "font":chosen,"repro_checks":f"{sum(x['exact_match'] for x in repro)}/{len(repro)}"},ensure_ascii=False))
    if not passed: raise SystemExit(2)

if __name__=="__main__":
    main()
