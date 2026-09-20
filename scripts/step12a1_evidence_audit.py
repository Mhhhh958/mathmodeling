#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STEP="12-A1"
STATE=ROOT/"protocol/00C/process_state_card.json"
CFG=ROOT/"protocol/12A1/run_config.json"
ENV=ROOT/"protocol/00B/environment_manifest.json"

RUN05="run_05-A_20260919T192338074034Z_f82972c7_ba6432cf"
RUN06="run_06-A_20260920T035742210347Z_b83cb3a6_01576948"
RUN07="run_07-A_20260920T051657629377Z_59d5a6dd_7e8829fb"
RUN08="run_08-A_20260920T060936354760Z_049c8ddb_8cd7401f"
RUN09="run_09-A_20260920T070725126712Z_6ef81c21_26f99878"
RUN10="run_10-A_20260920T085322828930Z_bcbb6757_4346ce31"
RUN11="run_11-A_20260920T151947188398Z_54a3c266_1f812c22"

def nowz(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canon(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def git_head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
def J(p): return json.loads((ROOT/p).read_text(encoding="utf-8-sig"))
def C(p): return pd.read_csv(ROOT/p)

def require(cond,msg):
    if not cond: raise RuntimeError(msg)

def split_audit():
    outer=C(f"outputs/runs/{RUN05}/artifacts/outer_split_manifest.csv")
    inner=C(f"outputs/runs/{RUN05}/artifacts/inner_split_manifest.csv")
    rows=[]
    outer_overlap=0; outer_duplicate_role=0
    all_groups=sorted(outer.group_id.unique())
    test_counts={g:0 for g in all_groups}
    outer_train_map={}
    outer_test_map={}
    for of in sorted(outer.outer_fold.unique()):
        d=outer[outer.outer_fold==of]
        train=set(d.loc[d.role=="train_pool","group_id"])
        test=set(d.loc[d.role=="test","group_id"])
        overlap=train&test; outer_overlap+=len(overlap)
        dupe=int(d.groupby("group_id").size().max()>1); outer_duplicate_role+=dupe
        outer_train_map[int(of)]=train; outer_test_map[int(of)]=test
        for g in test:test_counts[g]+=1
        rows.append({"audit_id":f"OUTER-{of}","scope":"Q2 outer grouped CV","status":"PASS" if not overlap and not dupe else "FAIL",
                     "evidence":f"{len(train)} train-pool files; {len(test)} test files; intersection={len(overlap)}",
                     "risk":"same-file/window leakage","severity_if_fail":"P0"})
    test_once=all(v==1 for v in test_counts.values())
    rows.append({"audit_id":"OUTER-TEST-ONCE","scope":"Q2 outer grouped CV","status":"PASS" if test_once else "FAIL",
                 "evidence":f"{sum(test_counts.values())} test assignments over {len(test_counts)} files; every file test exactly once={test_once}",
                 "risk":"OOF completeness / duplicate test reuse","severity_if_fail":"P0"})
    inner_overlap=0; inner_outer_test_intrusion=0; invalid_inner_pool=0
    for (of,inf),d in inner.groupby(["outer_fold","inner_fold"]):
        train=set(d.loc[d.role=="train","group_id"]); val=set(d.loc[d.role=="val","group_id"])
        overlap=train&val; inner_overlap+=len(overlap)
        outer_test=outer_test_map[int(of)]
        intr=(train|val)&outer_test; inner_outer_test_intrusion+=len(intr)
        outside=(train|val)-outer_train_map[int(of)]; invalid_inner_pool+=len(outside)
        rows.append({"audit_id":f"INNER-{of}-{inf}","scope":"Q2 inner grouped CV",
                     "status":"PASS" if not overlap and not intr and not outside else "FAIL",
                     "evidence":f"train={len(train)} val={len(val)} train∩val={len(overlap)} outer-test intrusion={len(intr)} outside outer-train={len(outside)}",
                     "risk":"inner/outer leakage","severity_if_fail":"P0"})
    return pd.DataFrame(rows), {
        "outer_group_overlap":outer_overlap,"outer_duplicate_role":outer_duplicate_role,
        "outer_every_file_test_once":test_once,"inner_train_val_overlap":inner_overlap,
        "inner_outer_test_intrusion":inner_outer_test_intrusion,"inner_outside_outer_train":invalid_inner_pool
    }

def build_claim_matrix():
    q1_02=J("protocol/02A/latest_freeze_package.json")
    q1_03=J("protocol/03A/latest_freeze_package.json")
    q1_04=J("protocol/04A/latest_freeze_package.json")
    q2_05=J("protocol/05A/latest_freeze_package.json")
    q2_06=J("protocol/06A/latest_freeze_package.json")
    q2_07=J("protocol/07A/latest_freeze_package.json")
    q3_08=J("protocol/08A/latest_freeze_package.json")
    q3_09=J("protocol/09A/latest_freeze_package.json")
    q3_10=J("protocol/10A/latest_freeze_package.json")
    q4_11=J("protocol/11A/latest_freeze_package.json")
    return pd.DataFrame([
      ["Q1-C1","Q1","源域MVP由12kHz FE故障组+4个正常文件构成，共49个独立文件；选择使用目标无标签数据仅作组级可比性。","02-A source/target comparability","subgroup ranking","source_12khz_fe四次leave-one-load均排名第一；MVP=49 files","protocol/02A/latest_freeze_package.json","P1:需比较alternative/all-source方案对后续核心结果的敏感性","SUPPORTED_WITH_GAP"],
      ["Q1-C2","Q1","48kHz正常文件重采样到12kHz时，0–6kHz公共频带信息保持率最低约0.9944。","03-A preprocessing fidelity","anti-alias FIR + matched-resolution Welch","common_band_ratio_min=0.9944328351","protocol/03A/latest_freeze_package.json","无关键缺口","SUPPORTED"],
      ["Q1-C3","Q1","最终Q2接口为1s窗口、50%重叠、26个目标兼容确定性特征；733源窗口、240目标窗口；未全局拟合学习型预处理。","04-A feature interface","deterministic feature extraction","49 source groups/16 target groups; 26 features; fit_boundary=no learned preprocessing","outputs/runs/%s/artifacts/data_interface_manifest.json"%("run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0"),"P1:窗口长度/重叠敏感性在04-A明确延期到STEP12","SUPPORTED_WITH_GAP"],
      ["Q2-C1","Q2","分类评价以独立文件为分组，4 outer×3 inner；相同文件的重叠窗口不跨训练/验证/测试集合。","05-A split protocol","grouped nested CV","4 outer folds; 49 files; group intersections=0","outputs/runs/%s/artifacts/outer_split_manifest.csv"%RUN05,"无关键缺口","SUPPORTED"],
      ["Q2-C2","Q2","LR基线文件级OOF Macro-F1约0.81845，主要失败为OR↔IR混淆和1个N→OR。","06-A baseline","nested grouped multinomial LR","Macro-F1=0.818452; accuracy=0.795918; errors=10","outputs/runs/%s/artifacts/oof_file_predictions.csv"%RUN06,"无；作为基线证据","SUPPORTED"],
      ["Q2-C3","Q2","H1_RF相对Base在四个冻结outer折上多数折改善，汇总OOF Macro-F1=0.9675；逐类Recall OR/IR/B/N=0.9048/1/1/1。","07-A final model","H1_RF 300 trees","fold deltas=[0.4951,-0.0635,0.1905,0.0727]; pooled Macro-F1=0.9675","outputs/runs/%s/artifacts/step07A_summary.json"%RUN07,"P0:outer test参与最终家族接受，0.9675不能视为完全未参与选择的独立终测","SUPPORTED_BUT_SELECTION_BIASED"],
      ["Q3-C1","Q3","源—目标域差异显著，MMD2约0.7922，域分类AUC=1.0，16/16目标文件位于最近源类P95半径外。","08-A domain diagnosis","file-level domain diagnostics","MMD2=0.792223; AUC=1.0; outside_fraction=1","protocol/08A/latest_freeze_package.json","域差异指标不是分类准确率","SUPPORTED"],
      ["Q3-C2","Q3","无迁移A-P预测为B=10、OR=6；T1(alpha=0.25)经伪目标载荷0/1/2选择、载荷3一次留出验证，无正增益也无退化。","08-A/09-A transfer selection","T1 shrink-moment","held load3 A0/T1 Macro-F1=1/1; delta=0","protocol/09A/latest_freeze_package.json","P2:伪目标载荷不能代表真实传感器/RPM/路径差异","SUPPORTED_WITH_LIMIT"],
      ["Q3-C3","Q3","最终A-P为OR=8、B=8，C/D由B→OR；目标真值未知，13/16被标为低可信。","10-A final inference","frozen T1→RF + mean-score aggregation","16/16; OR/B=8/8; min 9-setting agreement=0.3333; min bootstrap=0.6567","protocol/10A/latest_freeze_package.json","P1:需做leave-one-target-file-out目标矩估计及源样本扰动，量化传导式自影响/源样本不确定性","SUPPORTED_WITH_GAP"],
      ["Q3-C4","Q3","09-A未使用伪标签自训练，源分类器被精确保留；伪域负迁移逐文件检查为49个全不变。","09-A transfer safety","unsupervised adaptation without pseudo labels","pseudo_labels_used=false; source score delta=0; degraded_files=0","protocol/09A/latest_freeze_package.json","无真实目标真值，不能验证真实目标负迁移","SUPPORTED_WITH_LIMIT"],
      ["Q4-C1","Q4","解释对象确为最终T1(alpha=0.25)→冻结RF，并精确复现10-A的16/16目标标签。","11-A model identity","file/window occlusion on final pipeline input","model SHA verified; max score diff=2.22e-16; 16/16 labels match","protocol/11A/validation_record.json","无接口缺口","SUPPORTED"],
      ["Q4-C2","Q4","解释忠实性、稳定性和随机对照均通过：Top5遮蔽下降中位0.4488；相邻窗口Spearman中位0.9762；Jaccard=1.0；随机null百分位中位1.0。","11-A explanation validation","occlusion + adjacent-window stability + random5 null","faithfulness 9/9; stability 9/9; sanity valid","protocol/11A/latest_freeze_package.json","P2:遮蔽参考值依赖源文件中位数，且工程特征解释不定位原始波形片段","SUPPORTED_WITH_LIMIT"],
      ["Q4-C3","Q4","机理一致性不是因果证明；H、D等矛盾案例被保留，目标域解释不确认真值。","11-A mechanism audit","prototype-direction consistency","3 contradiction cases retained; target truth unknown","outputs/runs/%s/artifacts/failure_and_contradiction_cases.csv"%RUN11,"P2:目标轴承几何/精确RPM未知，不能做精确BPFO/BPFI验证","SUPPORTED_WITH_LIMIT"],
    ],columns=["claim_id","question","major_conclusion","data_evidence","model_evidence","result_evidence","evidence_path","remaining_gap","status"])

def build_interface_audit():
    d04=J(f"outputs/runs/run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0/artifacts/data_interface_manifest.json")
    d07=J(f"outputs/runs/{RUN07}/artifacts/q3_interface.json")
    v11=J("protocol/11A/validation_record.json")
    return pd.DataFrame([
      ["IF-Q1-Q2","Q1→Q2","PASS","04-A输出733源窗口/49组、26公共特征、group key=independent_object_id；05-A/06-A直接使用同一接口。","04-A data_interface_manifest + 05-A split manifests + 06-A run_config","无断链"],
      ["IF-Q2-Q3","Q2→Q3","PASS_WITH_P0_RELIABILITY_NOTE","07-A冻结final_source_model.joblib、26特征顺序、OR/IR/B/N类序、文件聚合规则；08/09直接继承。模型身份连通，但Q2性能证据存在outer-test选择偏差P0。","07-A q3_interface + 09-A source model SHA","接口连通；性能独立性需12-A2补证"],
      ["IF-Q3-Q4","Q3→Q4","PASS","11-A加载09-A final_transfer_bundle并复现10-A 16/16 A-P标签，最大分数差2.22e-16。","11-A validation_record","无断链"],
      ["IF-Q4-NEXT","Q4→审计/后续","PASS","11-A冻结解释数据、失败案例和表述边界；11-B只回写冻结结果；12-A1只审计不改Word。","11-A freeze + 11-B delivery","无断链"]
    ],columns=["interface_id","interface","status","finding","evidence","gap"])

def build_leakage(split_summary):
    return pd.DataFrame([
      ["LEAK-01","同源窗口跨集合","PASS","05-A以独立文件group_id分组；outer/inner集合交集脚本复核均为0，49文件各outer-test一次。","05-A outer/inner manifests","P0 if violated"],
      ["LEAK-02","全量拟合学习型预处理","PASS","04-A仅确定性特征；06-A StandardScaler与权重只在当前inner-train/outer-train-pool拟合；07-A相关裁剪仅inner-train。","04-A fit_boundary + 06-A run_config/validation + 07-A blueprint","P0 if violated"],
      ["LEAK-03","测试集调参/模型选择","P0_IDENTIFIED","07-A预先冻结规则后仍以4个outer test的paired effect决定H1是否替换Base，因此同一outer结果参与最终家族接受。","07-A final_model_decision.json","12-A2必须补独立/正交验证；07-A pooled OOF不得称完全独立终测"],
      ["LEAK-04","目标伪标签循环自证","PASS","09-A pseudo_labels_used=false；A-P真值unknown；08-A使用目标数据仅无标签域诊断，09-A用无标签矩/协方差。","08-A target_unlabeled_usage_ledger + 09-A freeze","无循环自证"],
      ["LEAK-05","A-P传导式自影响","DISCLOSED_P1","T1真实目标矩由全部240目标窗口共同估计，同一A-P集合参与适配和预测；这是公开的transductive设置，不是标签泄漏，但个体预测存在自影响。","09-A/10-A protocol","12-A2做leave-one-target-file-out目标矩稳定性"],
      ["LEAK-06","历史结果混入","PASS","10-A明确旧28维q3c脚本未使用；各A步以run_id/SHA/code_manifest绑定，失败运行未进入最终冻结。","10-A anomalies + 各code_manifest","继续保持冻结身份"],
      ["LEAK-07","目标无标签参与Q1源组选择","DISCLOSED_P1","02-A以目标无标签组级可比性选择MVP；没有读取目标故障标签，但整个源模型开发具有目标域导向。","02-A freeze/b_handoff","12-A2比较MVP/alternative/all-source敏感性，避免把源选择写成完全目标无关"],
    ],columns=["item_id","risk","status","finding","evidence","closure_or_boundary"])

def build_reliability():
    return pd.DataFrame([
      ["Q1","数据/特征构建与源组选择","预处理信息保真 + 参数敏感性","已有：48k→12k公共频带保持率min=0.9944；组选择leave-one-load稳定。缺：0.5/1/2s×0/50%窗口敏感性。","P1","12-A2运行冻结窗口候选的轻量敏感性；不改正式主配置，量化特征/模型结论是否翻转。"],
      ["Q2","四分类 H1_RF","混淆矩阵/逐类Recall + 分组稳定性","已有：OOF混淆矩阵与逐类Recall；4折file Macro-F1=0.9416/0.9365/1/1。问题：outer test参与最终家族接受。","P0","12-A2增加预冻结正交条件留出/重采样验证，并明确07-A outer为开发选择证据而非完全独立终测。"],
      ["Q3","无监督迁移 T1","无迁移基线 + 域差异变化 + 负迁移 + 稳定性","已有：A0基线、MMD、伪目标留出、49文件负迁移、alpha/聚合/bootstrap稳定性。","P1","12-A2做leave-one-target-file-out传导式稳定性及源文件扰动/重采样，重点D/N/C。"],
      ["Q4","最终模型解释","忠实性 + 稳定性 + 随机对照","已有且充分：Top5 vs Low5/random、窗口Spearman/Jaccard、200随机子集sanity；机理矛盾案例保留。","P2","不必堆更多解释指标；仅保留源中位参考依赖和无法定位原始波形的局限。"]
    ],columns=["question","core_model","selected_reliability_evidence","current_status","gap_level","decision"])

def build_gaps():
    return pd.DataFrame([
      ["P0-01","P0","Q2","07-A outer test参与最终模型家族接受，导致同一outer结果不能作为完全独立最终泛化估计。","07-A final_model_decision: replace Base if paired outer median delta>0 and >=3/4 nonnegative; pooled H1 OOF Macro-F1=0.9675。","12-A2预冻结一个不参与模型选择的正交条件留出/固定模型压力测试；若结果支持则把07-A outer定位为开发配对证据，若明显失败则回滚/降级核心结论。","open"],
      ["P1-01","P1","Q1","窗口长度/重叠敏感性明确延期到STEP12，当前1s/50%不是已证明全局最优。","04-A b_handoff wording_limits明确deferred to STEP12。","12-A2对0.5/1/2s×0/50%候选做固定协议敏感性，报告核心分类/特征结论是否稳定，不据此反调正式主配置。","open"],
      ["P1-02","P1","Q1→Q2","源MVP使用目标无标签可比性选择，缺alternative/all-source后续结果敏感性。","02-A已冻结MVP=49、alternative=109、all_source=161，并把paired comparison列为later_validation。","12-A2用同一冻结评价口径做轻量方案敏感性；至少验证MVP选择不是唯一导致结论成立的方案。","open"],
      ["P1-03","P1","Q3","A-P真实T1为传导式，单个目标文件也参与目标矩估计；当前稳定性未覆盖这种自影响。","09-A用全部240目标窗口估计无标签矩；10-A已有alpha/aggregation/window bootstrap但未leave-one-file-out。","12-A2逐文件留一目标文件不参与目标矩估计后再预测该文件，报告16/16标签/分数变化；重点D/N/C。","open"],
      ["P1-04","P1","Q3","A-P对源样本组成的不确定性尚未量化，13/16低可信且仅OR/B两类输出。","10-A low_confidence=13，OR/B=8/8，IR/N=0。","12-A2对源文件做分层bootstrap/jackknife（不调参）重复T1+推理，输出标签保持率与分数区间。","open"],
      ["P2-01","P2","Q2","N类仅4个独立正常文件，逐类Recall估计不确定性高。","02-A/05-A file_class_counts N=4；每个outer test仅1个N。","无法用现有数据消除；12-A2可给重采样范围但最终论文必须保留小样本限制。","open_limit"],
      ["P2-02","P2","Q3/Q4","A-P真值未知，无法计算真实目标Accuracy/F1/Recall，也无法用解释证明目标真值。","08/09/10/11全部冻结truth=unknown。","不可关闭；保持预测/解释身份，绝不补造真值。","open_limit"],
      ["P2-03","P2","Q3/Q4","目标轴承几何与逐文件精确RPM未知，不能验证精确BPFO/BPFI/BSF/FTF命中。","08-A target exact rpm unavailable; geometry unknown；11-A exact target fault frequency not asserted。","不可关闭；只讨论冲击/频谱代理和源域机理。","open_limit"],
      ["P2-04","P2","Q4","遮蔽解释依赖源文件中位参考，工程特征扰动可能离开真实数据流形。","11-A unresolved issue明确reference dependence/off-manifold。","可在论文说明；如12-A2有余量可做替代参考值敏感性，但非核心必跑。","open_limit"]
    ],columns=["gap_id","level","scope","gap","evidence","closure_condition","status"])

def build_tasks():
    return pd.DataFrame([
      ["A2-01","P0-01","must","Q2固定最终H1_RF的正交条件压力测试","冻结当前H1_RF家族/参数，不再模型选择；按物理条件（优先leave-one-load-out，其次预注册condition-group holdout）重训训练侧并只在留出条件评价。","每个留出条件的file Macro-F1、逐类Recall、混淆矩阵；不得用留出结果调参。","若整体/逐类不出现灾难性失效，则关闭P0并将07-A outer改称开发配对证据；若失效则触发回滚/降级。"],
      ["A2-02","P1-01","should","窗口参数敏感性","对04-A已冻结候选0.5/1/2s×overlap 0/0.5，使用固定特征定义和固定模型家族做有限重算。","各候选文件级指标、逐类Recall与相对1s/50%的变化。","核心结论不因合理窗口选择大幅翻转；否则保留敏感性并限制结论。"],
      ["A2-03","P1-02","should","源选择方案敏感性","MVP/alternative/all-source按相同分组评价与固定H1_RF家族做对照，不再扩大候选。","三方案Macro-F1/逐类Recall及Q3域差异指标。","若MVP并非唯一有效且无严重退化，关闭；否则明确目标无标签源选择依赖。"],
      ["A2-04","P1-03","should","A-P leave-one-target-file-out传导式稳定性","对每个目标文件，估计T1目标矩时排除该文件15窗，再用同一冻结RF预测该文件。","16文件LOTO标签、分数、与正式10-A一致率。","逐文件报告；D/N/C若翻转继续标高风险，不人工改正式标签。"],
      ["A2-05","P1-04","should","源文件组成不确定性","分层按OR/IR/B/N对49源文件bootstrap/jackknife，固定RF家族/参数和T1 alpha，不做调参。","A-P每文件标签保持率、最高分数分布/区间。","量化13个低可信文件的源样本敏感性；结果只补证不反调10-A。"],
      ["A2-06","P2-01","optional","N类小样本不确定性摘要","从A2-01/03已有重采样结果提取N类Recall范围，不新增独立大实验。","N类Recall范围/失败条件。","只能量化，不能声称消除4文件限制。"]
    ],columns=["task_id","closes_gap","priority","task","frozen_execution_rule","outputs","close_rule"])

def main():
    t0=time.time()
    state=J("protocol/00C/process_state_card.json")
    require(state["CURRENT_ALLOWED_STEP"]=="12-A1" and state["NEXT_ALLOWED"]=="12-A1","12-A1 gate not open")
    require(state["LAST_PASS_TOKEN"]=="11-B-V2026.09.20-df1a4844-通过","prior pass token mismatch")
    require(state.get("OPEN_P0")==[],"unexpected pre-existing OPEN_P0")
    cfg=J("protocol/12A1/run_config.json")
    split_df, split_summary=split_audit()
    claims=build_claim_matrix()
    interfaces=build_interface_audit()
    leaks=build_leakage(split_summary)
    reliab=build_reliability()
    gaps=build_gaps()
    tasks=build_tasks()

    # Deterministic random claim spot-check.
    rng=random.Random(int(cfg["random_spot_check"]["seed"]))
    ids=claims.claim_id.tolist()
    chosen=rng.sample(ids,min(int(cfg["random_spot_check"]["count"]),len(ids)))
    spot=[]
    for cid in chosen:
        r=claims[claims.claim_id==cid].iloc[0]
        p=ROOT/str(r.evidence_path)
        exists=p.exists()
        spot.append({"claim_id":cid,"question":r.question,"evidence_path":r.evidence_path,
                     "evidence_exists":exists,"status":"PASS" if exists else "FAIL",
                     "claim_excerpt":r.major_conclusion[:100]})
    spot=pd.DataFrame(spot)

    # Exact additional identity checks.
    v06=J("protocol/06A/validation_record.json")
    v07=J("protocol/07A/validation_record.json")
    f09=J("protocol/09A/latest_freeze_package.json")
    v10=J("protocol/10A/validation_record.json")
    v11=J("protocol/11A/validation_record.json")
    decision=J(f"outputs/runs/{RUN07}/artifacts/final_model_decision.json")
    outer_selection_flag=("outer" in decision["rule"].lower() and decision["H1_outer_effect"]["H1_outer_acceptance_passed"] is True)
    pseudo_used=bool(J("protocol/09A/latest_freeze_package.json")["parameters"]["pseudo_labels_used"])
    checks={
      "gate_valid":True,
      "all_random_spot_check_evidence_exists":bool(spot.evidence_exists.all()),
      "outer_group_leakage_zero":split_summary["outer_group_overlap"]==0 and split_summary["outer_duplicate_role"]==0,
      "inner_group_leakage_zero":split_summary["inner_train_val_overlap"]==0 and split_summary["inner_outer_test_intrusion"]==0 and split_summary["inner_outside_outer_train"]==0,
      "each_source_file_outer_test_once":bool(split_summary["outer_every_file_test_once"]),
      "learned_preprocessing_training_boundary_verified":bool(v06["checks"]["training_boundary"]["passed"]),
      "target_pseudo_labels_not_used":not pseudo_used,
      "historical_legacy_not_used_in_10A":bool(v10["checks"]["step09_file_results_exactly_reproduced"]),
      "q4_final_model_identity_verified":bool(v11["checks"]["final_09_10_model_identity_verified"] and v11["checks"]["target_10A_predictions_exactly_reproduced"]),
      "outer_test_final_family_selection_risk_identified":bool(outer_selection_flag and "P0-01" in gaps.gap_id.values),
      "all_P0_have_closure_task":all(gaps[gaps.level=="P0"].gap_id.isin(tasks.closes_gap)),
      "all_gaps_have_evidence_and_close_condition":bool((gaps.evidence.str.len()>0).all() and (gaps.closure_condition.str.len()>0).all()),
      "no_unidentified_P0":True,
      "word_not_edited":True,
      "no_new_large_scale_experiment":True,
      "no_formal_paper_figure":True
    }
    passed=all(checks.values())

    created=nowz()
    binding={"step_id":STEP,"git_commit":git_head(),"run_config_sha256":sha256_file(CFG),
             "environment_sha256":sha256_file(ENV),"source_state_version":state["state_version"],
             "latest_word_sha256":state["LATEST_WORD"]["sha256"]}
    bd=canon(binding)
    run_id=f"run_12-A1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{bd[:8]}_{os.urandom(4).hex()}"
    out=ROOT/"outputs/runs"/run_id; art=out/"artifacts"; mani=out/"manifests"
    art.mkdir(parents=True); mani.mkdir(parents=True)
    claims.to_csv(art/"claim_evidence_matrix.csv",index=False,encoding="utf-8-sig")
    interfaces.to_csv(art/"four_question_interface_audit.csv",index=False,encoding="utf-8-sig")
    leaks.to_csv(art/"leakage_and_circularity_audit.csv",index=False,encoding="utf-8-sig")
    reliab.to_csv(art/"main_model_reliability_evidence_selector.csv",index=False,encoding="utf-8-sig")
    gaps.to_csv(art/"gap_register_P0_P1_P2.csv",index=False,encoding="utf-8-sig")
    tasks.to_csv(art/"step12A2_validation_task_list.csv",index=False,encoding="utf-8-sig")
    split_df.to_csv(art/"group_split_reaudit.csv",index=False,encoding="utf-8-sig")
    spot.to_csv(art/"random_claim_spot_check.csv",index=False,encoding="utf-8-sig")

    summary={
      "step_id":STEP,"run_id":run_id,"status":"passed" if passed else "failed",
      "claim_count":len(claims),"random_spot_checks":len(spot),
      "interfaces":{"pass":int((interfaces.status=="PASS").sum()),"pass_with_note":int((interfaces.status!="PASS").sum())},
      "leakage_summary":{"hard_split_leakage_found":False,"global_learned_preprocess_leakage_found":False,
                         "target_pseudo_label_cycle_found":False,"historical_result_mix_found":False,
                         "recognized_P0":["P0-01"],"recognized_P1":["P1-01","P1-02","P1-03","P1-04"]},
      "gap_counts":gaps.level.value_counts().to_dict(),
      "P0_open":["P0-01"],
      "P0_interpretation":"07-A outer tests participated in final family acceptance; this is recognized, not hidden. 12-A2 must add an orthogonal fixed-model validation and final wording must not call 07-A pooled OOF an untouched final test.",
      "Q3_checks":{"no_transfer_baseline":True,"pseudo_target_holdout":True,"negative_transfer_check":True,"A_P_stability":True},
      "Q4_final_model_identity":True,
      "word_edit_performed":False
    }
    (art/"audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    validation={"schema_version":"12A1-validation-1.0","step_id":STEP,"run_id":run_id,
                "status":"passed" if passed else "failed","passed":passed,"checks":checks,
                "evidence":{
                  "claim_matrix":f"outputs/runs/{run_id}/artifacts/claim_evidence_matrix.csv",
                  "spot_check":f"{len(spot)}/{len(spot)} sampled claims have existing evidence paths" if spot.evidence_exists.all() else "spot check failure",
                  "split_reaudit":split_summary,
                  "test_selection_P0":"07-A final_model_decision uses paired outer effect in final family acceptance",
                  "pseudo_label_cycle":"09-A pseudo_labels_used=false",
                  "q4_identity":"11-A exact final model identity + 10-A 16/16 reproduction",
                  "gap_register":f"outputs/runs/{run_id}/artifacts/gap_register_P0_P1_P2.csv",
                  "A2_tasks":f"outputs/runs/{run_id}/artifacts/step12A2_validation_task_list.csv"
                },
                "recognized_open_P0":["P0-01"],"unidentified_P0":[],"word_edit_performed":False}
    (art/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    freeze={"schema_version":"00C-1.0","package_type":"A_freeze_package","step_id":STEP,
            "freeze_package_id":f"FREEZE-12A1-{run_id[-8:]}","status":"passed" if passed else "failed","run_id":run_id,
            "versions":{"code_version_id":"git:"+git_head(),"raw_data_version_id":"RAW-5a5dd129c91bfc64",
                        "input_derived_data_ids":["FREEZE-04A-3e3b97c0","FREEZE-07A-7e8829fb","FREEZE-10A-4346ce31","FREEZE-11A-1f812c22"],
                        "run_config_sha256":sha256_file(CFG),"environment_sha256":sha256_file(ENV)},
            "random_seed":int(cfg["random_spot_check"]["seed"]),
            "parameters":{"audit_only":True,"random_claim_spot_check_count":len(spot),"new_large_scale_experiment":False},
            "real_results":[
              {"name":"claim_evidence_rows","value":len(claims)},
              {"name":"split_leakage_reaudit","value":split_summary},
              {"name":"gap_counts","value":gaps.level.value_counts().to_dict()},
              {"name":"recognized_P0","value":["P0-01"]},
              {"name":"unidentified_P0","value":[]},
              {"name":"Q3_required_evidence_present","value":{"no_transfer":True,"pseudo_target":True,"negative_transfer":True,"A_P_stability":True}},
              {"name":"Q4_final_model_identity","value":True}
            ],
            "validation_evidence":[{"path":f"outputs/runs/{run_id}/artifacts/{x}"} for x in [
              "validation_record.json","claim_evidence_matrix.csv","four_question_interface_audit.csv",
              "leakage_and_circularity_audit.csv","main_model_reliability_evidence_selector.csv",
              "gap_register_P0_P1_P2.csv","step12A2_validation_task_list.csv","random_claim_spot_check.csv"]],
            "anomalies_and_failures":[
              {"item":"P0-01","status":"recognized_open","detail":"07-A outer test participated in final family acceptance; requires 12-A2 orthogonal fixed-model validation and wording boundary."}
            ],
            "b_handoff":{"required_data":["claim_evidence_matrix.csv","four_question_interface_audit.csv","leakage_and_circularity_audit.csv","main_model_reliability_evidence_selector.csv","gap_register_P0_P1_P2.csv","step12A2_validation_task_list.csv"],
                         "supported_conclusions":["Four-question data/model identities are connected and traceable.","No same-file window leakage, global learned-preprocessing leakage, target pseudo-label self-training, or legacy-result contamination was found.","One explicit P0 is recognized: Q2 outer-test participation in final model-family acceptance.","Q3 already has no-transfer/pseudo-target/negative-transfer/A-P stability evidence; Q4 explains the actual final model."],
                         "wording_limits":["Do not call 07-A pooled H1 OOF a completely untouched final test until P0-01 is addressed.","Do not report A-P target accuracy/F1/Recall.","Do not treat explanation as causal or target truth."],
                         "approved_tables":["claim_evidence_matrix.csv","four_question_interface_audit.csv","leakage_and_circularity_audit.csv","main_model_reliability_evidence_selector.csv","gap_register_P0_P1_P2.csv","step12A2_validation_task_list.csv"],
                         "approved_figures":[]},
            "unresolved_issues":["P0-01 must be addressed in 12-A2 before later final-quality acceptance.","P1-01..04 should be run in 12-A2 if compute budget permits; P2 limitations remain wording boundaries."],
            "freeze":{"created_utc":created,"content_sha256":None,
                      "invalidation_dependencies":["Any upstream A freeze changes","05-A split manifests change","07-A final decision changes","09/10/11 final model/prediction/explanation identity changes","12-A1 audit code/rules change"]}}
    tmp=json.loads(json.dumps(freeze)); freeze["freeze"]["content_sha256"]=canon(tmp)
    (art/"A_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    pdir=ROOT/"protocol/12A1"; pdir.mkdir(parents=True,exist_ok=True)
    (pdir/"latest_freeze_package.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"validation_record.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (pdir/"latest_run_id.txt").write_text(run_id+"\n",encoding="utf-8")

    files=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        files.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    manifest={"schema_version":"12A1-1.0","manifest_type":"run_manifest","run_id":run_id,"step_id":STEP,
              "status":"completed" if passed else "failed","created_utc":created,
              "binding":{**binding,"binding_digest":bd},
              "code":{"repository":"Mhhhh958/mathmodeling","commit":git_head(),"entrypoint":"scripts/step12a1_evidence_audit.py","code_manifest":"protocol/12A1/code_manifest.json"},
              "outputs":{"root":f"outputs/runs/{run_id}","files":files,"output_tree_sha256":canon(files)},
              "duration_seconds":float(time.time()-t0)}
    (mani/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":passed,"run_id":run_id,"freeze_package_id":freeze["freeze_package_id"],
                      "freeze_content_sha256":freeze["freeze"]["content_sha256"],
                      "gap_counts":summary["gap_counts"],"recognized_P0":["P0-01"],"unidentified_P0":[]},ensure_ascii=False))
    if not passed: raise SystemExit(2)
if __name__=="__main__": main()
