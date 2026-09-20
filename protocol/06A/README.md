# STEP06-A 最小可行基线实算冻结摘要

- run_id: run_06-A_20260920T035742210347Z_b83cb3a6_01576948
- model: multinomial LogisticRegression
- input: FREEZE-05A-ba6432cf + FEAT-23d45f5649dcd5f1
- features: 26 common features
- code commit: 9313b5e5aa5734ef6088626fcad1904bb7ead347
- evidence commit: 560888d768e4de2283df98102740ca47e7d344ef
- GitHub Actions: 106017506342 / success
- freeze SHA256: e55b030399965f25849fff60f8632495fbfd6763c974b49923177033e2b18af2

正式训练前先完成outer0/inner0边界内的48行训练+16行验证smoke test，26维、四类标签和4列合法模型分数全部通过；未访问outer-test。

正式基线只运行LR：C={0.1,1,10}×两种训练权重，共6配置；4个外层×3个内层=72次内层拟合，之后4次外层重拟合。StandardScaler和权重均只由当前训练组拟合/计算；不做插补、特征筛选、SMOTE或窗口复制。

四个外层文件级Macro-F1：0.446429、1.000000、0.809524、0.927273；中位数0.868398，min-max=0.446429–1.000000（不是置信区间）。49文件OOF汇总Macro-F1=0.818452，Accuracy=0.795918。逐类Recall：OR=0.714286，IR=0.75，B=1.0，N=0.75。

文件级混淆矩阵（行真值OR/IR/B/N，列预测OR/IR/B/N）：
[[15,6,0,0],[3,9,0,0],[0,0,12,0],[1,0,0,3]]。
共10个文件误判：OR→IR 6，IR→OR 3，N→OR 1。IR007_0.mat的窗口标签和group回溯检查无错位，真实IR但本轮预测OR；N_0.mat真实N但预测OR。

独立复算Macro-F1与sklearn结果差0，混淆矩阵完全一致；典型正确文件N_1_(1772rpm).mat及误判IR007_0.mat的窗口→文件均值聚合复算均一致。

LR的predict_proba输出未做校准；论文/后续步骤只称“模型分数”，不得称校准概率或置信度。

本A步骤未编辑Word、未更新五本账。
PASS_TOKEN: 06-A-V2026.09.20-e55b0303-通过
NEXT_ALLOWED: 06-B
