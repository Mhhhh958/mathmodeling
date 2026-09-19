# STEP04-A 切片与全数据特征提取冻结摘要

- run_id: run_04-A_20260919T182731095541Z_8ac2c328_3e3b97c0
- feature_data_version_id: FEAT-23d45f5649dcd5f1
- code_version_id: git:e8d1ed050b609833043653dae736f8d2323e4fbe
- evidence commit: 8e1fc48dd5e62a5eb9abbc4c12b5f3c50a0f8253
- GitHub Actions: 105945347823 / success
- freeze SHA256: d60f9e54ae8aa2e216c4d4540f0c4fda90f64464fc16769c4fa428b41b1bd567
- PASS_TOKEN: 04-A-V2026.09.20-d60f9e54-通过
- NEXT_ALLOWED: 04-B

窗口口径：共同12 kHz；1.0 s窗=12000点；50%重叠，步长0.5 s=6000点；不补零。
选择理由：源域约28.75–29.97 Hz转频，1 s至少覆盖约11个最慢FTF周期；目标约600 rpm时约10个轴转周期。0.5/1.0/2.0 s与0/50%重叠已冻结为STEP12敏感性候选，本步不宣称1 s/50%全局最优。

数量：49个源域独立文件→733窗（OR319/IR180/B180/N54）；16个目标文件→240窗；合计973窗。独立对象仍是65个原始文件，不是973个独立样本。

公共特征26个，全部源/目标兼容且NaN/Inf=0、常数特征=0。另有5个源域机理辅助特征，因目标几何/精确RPM不可用且存在采集配置混杂风险，默认Q2接口明确排除。

Q2接口：X_source_common.csv + y_source_labels.csv + groups_source.csv，任何插补/缩放/筛选均尚未fit，后续必须先按group_id划分，再只在训练组拟合。

验证：随机8窗回溯通过；RMS、峰值因子、谱质心3项独立复算最大绝对误差=0；默认X无label/file/group/load/channel/RPM/bearing等泄漏列。

第一次运行因验证侧谱质心分母写法与冻结公式不完全一致而失败；保持1e-10阈值不变，修正验证实现后完整重跑通过。失败运行已保存在code_manifest。
