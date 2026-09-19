# STEP02-A 面向迁移的源域筛选冻结摘要

- 蓝图先于执行冻结：4ba829a40cf9526f95677c9f6d8b75958106f643
- run_id: run_02-A_20260919T162016957185Z_75708ffe_6f36d0b4
- raw_data_version_id: RAW-5a5dd129c91bfc64
- code_version_id: git:60d5e453b9e4d007971dac95bf1421d6ff920b67
- evidence commit: f4082f1f22553a2f11d8952957755af164a729ce
- GitHub Actions: 105927559135 / success
- 目标A-P只以无标签信号参与组级可比性，不读取历史预测标签。
- 共同分析口径：12kHz，0–6kHz。
- 故障子域内部距离排名：12kHz FE=0.859955 < 12kHz DE=0.896483 < 48kHz DE=0.947292。
- 留一0/1/2/3hp四次复算中，12kHz FE均排名第一（4/4）。
- MVP：source_12khz_fe + 全部4个normal，共49文件，OR/IR/B/N=21/12/12/4。
- 替代方案：source_12khz_fe + source_12khz_de + 全部4个normal，共109文件，OR/IR/B/N=49/28/28/4，并恢复0.028 in故障尺寸覆盖。
- 正常类只有4个文件且采集配置独特，是后续四分类必须单独报告的风险。
- 目标约600rpm仅作粗粒度域差异；目标传感器位置和确切轴承型号未冻结，不作为评分依据。
- 本A步骤未编辑Word、未更新五本账、未训练最终分类模型。
- PASS_TOKEN: 02-A-V2026.09.20-f9ed0ecc-通过
- NEXT_ALLOWED: 02-B
