# STEP05-B 论文回写与Word交付

- source A freeze: FREEZE-05A-ba6432cf
- source A run: run_05-A_20260919T192338074034Z_f82972c7_ba6432cf
- source word: V0.5 / dfbb5d0ed45b47ef30d0336d273e2d425d5d43f57e24f2a9246ce8da9e68feee
- output word: V0.6 / f1131a5f160bf53821d87e88be611b93f26f916d5941596bc58387f26bcf3026
- run_id: run_05-B_20260919T195006000000Z_338d8e27_f1131a5f
- 仅增量编辑；未训练、未调参、未选模，未写入模型成绩。
- 新增：5.2.1“训练测试划分与评价协议”、表14、表15。
- 四分类独立文件数：OR/IR/B/N=21/12/12/4；采用4个外层文件级留出折，每折测试1个N；每个外层训练池再做3个内层验证折，每折验证1个剩余N。
- 主指标为文件级Macro-F1；同一文件窗口概率取均值后argmax；窗口级指标仅辅助；外层测试不参与模型选择。
- 候选冻结为LR/RF共12配置；最多144次内层拟合+4次外层重拟合；不自适应扩网格。
- 未制作协议图，因表格与文字已能无歧义表达边界；05-A验证确认所有group交集为0。
- Word最终渲染21页并逐页检查；既有媒体字节保持，media 7→7，drawing 6→6，tables 14→16，math 5→5，sections 6→6。
- styles/numbering/theme/fontTable/[Content_Types]与V0.5一致。
- PASS_TOKEN: 05-B-V2026.09.20-f1131a5f-通过
- NEXT_ALLOWED: 06-A
