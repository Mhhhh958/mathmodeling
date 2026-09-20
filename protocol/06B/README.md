# STEP06-B 论文回写与Word交付

- source A freeze: FREEZE-06A-01576948
- source A run: run_06-A_20260920T035742210347Z_b83cb3a6_01576948
- source word: V0.6 / f1131a5f160bf53821d87e88be611b93f26f916d5941596bc58387f26bcf3026
- output word: V0.7 / 0e671b79e603713d3c3a9924cfc8ad41de52f329a7b89166ddfe7a8ffae0b650
- run_id: run_06-B_20260920T042258000000Z_e55b0303_0e671b79
- 仅增量编辑；未重新训练、调参或选模。
- 新增：5.2.2“基线模型与真实评价”、表16-表18、图4-图6。
- 文件级主结果来自06-A当前运行：OOF Macro-F1=0.818452，Accuracy=0.795918；OR/IR/B/N Recall=0.714286/0.75/1.0/0.75。
- 文件级误判10个：OR→IR 6，IR→OR 3，N→OR 1；IR007_0.mat与N_0.mat作为真实失败案例保留。
- 窗口级指标只作为辅助；LR输出未校准，只称模型分数，不称校准概率/置信度。
- Word最终渲染24页；caption重复自动编号问题已修复并重渲染。media 7→10，drawing 6→9，tables 16→19，math 5→5，sections 6→6；既有媒体字节和00-D母版核心包保持。
- PASS_TOKEN: 06-B-V2026.09.20-0e671b79-通过
- NEXT_ALLOWED: 07-A
