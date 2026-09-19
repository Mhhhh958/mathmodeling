# 流程状态卡

- CURRENT_ALLOWED_STEP: `00-D`
- LAST_PASS_TOKEN: `00-C-V2026.09.19-r1-通过`
- LATEST_WORD: `null`（尚未进入需要母稿的B步骤）
- LATEST_FREEZE_PACKAGE: `null`（尚未产生后续A冻结包）
- OPEN_P0: `[]`
- NEXT_ALLOWED: `00-D`

## 闸门规则
每一步开始前必须核验 requested step == CURRENT_ALLOWED_STEP，同时核验 LAST_PASS_TOKEN / NEXT_ALLOWED 与前一步一致。非法跳步立即停止。

若某步骤按低负载协议暂停，则：
- PASS_TOKEN=无
- NEXT_ALLOWED=当前步骤
- 必须输出并保存 RESUME_TOKEN、已完成批次和未完成批次。

若存在影响当前通过条件的 OPEN_P0，则不得发放 PASS_TOKEN。

## 冻结失效
步骤13后，若数据、代码、模型、关键参数或核心结果发生实质变化，则回退到最早受影响A步骤，并重新执行步骤13及其后依赖步骤。
