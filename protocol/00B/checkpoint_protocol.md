# 低负载断点协议（00-B 冻结）

1. 仅在“自然批次”边界暂停；自然批次须在该步骤 run_config 事先定义，如文件批、fold、epoch 段、参数块或图表组。
2. 暂停前写入 outputs/runs/<run_id>/checkpoints/<batch_id>/，至少包含 checkpoint_manifest.json、已完成批次、未完成批次、恢复所需中间状态及其 SHA256。
3. RESUME_TOKEN 格式：
   RESUME_<run_id>_<batch_id>_<checkpoint_digest前12位>
4. 暂停时 run_manifest 必须写 status=paused，并记录 resume_token、completed_batches、pending_batches。
5. 暂停时流程闸门固定：
   NEXT_ALLOWED=<当前步骤>
   不得跳到后续步骤。
6. 恢复前必须验证 checkpoint SHA、code_version_id、data_version、run_config_sha、environment_sha；任一不符则不得续跑，必须新建 run_id。
7. 全部批次完成后才允许将 run_manifest 改为 completed 并进入下一流程步骤。
