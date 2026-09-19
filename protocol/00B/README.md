# STEP 00-B 运行与版本基线包

状态：冻结基线（00-B）。本包不训练模型、不建立论文 Word、不正式成图。

## 默认执行模式
- 默认：Chat + Git。
- 仓库：Mhhhh958/mathmodeling。
- Git 模式以实际运行所用 commit 作为 code_version_id；Chat 负责流程闸门和验收，Git 负责代码、数据版本与可复现产物。
- Chat 直传仅作备选；直传模式必须用 code_manifest + SHA256 生成 code_version_id。

## 执行环境基线
- OS：Ubuntu 24.04（GitHub Actions 基线）
- 架构：x86_64
- Python：3.11；每次运行记录实际 patch 版本
- CPU：默认 CPU-only；实际型号和核数逐 run 采集
- GPU：默认不要求
- CUDA：默认 None；以后若显式启用 GPU，须先新建环境版本并冻结 torch/CUDA/驱动
- 全局默认随机种子：20260919
- manifest 时间统一 UTC
- 依赖：核心库用 requirements-lock-00b.txt 精确固定；每次真实运行另存 pip_freeze.txt

## run_id 与绑定
每次真实运行必须绑定：
code_version_id + data_version_id + run_config_sha256 + seed + environment_sha256。
对上述 binding 做 canonical JSON 后计算 SHA256，得到 binding_digest。

run_id：
run_<STEP>_<UTC微秒时间>_<binding前8位>_<随机8位>

输出只允许进入：
outputs/runs/<run_id>/
禁止覆盖其他 run。

## 数据版本
- 00-B 仅建立原始数据指纹规则。
- raw_data_version_id 在 01-A 后冻结。
- data_manifest.draft.json 只保存 00-A 已核实的目录/数量/Git 元数据，不代表已审计 MAT 内部内容。
- 派生数据必须有 derived_data_id，并记录父数据、producer_run_id、code_version_id、配置 SHA 和输出 SHA256。

## 代码落盘
凡流程中实际执行的分析/训练/绘图代码，必须保存为仓库真实 .py/.ipynb 文件、进入 code_manifest，并在真实运行前提交 Git。聊天临时代码不得作为正式结果依据。

命名：
- 正式步骤脚本：scripts/stepXX_<scope>_vNN.py
- 协议工具：scripts/protocol/<name>.py
- 探索 notebook：notebooks/stepXX_<scope>_vNN.ipynb；若结果进入正式结论，必须完整纳入 code_manifest 或转为可复现脚本。

## 输出目录
每个 run 根目录可含：
manifests/、logs/、artifacts/、tables/、figures/、checkpoints/。
最终 run_manifest 必须记录输出文件 SHA256、大小和 output_tree_sha256。

## 低负载断点
只能在自然批次边界暂停。暂停前落盘 checkpoint、已完成/未完成清单和 RESUME_TOKEN；NEXT_ALLOWED 保持当前步骤，直到该步骤全部完成。详见 checkpoint_protocol.md。
