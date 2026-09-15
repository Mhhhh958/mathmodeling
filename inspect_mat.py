from pathlib import Path
from collections import Counter, defaultdict
import scipy.io as sio
import csv

ROOT = Path("data/raw")
OUT_TXT = Path("mat_audit_summary.txt")
OUT_CSV = Path("mat_audit_detail.csv")

rows = []
errors = []
dir_counts = Counter()
var_patterns = Counter()
target_lengths = {}
source_rpm = {}

mat_files = sorted(ROOT.rglob("*.mat"))

for i, f in enumerate(mat_files, 1):
    rel = f.relative_to(ROOT)
    folder = str(rel.parent)
    dir_counts[folder] += 1

    try:
        info = sio.whosmat(f)
        vars_info = []
        signal_lengths = []

        for name, shape, dtype in info:
            vars_info.append(f"{name}:{shape}:{dtype}")

            # 记录一维/二维向量长度
            if len(shape) == 2 and 1 in shape:
                signal_lengths.append(max(shape))

        var_names = tuple(sorted(x[0] for x in info))
        var_patterns[var_names] += 1

        # 目标域信号长度
        if "target_domain" in str(rel):
            lengths = [max(shape) for name, shape, dtype in info
                       if len(shape) == 2 and 1 in shape]
            target_lengths[f.stem] = lengths

        # 尝试读取源域 RPM
        rpm_value = ""
        rpm_vars = [name for name, shape, dtype in info
                    if "RPM" in name.upper()]

        if rpm_vars:
            try:
                d = sio.loadmat(f, variable_names=rpm_vars)
                vals = []
                for rv in rpm_vars:
                    x = d.get(rv)
                    if x is not None and x.size > 0:
                        vals.append(float(x.ravel()[0]))
                if vals:
                    rpm_value = ";".join(f"{x:.3f}" for x in vals)
                    source_rpm[str(rel)] = rpm_value
            except Exception:
                pass

        rows.append({
            "file": str(rel),
            "vars": " | ".join(vars_info),
            "vector_lengths": ";".join(map(str, signal_lengths)),
            "rpm": rpm_value,
            "status": "OK"
        })

    except Exception as e:
        errors.append((str(rel), repr(e)))
        rows.append({
            "file": str(rel),
            "vars": "",
            "vector_lengths": "",
            "rpm": "",
            "status": "ERROR: " + repr(e)
        })

with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as fp:
    w = csv.DictWriter(
        fp,
        fieldnames=["file", "vars", "vector_lengths", "rpm", "status"]
    )
    w.writeheader()
    w.writerows(rows)

with OUT_TXT.open("w", encoding="utf-8") as fp:
    fp.write("=== E题 MAT 数据完整性审计 ===\n\n")

    fp.write(f"MAT文件总数: {len(mat_files)}\n")
    fp.write(f"成功读取: {len(mat_files)-len(errors)}\n")
    fp.write(f"读取失败: {len(errors)}\n\n")

    fp.write("=== 各目录文件数 ===\n")
    for k, v in sorted(dir_counts.items()):
        fp.write(f"{k}: {v}\n")

    fp.write("\n=== 变量结构模式 ===\n")
    for pattern, count in var_patterns.most_common():
        fp.write(f"{count} 个文件: {pattern}\n")

    fp.write("\n=== 目标域 A-P 信号长度 ===\n")
    for k in sorted(target_lengths):
        fp.write(f"{k}: {target_lengths[k]}\n")

    fp.write("\n=== 源域 RPM 示例（前20个） ===\n")
    for k, v in list(source_rpm.items())[:20]:
        fp.write(f"{k}: {v}\n")

    fp.write("\n=== 读取失败文件 ===\n")
    if errors:
        for f, e in errors:
            fp.write(f"{f}: {e}\n")
    else:
        fp.write("无\n")

print("检查完成")
print("请把这个文件发给我：", OUT_TXT.resolve())
print("详细结果另存为：", OUT_CSV.resolve())
