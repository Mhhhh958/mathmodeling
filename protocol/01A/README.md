# STEP01-A 原始MAT逐文件审计冻结摘要

- run_id: run_01-A_20260919T155449028094Z_4e160e7d_85d31497
- raw_data_version_id: RAW-5a5dd129c91bfc64
- code_version_id: git:4f6022e0203c41705242dcc543a4ea9dff28b12a
- evidence commit: 9b51c003123cab393ae72db52c51dd4feec7bfea
- GitHub Actions check: 105923977890 / success
- 读取：177/177成功；源域161、目标域16。
- 源域类别：OR 77、IR 40、B 40、N 4，与题目口径完全一致。
- 目标A-P：16/16均256000点；32kHz下均8秒；真值均unknown；约600rpm只保留为近似工况。
- 源域RPM：161/161可获得（MAT变量或文件名注记）。
- SHA检查：原始文件字节重复组0；主信号内容重复组0。
- 非有限值文件0；常量文件0；严重错配0。
- 独立样本：一个原始MAT文件=一个独立采集对象；唯一ID使用data/raw相对路径。
- 非阻塞结构/长度差异详见 observed_irregularities.json；无文件被静默删除。
- 本A步骤未编辑Word、未更新五本账、未训练模型。
