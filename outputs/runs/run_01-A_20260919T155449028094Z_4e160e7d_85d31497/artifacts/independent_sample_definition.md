# 独立样本口径

一个原始MAT文件=一个独立采集对象。共177个：源域161、目标域16。
唯一ID为data/raw下POSIX相对路径。不同子目录存在同名basename，因此禁止只用文件名作为ID。
后续所有窗口/分段必须携带independent_object_id；同一MAT派生窗口必须同组切分，不得跨训练/验证/测试。
DE/FE/BA是传感器位置；fault_bearing_location是发生故障的轴承位置，两者严格分离。
目标A-P真值未知；约600rpm只作近似工况，不作逐时精确转速。
