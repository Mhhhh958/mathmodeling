# STEP02-A 面向迁移的源域筛选结果

run_id: run_02-A_20260919T162016957185Z_75708ffe_6f36d0b4
raw_data_version_id: RAW-5a5dd129c91bfc64
full ranking: [{'subgroup': 'source_12khz_fe', 'robust_group_distance': 0.8599550818244359, 'rank': 1}, {'subgroup': 'source_12khz_de', 'robust_group_distance': 0.8964830009225436, 'rank': 2}, {'subgroup': 'source_48khz_de', 'robust_group_distance': 0.9472915828793357, 'rank': 3}]
leave-one-load winner counts: {'source_12khz_fe': 4}
selection mode: single_fault_subgroup_plus_all_normals
MVP groups: ['source_12khz_fe', 'source_48khz_normal']; files=49
alternative groups: ['source_12khz_fe', 'source_12khz_de', 'source_48khz_normal']; files=109

目标A-P只以无标签信号参与组级分布可比性分析，不使用任何历史/预测类别。
共同分析口径为12kHz、0-6kHz。幅值差单独报告，不参与筛选距离；目标约600rpm只报告域差异，不作精确阶次归一化。
正常类只有4个独立文件，所有可行方案均强制全部保留。
