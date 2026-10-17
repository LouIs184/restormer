# 文档索引

| 文档 | 内容 |
|------|------|
| [1-RUN_GUIDE.md](1-RUN_GUIDE.md) | 完整运行指南：图像域/投影域选择、数据准备、环境、训练、推理、常见问题 |
| [2-EXECUTION.md](2-EXECUTION.md) | 执行文档：命令优先，从装环境到开训、监控、续训、推理（图像域 + 投影域） |
| [3-TRAINING_LOGISTICS.md](3-TRAINING_LOGISTICS.md) | 训练后勤：产物位置、TensorBoard、监控、断点续训、异常处理 |
| [4-CONFIG_PARAMS.md](4-CONFIG_PARAMS.md) | `Options/CT_ImageDomain_Restormer.yml` 与 `Options/CT_ProjectionDomain_Restormer.yml` 逐字段参数说明与设计理由 |
| [5-PROJECTION_DOMAIN.md](5-PROJECTION_DOMAIN.md) | 投影域（post-log sinogram）训练适配方案：方案 B（按视角）+ `domain` 域切换 + 代码/配置细节 |
| [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) | 适配西门子 PBI 投影数据（`ximenzi_pbi_data`）：形状参数化改动、裁剪量实测依据、`clip_max` 标定流程 |
| [7-TROUBLESHOOTING.md](7-TROUBLESHOOTING.md) | 排错手册：通用分层排查法 + 实战案例（`No module named 'basicsr'`、cuDNN 冲突导致 illegal memory access、`gt_size` 越界等）+ 快速对照表 |
