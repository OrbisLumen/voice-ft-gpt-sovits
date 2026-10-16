# 其余六角色 · SoVITS20 / GPT10（2026-10-16）

按用户“把其余角色也按 20 + 10 e 跑完”的要求，沿用蓝的流程接受现有自动转写用于训练；这不代表逐句人工听写校对。6条空转写已从训练候选排除，原音频保留。修改前的 review.csv、dataset_manifest.json 保存在各任务 runs 目录的 before_training_acceptance_20261016/。

开始：2026-10-16T10:29:05.564332+08:00；全部完成：2026-10-16T11:58:19.287380+08:00。

使用固定 v2Pro 环境，batch2、半精度、梯度检查点、GPT workers0，串行训练。每个角色均保留 SoVITS 1–20e 与 GPT 1–10e 的推理权重；20/10 默认选择与参考音频已保存。

| 角色 | 任务 | 训练切片 | 秒数 | 排除空转写 | SoVITS / GPT | 样音 |
|---|---|---:|---:|---:|---|---:|
| 夏目雫 | `shizuku_gsv_v2pro_v1` | 65 | 225.46 | 1 | 20 / 10 | 3 |
| 禀 | `rin_gsv_v2pro_v1` | 68 | 218.26 | 0 | 20 / 10 | 3 |
| 圭 | `kei_gsv_v2pro_v1` | 58 | 220.37 | 0 | 20 / 10 | 3 |
| 里奈 | `rina_gsv_v2pro_v1` | 71 | 210.48 | 0 | 20 / 10 | 3 |
| 优美 | `yuumi_gsv_v2pro_v1` | 62 | 226.50 | 1 | 20 / 10 | 3 |
| 真琴 | `makoto_gsv_v2pro_v1` | 63 | 199.17 | 4 | 20 / 10 | 3 |

每个角色的最终权重均实际加载，生成日文日常句、中文日常句和中文问句。共18条样音，均为32kHz、数值有限且非静音；这验证推理链路与文件有效性，音色及发音尚待人工试听。

- 权重：`models/<任务ID>/{sovits,gpt}/`。
- 样音：`outputs/<任务ID>/samples_s20_g10/`，含3个WAV及 manifest.json。
- 每个角色的结果、哈希及参考：`runs/<任务ID>/training_result.json`。
- 接受现有转写的记录：`runs/<任务ID>/training_acceptance.json`；被排除行见对应数据集的 excluded_review.csv。
- 整批状态：`runs/remaining_20_10_status.json`；日志：`runs/remaining_20_10.log`，各阶段详细日志在各任务 logs/。
- 统一核验：`runs/remaining_20_10_verification.json`，覆盖全部轮次导出、最终权重及样音SHA256、训练音频审计、标注备份和默认模型选择。

数据检查入口 `data/datasets/index.html` 已刷新。推理界面刷新角色和模型列表后，选择对应角色并加载20/10即可。
