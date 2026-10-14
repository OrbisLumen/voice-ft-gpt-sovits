# 数据与训练

当前任务 `misuzu_gsv_v2pro_v1`：18段人工挑选的日语原音，64条有效32kHz单声道切片，约203.27秒，全部完成校对。5段源级留出音频没有加入训练。

全部原音库保留在 `data/source_library/`，含心铃、雫及旧选材副本；这些是后续分割与标注素材，旧选材train/holdout目录不是新任务必须沿用的划分。

## 当前文件

- 源索引：`data/manifests/misuzu_sources.json`，记录当前18段训练源、5段留出源、选择标记及音频SHA256。
- 切片：`data/datasets/misuzu_gsv_v2pro_v1/wavs/`。
- 人工校对：`runs/misuzu_gsv_v2pro_v1/review.csv`；日语language=ja，64条reviewed=true。
- 原转写、朋友编码错误的原件、修复与审计记录：同任务的 `asr_drafts/`、`encoding_backup_20261013/`、`encoding_repair_20261013.json` 等。
- 特征：同任务 `features/`，含音素、中文HuBERT、32k音频、语义token、说话人嵌入。已核对64条无遗漏及有限数值。
- 续训状态：`features/logs_s2_v2Pro/` 的SoVITS G/D，`logs_s1_v2Pro/ckpt/epoch=9-step=150.ckpt` 的GPT完整状态。
- 推理权重：`models/misuzu_gsv_v2pro_v1/sovits/` 与 `gpt/`，保留SoVITS1–20轮与GPT1–10轮。

## 阶段命令

```powershell
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py doctor
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py prepare
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py audit
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py train-sovits --epochs 20
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py train-gpt --epochs 10
```

总轮数是上限，目标权重已存在时复用；本次整理检查确认20/10都复用原权重，没有额外训练。特征、音频哈希及校对状态仍受检查，未校对标签不能进入训练。

新增素材应另建任务名与源索引，不改当前任务的数据选择后直接续训。新任务依次prepare、transcribe，人工校对review.csv，再audit、features、train-sovits、train-gpt。使用源级划分，避免同一原音的相邻切片跨训练和留出集。

## Notebook与历史

当前 `notebooks/finetune_misuzu_tts.ipynb` 默认20/10，准备、转写与特征提取关闭，训练与试听开启。使用Python (voice)内核，计算交给独立GPT环境。

原已执行notebook及完整输出原样保存在 `notebooks/history/finetune_before_reorganization_20261014.ipynb`。历史日志和二进制检查点中的旧绝对路径保留作历史记录；当前配置、校对、train.list及特征清单已迁移，路径对应关系见 `runs/misuzu_gsv_v2pro_v1/maintenance/migration_manifest.json`。

当前参数batch2、半精度、梯度检查点、GPT workers0。RTX5060 Laptop约8GB环境已完成真实两阶段训练，后续扩大数据或更改batch时仍需观察显存与试听结果。
