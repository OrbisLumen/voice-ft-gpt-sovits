# 角色首版数据集准备（2026-10-15）

后续进展：蓝已按用户授权完成83条数据的20/10训练，见 [训练记录](experiments/2026-10-15-ai.md)。2026-10-16，其余六角色也按用户授权接受现有转写，排除6条空文本后以387条数据完成20/10训练及18条样音检查，见 [六角色训练记录](experiments/2026-10-16-remaining-characters.md)。以下保留首版准备时的流程说明；数据检查页现已反映当前接受状态。

8个角色的统一入口：`data/datasets/index.html`。禀、圭、蓝、里奈、雫、优美、真琴已建立与心铃平行的 `<角色ID>_gsv_v2pro_v1` 任务。

每个新角色按心铃规模，自动筛选18段训练源和5段独立留出源。候选源要求8–20秒、数值有效、达到最低信号强度、削波比例低于0.5%，按文件名区间分散取材并优先选择约14秒的源。源文件按SHA256去重，同一源不会跨训练和留出集。原素材全部保留，不复制整个原音库进训练集。

雫的 `szd`、`kzd` 音库已由用户确认属于同一角色、等价使用，两者共同进入筛选范围。仅使用 `szd` 的中间草稿保存在 `data/datasets/_superseded/` 和 `runs/_superseded/`，不参与当前训练。

自动声学筛选尚未经过人工听音；它不能判定所有背景音乐、重叠人声、角色语气及非语言声音。源清单的 `selection_method` 记录自动筛选来源；其中 `user_selected=true` 是现有 prepare 阶段选入训练的技术字段，不代表用户已经试听认可。

切片遵循心铃现有规则：32kHz单声道PCM16，按静音分割，1–15秒的有效切片进入标注表。使用本地 faster-whisper-small、日语、beam5 生成标注草稿；空结果会再尝试不使用VAD识别，并在检查页面提示。

| 文件 | 用途 |
|---|---|
| `data/datasets/<任务ID>/wavs/` | 训练候选切片 |
| `data/datasets/<任务ID>/review.csv` | 唯一的人工校对输入，UTF-8 BOM，方便表格软件打开日文 |
| `data/datasets/<任务ID>/review.html` | 带音频播放、文本和检查提示的只读页面 |
| `data/datasets/<任务ID>/review_flags.json` | 空转写、较低识别置信度等检查提示 |
| `data/datasets/<任务ID>/train.list` | 全部人工确认后，由 audit 生成的正式训练文本 |
| `data/manifests/<任务ID>.json` | 训练源、留出源、哈希和自动筛选记录 |
| `runs/<任务ID>/dataset_manifest.json` | 切片时间位置、时长、哈希及是否满足时长要求 |
| `runs/<任务ID>/asr_drafts/` | 原始自动转写与置信度 |
| `runs/<任务ID>/train_sovits.json`、`train_gpt.yaml` | 已生成的训练配置；未启动训练 |
| `notebooks/finetune_<角色ID>_tts.ipynb` | 各角色独立操作入口；所有阶段开关默认关闭 |

检查时在 review.csv 修改 `text`，逐条试听确认后才设置 `reviewed=true`。保持 `name`、`language`（ja）、`audio_path` 不变。不要把所有草稿直接批量标为已确认。无法辨认或声线不符的片段先保持 false，重新选材需另建任务。

修改CSV后刷新只读页面：

```powershell
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts/prepare_character_datasets.py --pages-only
```

本次没有提取新角色训练特征，也没有启动GPT或SoVITS训练。人工确认之前，audit、features 和两个训练入口均受 reviewed 检查约束。训练参数沿用心铃：SoVITS20轮/GPT10轮、batch2、半精度、梯度检查点。

心铃 review.csv、train.list 已从 runs 迁入自身数据集目录，逐字节核对一致；原有64条人工校对状态、音频、训练特征与模型保留。当前源索引中旧英文目录已按音频SHA256映射到中文素材目录。

最终各角色条数、时长、空转写和提示数量见 `data/datasets/preparation_summary.json`；数据完整性检查见 `runs/dataset_preparation_verification.json`。数据、标注、权重和日志继续仅存本机，Git不收录这些文件。
