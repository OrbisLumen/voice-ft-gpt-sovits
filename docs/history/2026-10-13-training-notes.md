# Historical training notes (2026-10-13)

Historical budgets and observations; use ../training.md for current commands.

# 心铃本地 TTS 微调任务

## 2026-10-13 实测进度

Jupyter续训已完成SoVITS10轮（640个batch）并保存；随后GPT恢复第一轮检查点时遇到PyTorch weights_only加载拒绝pathlib.WindowsPath。setup适配为仅在trainer.fit恢复期间允许标准Path实际类型，并显式保留weights_only=True；未关闭安全加载、未重训或删除检查点。真实恢复检查已加载模型、优化器及调度器，日志含Restored all states。重新从notebook参数单元格Run All即可跳过已完成SoVITS并继续GPT到5轮；无需重复特征提取。

朋友校对的64条日语台词已导入；GBK文件转为UTF-8，修复一处中点编码丢失，保留三个名字修正。原件和原草稿在work/encoding_backup_20261013，详情在encoding_repair_20261013.json。

64组音素、HuBERT、说话人嵌入、32k音频及语义token已提取，未遗漏切片；真实特征和音频通过有限数值检查。SoVITS与GPT各完成1轮、64个训练batch，均加载官方预训练权重并完成GPU反向传播、优化器更新及保存。在本机batch2可运行，首轮并非质量达标证明。

GPT的Windows单卡DDP初始化曾发生原生崩溃，已改为仅多GPU启用DDP，采样器在未初始化分布式时使用rank0/world1；num_workers0配套关闭persistent_workers/prefetch。适配已纳入setup，原上游文件有备份。

一轮检查权重位于outputs/misuzu_gsv_v2pro_v1：SoVITS `sovits/misuzu_gsv_v2pro_v1_e1_s64.pth`，GPT `gpt/misuzu_gsv_v2pro_v1-e1.ckpt`。训练日志在work/logs；这次不再对未校对标签训练。

按用户要求，后续交给Jupyter挂机。Notebook已配置RUN_PREPARE、RUN_ASR、RUN_FEATURES为False，RUN_TRAIN、RUN_SAMPLE为True；总上限为SoVITS10轮、GPT5轮，已有检查点会续训。训练完成后加载明确的10/5轮权重并生成短句，分别评价音色、语气、中文清晰度。不要同时运行其他GPU训练。

一轮实际推理已完成日语日常、中文日常与中文疑问三条短句，分别3.72、3.24、2.68秒，均为32k PCM16有效非静音音频。这里只验证生成链路，不宣称内容完整、角色相似度或语气质量达标。结果与推理参数见 `outputs/misuzu_gsv_v2pro_v1/samples_s1_g1/manifest.json`。

推理补齐resampy、官方G2PW中文前端与公开FastText语言识别模型，全部本地加载。下载入口：`python scripts/setup_tts.py --models-only --chinese-frontend`。运行时FFmpeg路径适配为本仓库已有的ffmpeg.exe。

一轮推理检查：`python scripts/tts_sample.py`；指定正式训练版本：`python scripts/tts_sample.py --sovits-epoch 10 --gpt-epoch 5`。输出samples_s1_g1或samples_s10_g5，manifest记录输入文本、参考音频、参考台词、所用权重和音频有效性。脚本不默默回退到底模。

## 选择与目标


本机为Windows、RTX 5060 Laptop、7.96 GiB显存，现有Python 3.12。选择v2Pro而非直接全量微调1.7B语音模型：官方提供Windows流程和中日跨语言合成，v2Pro保持v2级别的硬件成本，适合先验证小批量任务。音色和语气能否同时达到心铃目标仍需生成音频盲测，不能由架构名称推定。


## 首轮素材与标签

采用既有A/B素材清单中用户选定的18段心铃日语。保持原文件与哈希，用静音切分产生32kHz单声道PCM16候选，1–15秒片段进入标签校对；较长/过短片段保留但不进入首轮训练。未做变调、语气处理或额外降噪。

首轮切出64段候选，符合条件的有效音频约203.27秒。源文件与切片哈希、源内时间位置、排除片段和时长记在 `runs/misuzu_gsv_v2pro_v1/dataset_manifest.json`。既有5段心铃holdout按原录音级别保留在清单中，不将同一原录音的兄弟切片随机拆入验证集。

Whisper small在本地按日语、beam5转写，输出 `runs/misuzu_gsv_v2pro_v1/review.csv`。这是草稿，不能视为准确原作台词；名字、语气词、喘息/笑声和背景声尤其需要人工试听校正。

CSV列：`name,language,text,reviewed,audio_path`。保留name及language=ja；修改text，确认准确后将reviewed设为true。审计要求每个有效片段恰好出现一次、文本非空、音频哈希不变、全部已校对，之后才生成上游格式 `绝对音频路径|说话人|ja|文本` 的 `train.list`。音频并非中文，不为了生成中文而将训练语言改成zh。

## 独立环境与本机适配

- 上游与运行时：`vendor/GPT-SoVITS/`、独立 `.venv/Scripts/python.exe`。
- PyTorch / torchaudio 2.11.0、CUDA 12.8；CUDA卷积反向传播与日语音素处理已实测通过。正式TTS训练的显存与收敛尚需验证。
- 官方pyopenjtalk需要本机没有的C++编译器。改用PyPI预编译 `pyopenjtalk-plus`，仍导入pyopenjtalk；验证过日语g2p。参见[维护者说明](https://github.com/taihei-05/pyopenjtalk-plus)。
- torchaudio 2.11的音频读取需要额外TorchCodec；本任务的本地参考WAV在特征提取和推理入口用SoundFile读取为同形状tensor。Whisper直接接收16k波形，避开PyAV接口变更；适配由setup重复应用，原代码备份到work。
- 实际安装依赖锁定在 `config/requirements-windows.lock.txt`，本机完整freeze另存在work。安装器复用锁文件。

## Notebook入口

打开 `notebooks/finetune_misuzu_tts.ipynb`，选择Python (voice)。它用独立TTS Python执行计算，日志保存在work/tts下。本机已完成校对与准备，默认只做环境检查、续训和合成；Run → Run All Cells即可。插电并关闭自动睡眠，保持Jupyter服务与内核运行。运行结束可在notebook播放音频。

若在新的工作区从头准备数据，先关闭RUN_TRAIN、RUN_SAMPLE，并按下方步骤开启准备与转写。

1. 完成环境检查、切分与转写草稿。
2. 试听并校对review.csv的每条日语台词，标记reviewed=true。
3. 开启RUN_FEATURES，提取音素、HuBERT、32k音频、说话人嵌入和语义token；核对所有切片均成功产出，避免上游忽略单文件异常。

命令行同等入口（项目根目录、voice环境）：

```powershell
python scripts/setup_tts.py                 # 首次联网安装与下载
python scripts/tts_finetune.py doctor
python scripts/tts_finetune.py prepare
python scripts/tts_finetune.py transcribe
# 先校对runs/misuzu_gsv_v2pro_v1/review.csv
python scripts/tts_finetune.py audit
python scripts/tts_finetune.py features
python scripts/tts_finetune.py train-sovits
python scripts/tts_finetune.py train-gpt
python scripts/tts_finetune.py status
```

如果只缺权重：`python scripts/setup_tts.py --models-only`；转写小检查可用 `transcribe --limit 3`。重复转写复用已生成草稿，保留CSV里的人工编辑。阶段具有互斥锁；中断会结束自己启动的子进程树。

## 微调预算与产物

初始参数：SoVITS batch2、10轮、fp16、梯度检查点；GPT batch2、5轮、16-mixed、num_workers0；每轮保存最新续训检查点与推理权重。两阶段依次执行，属于微调而非从零预训练。8GB能否承受该任务需实际反向传播验证，发生OOM时优先减少batch，不能据CUDA小测试断言完整训练可运行。



## 验证标准


2026-10-12构建阶段准备了环境、数据、草稿校对与训练入口；2026-10-13标签校对与两阶段一轮短训已完成，后续正式微调和质量试听仍需进行。运行记录见work，Git只管理代码、配置、依赖锁和文档；权重、音频和含完整台词的标签保存在本机忽略目录。

## Git阶段
