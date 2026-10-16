# SoVITS20 / GPT10：序与 PicaPica 长文本试听

本实验使用心铃的 SoVITS20轮和 GPT10轮权重进行中文长文推理，检查连贯性、停顿、读音及首尾音色。只使用已有模型，不涉及训练。

2026-10-16更新：朗读文本统一从 `texts/narration_v1.md` 提取，《序》不再使用独立短版文件。下方复现步骤使用当前文稿；2026-10-14生成的音频保留为历史试听结果，本次文档更新没有重新合成。

## 当前文稿

文本来源为 [narration_v1.md（口播稿第一版）](../../texts/narration_v1.md)，原稿保存在 [original.md](../../texts/original.md)。提取配置为 [config/narration.json](../../config/narration.json)。`chapter_overrides` 为空，直接读取 Markdown 二级章节标题下的正文。后续口播稿可按 `narration_v2.md`、`narration_v3.md` 继续编号；切换用于推理的版本时，同步修改配置中的 `source`。

| 章节 | 来源 | 提取后字符数 | 推理输入 |
|---|---|---:|---|
| 序 | `## 序` 下的完整正文 | 95 | `01_intro.txt` |
| PicaPica | `## PicaPica` 下的完整正文 | 362 | `02_PicaPica.txt` |

字符数包含标点，不含排版换行。提取时移除制作注释，不朗读 Markdown 标题，合并正文换行；中文副标题与引文保留，正文不额外改写。

当前《序》的朗读内容为：

> 如是一切是我中的众生，故也是众生各自中的一切。
>
> 这句来自《春与修罗·序》的句子，成为了樱之诗的独特意象。
>
> 这个视频里，我想聊聊《樱之诗》中，那些关于天才、艺术与陪伴的故事。注意以下内容涉及剧透。

## 模型与推理参数

任务配置为 `config/task.json`，任务 ID 为 `misuzu_gsv_v2pro_v1`。使用该任务的 SoVITS20 / GPT10 权重，参考音频为 `data/datasets/misuzu_gsv_v2pro_v1/wavs/fem_mis_00025_000.wav`，日语参考台词为 `校長先生からは許可書を頂いてますが`。

推理参数：中文模式 `all_zh`、seed=20261012、top_k=15、top_p=1、temperature=1、speed=1、batch1、cut5、串行推理。cut5 按标点切分，上游按正文顺序拼接，片段间隔使用默认0.3秒。

## 使用当前文稿复现

在项目根目录执行，先提取《序》和 PicaPica，再生成长文语音：

```powershell
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts/prepare_narration.py --chapters 1 2
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts/tts_sample.py --sovits-epoch 20 --gpt-epoch 10 --text-files runs/misuzu_gsv_v2pro_v1/narration_inputs/01_intro.txt runs/misuzu_gsv_v2pro_v1/narration_inputs/02_PicaPica.txt
```

第一步将当前正文、字符数及源文件 SHA256 写入 `runs/misuzu_gsv_v2pro_v1/narration_inputs/`。第二步读取这两份 UTF-8 纯文本，输出到 `outputs/misuzu_gsv_v2pro_v1/narration_s20_g10/`，保存32kHz单声道 PCM16 音频、实际朗读文本和推理 `manifest.json`。

上述命令会覆盖对应输入快照和原始长文输出，需要保留此前实验时应先备份这两个目录。命令不会更新已有 `final/` 后处理音频、`final_manifest.json` 或 `asr_check.json`；这些文件须在新音频生成后另行重建。

试听时重点核对漏字、重复、人名及作品名读音、逗号停顿密度，以及首尾音色是否一致。新《序》的时长和切分数量应以重跑结果为准。

## 2026-10-14历史结果

当时的《序》采用88字符短版，PicaPica 为362字符完整正文。历史输入快照位于 `runs/misuzu_gsv_v2pro_v1/narration_inputs/`，输出目录也保存了当时实际朗读的文本；重跑前仍可据此核对旧音频。

| 内容 | 历史试听文件（相对输出目录） | 时长 | cut5片段数 |
|---|---|---:|---:|
| 序（88字符短版） | `final/01_intro.wav` | 17.14秒 | 7 |
| PicaPica完整正文 | `final/02_PicaPica.wav` | 72.34秒 | 31 |

输出目录为 `outputs/misuzu_gsv_v2pro_v1/narration_s20_g10/`。历史试听版本经后处理统一为 -22 LUFS、48kHz、单声道 PCM16。旧《序》的17.14秒结果对应当时的短版文本，不能作为当前95字符正文的生成结果。

历史检查记录：

- 推理及后处理均退出码0。原始音频检查通过32kHz单声道、有限值及非静音；最终音频检查通过48kHz、有限值、非静音，以及整理前后时长一致。
- `final_manifest.json` 保存模型、原始及最终音频哈希、实际朗读文本与时长。
- `asr_check.json` 为本地 faster-whisper-small 自动转写。两段开头和结尾均有覆盖，人名、作品名及部分词语存在同音字或近音混淆；该检查仅辅助人工核对，不能证明逐字准确。
- PicaPica 日志记录前端约1秒、语义预测约12秒、音频解码约394秒，主要耗时在音频解码。

推理日志为 `runs/misuzu_gsv_v2pro_v1/logs/narration_s20_g10.log`；历史后处理及自动检查程序为同任务的 `check_narration.py`，日志为 `logs/check_narration_s20_g10.log`。本实验没有调整模型或参考语气，也未完成人工试听质量评价。
