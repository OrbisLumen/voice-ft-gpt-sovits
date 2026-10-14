# SoVITS20 / GPT10：序与PicaPica长文本试听

本轮只进行推理，使用已训练的SoVITS20轮和GPT10轮，检查实际视频文案的连贯性、停顿及较长文本中的读音稳定性。

## 文稿

- 《序》：沿用 `texts/序_短视频版.txt`，88字符，包含剧透提示。
- 第一节：`texts/口播稿.md` 中PicaPica的完整正文，362字符，包含中文副标题与引文；外文章标题不朗读。
- 沿用现有 `config/pipeline.json` 的章节提取规则，合并排版换行、排除Markdown标题与制作注释，正文没有额外改写。
- 输入及原稿SHA256记录在 `runs/misuzu_gsv_v2pro_v1/narration_inputs/manifest.json`。输入纯文本保存在同目录，也复制到输出目录供逐句核对。

## 模型与推理

沿用上一轮参考片段 `fem_mis_00025_000.wav` 及日语参考台词；中文模式all_zh，seed=20261012、top_k=15、top_p=1、temperature=1、speed=1、batch1、cut5、串行推理。cut5按标点切分，再由上游恢复原文顺序拼成整篇；片段间隔为上游默认0.3秒。

原始32kHz单声道PCM16、实际朗读文本和推理manifest保存在 `outputs/misuzu_gsv_v2pro_v1/narration_s20_g10/`。用于试听的版本在其 `final/` 子目录，统一-22 LUFS、48kHz、单声道PCM16。完整推理日志在 `runs/misuzu_gsv_v2pro_v1/logs/narration_s20_g10.log`。

## 重现

新增 `scripts/tts_sample.py --text-files` 可直接读取UTF-8纯文本进行中文长文推理，输出独立于短句对照目录。

```powershell
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts/tts_sample.py --sovits-epoch 20 --gpt-epoch 10 --text-files runs/misuzu_gsv_v2pro_v1/narration_inputs/01_intro.txt runs/misuzu_gsv_v2pro_v1/narration_inputs/02_PicaPica.txt
```

此命令会重建对应长文样音。需试听是否漏字、重复、人名读音不稳、逗号停顿过密，以及首尾音色是否一致。

## 完成结果与检查

- 《序》：`final/01_intro.wav`，17.14秒。
- PicaPica完整正文：`final/02_PicaPica.wav`，72.34秒。
- 终端推理及后处理均退出码0。原始音频检查32kHz单声道、有限值及非静音；最终音频检查48kHz、有限值、非静音及整理前后时长一致。模型、参考与推理控制参数和上一轮20/10短句试听相同。
- `final_manifest.json`保存模型哈希、原始及最终音频哈希、实际朗读文本与时长。
- 本地faster-whisper-small生成 `asr_check.json`，自动转写覆盖了两段的开头和结尾，但人名、作品名及部分词语有较多同音字或近音混淆。自动转写仅用于辅助人工核对，不证明所有字词准确，也不能区分口音读音偏差和识别错误。
- 本轮cut5将《序》和PicaPica分别切为7和31个片段，分桶后由上游恢复正文顺序。PicaPica日志记录前端约1秒、语义预测约12秒、音频解码约394秒，当前主要耗时在音频解码。

后处理及自动检查程序保留在本机 `runs/misuzu_gsv_v2pro_v1/check_narration.py`，日志为同任务 `logs/check_narration_s20_g10.log`。本次没有调整模型或参考语气，也未人工确认新长文的质量优劣。

## 2026-10-14目录整理

以上为本次实验的历史记录；当前默认模型为SoVITS20/GPT10，任务配置在config/task.json。目录已更新，权重现存models/，运行记录在runs/，试听在outputs/。
