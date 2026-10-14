# 多角色本地配音 · GPT-SoVITS v2Pro

项目根目录：`D:\Project\voice`。当前采用本地GPT-SoVITS，SoVITS20轮 + GPT10轮，直接生成中文和日语语音。

## 网页界面

- 双击 **[Start-Inference.cmd](Start-Inference.cmd)**：带角色管理的官方快速推理界面，浏览器打开 `http://127.0.0.1:9872`。选择角色、任务与两份权重，点击“加载所选模型”。当前加载心铃20/10，后续记住每个任务的模型选择。
- 双击 **[Start-WebUI.cmd](Start-WebUI.cmd)**：官方训练与数据处理主界面，`http://127.0.0.1:9874`。
- 用完在对应终端按Ctrl+C关闭。两个界面使用同一独立Python环境，不必打开Jupyter。

参考音频：`data/datasets/misuzu_gsv_v2pro_v1/wavs/fem_mis_00025_000.wav`；参考语种选日文，文本填 `校長先生からは許可書を頂いてますが`。目标语种选中文，输入文稿并合成。参数与局部语速调整见 [界面说明](docs/inference.md)。

展开页面“管理角色、数据与参考音频”可登记任意数量的角色，每个角色支持多个独立训练任务、模型导入和参考保存。心铃数据与全部权重保留；雫已登记且未训练。操作与训练命令见 [多角色管理](docs/characters.md)。

## 目录

| 目录 | 内容 |
|---|---|
| `data/source_library/` | 全部原角色音声及既有选材副本，供重新分割、标注 |
| `data/manifests/` | 当前18段训练源、5段留出源的索引与哈希 |
| `data/datasets/` | 当前64条32kHz训练切片 |
| `models/` | SoVITS1–20轮、GPT1–10轮推理权重 |
| `runs/` | 人工校对、草稿和编码修复备份、特征、训练状态、日志、迁移核验 |
| `outputs/` | 短句、轮次对比、长文、自动转写和最终音频 |
| `vendor/GPT-SoVITS/` | 官方克隆、固定提交、独立环境、预训练模型及Windows适配 |
| `config/` | 本地任务、模型下载revision、依赖锁、文稿与响度配置 |
| `texts/` | 原稿、口播稿、《序》短视频稿 |
| `scripts/` | 安装、校对审计、训练、推理、WebUI、文稿提取、音频整理 |
| `notebooks/` | 当前训练入口；`history/`完整保留原已执行notebook |
| `docs/` | 当前操作指南、GPT阶段实验与训练历史 |

## 命令行

在项目根目录运行，以下命令使用现有独立环境：

```powershell
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_webui.py
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py status
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py train-sovits --epochs 20
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe scripts\tts_finetune.py train-gpt --epochs 10
```

轮数是总上限；目标权重存在时复用。训练步骤及从头重建见 [训练与数据](docs/training.md)，环境安装与维护见 [运行环境](docs/setup.md)。

原始音声全部保留；旧远端配音、密钥环境及旧转换工作已退役。Git从本轮整理重新开始，只管理代码、配置、文稿和文档；音频、权重、特征、运行记录及内嵌音频的历史notebook均留在本机，不提交。

官方仓库：[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。当前固定提交 `1b47ca07ef43a284e0b8bd7f29d3e1e3ff76082a`，保留验证过的v2Pro环境。
