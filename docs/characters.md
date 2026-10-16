# 多角色与训练任务管理

打开 `http://127.0.0.1:9872/`，页面顶部选择角色、训练任务，下方选择 GPT 和 SoVITS 权重，再点击 **加载所选模型**。角色数量、任务数量由配置登记，添加角色无需改代码。当前使用已验证的 v2Pro 架构；不同架构的权重仍需对应运行环境与适配，不能混装。

心铃保留原数据、校对、全部 GPT 1–10e 和 SoVITS 1–20e，当前加载 20e/10e。蓝的 `ai_gsv_v2pro_v1` 已按用户授权采用现有标注，排除1条空转写后完成83条数据的 SoVITS20轮/GPT10轮训练。雫、禀、圭、里奈、优美、真琴也已按用户授权采用现有转写，排除6条空文本后以387条数据完成20/10训练，各自保存默认模型选择、参考音频与3条中日文样音；详见 [六角色训练记录](experiments/2026-10-16-remaining-characters.md)。无配套模型的任务会显示未训练并禁止合成，不会借用心铃模型。

## 网页操作

1. 展开“管理角色、数据与参考音频”。输入唯一角色 ID、显示名称、素材语种，点击“登记新角色”。ID 使用小写英文开头，可含数字、下划线和短横线。
2. 同一角色要改变素材、分割参数或做独立实验，填写新任务后缀，点击“为当前角色新建任务”。旧任务全部保留，新任务从空数据和空模型目录开始。
3. 上传选好的原音声，选择训练或留出划分，点击“保存源素材到当前任务”。素材复制进角色原素材库，以哈希去重；留出以源文件划分，避免同一文件的兄弟切片泄漏。登记角色不会自动把整个素材库加入训练。
4. 已有模型可同时上传 `.ckpt` 和推理 `.pth`，点击“导入配套模型”，再加载。支持当前 v2Pro 导出格式和不同文件名；续训 G/D、Lightning 状态以及不兼容模型会拒绝。已有同名不同内容文件不会被覆盖。
5. 下方填写 3–10 秒参考音频、准确台词及语种，点击管理区的保存参考按钮。下次切回该任务会自动填入，并清空前一个角色的辅参考音频。当前保存参考支持日、中、英、韩、粤语。
6. 训练完成或外部添加权重后，点“刷新模型路径”；外部登记角色后，点“刷新角色列表”。GPT、SoVITS 列表只扫描当前任务，按识别到的 epoch 数字排序；无法识别轮数的文件显示为导入模型。

模型选择改变后需要显式加载。加载和合成共享模型锁；加载失败时停止合成，避免半切换的模型继续生成。每个任务记住上次成功加载的两份权重，重启后使用有效保存选择，失效文件则重新选择配置轮次或最新文件。界面在一块 GPU 上轮流加载角色，角色数量不意味着所有模型同时驻留显存。

## 配置与目录

`config/characters.json` 是角色目录，保存显示名称、原素材目录、当前任务、各任务配置入口和参考信息。`config/task_template.json` 是新任务模板，角色不复制其他角色的数据或模型。

| 内容 | 每个角色/任务的位置 |
|---|---|
| 新任务配置 | `config/characters/<角色ID>/<任务后缀>.json` |
| 全量原素材 | `data/source_library/<角色素材目录>/` |
| 选材索引 | `data/manifests/<任务ID>.json` |
| 切片 | `data/datasets/<任务ID>/wavs/` |
| 人工校对 | `data/datasets/<任务ID>/review.csv` |
| 特征、续训状态、日志 | `runs/<任务ID>/` |
| 推理权重 | `models/<任务ID>/{gpt,sovits}/` |
| 脚本生成音频 | `outputs/<任务ID>/` |
| 保存的参考音频 | `data/references/<角色ID>/` |

心铃原任务继续使用 `config/task.json`；其他任务统一使用 `<角色ID>_gsv_v2pro_v1`。标注文件 review.csv 与训练文本 train.list 统一放在各自数据集目录，心铃已迁移并保持字节一致；音频、特征和模型未变动。网页合成沿用官方临时缓存和下载按钮，需手动下载要保留的结果。管理界面不会自动启动训练。

## 新角色训练

在项目根目录使用同一个独立 Python 环境。以下以雫第一任务为例；换成其他角色 ID 和任务 ID 即可。现有首版任务均已完成20/10训练；以下完整命令供新任务参考，已有任务的目标权重会复用。训练前双击 Stop-WebUI 释放推理显存，训练完成后重新打开 Start-Inference。

```powershell
$python = 'D:\Project\voice\vendor\GPT-SoVITS\.venv\Scripts\python.exe'
& $python scripts/character_manager.py list
# 本地文件可直接登记选材，无需通过网页复制上传
& $python scripts/character_manager.py sources shizuku 'D:\Project\voice\data\source_library\雫\选好的音声.wav' --task shizuku_gsv_v2pro_v1
& $python scripts/tts_finetune.py doctor --character shizuku --task shizuku_gsv_v2pro_v1
& $python scripts/tts_finetune.py prepare --character shizuku --task shizuku_gsv_v2pro_v1
& $python scripts/tts_finetune.py transcribe --character shizuku --task shizuku_gsv_v2pro_v1
# 人工校对 data/datasets/shizuku_gsv_v2pro_v1/review.csv，确认 text/language 并设置 reviewed=true
& $python scripts/tts_finetune.py audit --character shizuku --task shizuku_gsv_v2pro_v1
& $python scripts/tts_finetune.py features --character shizuku --task shizuku_gsv_v2pro_v1
& $python scripts/tts_finetune.py train-sovits --character shizuku --task shizuku_gsv_v2pro_v1
& $python scripts/tts_finetune.py train-gpt --character shizuku --task shizuku_gsv_v2pro_v1
```

新任务的训练总轮数、batch size、切片时长、种子可在各自 JSON 配置中调整。已准备数据的任务不允许再次添加素材；新建任务后重新选材。所有训练命令支持 `--config`，也支持角色/任务解析，仍保留人工校对门槛。命令行 `tts_sample.py --character <ID> --task <任务ID>` 使用该任务的模型选择和已保存参考，未保存参考时从该任务已校对切片选取。

命令行登记也支持任意角色：

```powershell
& $python scripts/character_manager.py add new_voice '新角色名称' --language ja
& $python scripts/character_manager.py new-task new_voice alternate
```

同一角色语言作为新任务默认语言，任务实际语言从任务配置读取。音声、权重和运行记录留在本机；Git 管理登记配置、模板、代码和文档。

本次验证：13 项离线检查通过，包含 5 个角色 × 2 个任务的路径隔离、跨角色/任务权重拒绝、排序、选择持久化与防覆盖；真实 UI API 完成 10e/5e → 20e/10e 切换和中文生成，未训练雫合成被拒绝；现有 v2Pro 推理导出格式可导入。
