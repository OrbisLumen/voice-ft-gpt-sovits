# 官方仓库与运行环境

本机重新克隆官方 `https://github.com/RVC-Boss/GPT-SoVITS.git` 到 `vendor/GPT-SoVITS/`，固定提交 `1b47ca07ef43a284e0b8bd7f29d3e1e3ff76082a`。复用已验证的独立Python3.12环境、预训练模型与FFmpeg；没有重新下载或重训推理权重。

## 当前状态

- Python：`vendor/GPT-SoVITS/.venv/Scripts/python.exe`。
- PyTorch / torchaudio 2.11.0+cu128；RTX5060 Laptop CUDA卷积反向传播、日语音素处理已通过。
- 完整依赖锁：`config/requirements-windows.lock.txt`；预训练模型revision：`config/model_sources.json`。
- WebUI为Gradio4.44.1 / gradio_client1.3.0。固定FastAPI0.115.6 / Starlette0.41.3；原Starlette1.7.0删除了旧TemplateResponse调用约定，造成首页500和误导性的localhost不可访问报错。固定版本后首页与合成API实测正常，pip check通过。

## 重建

使用Python3.12在根目录运行 `python scripts/setup_tts.py`。已有环境使用对应独立Python执行即可。`--models-only`只下载缺失权重，`--models-only --chinese-frontend`安装中文G2PW与语言识别模型。

FFmpeg和FFprobe需在系统PATH，或放入 `vendor/GPT-SoVITS/`；本机两个exe已保留并验证音频整理正常。安装器不依赖其他语音转换仓库。

## Windows适配

`scripts/setup_tts.py`重复应用：pyopenjtalk-plus wheel、纯Python jieba、SoundFile本地音频读取、GPT workers0配置、单GPU不创建DDP进程组、恢复checkpoint时仅允许标准Path类型且保留weights_only=True、独立FFmpeg路径。

上游原文件备份在 `runs/misuzu_gsv_v2pro_v1/upstream_originals/`，迁移前Windows差异另存 `maintenance/runtime_windows_before_reclone.patch`。

官方快速推理UI增加本地默认参考音频与台词、日语参考语种、batch1、固定种子20261012；这些默认值仍可在原生界面自由修改。主WebUI提供官方数据处理与训练页面，现有64条任务的恢复入口优先使用training.md中的配置化命令。
