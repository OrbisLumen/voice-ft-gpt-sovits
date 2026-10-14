# 本地GPT-SoVITS项目整理记录

## 最终结构与入口

根目录迁到 `D:\Project\voice`，已移除空的中间项目目录。重新克隆官方GPT-SoVITS到vendor，固定已验证提交；数据data、权重models、运行状态runs、试听outputs分别管理。

根目录 `Start-Inference.cmd` 为官方快速推理入口（9872），`Start-WebUI.cmd` 为官方数据与训练主界面（9874），`Stop-WebUI.cmd` 停止本项目界面进程。两页实测HTTP200，默认参考音频也可正常播放。后台服务已保持运行。

推理默认SoVITS20/GPT10，参考片段及日语台词与此前试听一致，batch1、seed20261012、保持随机关闭。所有参数可在官方界面修改。

## 保留

- 完整原始数据目录：心铃、雫、之前的选材副本，迁到 `data/source_library/`，未删原音声，可供重新分割与标注。
- 当前18段训练源、5段源级留出、64条已校对切片及原始转写、编码错误原件、修复和审计记录。
- 全部GPT阶段特征、SoVITS G/D及GPT完整训练状态，推理SoVITS1–20和GPT1–10权重。
- 短句、轮次比较、《序》与PicaPica长文、自动转写、响度整理结果、日志和Windows适配。
- 原已执行notebook完整输出原样放入 `notebooks/history/`。当前notebook按新目录及20/10配置重新生成。

迁移前后对14,103个原音声、切片、模型、特征、试听及原notebook文件进行SHA256核对，全部一致，共16,120,761,095字节。验证文件见 `runs/misuzu_gsv_v2pro_v1/maintenance/verification.json`，逐文件对应关系和哈希在同目录migration_manifest.json。

活动配置、源索引、review.csv、train.list、训练配置、输出manifest及采样脚本已更新路径。历史日志、原编码备份和二进制检查点中的旧路径保留为历史记录，不重写权重字节。

## 清理与Git

按用户确认清理旧远端TTS音频、请求和代码、`.env`及示例、旧语音转换模型/索引、环境、工作目录、实验输出、文档、notebook与相关测试。清理核验后的明确名单记录在 `maintenance/cleanup_completed.json`；删除旧文件16,782,497,949字节，约15.63GiB，另清理了小的旧入口文件。

旧项目Git目录及历史对象、分支、标签和remote已移除；在voice根目录重新初始化main，代码、配置、文稿和文档作为新基线。原始数据、运行产物、vendor官方克隆和历史notebook不提交。未设置remote、未推送。

## 验证

- 原始数据和GPT关键产物完整哈希核验。
- prepare复用并验证64条切片；audit核对64条人工标签与音频。
- CUDA卷积反向传播、日语g2p及依赖导入通过，预训练模型齐全。
- 移动后20/10训练命令复用原推理权重，没有追加训练。
- 9项离线测试通过。
- 主WebUI和推理网页、参考音频HTTP200；官方Gradio推理API加载20/10，实测中文“今天的天气很好。”生成1.7秒32kHz有效非静音WAV。
- Gradio4.44.1的Starlette接口不兼容已通过固定FastAPI0.115.6/Starlette0.41.3解决，pip check通过，锁文件更新。

未在官方主WebUI里重新训练现有数据；现有训练状态、路径与标签继续由配置化脚本管理。主界面用于后续新数据处理和训练时，应确认新任务输出目录，避免覆盖已有推理权重。
