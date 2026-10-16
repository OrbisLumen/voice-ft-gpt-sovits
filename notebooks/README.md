# Notebook

`finetune_misuzu_tts.ipynb` 为当前可运行入口，使用 `config/task.json`。当前已完成20/10；已有目标权重会复用，不会额外增加轮数。

其他角色使用 `finetune_<角色ID>_tts.ipynb`，读取各自 `config/characters/<角色ID>/v1.json`。现有8个角色均已完成 SoVITS20/GPT10；数据接受依据和结果见各任务 `runs/` 记录及 [训练文档](../docs/training.md)。所有 Notebook 的准备、转写、特征、训练及试听开关默认关闭，按需手动开启。

更新生成器或角色登记后，可运行 `python scripts/build_tts_notebook.py --all` 重新生成所有入口；生成器不会启动训练，生成文件不包含执行输出。

迁移前已执行的notebook及全部输出原样保存在 `history/finetune_before_reorganization_20261014.ipynb`，仅作历史阅读，不运行其旧路径单元格。该文件不提交Git。

网页推理优先使用根目录 `Start-Inference.cmd`，无需Jupyter。
