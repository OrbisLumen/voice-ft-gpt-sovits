# 仓库维护与 Git 提交

## 文件边界

Git 管理脚本、配置、测试、文稿、文档，以及清空执行输出的 Notebook。音频、模型权重、特征、转写草稿、运行日志和历史已执行 Notebook 保存在本机，遵循 `.gitignore`，不要使用 `git add -f` 纳入提交。

文稿保存在 `texts/`：`original.md` 为原稿，`narration_v1.md` 为口播稿第一版。后续口播稿可按 `narration_v2.md`、`narration_v3.md` 编号；切换推理版本时更新 `config/narration.json` 的 `source` 和相关文档。

## 提交规范

使用 Conventional Commits 格式：`type(scope): summary`。scope 可省略；摘要清楚描述实际变更，每次提交只覆盖一个相关主题。

| 类型 | 用途 | 示例 |
|---|---|---|
| `feat` | 新功能或新工作流 | `feat(dataset): add character review pages` |
| `fix` | 修复错误行为 | `fix(ui): restore synthesis after model loading` |
| `refactor` | 不改变行为的代码调整 | `refactor(config): centralize task path resolution` |
| `docs` | 使用说明或实验记录 | `docs(training): record completed character runs` |
| `test` | 测试用例调整 | `test(training): cover unreviewed transcript rejection` |
| `chore` | 仓库、文件及工具维护 | `chore(texts): version English manuscript filenames` |
| `ci` | 持续集成配置 | `ci(tests): install UI test dependencies` |

按主题显式暂存文件，检查 `git diff --cached` 后提交；不把所有变更无差别合成一次提交。训练结果的提交记录应说明数据接受依据、轮数和验证范围，运行产物继续留在忽略目录中。已有历史提交不为统一前缀而改写。

## 提交前验证

在项目根目录使用已有独立环境：

```powershell
& .\vendor\GPT-SoVITS\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
git diff --cached --check
git status --short
```

没有完整 GPU 环境时，可在 Python3.12 环境安装 `config/requirements-test.txt` 后运行同一测试命令。测试使用临时数据和模拟推理对象，不启动训练或下载模型。GitHub Actions 在 Linux 和 Windows 上执行这些检查。

修改 Notebook 生成器后运行 `python scripts/build_tts_notebook.py --all`，检查生成文件没有执行输出、训练开关默认关闭。修改文稿路径时验证章节提取；不为文档整理重跑模型训练或覆盖历史音频。
