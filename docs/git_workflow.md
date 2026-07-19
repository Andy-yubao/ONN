# Git 工作流规范

## 为什么频繁 Push

本项目处于早期探索阶段，代码和文档迭代速度快。**频繁 Push 到 GitHub** 有两大目的：

1. **AI 辅助阅读**：ChatGPT 等 AI 工具可通过 GitHub 仓库链接读取最新代码，提供更准确的辅助开发建议
2. **版本追溯**：记录每一步实验和决策，方便回溯

## 分支策略

现阶段为单人开发，**直接在 `main`（或 `master`）分支上工作并推送**。不需要复杂的分支管理，保持简单。

```
git add .
git commit -m "类型: 简述修改内容"
git push
```

## Commit Message 规范

格式：`<类型>: <一句话描述>`

| 类型 | 适用场景 | 示例 |
|------|---------|------|
| `feat` | 新增功能/代码 | `feat: 添加 CNN 基础训练脚本` |
| `doc` | 文档改动 | `doc: 更新系统架构图` |
| `exp` | 实验记录 | `exp: 记录 LTP/LTD 参数拟合结果` |
| `refactor` | 重构 | `refactor: 统一数据加载接口` |
| `fix` | 修 Bug | `fix: 修复权重初始化越界` |
| `chore` | 杂项（配置、gitignore 等） | `chore: 添加 essay 到 gitignore` |

描述用**中文**，简洁即可，不要求完整句子。

## 推送频率建议

- **代码/文档有实质性改动后** → 立即推送
- **每天至少推送 1 次** → 保证 AI 工具能读到最新版本
- **实验前后各推送 1 次** → 记录实验起点和结果

## 常用命令速查

```bash
# 查看当前状态
git status

# 添加所有改动并提交
git add .
git commit -m "feat: 你的修改描述"

# 推送到 GitHub
git push

# 一条龙（暂存 + 提交 + 推送）
git add . && git commit -m "feat: 描述" && git push
```

## ChatGPT 读取本项目

将仓库链接发给 ChatGPT（或配置 GitHub 集成），即可让它阅读最新代码辅助开发：

```
https://github.com/Andy-yubao/ONN
```
