# Intelligent Travel Planning Agent

一个基于 LangGraph 构建的智能旅行规划 Agent，支持结构化需求抽取、工具调用、长短期记忆、本地 RAG、MCP 服务、自动化评测和 LangSmith 可观测性。

> 当前项目使用本地演示数据，仅支持成都和北京，不代表真实价格或实时旅行信息。

## 功能

- 使用 Groq 大模型理解自然语言旅行需求
- 使用 Pydantic 抽取城市、天数、人数、预算、偏好和行程节奏
- 使用 LangGraph 实现条件路由和工具调用循环
- 使用 SQLite Checkpointer 保存会话短期记忆
- 使用 SQLite Store 保存跨会话用户偏好
- 使用 BGE 中文 Embedding 和 Chroma 构建本地 RAG
- 使用 MCP stdio Server 提供独立酒店查询服务
- 使用确定性工具计算住宿和门票预算
- 使用 pytest 实现单元测试、MCP 集成测试和端到端测试
- 使用 LangSmith 查看完整 Agent Trace
- 使用 LangSmith pytest Evaluation 构建离线评测数据集并记录多维评分

## 项目结构

```text
travel_agent/
├── travel_agent/
│   ├── graph.py
│   ├── main.py
│   ├── mcp_client.py
│   ├── rag.py
│   ├── state.py
│   └── tools.py
├── mcp_servers/
│   └── hotel_server.py
├── data/
│   └── knowledge/
├── scripts/
│   ├── build_index.py
│   └── test_mcp_server.py
└── tests/
    ├── test_tools.py
    ├── test_mcp.py
    └── test_e2e_agent.py
```

## Agent 流程

```text
用户输入
  ↓
结构化需求抽取
  ↓
字段校验
  ├── 信息缺失 → 请求补充
  └── 信息完整
        ↓
   LangGraph Agent
        ├── 景点工具
        ├── MCP 酒店服务
        ├── RAG 知识库
        └── 预算工具
        ↓
   最终旅行方案
```

## 环境要求

- Python 3.11
- conda
- Groq API Key
- LangSmith API Key（可选）

## 安装

```bash
conda create -n agent python=3.11
conda activate agent

pip install -r requirements.txt
```

复制环境变量模板：

```bash
cp .env.example .env
```

然后在 `.env` 中配置自己的密钥。不要提交真实 `.env` 文件。

## 构建本地知识库

```bash
python -m scripts.build_index
```

## 运行

```bash
python -m travel_agent.main \
  --user-id demo-user \
  --thread-id demo-trip \
  --query "去成都3天，两个人，预算3000元，喜欢美食和文化，行程轻松。"
```

## 测试

运行不调用大模型的确定性测试：

```bash
python -m pytest -m "not e2e" -v
```

运行完整端到端测试：

```bash
python -m pytest -m e2e -v -s
```

端到端测试会调用 Groq API，并启动本地 MCP 子进程。

运行 LangSmith 离线数据集评测：

```bash
set -a
source .env
set +a

LANGSMITH_TEST_SUITE="travel-agent-evaluation" \
LANGSMITH_EXPERIMENT="baseline-grounded-v1" \
python -m pytest tests/test_langsmith_eval.py -m evaluation -v

## 安全说明

- API Key 只保存在 `.env`
- `.env` 不会提交到 Git
- 仓库只提供 `.env.example`
- 本地数据库和 Chroma 索引不会提交

## Roadmap

- 最终答案事实校验
- FastAPI 服务
- Docker 部署
- PostgreSQL / pgvector

```markdown
## 已知问题

当前 LangGraph 版本在序列化非空 Runtime context 时可能输出
`PydanticSerializationUnexpectedValue` 警告。该警告不影响
context 读取、Memory、Agent 执行和测试结果。

相关上游问题：
https://github.com/langchain-ai/langgraph/issues/8417