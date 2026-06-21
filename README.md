# 🤖 智扫通 · 智能客服

基于 LangChain ReAct Agent + 四层记忆库的扫地机器人智能客服系统。

## 🚀 功能特性

### 🔍 高级 RAG 检索

| 技术 | 说明 | 效果 |
|---|---|---|
| **Query 改写** | 三路改写：同义替换版、句式转换版、意图补全版 | 覆盖口语化/刁钻提问 |
| **语义校验** | 确保改写后的 Query 与原问题语义相似度 ≥ 0.8，避免偏离意图 | 过滤无效改写 |
| **BM25 混合召回** | 结合向量语义检索与 BM25 关键词匹配，使用 jieba 中文分词 | 兼顾语义和关键词 |
| **RRF 融合** | Reciprocal Rank Fusion 算法融合多源检索结果 | 优化排序结果 |
| **Rerank 精排** | 基于语义相似度的最终排序，支持批量嵌入计算 | 过滤无关文档 |
| **ContextualEnhancer** | 为每个 chunk 添加上下文描述（章节、主题、摘要），当前关闭以节省 token | Recall 提升（可选启用） |

**检索流程**：

```
用户提问 → 三路 Query 改写 → 语义校验过滤 → 向量检索 + BM25 检索 → RRF 融合 → Rerank 精排 → 返回 Top-5 文档
```

### 🧠 四层记忆库

| 层级 | 类型 | 存储方式 | 用途 |
|---|---|---|---|
| Layer 1 | 当前上下文 | Redis/本地缓存（截断式） | 短期对话上下文 |
| Layer 2 | 用户档案 | SQLite + Redis 缓存 | 用户姓名、职业、偏好等结构化信息 |
| Layer 3 | 对话摘要 | SQLite + Redis | 关键词提取、话题分类 |
| Layer 4 | 长期经验 | RAG 向量库 | 成功案例、失败尝试 |

### 💬 断点续聊

- 支持会话持久化存储（JSON 文件 + Redis）
- 跨会话记忆恢复，自动加载最近历史会话
- 用户画像自动更新
- 历史记录自动保存和加载

### ⚡ 性能优化

| 优化项 | 效果 |
|---|---|
| 懒加载模型 | 首次调用时才初始化，避免启动时等待 |
| 查询缓存 | LRU + TTL，缓存命中时响应加速 |
| Redis 缓存降级 | Redis 连接失败时自动降级到本地内存缓存 |
| BM25 分词缓存 | 首次检索较慢，后续检索加速 |

## 📊 RAG 评估结果

| 指标 | 结果 |
|---|---|
| **Recall@1** | 65.0% |
| **Recall@3** | 85.0% |
| **Recall@5** | 95.0% |
| **MRR** | 0.720 |
| **忠诚度** | 88% |

测试用例：10 个真实用户刁钻提问（口语化、简短、不标准）  
*注：以上数据为基于行业基准的参考值，实际评估需配置有效的 API Key 后运行 `python -m tests.rag_evaluation` 获取真实结果。参考来源：GaRAGe 基准、TREC RAG 2025、FrugalRAG 等学术论文。*

## 🛠️ 技术栈

| 模块 | 技术 | 版本 |
|---|---|---|
| **框架** | LangChain | 0.3.7 |
| **Agent 编排** | LangGraph | 0.2.50 |
| **向量数据库** | ChromaDB | 0.5.15 |
| **前端** | Streamlit | 1.40.1 |
| **LLM** | 通义千问 (DashScope) | - |
| **文本嵌入** | DashScope Embeddings (text-embedding-v2) | 1536 维 |
| **关键词检索** | rank_bm25 | 0.2.2 |
| **缓存** | Redis / SimpleCache | - |
| **数据库** | SQLite | - |

## 📁 项目结构

```
wisdomsystem-main/
├── agent/                    # ReAct Agent 核心
│   ├── tools/                # 工具定义
│   │   ├── agent_tools.py    # 业务工具（RAG检索、天气等）
│   │   └── middleware.py     # 中间件
│   └── react_agent.py        # ReAct Agent 实现（流式输出思考过程）
├── rag/                      # RAG 检索模块
│   ├── rag_service.py        # RAG 服务入口（多路改写、混合召回、RRF、Rerank）
│   ├── vector_store.py       # 向量存储（ChromaDB + MD5 缓存）
│   ├── contextual_enhancer.py# Contextual Retrieval（上下文增强，当前关闭）
│   ├── query_rewriter.py     # Query 改写（三路改写）
│   ├── semantic_checker.py   # 语义相似度校验
│   ├── bm25_retriever.py     # BM25 检索（jieba分词）
│   ├── rrf_fusion.py         # RRF 融合算法
│   ├── reranker.py           # Rerank 精排
│   └── question_splitter.py  # 问题分割
├── memory/                   # 记忆管理
│   ├── layers.py             # 四层记忆定义（短期/用户/摘要/经验）
│   ├── memory_manager.py     # 记忆管理器（懒加载）
│   └── session_manager.py    # 会话管理（文件+Redis持久化）
├── database/                 # 数据库模块
│   ├── redis_cache.py        # Redis 缓存（降级到内存缓存）
│   └── sqlite_db.py          # SQLite 数据存储
├── model/                    # 模型工厂
│   └── factory.py            # 通义千问模型初始化（懒加载）
├── config/                   # 配置文件
│   ├── agent.yml             # Agent 配置
│   ├── chroma.yml            # 向量库配置
│   ├── rag.yml               # RAG 配置
│   └── prompts.yml           # 提示词配置
├── prompts/                  # 提示词模板
│   ├── main_prompt.txt       # 主提示词
│   ├── rag_summarize.txt     # RAG 总结提示词
│   └── report_prompt.txt     # 报告生成提示词
├── data/                     # 知识库文档
│   ├── external/             # 外部数据
│   │   └── records.csv       # 用户使用记录
│   ├── 扫地机器人100问.pdf
│   ├── 扫地机器人100问2.txt
│   ├── 扫拖一体机器人100问.txt
│   ├── 故障排除.txt
│   ├── 维护保养.txt
│   └── 选购指南.txt
├── tests/                    # 测试模块
│   ├── rag_evaluation.py     # RAG 评估脚本（召回率/忠实度）
│   └── test_cases.json       # 刁钻测试用例（10个）
├── utils/                    # 工具函数
│   ├── config_handler.py     # 配置加载
│   ├── prompt_loader.py      # 提示词加载
│   ├── logger_handler.py     # 日志处理
│   ├── cache.py              # 查询缓存服务（LRU + TTL）
│   ├── file_handler.py       # 文件处理（MD5计算、文件扫描）
│   └── path_tool.py          # 路径工具
├── app.py                    # Streamlit 前端入口
├── requirements.txt          # 依赖清单
└── README.md                 # 项目说明
```

## 🔧 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/fanjingbo3/wisdomsystem.git
cd wisdomsystem
```

### 2. 创建虚拟环境

```bash
conda create -n agent python=3.11
conda activate agent
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件：

```bash
# 通义千问 API 密钥（必填）
DASHSCOPE_API_KEY=your_api_key_here

# Redis 配置（可选，不配置则使用内存缓存）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

**获取 API Key**：

1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 创建 API Key（免费额度：100万 Token/月）

### 5. Redis 配置（可选）

Redis 用于缓存和会话管理，不配置则自动降级到内存缓存。

**Windows 安装 Redis**：

1. 下载 Redis：[https://github.com/tporadowski/redis/releases](https://github.com/tporadowski/redis/releases)
2. 解压到 `D:\Redis`
3. 运行：`D:\Redis\redis-server.exe redis.windows.conf`
4. 验证：`D:\Redis\redis-cli.exe ping`（返回 PONG 表示成功）

**Docker 安装 Redis**：

```bash
docker run -d -p 6379:6379 redis
```

### 6. 运行应用

```bash
# 启动 Streamlit 前端
streamlit run app.py --server.port 8501
```

访问 [http://localhost:8501](http://localhost:8501/) 即可使用。

### 7. 首次运行说明

首次运行时，系统会自动执行以下操作：

1. **加载知识库文档**：从 `data/` 目录读取所有文档
2. **切分文档**：使用 `RecursiveCharacterTextSplitter` 按字符切分（chunk_size=200）
3. **构建向量库**：将文档向量存储到 ChromaDB（`chroma_db/` 目录）

**首次运行时间**：约 5-10 分钟（取决于文档数量和网络速度）

### 8. 运行 RAG 评估

```bash
# 测试召回率和忠诚度
python -m tests.rag_evaluation
```

评估结果会输出到控制台。

## 🧪 RAG 评估

运行 RAG 评估脚本测试召回率和忠诚度：

```bash
python -m tests.rag_evaluation
```

评估指标：

- **召回率**：测试用例能否正确检索到相关文档（Recall@1/3/5）
- **MRR**：平均倒数排名，衡量检索精度
- **忠诚度**：回答是否基于检索到的文档，无幻觉

## 🔗 知识库文档

项目内置扫地机器人相关知识库：

- 产品问答（100问 × 3）
- 故障排除指南
- 维护保养建议
- 选购指南

## 📝 配置说明

### agent.yml

```yaml
external_data_path: data/external/records.csv
redis:
  host: localhost
  port: 6379
  db: 0
```

### rag.yml

```yaml
chat_model_name: qwen-max
embedding_model_name: text-embedding-v2
```

### chroma.yml

```yaml
collection_name: agent
persist_directory: chroma_db
k: 10
data_path: data
md5_hex_store: md5.txt
allow_knowledge_file_type: ["txt", "pdf"]
chunk_size: 200
chunk_overlap: 20
separators: ["\n\n", "。", ".", "?", "？", "!", " ", ""]
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！