# 🦌 DeerFlow

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](./README.md) | [简体中文](./README_zh.md) | [日本語](./README_ja.md) | [Deutsch](./README_de.md) | [Español](./README_es.md) | [Русский](./README_ru.md) |[Portuguese](./README_pt.md)

> 源于开源，回馈开源。

**DeerFlow**（**D**eep **E**xploration and **E**fficient **R**esearch **Flow**）是一个社区驱动的深度研究框架，它建立在开源社区的杰出工作基础之上。我们的目标是将语言模型与专业工具（如网络搜索、爬虫和 Python 代码执行）相结合，同时回馈使这一切成为可能的社区。

请访问[DeerFlow 的官方网站](https://deerflow.tech/)了解更多详情。

## 演示

### 视频

https://github.com/user-attachments/assets/f3786598-1f2a-4d07-919e-8b99dfa1de3e

在此演示中，我们展示了如何使用 DeerFlow：

- 无缝集成 MCP 服务
- 进行深度研究过程并生成包含图像的综合报告
- 基于生成的报告创建播客音频

### 回放示例

- [埃菲尔铁塔与最高建筑相比有多高？](https://deerflow.tech/chat?replay=eiffel-tower-vs-tallest-building)
- [GitHub 上最热门的仓库有哪些？](https://deerflow.tech/chat?replay=github-top-trending-repo)
- [撰写关于南京传统美食的文章](https://deerflow.tech/chat?replay=nanjing-traditional-dishes)
- [如何装饰租赁公寓？](https://deerflow.tech/chat?replay=rental-apartment-decoration)
- [访问我们的官方网站探索更多回放示例。](https://deerflow.tech/#case-studies)

---

## 📑 目录

- [🚀 快速开始](#快速开始)
- [🌟 特性](#特性)
- [🏗️ 架构](#架构)
- [🛠️ 开发](#开发)
- [🗣️ 文本转语音集成](#文本转语音集成)
- [📚 示例](#示例)
- [❓ 常见问题](#常见问题)
- [📜 许可证](#许可证)
- [💖 致谢](#致谢)
- [⭐ Star History](#star-History)

## 快速开始

DeerFlow 使用 Python 开发，并配有用 Node.js 编写的 Web UI。为确保顺利的设置过程，我们推荐使用以下工具：

### 推荐工具

- **[`uv`](https://docs.astral.sh/uv/getting-started/installation/):**
  简化 Python 环境和依赖管理。`uv`会自动在根目录创建虚拟环境并为您安装所有必需的包—无需手动安装 Python 环境。

- **[`nvm`](https://github.com/nvm-sh/nvm):**
  轻松管理多个 Node.js 运行时版本。

- **[`pnpm`](https://pnpm.io/installation):**
  安装和管理 Node.js 项目的依赖。

### 环境要求

确保您的系统满足以下最低要求：

- **[Python](https://www.python.org/downloads/):** 版本 `3.12+`
- **[Node.js](https://nodejs.org/en/download/):** 版本 `22+`

### 安装

```bash
# 克隆仓库
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow

# 安装依赖，uv将负责Python解释器和虚拟环境的创建，并安装所需的包
uv sync

# 使用您的API密钥配置.env
# Tavily: https://app.tavily.com/home
# Brave_SEARCH: https://brave.com/search/api/
# 火山引擎TTS: 如果您有TTS凭证，请添加
cp .env.example .env

# 查看下方的"支持的搜索引擎"和"文本转语音集成"部分了解所有可用选项

# 为您的LLM模型和API密钥配置conf.yaml
# 请参阅'docs/configuration_guide.md'获取更多详情
cp conf.yaml.example conf.yaml

# 安装marp用于PPT生成
# https://github.com/marp-team/marp-cli?tab=readme-ov-file#use-package-manager
brew install marp-cli
```

可选，通过[pnpm](https://pnpm.io/installation)安装 Web UI 依赖：

```bash
cd deer-flow/web
pnpm install
```

### 配置

请参阅[配置指南](docs/configuration_guide.md)获取更多详情。

> [!注意]
> 在启动项目之前，请仔细阅读指南，并更新配置以匹配您的特定设置和要求。

### 控制台 UI

运行项目的最快方法是使用控制台 UI。

```bash
# 在类bash的shell中运行项目
uv run main.py
```

### Web UI

本项目还包括一个 Web UI，提供更加动态和引人入胜的交互体验。

> [!注意]
> 您需要先安装 Web UI 的依赖。

```bash
# 在开发模式下同时运行后端和前端服务器
# 在macOS/Linux上
./bootstrap.sh -d

# 在Windows上
bootstrap.bat -d
```

打开浏览器并访问[`http://localhost:3000`](http://localhost:3000)探索 Web UI。

在[`web`](./web/)目录中探索更多详情。

## 支持的搜索引擎

DeerFlow 支持多种搜索引擎，可以在`.env`文件中通过`SEARCH_API`变量进行配置：

- **Tavily**（默认）：专为 AI 应用设计的专业搜索 API

  - 需要在`.env`文件中设置`TAVILY_API_KEY`
  - 注册地址：https://app.tavily.com/home

- **DuckDuckGo**：注重隐私的搜索引擎

  - 无需 API 密钥

- **Brave Search**：具有高级功能的注重隐私的搜索引擎

  - 需要在`.env`文件中设置`BRAVE_SEARCH_API_KEY`
  - 注册地址：https://brave.com/search/api/

- **Arxiv**：用于学术研究的科学论文搜索
  - 无需 API 密钥
  - 专为科学和学术论文设计

要配置您首选的搜索引擎，请在`.env`文件中设置`SEARCH_API`变量：

```bash
# 选择一个：tavily, duckduckgo, brave_search, arxiv
SEARCH_API=tavily
```

## 特性

### 核心能力

- 🤖 **LLM 集成**
  - 通过[litellm](https://docs.litellm.ai/docs/providers)支持集成大多数模型
  - 支持开源模型如 Qwen
  - 兼容 OpenAI 的 API 接口
  - 多层 LLM 系统适用于不同复杂度的任务

  一般来说，使用litellm的方式是：
  ```
   import litellm

   response = litellm.completion(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "你好，litellm 是什么？"}],
       api_key="你的OpenAI_API_KEY"
   )
   print(response['choices'][0]['message']['content'])
   ```

### 工具和 MCP 集成

- 🔍 **搜索和检索**

  - 通过 Tavily、Brave Search 等进行网络搜索
  - 使用 Jina 进行爬取
  - 高级内容提取

- 🔗 **MCP 无缝集成**
  - 扩展私有域访问、知识图谱、网页浏览等能力
  - 促进多样化研究工具和方法的集成

### 人机协作

- 🧠 **人在环中**

  - 支持使用自然语言交互式修改研究计划
  - 支持自动接受研究计划

- 📝 **报告后期编辑**
  - 支持类 Notion 的块编辑
  - 允许 AI 优化，包括 AI 辅助润色、句子缩短和扩展
  - 由[tiptap](https://tiptap.dev/)提供支持（可以编辑，可以插件扩展从而引入AI功能）

### 内容创作

- 🎙️ **播客和演示文稿生成**
  - AI 驱动的播客脚本生成和音频合成
  - 自动创建简单的 PowerPoint 演示文稿
  - 可定制模板以满足个性化内容需求

## 架构

DeerFlow 实现了一个模块化的多智能体系统架构，专为自动化研究和代码分析而设计。该系统基于 LangGraph（LangChain的升级版） 构建，实现了灵活的基于状态的工作流，其中组件通过定义良好的消息传递系统进行通信。

![架构图](./assets/architecture.png)

可以从图中读到：

---

1. coordinator（协调器）

**职责：**
- 作为整个多智能体流程的入口和总控，负责接收用户的初始请求，启动整个研究流程。
- 协调各个模块之间的流转，决定何时将任务交给 planner（规划器）、research_team（研究团队）等。

**输入：**
- 用户的研究请求（如一个问题或任务描述）。

**输出：**
- 将用户请求转发给 planner 进行任务分解和规划。
- 在流程中根据需要将中间结果或反馈分发给下游模块（如 human_feedback、reporter、research_team）。

---

2. research_team（研究团队）

**职责：**
- 负责执行 planner 制定的研究计划，具体完成信息检索、数据收集、分析等任务。
- 研究团队内部可以包含多个智能体（如 researcher、coder），分别负责不同类型的子任务。

**输入：**
- 来自 planner 的研究计划（具体的任务分解和执行步骤）。

**输出：**
- 研究和分析的中间结果或最终结果，返回给 reporter 进行汇总，或直接反馈给 coordinator 以便流程推进。

---

3. reporter（报告员）

**职责：**
- 汇总 research_team 的研究成果，对收集到的信息进行整理、归纳和结构化。
- 生成最终的研究报告或输出文档，便于用户理解和后续使用。

**输入：**
- 来自 research_team 的研究结果和分析数据。

**输出：**
- 结构化的研究报告或总结，最终返回给 coordinator，由其输出给用户或进入下一个流程环节。

---

> 在[deerflow.tech](https://deerflow.tech/#multi-agent-architecture)上查看实时演示

系统采用了精简的工作流程，包含以下组件：

1. **协调器**：管理工作流生命周期的入口点

   - 根据用户输入启动研究过程
   - 在适当时候将任务委派给规划器
   - 作为用户和系统之间的主要接口

2. **规划器**：负责任务分解和规划的战略组件

   - 分析研究目标并创建结构化执行计划
   - 确定是否有足够的上下文或是否需要更多研究
   - 管理研究流程并决定何时生成最终报告

3. **研究团队**：执行计划的专业智能体集合：

   - **研究员**：使用网络搜索引擎、爬虫甚至 MCP 服务等工具进行网络搜索和信息收集。
   - **编码员**：使用 Python REPL 工具处理代码分析、执行和技术任务。
     每个智能体都可以访问针对其角色优化的特定工具，并在 LangGraph 框架内运行

4. **报告员**：研究输出的最终阶段处理器
   - 汇总研究团队的发现
   - 处理和组织收集的信息
   - 生成全面的研究报告

## 开发

### 测试

运行测试套件：

```bash
# 运行所有测试
make test

# 运行特定测试文件
pytest tests/integration/test_workflow.py

# 运行覆盖率测试
make coverage
```

### 代码质量

```bash
# 运行代码检查
make lint

# 格式化代码
make format
```

### 使用 LangGraph Studio 进行调试

DeerFlow 使用 LangGraph 作为其工作流架构。您可以使用 LangGraph Studio 实时调试和可视化工作流。

#### 本地运行 LangGraph Studio

DeerFlow 包含一个`langgraph.json`配置文件，该文件定义了 LangGraph Studio 的图结构和依赖关系。该文件指向项目中定义的工作流图，并自动从`.env`文件加载环境变量。

##### Mac

```bash
# 如果您没有uv包管理器，请安装它
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖并启动LangGraph服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
```

##### Windows / Linux

```bash
# 安装依赖
pip install -e .
pip install -U "langgraph-cli[inmem]"

# 启动LangGraph服务器
langgraph dev
```

启动 LangGraph 服务器后，您将在终端中看到几个 URL：

- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API 文档: http://127.0.0.1:2024/docs

在浏览器中打开 Studio UI 链接以访问调试界面。

#### 使用 LangGraph Studio

在 Studio UI 中，您可以：

1. 可视化工作流图并查看组件如何连接
2. 实时跟踪执行情况，了解数据如何在系统中流动
3. 检查工作流每个步骤的状态
4. 通过检查每个组件的输入和输出来调试问题
5. 在规划阶段提供反馈以完善研究计划

当您在 Studio UI 中提交研究主题时，您将能够看到整个工作流执行过程，包括：

- 创建研究计划的规划阶段
- 可以修改计划的反馈循环
- 每个部分的研究和写作阶段
- 最终报告生成

### 启用 LangSmith 追踪

DeerFlow 支持 LangSmith 追踪功能，帮助您调试和监控工作流。要启用 LangSmith 追踪：

1. 确保您的 `.env` 文件中有以下配置（参见 `.env.example`）：
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   LANGSMITH_API_KEY="xxx"
   LANGSMITH_PROJECT="xxx"
   ```

2. 通过运行以下命令本地启动 LangSmith 追踪：
   ```bash
   langgraph dev
   ```

这将在 LangGraph Studio 中启用追踪可视化，并将您的追踪发送到 LangSmith 进行监控和分析。

## Docker

您也可以使用 Docker 运行此项目。

首先，您需要阅读下面的[配置](#配置)部分。确保`.env`和`.conf.yaml`文件已准备就绪。

其次，构建您自己的 Web 服务器 Docker 镜像：

```bash
docker build -t deer-flow-api .
```

最后，启动运行 Web 服务器的 Docker 容器：

```bash
# 将deer-flow-api-app替换为您首选的容器名称
docker run -d -t -p 8000:8000 --env-file .env --name deer-flow-api-app deer-flow-api

# 停止服务器
docker stop deer-flow-api-app
```

### Docker Compose

您也可以使用 docker compose 设置此项目：

```bash
# 构建docker镜像
docker compose build

# 启动服务器
docker compose up
```

## 文本转语音集成

DeerFlow 现在包含一个文本转语音(TTS)功能，允许您将研究报告转换为语音。此功能使用火山引擎 TTS API 生成高质量的文本音频。速度、音量和音调等特性也可以自定义。

### 使用 TTS API

您可以通过`/api/tts`端点访问 TTS 功能：

```bash
# 使用curl的API调用示例
curl --location 'http://localhost:8000/api/tts' \
--header 'Content-Type: application/json' \
--data '{
    "text": "这是文本转语音功能的测试。",
    "speed_ratio": 1.0,
    "volume_ratio": 1.0,
    "pitch_ratio": 1.0
}' \
--output speech.mp3
```

## 示例

以下示例展示了 DeerFlow 的功能：

### 研究报告

1. **OpenAI Sora 报告** - OpenAI 的 Sora AI 工具分析

   - 讨论功能、访问方式、提示工程、限制和伦理考虑
   - [查看完整报告](examples/openai_sora_report.md)

2. **Google 的 Agent to Agent 协议报告** - Google 的 Agent to Agent (A2A)协议概述

   - 讨论其在 AI 智能体通信中的作用及其与 Anthropic 的 Model Context Protocol (MCP)的关系
   - [查看完整报告](examples/what_is_agent_to_agent_protocol.md)

3. **什么是 MCP？** - 对"MCP"一词在多个上下文中的全面分析

   - 探讨 AI 中的 Model Context Protocol、化学中的 Monocalcium Phosphate 和电子学中的 Micro-channel Plate
   - [查看完整报告](examples/what_is_mcp.md)

4. **比特币价格波动** - 最近比特币价格走势分析

   - 研究市场趋势、监管影响和技术指标
   - 基于历史数据提供建议
   - [查看完整报告](examples/bitcoin_price_fluctuation.md)

5. **什么是 LLM？** - 对大型语言模型的深入探索

   - 讨论架构、训练、应用和伦理考虑
   - [查看完整报告](examples/what_is_llm.md)

6. **如何使用 Claude 进行深度研究？** - 在深度研究中使用 Claude 的最佳实践和工作流程

   - 涵盖提示工程、数据分析和与其他工具的集成
   - [查看完整报告](examples/how_to_use_claude_deep_research.md)

7. **医疗保健中的 AI 采用：影响因素** - 影响医疗保健中 AI 采用的因素分析

   - 讨论 AI 技术、数据质量、伦理考虑、经济评估、组织准备度和数字基础设施
   - [查看完整报告](examples/AI_adoption_in_healthcare.md)

8. **量子计算对密码学的影响** - 量子计算对密码学影响的分析

   - 讨论经典密码学的漏洞、后量子密码学和抗量子密码解决方案
   - [查看完整报告](examples/Quantum_Computing_Impact_on_Cryptography.md)

9. **克里斯蒂亚诺·罗纳尔多的表现亮点** - 克里斯蒂亚诺·罗纳尔多表现亮点的分析
   - 讨论他的职业成就、国际进球和在各种比赛中的表现
   - [查看完整报告](examples/Cristiano_Ronaldo's_Performance_Highlights.md)

要运行这些示例或创建您自己的研究报告，您可以使用以下命令：
（这些命令能说明 main.py是入口文件，interactive表示交互模型）

```bash
# 使用特定查询运行
uv run main.py "哪些因素正在影响医疗保健中的AI采用？"

# 使用自定义规划参数运行
uv run main.py --max_plan_iterations 3 "量子计算如何影响密码学？"

# 在交互模式下运行，带有内置问题
uv run main.py --interactive

# 或者使用基本交互提示运行
uv run main.py

# 查看所有可用选项
uv run main.py --help
```

### 交互模式

应用程序现在支持带有英文和中文内置问题的交互模式：

1. 启动交互模式：

   ```bash
   uv run main.py --interactive
   ```

2. 选择您偏好的语言（English 或中文）

3. 从内置问题列表中选择或选择提出您自己问题的选项

4. 系统将处理您的问题并生成全面的研究报告

### 人在环中

DeerFlow 包含一个人在环中机制，允许您在执行研究计划前审查、编辑和批准：

1. **计划审查**：启用人在环中时，系统将在执行前向您展示生成的研究计划

2. **提供反馈**：您可以：

   - 通过回复`[ACCEPTED]`接受计划
   - 通过提供反馈编辑计划（例如，`[EDIT PLAN] 添加更多关于技术实现的步骤`）
   - 系统将整合您的反馈并生成修订后的计划

3. **自动接受**：您可以启用自动接受以跳过审查过程：

   - 通过 API：在请求中设置`auto_accepted_plan: true`

4. **API 集成**：使用 API 时，您可以通过`feedback`参数提供反馈：
   ```json
   {
     "messages": [{ "role": "user", "content": "什么是量子计算？" }],
     "thread_id": "my_thread_id",
     "auto_accepted_plan": false,
     "feedback": "[EDIT PLAN] 包含更多关于量子算法的内容"
   }
   ```

### 命令行参数

应用程序支持多个命令行参数来自定义其行为：

- **query**：要处理的研究查询（可以是多个词）
- **--interactive**：以交互模式运行，带有内置问题
- **--max_plan_iterations**：最大规划周期数（默认：1）
- **--max_step_num**：研究计划中的最大步骤数（默认：3）
- **--debug**：启用详细调试日志

## 常见问题

请参阅[FAQ.md](docs/FAQ.md)获取更多详情。
