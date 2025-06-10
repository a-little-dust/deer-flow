# 配置指南

## 快速设置

将 `conf.yaml.example` 文件复制为 `conf.yaml`，并根据你的具体设置和需求修改配置。

```bash
cd deer-flow
cp conf.yaml.example conf.yaml
```

## DeerFlow 支持哪些模型？

在 DeerFlow 中，目前我们只支持非推理类模型，这意味着像 OpenAI 的 o1/o3 或 DeepSeek 的 R1 这样的模型暂时还不支持，但我们未来会增加对它们的支持。

### 支持的模型

`doubao-1.5-pro-32k-250115`、`gpt-4o`、`qwen-max-latest`、`gemini-2.0-flash`、`deepseek-v3`，以及理论上任何实现了 OpenAI API 规范的非推理类对话模型。

> [!注意]
> 深度研究流程需要模型具备**更长的上下文窗口**，并非所有模型都支持。
> 一个变通方法是在网页右上角的设置对话框中将"研究计划的最大步骤数"设置为 2，
> 或者在调用 API 时设置 `max_step_num` 为 2。

### 如何切换模型？

你可以通过修改项目根目录下的 `conf.yaml` 文件来切换所用模型，配置格式采用 [litellm 格式](https://docs.litellm.ai/docs/providers/openai_compatible)。

---

### 如何使用 OpenAI 兼容模型？

DeerFlow 支持集成 OpenAI 兼容模型，即实现了 OpenAI API 规范的模型。这包括各种开源和商用模型，只要它们提供兼容 OpenAI 格式的 API 接口即可。详细文档可参考 [litellm OpenAI-Compatible](https://docs.litellm.ai/docs/providers/openai_compatible)。
以下是 `conf.yaml` 配置 OpenAI 兼容模型的示例：

```yaml
# 以火山引擎 Doubao 模型为例
BASIC_MODEL:
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  model: "doubao-1.5-pro-32k-250115"
  api_key: YOUR_API_KEY

# 以阿里云模型为例
BASIC_MODEL:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-max-latest"
  api_key: YOUR_API_KEY

# 以 deepseek 官方模型为例
BASIC_MODEL:
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
  api_key: YOUR_API_KEY

# 以 Google Gemini 使用 OpenAI 兼容接口为例
BASIC_MODEL:
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-2.0-flash"
  api_key: YOUR_API_KEY
```

### 如何使用 Ollama 模型？

DeerFlow 支持集成 Ollama 模型。可参考 [litellm Ollama](https://docs.litellm.ai/docs/providers/ollama)。<br>
以下是 `conf.yaml` 配置 Ollama 模型的示例：

```yaml
BASIC_MODEL:
  model: "ollama/ollama-model-name"
  base_url: "http://localhost:11434" # Ollama 的本地服务地址，可通过 ollama serve 启动/查看
```

### 如何使用 OpenRouter 模型？

DeerFlow 支持集成 OpenRouter 模型。可参考 [litellm OpenRouter](https://docs.litellm.ai/docs/providers/openrouter)。如需使用 OpenRouter 模型，请：

1. 从 OpenRouter（https://openrouter.ai/）获取 OPENROUTER_API_KEY，并设置到环境变量中。
2. 在模型名称前加上 `openrouter/` 前缀。
3. 配置正确的 OpenRouter base URL。

以下是使用 OpenRouter 模型的配置示例：

1. 在环境变量（如 `.env` 文件）中配置 OPENROUTER_API_KEY

```ini
OPENROUTER_API_KEY=""
```

2. 在 `conf.yaml` 中设置模型名称

```yaml
BASIC_MODEL:
  model: "openrouter/google/palm-2-chat-bison"
```

注意：可用模型及其准确名称可能会随时间变化。请在 [OpenRouter 官方文档](https://openrouter.ai/docs) 中核查当前可用模型及其正确标识。

### 如何使用 Azure 模型？

DeerFlow 支持集成 Azure 模型。可参考 [litellm Azure](https://docs.litellm.ai/docs/providers/azure)。`conf.yaml` 配置示例：

```yaml
BASIC_MODEL:
  model: "azure/gpt-4o-2024-08-06"
  api_base: $AZURE_API_BASE
  api_version: $AZURE_API_VERSION
  api_key: $AZURE_API_KEY
```
