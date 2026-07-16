# RAGFlow Embedding 路由设计

## 一句话
RAGFlow 把每个 Embedding 提供商写成继承 `Base` 的类、用 `_FACTORY_NAME` 类变量挂"对外名"，`__init__.py` 在 import 时扫一遍自动塞字典；外部按字符串名查表拿到对应类，零条件分支。

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 模块：`rag/llm/embedding_model.py`（各 Provider 类）、`rag/llm/__init__.py`（dispatch 注册）
- 关联：本仓库 s04 `embed()` 字典分发（最小版）

## dispatch 循环

```python
    base_class = None
    lite_llm_base_class = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if name == "Base":
                base_class = obj
            elif name == "LiteLLMBase":
                lite_llm_base_class = obj
                assert hasattr(obj, "_FACTORY_NAME"), "LiteLLMbase should have _FACTORY_NAME field."
                if hasattr(obj, "_FACTORY_NAME"):
                    if isinstance(obj._FACTORY_NAME, list):
                        for factory_name in obj._FACTORY_NAME:
                            mapping_dict[factory_name] = obj
                    else:
                        mapping_dict[obj._FACTORY_NAME] = obj
```

## 示例类

```python
class OpenAIEmbed(Base):
    _FACTORY_NAME = "OpenAI"

    def __init__(self, key, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1"):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
```

## 为什么这样写

- **声明式注册，不要 `if/elif` 链**。新增一个提供商只需要"写一个类 + 给它 `_FACTORY_NAME = "Xxx"`"，import 时 `inspect.getmembers` 自动把类塞进 `EmbeddingModel` 字典。本仓库 s04 的 `embed()` 字典分发（`{"local": ..., "openai": ..., "ollama": ...}`）是同样思路的最小版；RAGFlow 把这种模式扩展到 30+ 家提供商。
- **`_FACTORY_NAME` 可为 list 一对多**。同一 SDK 适配多家（如 `Astraflow` / `AstraflowCN` 都继承 `OpenAIEmbed`）只需在子类的列表里多写一个名字，主流程不用动。这把"维度继承"和"对外命名"解耦——子类继承父类所有行为，只换 base_url + 名。
- **错误统一为 `EmbeddingError`**。每个 `_call` / `_batched_encode` 都把任何异常包成 `EmbeddingError`，调用方只看一种异常类型就能做"换 provider 重试"或"降级回退"。这是 `EMBED_PROVIDER` 切换之外的第二条防线：同一 provider 内网络失败也走统一的 retry/降级路径。