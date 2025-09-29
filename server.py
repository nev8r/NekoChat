from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from model_utils import ModelWrapper
import yaml
import asyncio

app = FastAPI(title="NekoChat API Dynamic Config", version="1.0")

# ---------------- 配置加载 ----------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

default_model = config.get("default_model", "nekochat")
models_config = config.get("models", {})

# 模型池
model_pool: Dict[str, ModelWrapper] = {}

# ---------------- 请求模型输入格式 ----------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = default_model
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    max_new_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

# ---------------- 异步推理接口 ----------------
@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    model_name = req.model if req.model in models_config else default_model

    # 动态加载模型
    if model_name not in model_pool:
        cfg = models_config[model_name]
        model_pool[model_name] = ModelWrapper(cfg["path"], cfg.get("system_prompt", ""))

    model = model_pool[model_name]

    # 转换为 history
    history = [{"role": m.role, "content": m.content} for m in req.messages if m.role in ["user", "assistant"]]

    # 异步调用模型
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        None,
        lambda: model.chat(
            history,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens
        )
    )

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
