import logging, os
from fastapi import FastAPI, HTTPException
from .provider import ModelProvider
from .schemas import ChatRequest, ChatResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("llm_service")
app=FastAPI(title="LLM Service",version="0.1.0",description="Learning starter for stub or local open-weight LLM inference.")
provider=ModelProvider()

@app.get("/healthz",tags=["health"])
def healthz():
    return {"status":"ok"}

@app.get("/readyz",tags=["health"])
def readyz():
    if not provider.ready():
        raise HTTPException(status_code=503,detail="Model has not been loaded yet")
    return {"status":"ready","backend":provider.backend}

@app.post("/v1/chat/completions",response_model=ChatResponse,tags=["inference"])
def chat_completions(request: ChatRequest):
    count=sum(len(m.content) for m in request.messages)
    try:
        answer=provider.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=413,detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=503,detail="Model unavailable; check server logs") from exc
    return ChatResponse(content=answer,model=provider.model_id if provider.backend=="transformers" else "stub",
                        backend=provider.backend,input_characters=count)
