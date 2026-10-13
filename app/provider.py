import os
from threading import Lock
from .schemas import ChatRequest

class ModelProvider:
    """Provider adapter. Transformers and model weights load lazily on first request."""
    def __init__(self):
        self.backend=os.getenv("MODEL_BACKEND","stub").strip().lower()
        self.model_id=os.getenv("MODEL_ID","Qwen/Qwen2.5-0.5B-Instruct")
        self.device_setting=os.getenv("MODEL_DEVICE","auto")
        self.max_input_chars=int(os.getenv("MAX_INPUT_CHARS","12000"))
        self._model=None; self._tokenizer=None; self._torch=None; self._lock=Lock()
        if self.backend not in {"stub","transformers"}:
            raise ValueError("MODEL_BACKEND must be stub or transformers")
    def ready(self):
        return self.backend=="stub" or self._model is not None
    def _load(self):
        if self._model is not None: return
        with self._lock:
            if self._model is not None: return
            try:
                import torch
                from transformers import AutoModelForCausalLM,AutoTokenizer
                self._tokenizer=AutoTokenizer.from_pretrained(self.model_id)
                kwargs={"device_map":"auto"} if self.device_setting=="auto" else {}
                self._model=AutoModelForCausalLM.from_pretrained(self.model_id,**kwargs)
                if self.device_setting=="cpu": self._model.to("cpu")
                self._torch=torch; self._model.eval()
            except Exception as exc:
                raise RuntimeError(f"Could not load model {self.model_id}: {exc}") from exc
    def generate(self,request: ChatRequest):
        if sum(len(m.content) for m in request.messages)>self.max_input_chars:
            raise ValueError(f"Input exceeds MAX_INPUT_CHARS={self.max_input_chars}")
        if self.backend=="stub":
            question=next(m.content for m in reversed(request.messages) if m.role=="user")
            return "[STUB MODE — no language model was called] You asked: "+question+"\nSet MODEL_BACKEND=transformers and install requirements-model.txt to run an actual model."
        self._load()
        messages=[{"role":m.role,"content":m.content} for m in request.messages]
        if hasattr(self._tokenizer,"apply_chat_template"):
            prompt=self._tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        else:
            prompt="\n".join(f"{m['role']}: {m['content']}" for m in messages)+"\nassistant:"
        inputs=self._tokenizer(prompt,return_tensors="pt")
        device=next(self._model.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}
        with self._torch.inference_mode():
            output=self._model.generate(**inputs,max_new_tokens=request.max_tokens,
                do_sample=request.temperature>0,temperature=max(request.temperature,1e-5),
                pad_token_id=self._tokenizer.eos_token_id)
        generated=output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated,skip_special_tokens=True).strip()
