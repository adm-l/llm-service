from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Message(BaseModel):
    role: Literal["system","user","assistant"]
    content: str = Field(min_length=1,max_length=12000)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1,max_length=32)
    max_tokens: int = Field(default=256,ge=1,le=2048)
    temperature: float = Field(default=0.7,ge=0.0,le=2.0)
    @model_validator(mode="after")
    def must_include_user(self):
        if not any(m.role=="user" for m in self.messages):
            raise ValueError("At least one user message is required")
        return self

class ChatResponse(BaseModel):
    content: str
    model: str
    backend: str
    input_characters: int
