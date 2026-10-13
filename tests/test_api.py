from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)

def test_health_and_ready():
    assert client.get("/healthz").status_code==200
    assert client.get("/readyz").status_code==200

def test_chat_stub():
    r=client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"Hello"}],"max_tokens":20})
    assert r.status_code==200
    assert r.json()["backend"]=="stub"
    assert "Hello" in r.json()["content"]

def test_requires_user_message():
    r=client.post("/v1/chat/completions",json={"messages":[{"role":"assistant","content":"Hi"}]})
    assert r.status_code==422
