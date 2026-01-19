from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single_valid():
    payload = {"text": "Пример безопасного текста"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
    assert isinstance(j["prediction"]["score"], float)


def test_predict_batch_valid():
    payload = {"texts": ["Текст норма", "Текст с оскорблением"]}
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "results" in j
    assert len(j["results"]) == 2


def test_model_info():
    r = client.get("/model_info")
    assert r.status_code == 200
    j = r.json()
    assert "model_id" in j
    assert j["model_id"] == "s-nlp/russian_toxicity_classifier"


def test_predict_invalid_payload():
    r = client.post("/predict", json={"txt": "no key"})
    assert r.status_code == 422
