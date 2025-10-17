from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_missing_timetable():
    resp = client.get("/timetable/999")
    assert resp.status_code == 404
