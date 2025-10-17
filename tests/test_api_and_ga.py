import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def sample_payload():
    semesters = [
        {
            "id": 1,
            "name": "S1",
            "days": 5,
            "periods_per_day": 8,
            "subjects": [
                {"id": 101, "name": "Maths", "teacher_id": 1, "semester_id": 1, "weekly_min": 3, "weekly_max": 5},
                {"id": 102, "name": "Physics", "teacher_id": 2, "semester_id": 1, "weekly_min": 2, "weekly_max": 4},
            ],
        },
        {
            "id": 2,
            "name": "S2",
            "days": 5,
            "periods_per_day": 8,
            "subjects": [
                {"id": 201, "name": "Chemistry", "teacher_id": 3, "semester_id": 2, "weekly_min": 3, "weekly_max": 5},
                {"id": 202, "name": "Biology", "teacher_id": 4, "semester_id": 2, "weekly_min": 2, "weekly_max": 4},
            ],
        },
    ]
    teachers = [
        {"id": 1, "name": "T1", "max_classes_per_day": 2, "max_classes_per_week": 6},
        {"id": 2, "name": "T2", "max_classes_per_day": 2, "max_classes_per_week": 6},
        {"id": 3, "name": "T3", "max_classes_per_day": 2, "max_classes_per_week": 6},
        {"id": 4, "name": "T4", "max_classes_per_day": 2, "max_classes_per_week": 6},
    ]
    rooms = [
        {"id": 1, "name": "R1", "capacity": 60},
        {"id": 2, "name": "R2", "capacity": 60},
    ]
    return {"semesters": semesters, "teachers": teachers, "rooms": rooms}


def test_generate_endpoint_and_structure():
    resp = client.post("/generate", json=sample_payload())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "timetables" in data
    # two semesters
    assert "1" in data["timetables"] and "2" in data["timetables"]
    for sem_id, matrix in data["timetables"].items():
        assert len(matrix) == 5  # days
        for row in matrix:
            assert len(row) == 8  # periods with breaks
            # Check breaks at positions 2 and 5
            assert row[2]["type"] == "break"
            assert row[5]["type"] == "break"


def test_teacher_no_cross_semester_clash_and_load_and_no_consecutive():
    resp = client.post("/generate", json=sample_payload())
    data = resp.json()["timetables"]
    # Build teaching view (filter breaks)
    for sem_id, matrix in data.items():
        for day in range(5):
            # teacher daily counts
            daily_count = {}
            last_teacher_in_pair = None
            for p in range(8):
                cell = matrix[day][p]
                if cell.get("type") == "break":
                    last_teacher_in_pair = None
                    continue
                tid = cell.get("teacher_id")
                if tid is None:
                    continue
                daily_count[tid] = daily_count.get(tid, 0) + 1
                # check pairs (0,1), (3,4), (6,7) in 8-slot indexing
                if p in (0, 3, 6):
                    last_teacher_in_pair = tid
                else:
                    assert tid != last_teacher_in_pair, "Teacher has consecutive class without break"
            for tid, cnt in daily_count.items():
                assert cnt <= 2

    # cross-semester clash check at each non-break period
    for day in range(5):
        for p in [0,1,3,4,6,7]:
            seen = set()
            for sem_id, matrix in data.items():
                tid = matrix[day][p].get("teacher_id")
                if tid is None:
                    continue
                assert tid not in seen, "Teacher clash across semesters"
                seen.add(tid)


def test_subject_weekly_counts():
    resp = client.post("/generate", json=sample_payload())
    data = resp.json()["timetables"]
    # Map subjects per semester
    subj_by_sem = {
        1: {101: (3, 5), 102: (2, 4)},
        2: {201: (3, 5), 202: (2, 4)},
    }
    for sem_id_str, matrix in data.items():
        sem_id = int(sem_id_str)
        counts = {}
        for day in range(5):
            for p in [0,1,3,4,6,7]:
                sid = matrix[day][p].get("subject_id")
                if sid is not None:
                    counts[sid] = counts.get(sid, 0) + 1
        for sid, (mn, mx) in subj_by_sem[sem_id].items():
            assert counts.get(sid, 0) >= mn
            assert counts.get(sid, 0) <= mx


def test_room_conflicts():
    resp = client.post("/generate", json=sample_payload())
    data = resp.json()["timetables"]
    for day in range(5):
        for p in [0,1,3,4,6,7]:
            seen_rooms = set()
            for sem_id, matrix in data.items():
                rid = matrix[day][p].get("room_id")
                if rid is None:
                    continue
                assert rid not in seen_rooms
                seen_rooms.add(rid)


def test_breaks_inserted_and_update_slot_blocked_on_break():
    resp = client.post("/generate", json=sample_payload())
    assert resp.status_code == 200
    data = resp.json()["timetables"]
    sem_id = next(iter(data.keys()))
    tt = data[sem_id]
    for day in range(5):
        assert tt[day][2]["type"] == "break"
        assert tt[day][5]["type"] == "break"

    upd_resp = client.put("/update-slot", json={
        "semester_id": int(sem_id),
        "day": 0,
        "period": 2,
        "subject_id": 999,
    })
    assert upd_resp.status_code == 400
