from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas import (
    SemesterIn,
    GenerateRequest,
    TimetableResponse,
    UpdateSlotRequest,
)
from app.services.timetable_service import TimetableService

app = FastAPI(title="AI Timetable Generator Backend")

service = TimetableService()


@app.post("/generate", response_model=TimetableResponse)
def generate_timetable(payload: GenerateRequest):
    try:
        result = service.generate(payload)
        return TimetableResponse(timetables=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/timetable/{semester_id}", response_model=Dict[str, List[List[dict]]])
def get_timetable(semester_id: int):
    timetable = service.get_timetable(semester_id)
    if timetable is None:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return {"timetable": timetable}


@app.put("/update-slot")
def update_slot(req: UpdateSlotRequest):
    try:
        service.update_slot(req)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
