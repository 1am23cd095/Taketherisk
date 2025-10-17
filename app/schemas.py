from __future__ import annotations
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, conint


class Teacher(BaseModel):
    id: int
    name: str
    max_classes_per_day: int
    max_classes_per_week: int


class Room(BaseModel):
    id: int
    name: str
    capacity: int


class Subject(BaseModel):
    id: int
    name: str
    teacher_id: int
    semester_id: int
    weekly_min: int = 1
    weekly_max: int = 5
    fixed_slots: List[tuple[int, int]] = Field(default_factory=list, description="List of (day, period)")


class SemesterIn(BaseModel):
    id: int
    name: str
    subjects: List[Subject]
    days: conint(ge=1) = 5
    periods_per_day: conint(ge=1) = 8


class GenerateRequest(BaseModel):
    semesters: List[SemesterIn]
    teachers: List[Teacher]
    rooms: List[Room]


class TimetableSlot(BaseModel):
    day: int
    period: int
    subject_id: Optional[int] = None
    teacher_id: Optional[int] = None
    room_id: Optional[int] = None


class UpdateSlotRequest(BaseModel):
    semester_id: int
    day: int
    period: int
    subject_id: Optional[int] = None
    teacher_id: Optional[int] = None
    room_id: Optional[int] = None


class TimetableResponse(BaseModel):
    timetables: Dict[int, List[List[Dict]]]
