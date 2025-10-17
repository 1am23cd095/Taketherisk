from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.schemas import (
    GenerateRequest,
    UpdateSlotRequest,
)
from app.ga.engine import GeneticAlgorithm


BREAKS = {
    # day -> set(periods) which are breaks (0-indexed periods)
    # 8 periods/day, 55 mins each; snack break after period 2, lunch after period 4
    # According to user timing: periods 1,2, break, 3,4, lunch, 5,6, then end. We treat only 6 teaching periods usable indexes: 0..5
}


@dataclass
class State:
    timetables: Dict[int, List[List[Dict]]]  # semester_id -> matrix [days][periods]


class TimetableService:
    def __init__(self) -> None:
        self.state = State(timetables={})

    def _ensure_breaks(self, days: int, periods_per_day: int) -> Tuple[int, List[int]]:
        # We model 8 slots with breaks at 2 and 5 (0-indexed) => teaching slots: [0,1,3,4,6,7]
        if periods_per_day != 8:
            raise ValueError("This implementation expects exactly 8 periods per day to encode breaks.")
        teaching_periods = [0, 1, 3, 4, 6, 7]
        return periods_per_day, teaching_periods

    def generate(self, req: GenerateRequest) -> Dict[int, List[List[Dict]]]:
        # Build GA input
        ga = GeneticAlgorithm()
        result = ga.run(req)

        # Inject break placeholders into final matrix
        final_results: Dict[int, List[List[Dict]]] = {}
        for sem in req.semesters:
            days = sem.days
            periods = sem.periods_per_day
            _, teaching_periods = self._ensure_breaks(days, periods)
            teaching_matrix = result[sem.id]  # [days][6]
            # Convert to 8 with breaks at 2 and 5
            matrix8: List[List[Dict]] = []
            for d in range(days):
                row: List[Dict] = []
                t_idx = 0
                for p in range(periods):
                    if p in (2, 5):
                        row.append({"type": "break", "label": "Break" if p == 2 else "Lunch"})
                    else:
                        # Ensure keys exist for UI stability
                        cell = teaching_matrix[d][t_idx]
                        row.append({
                            "subject_id": cell.get("subject_id"),
                            "teacher_id": cell.get("teacher_id"),
                            "room_id": cell.get("room_id"),
                        })
                        t_idx += 1
                matrix8.append(row)
            final_results[sem.id] = matrix8
        self.state.timetables = final_results
        # Write to disk for verification convenience
        try:
            import json, os
            os.makedirs("output", exist_ok=True)
            with open("output/timetable.json", "w", encoding="utf-8") as f:
                json.dump(final_results, f)
        except Exception:
            pass
        return final_results

    def get_timetable(self, semester_id: int) -> Optional[List[List[Dict]]]:
        return self.state.timetables.get(semester_id)

    def update_slot(self, req: UpdateSlotRequest) -> None:
        tt = self.state.timetables.get(req.semester_id)
        if tt is None:
            raise ValueError("Timetable not found")
        if req.period in (2, 5):
            raise ValueError("Cannot update a break period")
        if not (0 <= req.day < len(tt)):
            raise ValueError("Invalid day")
        if not (0 <= req.period < len(tt[0])):
            raise ValueError("Invalid period")
        # map 8-slot period to teaching index
        mapping = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5}
        t_idx = mapping.get(req.period)
        if t_idx is None:
            raise ValueError("Invalid period mapping")
        # update teaching cell kept as dict
        cell = tt[req.day][req.period]
        if cell.get("type") == "break":
            raise ValueError("Cannot update a break period")
        if req.subject_id is not None:
            cell["subject_id"] = req.subject_id
        if req.teacher_id is not None:
            cell["teacher_id"] = req.teacher_id
        if req.room_id is not None:
            cell["room_id"] = req.room_id
