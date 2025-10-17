from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass

from app.schemas import GenerateRequest, SemesterIn, Subject, Teacher, Room


@dataclass
class Individual:
    # genome: semester_id -> [days][6 teaching periods]
    genome: Dict[int, List[List[Dict]]]
    fitness: float | None = None


class GeneticAlgorithm:
    def __init__(self, population_size: int = 200, generations: int = 100, mutation_rate: float = 0.05, crossover_rate: float = 0.7) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # Mapping from 8-slot indices to teaching-slot indices (exclude breaks at 2 and 5)
        self.period8_to_tslot: Dict[int, int] = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5}
        self.tslot_to_period8: Dict[int, int] = {v: k for k, v in self.period8_to_tslot.items()}

    # --- Public API ---
    def run(self, req: GenerateRequest) -> Dict[int, List[List[Dict]]]:
        semesters = {s.id: s for s in req.semesters}
        teachers = {t.id: t for t in req.teachers}
        rooms = {r.id: r for r in req.rooms}

        population = self._init_population(semesters, teachers, rooms)
        # Ensure fixed slots are respected from the start
        for ind in population:
            self._apply_fixed_slots(ind.genome, semesters, rooms)
        for _ in range(self.generations):
            self._evaluate_population(population, semesters, teachers, rooms)
            population = self._next_generation(population)
        # Evaluate once more to assign fitness to new population members without fitness
        self._evaluate_population(population, semesters, teachers, rooms)
        best = max(population, key=lambda ind: ind.fitness or float('-inf'))
        # Repair/normalize the best genome to strictly satisfy constraints where possible
        repaired = self._repair(best.genome, semesters, teachers, rooms)
        return repaired

    # --- GA internals ---
    def _init_population(self, semesters: Dict[int, SemesterIn], teachers: Dict[int, Teacher], rooms: Dict[int, Room]) -> List[Individual]:
        population: List[Individual] = []
        for _ in range(self.population_size):
            genome: Dict[int, List[List[Dict]]] = {}
            for sem_id, sem in semesters.items():
                days = sem.days
                teaching_periods_per_day = 6  # because 8 slots with breaks at 2 and 5
                subjects = list(sem.subjects)
                genome[sem_id] = []
                for d in range(days):
                    row: List[Dict] = []
                    for t in range(teaching_periods_per_day):
                        subj = random.choice(subjects) if subjects else None
                        if subj is None:
                            row.append({"subject_id": None, "teacher_id": None, "room_id": None})
                        else:
                            teacher_id = subj.teacher_id
                            room_id = random.choice(list(rooms.keys())) if rooms else None
                            row.append({"subject_id": subj.id, "teacher_id": teacher_id, "room_id": room_id})
                    genome[sem_id].append(row)
            population.append(Individual(genome=genome, fitness=None))
        return population

    def _evaluate_population(self, population: List[Individual], semesters: Dict[int, SemesterIn], teachers: Dict[int, Teacher], rooms: Dict[int, Room]) -> None:
        for ind in population:
            ind.fitness = self._fitness(ind.genome, semesters, teachers)

    def _fitness(self, genome: Dict[int, List[List[Dict]]], semesters: Dict[int, SemesterIn], teachers: Dict[int, Teacher]) -> float:
        score = 0.0
        penalty = 0.0

        # Constraint: cross-semester teacher clashes per day/period
        # For each day and teaching slot index (0..5), ensure a teacher teaches at most one semester
        for day in range(next(iter(semesters.values())).days):
            for tslot in range(6):
                seen_teachers: set[int] = set()
                for sem_id, matrix in genome.items():
                    cell = matrix[day][tslot]
                    teacher_id = cell.get("teacher_id")
                    if teacher_id is None:
                        continue
                    if teacher_id in seen_teachers:
                        penalty += 5
                    else:
                        seen_teachers.add(teacher_id)

        # Room conflicts across semesters at same time
        for day in range(next(iter(semesters.values())).days):
            for tslot in range(6):
                seen_rooms: set[int] = set()
                for sem_id, matrix in genome.items():
                    cell = matrix[day][tslot]
                    room_id = cell.get("room_id")
                    if room_id is None:
                        continue
                    if room_id in seen_rooms:
                        penalty += 3
                    else:
                        seen_rooms.add(room_id)

        # Teacher daily and weekly load, and no consecutive classes (teach-break-teach constraint)
        for sem_id, sem in semesters.items():
            matrix = genome[sem_id]
            teacher_day_load: Dict[int, int] = {}
            teacher_week_load: Dict[int, int] = {}
            for day in range(sem.days):
                teacher_day_count: Dict[int, int] = {}
                # Track last teacher within contiguous pairs: (0,1), (2,3), (4,5)
                last_teacher_in_pair: Optional[int] = None
                for tslot in range(6):
                    cell = matrix[day][tslot]
                    tid = cell.get("teacher_id")
                    if tid is None:
                        # Reset within pair boundaries when starting new pair
                        if tslot in (0, 2, 4):
                            last_teacher_in_pair = None
                        continue
                    teacher_day_count[tid] = teacher_day_count.get(tid, 0) + 1
                    teacher_week_load[tid] = teacher_week_load.get(tid, 0) + 1
                    # consecutive class penalty only within contiguous pairs
                    if tslot in (0, 2, 4):
                        # start of pair
                        last_teacher_in_pair = tid
                    else:
                        # second slot of pair
                        if tid == last_teacher_in_pair:
                            penalty += 4
                        # update for completeness
                        last_teacher_in_pair = tid
                # daily limit
                for tid, count in teacher_day_count.items():
                    max_per_day = teachers[tid].max_classes_per_day
                    if count > max_per_day:
                        penalty += (count - max_per_day) * 2
            # weekly limit
            for tid, count in teacher_week_load.items():
                max_per_week = teachers[tid].max_classes_per_week
                if count > max_per_week:
                    penalty += (count - max_per_week) * 2

        # Subject weekly min/max counts
        for sem_id, sem in semesters.items():
            matrix = genome[sem_id]
            counts: Dict[int, int] = {subj.id: 0 for subj in sem.subjects}
            for day in range(sem.days):
                for tslot in range(6):
                    sid = matrix[day][tslot].get("subject_id")
                    if sid is not None and sid in counts:
                        counts[sid] += 1
            for subj in sem.subjects:
                if counts[subj.id] < subj.weekly_min:
                    penalty += (subj.weekly_min - counts[subj.id]) * 2
                if counts[subj.id] > subj.weekly_max:
                    penalty += (counts[subj.id] - subj.weekly_max) * 2

        # Simple reward for filled slots
        filled = 0
        total = 0
        for sem_id, sem in semesters.items():
            for day in range(sem.days):
                for tslot in range(6):
                    total += 1
                    cell = genome[sem_id][day][tslot]
                    if cell.get("subject_id") is not None:
                        filled += 1
        score += filled / max(1, total)

        return score - penalty

    def _next_generation(self, population: List[Individual]) -> List[Individual]:
        new_pop: List[Individual] = []
        while len(new_pop) < len(population):
            parent1 = self._tournament(population)
            parent2 = self._tournament(population)
            child1_genome, child2_genome = parent1.genome, parent2.genome
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self._crossover(parent1.genome, parent2.genome)
            child1_genome = self._mutate(child1_genome)
            child2_genome = self._mutate(child2_genome)
            new_pop.append(Individual(genome=child1_genome))
            if len(new_pop) < len(population):
                new_pop.append(Individual(genome=child2_genome))
        return new_pop

    def _tournament(self, population: List[Individual], k: int = 3) -> Individual:
        competitors = random.sample(population, k)
        return max(competitors, key=lambda ind: ind.fitness or float('-inf'))

    def _crossover(self, g1: Dict[int, List[List[Dict]]], g2: Dict[int, List[List[Dict]]]) -> Tuple[Dict[int, List[List[Dict]]], Dict[int, List[List[Dict]]]]:
        child1: Dict[int, List[List[Dict]]] = {}
        child2: Dict[int, List[List[Dict]]] = {}
        for sem_id in g1.keys():
            a = g1[sem_id]
            b = g2[sem_id]
            days = len(a)
            cut = random.randint(1, days - 1) if days > 1 else 1
            child1[sem_id] = a[:cut] + b[cut:]
            child2[sem_id] = b[:cut] + a[cut:]
        return child1, child2

    def _mutate(self, genome: Dict[int, List[List[Dict]]]) -> Dict[int, List[List[Dict]]]:
        # Deep-ish copy rows and cells
        g = {sem_id: [[cell.copy() for cell in row] for row in grid] for sem_id, grid in genome.items()}
        for sem_id, grid in g.items():
            days = len(grid)
            for d in range(days):
                for t in range(len(grid[d])):
                    if random.random() < self.mutation_rate:
                        # swap with random cell same semester
                        d2 = random.randrange(days)
                        t2 = random.randrange(len(grid[d]))
                        grid[d][t], grid[d2][t2] = grid[d2][t2], grid[d][t]
        return g

    # --- Helpers ---
    def _apply_fixed_slots(self, genome: Dict[int, List[List[Dict]]], semesters: Dict[int, SemesterIn], rooms: Dict[int, Room]) -> None:
        if not rooms:
            return
        room_ids = list(rooms.keys())
        for sem_id, sem in semesters.items():
            grid = genome[sem_id]
            for subj in sem.subjects:
                for (day8, period8) in getattr(subj, "fixed_slots", []) or []:
                    if day8 < 0 or day8 >= sem.days:
                        continue
                    # Map to teaching slot; skip if a break index
                    if period8 not in self.period8_to_tslot:
                        continue
                    tslot = self.period8_to_tslot[period8]
                    chosen_room = random.choice(room_ids)
                    grid[day8][tslot] = {
                        "subject_id": subj.id,
                        "teacher_id": subj.teacher_id,
                        "room_id": chosen_room,
                        "_locked": True,  # mark as locked to protect during repair
                    }

    def _repair(
        self,
        genome: Dict[int, List[List[Dict]]],
        semesters: Dict[int, SemesterIn],
        teachers: Dict[int, Teacher],
        rooms: Dict[int, Room],
    ) -> Dict[int, List[List[Dict]]]:
        # Deep copy
        g = {sem_id: [[cell.copy() for cell in row] for row in grid] for sem_id, grid in genome.items()}
        room_ids = list(rooms.keys())

        # Pre-calculate per-subject min/max
        subj_by_sem: Dict[int, Dict[int, Subject]] = {sem_id: {s.id: s for s in sem.subjects} for sem_id, sem in semesters.items()}

        days = next(iter(semesters.values())).days

        # First pass: clear conflicts and enforce teacher loads and non-consecutive constraints, and room uniqueness
        # Track teacher loads aggregated across semesters
        teacher_week_count: Dict[int, int] = {}
        teacher_day_count: List[Dict[int, int]] = [dict() for _ in range(days)]

        for day in range(days):
            # Reset day counts for day
            teacher_day_count[day] = {}
            seen_teacher_at_tslot: List[set[int]] = [set() for _ in range(6)]
            seen_room_at_tslot: List[set[int]] = [set() for _ in range(6)]
            # Track last teacher per semester within pairs
            last_teacher_in_pair: Dict[int, Optional[int]] = {sem_id: None for sem_id in g.keys()}

            for tslot in range(6):
                # Reset pair boundary
                if tslot in (0, 2, 4):
                    for sem_id in last_teacher_in_pair.keys():
                        last_teacher_in_pair[sem_id] = None
                for sem_id, sem in semesters.items():
                    cell = g[sem_id][day][tslot]
                    if cell.get("subject_id") is None or cell.get("teacher_id") is None:
                        continue
                    tid = cell["teacher_id"]
                    rid = cell.get("room_id")
                    locked = cell.get("_locked", False)

                    # Cross-semester teacher clash
                    if tid in seen_teacher_at_tslot[tslot] and not locked:
                        cell.update({"subject_id": None, "teacher_id": None, "room_id": None})
                        continue
                    # Room conflict
                    if rid is not None and rid in seen_room_at_tslot[tslot] and not locked:
                        cell.update({"subject_id": None, "teacher_id": None, "room_id": None})
                        continue
                    # Teacher daily/weekly limits
                    max_day = teachers[tid].max_classes_per_day
                    max_week = teachers[tid].max_classes_per_week
                    day_count = teacher_day_count[day].get(tid, 0)
                    week_count = teacher_week_count.get(tid, 0)
                    if (day_count + 1) > max_day and not locked:
                        cell.update({"subject_id": None, "teacher_id": None, "room_id": None})
                        continue
                    if (week_count + 1) > max_week and not locked:
                        cell.update({"subject_id": None, "teacher_id": None, "room_id": None})
                        continue

                    # No consecutive classes within pairs
                    prev_tid = last_teacher_in_pair[sem_id]
                    if prev_tid is not None and prev_tid == tid and not locked:
                        cell.update({"subject_id": None, "teacher_id": None, "room_id": None})
                        continue

                    # Accept cell
                    seen_teacher_at_tslot[tslot].add(tid)
                    if rid is not None:
                        seen_room_at_tslot[tslot].add(rid)
                    teacher_day_count[day][tid] = day_count + 1
                    teacher_week_count[tid] = week_count + 1
                    # update last within pair
                    last_teacher_in_pair[sem_id] = tid

        # Second pass: enforce subject weekly maxima by removing excess (keep locked first)
        for sem_id, sem in semesters.items():
            # Count current per subject
            counts: Dict[int, int] = {s.id: 0 for s in sem.subjects}
            for day in range(sem.days):
                for tslot in range(6):
                    sid = g[sem_id][day][tslot].get("subject_id")
                    if sid is not None:
                        counts[sid] = counts.get(sid, 0) + 1
            # Remove extras
            for subj in sem.subjects:
                max_allowed = getattr(subj, "weekly_max", 5)
                current = counts.get(subj.id, 0)
                if current > max_allowed:
                    to_remove = current - max_allowed
                    for day in range(sem.days):
                        for tslot in range(6):
                            if to_remove <= 0:
                                break
                            cell = g[sem_id][day][tslot]
                            if cell.get("_locked"):
                                continue
                            if cell.get("subject_id") == subj.id:
                                # remove
                                cell.update({"subject_id": None, "teacher_id": None, "room_id": None})
                                to_remove -= 1
                        if to_remove <= 0:
                            break

        # Third pass: satisfy subject weekly minima by adding where safe
        # Rebuild availability maps for placement
        def build_used_maps() -> Tuple[List[Dict[int, int]], List[set[int]], List[set[int]]]:
            t_day_count: List[Dict[int, int]] = [dict() for _ in range(days)]
            used_teacher: List[set[int]] = [set() for _ in range(6 * days)]  # flatten by day*6 + tslot
            used_room: List[set[int]] = [set() for _ in range(6 * days)]
            for day in range(days):
                for tslot in range(6):
                    idx = day * 6 + tslot
                    for sem_id2 in g.keys():
                        cell = g[sem_id2][day][tslot]
                        tid = cell.get("teacher_id")
                        rid = cell.get("room_id")
                        if tid is not None:
                            used_teacher[idx].add(tid)
                            t_day_count[day][tid] = t_day_count[day].get(tid, 0) + 1
                        if rid is not None:
                            used_room[idx].add(rid)
            return t_day_count, used_teacher, used_room

        teacher_day_count2, used_teacher_idx, used_room_idx = build_used_maps()
        teacher_week_count2: Dict[int, int] = {}
        for day in range(days):
            for tslot in range(6):
                for sem_id2 in g.keys():
                    tid = g[sem_id2][day][tslot].get("teacher_id")
                    if tid is not None:
                        teacher_week_count2[tid] = teacher_week_count2.get(tid, 0) + 1

        for sem_id, sem in semesters.items():
            # Current counts
            counts: Dict[int, int] = {s.id: 0 for s in sem.subjects}
            for day in range(sem.days):
                for tslot in range(6):
                    sid = g[sem_id][day][tslot].get("subject_id")
                    if sid is not None:
                        counts[sid] += 1
            for subj in sem.subjects:
                min_needed = getattr(subj, "weekly_min", 1)
                current = counts.get(subj.id, 0)
                attempts = 0
                while current < min_needed and attempts < (days * 6):
                    placed = False
                    for day in range(sem.days):
                        for tslot in range(6):
                            cell = g[sem_id][day][tslot]
                            if cell.get("subject_id") is not None:
                                continue
                            # check teacher availability and load
                            tid = subj.teacher_id
                            max_day = teachers[tid].max_classes_per_day
                            max_week = teachers[tid].max_classes_per_week
                            idx = day * 6 + tslot
                            if tid in used_teacher_idx[idx]:
                                continue
                            if teacher_day_count2[day].get(tid, 0) + 1 > max_day:
                                continue
                            if teacher_week_count2.get(tid, 0) + 1 > max_week:
                                continue
                            # no consecutive within pair
                            if tslot in (1, 3, 5):
                                prev_cell = g[sem_id][day][tslot - 1]
                                if prev_cell.get("teacher_id") == tid:
                                    continue
                            # choose available room
                            rid: Optional[int] = None
                            for candidate in room_ids:
                                if candidate not in used_room_idx[idx]:
                                    rid = candidate
                                    break
                            if rid is None:
                                continue
                            # place
                            cell.update({
                                "subject_id": subj.id,
                                "teacher_id": tid,
                                "room_id": rid,
                            })
                            # update trackers
                            used_teacher_idx[idx].add(tid)
                            used_room_idx[idx].add(rid)
                            teacher_day_count2[day][tid] = teacher_day_count2[day].get(tid, 0) + 1
                            teacher_week_count2[tid] = teacher_week_count2.get(tid, 0) + 1
                            current += 1
                            placed = True
                            break
                        if placed:
                            break
                    if not placed:
                        attempts += 1
                        break

        # Cleanup: remove helper keys
        for sem_id in g.keys():
            for day in range(days):
                for tslot in range(6):
                    cell = g[sem_id][day][tslot]
                    if "_locked" in cell:
                        del cell["_locked"]

        return g
