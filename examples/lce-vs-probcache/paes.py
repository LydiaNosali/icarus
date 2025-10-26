# paes.py
from __future__ import annotations
import random
from collections import Counter
from typing import Callable, List, Tuple, Any

Objectives = Tuple[0, 0]   # e.g. (cost, carbon)
Solution = Any

def dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

class AdaptiveGridArchive:
    def __init__(self, max_size: int = 100, divisions: int = 12):
        self.max_size = max_size
        self.divisions = divisions
        self.items: List[Tuple[Solution, Objectives]] = []
        self.obj_min = None
        self.obj_max = None
        self.cells: List[Tuple[int, ...]] = []
        self.counts: Counter = Counter()

    def _update_bounds(self, objs: Objectives) -> None:
        if self.obj_min is None:
            self.obj_min = list(objs); self.obj_max = list(objs); return
        for i, v in enumerate(objs):
            if v < self.obj_min[i]: self.obj_min[i] = v
            if v > self.obj_max[i]: self.obj_max[i] = v

    def _cell_id_of(self, objs: Objectives):
        self._update_bounds(objs)
        ids = []
        for i, v in enumerate(objs):
            lo, hi = self.obj_min[i], self.obj_max[i]
            if hi == lo: idx = 0
            else:
                r = (v - lo) / (hi - lo)
                idx = min(self.divisions - 1, max(0, int(r * self.divisions)))
            ids.append(idx)
        return tuple(ids)

    def _reindex(self):
        self.counts = Counter()
        self.cells = [self._cell_id_of(objs) for _, objs in self.items]
        for cid in self.cells: self.counts[cid] += 1

    def _insert(self, sol: Solution, objs: Objectives):
        cid = self._cell_id_of(objs)
        self.items.append((sol, objs))
        self.cells.append(cid)
        self.counts[cid] += 1
        if len(self.items) > self.max_size:
            most_crowded, _ = self.counts.most_common(1)[0]
            idxs = [i for i, c in enumerate(self.cells) if c == most_crowded]
            victim = random.choice(idxs)
            victim_cell = self.cells[victim]
            self.counts[victim_cell] -= 1
            del self.items[victim]; del self.cells[victim]

    def consider(self, sol: Solution, objs: Objectives) -> bool:
        for _, aobjs in self.items:
            if dominates(aobjs, objs): return False
        keep = []; changed = False
        for (s, aobjs), cid in zip(self.items, self.cells):
            if dominates(objs, aobjs):
                self.counts[cid] -= 1; changed = True
            else:
                keep.append((s, aobjs))
        self.items = keep
        self.obj_min = self.obj_max = None
        for _, aobjs in self.items: self._update_bounds(aobjs)
        self._reindex()
        self._insert(sol, objs)
        return True

    def cell_density(self, objs: Objectives) -> int:
        return self.counts.get(self._cell_id_of(objs), 0)

    def as_pareto_set(self) -> List[Tuple[Solution, Objectives]]:
        return list(self.items)

class PAES:
    def __init__(self, init_fn, mutate_fn, eval_fn,
                 archive_size=40, grid_divisions=12,
                 max_evaluations=120, seed: int | None = 0):
        if seed is not None: random.seed(seed)
        self.init_fn = init_fn; self.mutate_fn = mutate_fn; self.eval_fn = eval_fn
        self.archive = AdaptiveGridArchive(archive_size, grid_divisions)
        self.max_evaluations = max_evaluations

    def run(self):
        parent = self.init_fn(); f_parent = self.eval_fn(parent)
        self.archive.consider(parent, f_parent); evaluations = 1
        while evaluations < self.max_evaluations:
            child = self.mutate_fn(parent); f_child = self.eval_fn(child); evaluations += 1
            if dominates(f_child, f_parent):
                parent, f_parent = child, f_child
                self.archive.consider(child, f_child)
                continue
            if dominates(f_parent, f_child):
                self.archive.consider(child, f_child)
                continue
            if self.archive.cell_density(f_child) < self.archive.cell_density(f_parent):
                parent, f_parent = child, f_child
            else:
                # 👇 Add exploration: 10% chance to accept worse child anyway
                if random.random() < 0.1:
                    parent, f_parent = child, f_child
            self.archive.consider(child, f_child)
        return self.archive.as_pareto_set()

