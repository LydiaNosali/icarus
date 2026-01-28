# paes.py
from __future__ import annotations
import random
from collections import Counter
from typing import List, Tuple, Any

Objectives = Tuple[float, float, float]   # (hit, cost, carbon)
Solution = Any

def dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

class AdaptiveGridArchive:
    def __init__(self, max_size: int = 100, divisions: int = 12, local_random=None):
        self.max_size = max_size
        self.divisions = divisions
        self.items: List[Tuple[Solution, Objectives]] = []
        self.obj_min = None
        self.obj_max = None
        self.cells: List[Tuple[int, ...]] = []
        self.counts: Counter = Counter()
        self.random = local_random or random

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
            victim = self.random.choice(idxs)
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
                 max_evaluations=120, seed: int | None = 0,
                 init_fns_extra=None):
        if seed is not None:
            self.local_random = random.Random(seed)
        self.init_fn = init_fn
        self.init_fns_extra = init_fns_extra or []  # 👈 NEW
        self.mutate_fn = mutate_fn
        self.eval_fn = eval_fn
        self.archive = AdaptiveGridArchive(archive_size, grid_divisions)
        self.max_evaluations = max_evaluations

        # cache of evaluated solutions to avoid re-running Icarus
        # key must be hashable: use tuple(allocations) or any canonical form
        self._evaluated: dict[tuple, Objectives] = {}

    def _key(self, sol: Solution) -> tuple:
        """
        Turn a solution into a hashable key.
        Adapt this to your real solution structure.
        """
        allocs = sol["allocations"]
        return tuple(allocs)

    def _get_or_eval(self, sol: Solution) -> Objectives:
        k = self._key(sol)
        if k in self._evaluated:
            return self._evaluated[k]
        objs = self.eval_fn(sol)
        self._evaluated[k] = objs
        return objs

    def run(self):
        # =====================================================
        # 1) Explicit archive seeding (CRITICAL FIX)
        # =====================================================
        init_solutions = []

        # baseline
        init_solutions.append(self.init_fn())

        # additional extremes
        for fn in self.init_fns_extra:
            init_solutions.append(fn())

        evaluations = 0
        parent = None
        f_parent = None

        for sol in init_solutions:
            k = self._key(sol)

            if k in self._evaluated:
                f = self._evaluated[k]
            else:
                f = self.eval_fn(sol)
                self._evaluated[k] = f
                evaluations += 1

            self.archive.consider(sol, f)

            # pick first solution as parent
            if parent is None:
                parent, f_parent = sol, f

        # =====================================================
        # 2) Standard PAES loop (unchanged)
        # =====================================================
        while evaluations < self.max_evaluations:
            child = self.mutate_fn(parent, self.local_random)

            parent_key = self._key(parent)
            child_key = self._key(child)

            if parent_key == child_key:
                continue

            if child_key in self._evaluated:
                f_child = self._evaluated[child_key]
            else:
                f_child = self.eval_fn(child)
                self._evaluated[child_key] = f_child
                evaluations += 1

            # always consider for archive
            self.archive.consider(child, f_child)

            dens_child = self.archive.cell_density(f_child)
            dens_parent = self.archive.cell_density(f_parent)

            if dens_child < dens_parent or self.local_random.random() < 0.10:
                parent, f_parent = child, f_child
            elif self.local_random.random() < 0.05:
                parent, f_parent = child, f_child

        return self.archive.as_pareto_set()
