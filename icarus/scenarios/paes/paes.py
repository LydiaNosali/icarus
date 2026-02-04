# paes.py
from __future__ import annotations
import random
from collections import Counter
from typing import List, Tuple, Any

Objectives = Tuple[float, float, float]   # (carbon, hit, cost)
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
                 batch_eval_fn=None, # NEW
                 batch_size=4,
                 batch_init_fn=None):
        if seed is not None:
            self.local_random = random.Random(seed)
        self.init_fns = list(init_fn) # 👈 NEW
        self.mutate_fn = mutate_fn
        self.eval_fn = eval_fn
        self.batch_eval_fn = batch_eval_fn      # NEW
        self.batch_size = batch_size
        self.batch_init_fn = batch_init_fn
        self.archive = AdaptiveGridArchive(archive_size, grid_divisions)
        self.max_evaluations = max_evaluations

        # cache of evaluated solutions to avoid re-running Icarus
        # key must be hashable: use tuple(allocations) or any canonical form
        self.evaluated: dict[tuple, Objectives] = {}

    def key(self, sol):
        """Turn a solution into a hashable key."""
        allocs = sol['allocations']
        return tuple(allocs)

    def _init_batch(self, init_fns):
        for fn in init_fns:
            if not callable(fn):
                raise TypeError(
                    f"_init_batch expects callables, got {type(fn)}"
                )
        return self.batch_init_fn(init_fns)
    
    def _eval_one(self, sol):
        """Evaluate a single solution (sequential fallback)"""
        k = self.key(sol)
        if k in self.evaluated:
            return self.evaluated[k]
        objs = self.eval_fn(sol)
        self.evaluated[k] = objs
        return objs

    def _eval_batch(self, sols):
        if self.batch_eval_fn is None:
            return [self._eval_one(s) for s in sols]

        to_eval = []
        to_eval_idx = []
        results = [None] * len(sols)

        for i, sol in enumerate(sols):
            k = self.key(sol)
            if k in self.evaluated:
                results[i] = self.evaluated[k]
            else:
                to_eval.append(sol)
                to_eval_idx.append(i)

        if to_eval:
            batch_objs = self.batch_eval_fn(to_eval)
            for j, objs in enumerate(batch_objs):  # FIXED: use enumerate
                i = to_eval_idx[j]
                k = self.key(to_eval[j])  # FIXED: use j for to_eval indexing
                self.evaluated[k] = objs
                results[i] = objs

        return results

    def run(self):
        evaluations = 0
        parent = None
        fparent = None
        init_solutions = []
        init_objectives = []
        # Initialize solutions (batch the INIT CALLABLES)
        init_callables = list(self.init_fns)

        for i in range(0, len(init_callables), self.batch_size):
            fn_batch = init_callables[i : i + self.batch_size]   # list[callable]

            # Generate solutions (parallel)
            sols = self._init_batch(fn_batch)                    # list[dict]

            # Evaluate solutions (parallel)
            f_sols = self._eval_batch(sols)

            new_evals = 0

            for sol, f in zip(sols, f_sols):
                if f is None:
                    continue

                k = self.key(sol)

                # _eval_batch already cached, but keep safe:
                if k not in self.evaluated:
                    self.evaluated[k] = f
                    new_evals += 1
                else:
                    f = self.evaluated[k]   # ensure consistency

                # ALWAYS archive + ALWAYS record for parent selection
                self.archive.consider(sol, f)
                init_solutions.append(sol)
                init_objectives.append(f)
        
        if not init_solutions:
            raise RuntimeError("PAES init produced no valid solutions")

        # Pick best init solution by archive density (PAES rule)
        best_idx = None
        best_dens = float("inf")

        for i, f in enumerate(init_objectives):
            dens = self.archive.cell_density(f)
            if dens < best_dens:
                best_dens = dens
                best_idx = i

        parent = init_solutions[best_idx]
        fparent = init_objectives[best_idx]

        # BATCHED MAIN LOOP
        while evaluations < self.max_evaluations:
            B = min(self.batch_size, self.max_evaluations - evaluations)

            # Generate batch from current parent
            batch = [self.mutate_fn(parent, self.local_random) for _ in range(B)]

            # Evaluate batch in parallel
            f_batch = self._eval_batch(batch)
            evaluations += len([f for f in f_batch if f is not None])

            # Update archive with all children
            for child, f_child in zip(batch, f_batch):
                if f_child is not None:
                    self.archive.consider(child, f_child)

            # Parent replacement: pick best child by density
            parent_dens = self.archive.cell_density(fparent)
            best_child = None
            best_child_f = None
            best_dens = float('inf')

            for child, f_child in zip(batch, f_batch):
                if f_child is None:
                    continue
                dens_child = self.archive.cell_density(f_child)
                if dens_child < best_dens:
                    best_dens = dens_child
                    best_child = child
                    best_child_f = f_child

            # Apply PAES acceptance (same logic as original)
            if best_child is not None:
                dens_child = self.archive.cell_density(best_child_f)
                dens_parent = self.archive.cell_density(fparent)
                if dens_child < dens_parent or self.local_random.random() < 0.10:
                    parent, fparent = best_child, best_child_f

        return self.archive.as_pareto_set()
