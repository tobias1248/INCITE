import time
import numpy as np
import cv2
import os
import json
from pathlib import Path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)

class ConcolicTestRecorder:
    def __init__(self, save_dir, input_name):
        # save
        self.save_dir = save_dir
        
        # iters
        self.sat = []
        self.unsat = []
        self.unknown = []
        self.gen_constraint = []
        self.solve_constraint = []
        self.iter_wall_time = []
        self.iter_cpu_time = []
        self.execute_wall_time = []
        self.execute_cpu_time = []
        self.solve_constraint_wall_time = []
        self.solve_constraint_cpu_time = []
        self.sat_inputs = []
        
        # total
        self.total_wall_time = None
        self.total_cpu_time = None
        self.total_iter = -1
        
        # meta
        self.input_name = input_name
        self.input_shape = None
        self.original_label = None 
        self.attack_label = None
        self.adversarial_input = None
        self.original_input = None
        self.is_finish = False # finish all iteration or generate an adversarial input
        self.is_timeout = False
        self.solve_all_ctr = False # solve all constraints

        # extra metadata for compact stats
        self.extra_meta = {}
        self.queue_max = 0
        self.queue_last = 0

        # calculation
        self._pre_sat = 0
        self._pre_unsat = 0
        self._pre_unk = 0


    def iter_start(self, solver):
        self._iter_start_wall_time = time.time()
        self._iter_start_cpu_time = time.process_time()
        
        solver.iter = self.total_iter+1
        solver.iter_count = 1
        
        
    def iter_end(self, solver_stats, solve_constr_num):
        self.iter_wall_time.append(time.time() - self._iter_start_wall_time)
        self.iter_cpu_time.append(time.process_time() - self._iter_start_cpu_time)

        self.solve_constraint.append(solve_constr_num)

        # sat, unsat, unknown
        self.sat.append(solver_stats['sat_number'] - self._pre_sat)
        self.unsat.append(solver_stats['unsat_number'] - self._pre_unsat)
        self.unknown.append(solver_stats['otherwise_number'] - self._pre_unk)

        self._pre_sat = solver_stats['sat_number']
        self._pre_unsat = solver_stats['unsat_number']
        self._pre_unk = solver_stats['otherwise_number']
        
        self.total_iter += 1


    def execution_start(self):
        self._execute_wall_time = time.time()
        self._execute_cpu_time = time.process_time()

    def execution_end(self):
        self.execute_wall_time.append(time.time() - self._execute_wall_time)
        self.execute_cpu_time.append(time.process_time() - self._execute_cpu_time)
        

    def solve_constr_start(self):
        self._solve_wall_time = time.time()
        self._solve_cpu_time = time.process_time()

    def solve_constr_end(self):
        self.solve_constraint_wall_time.append(time.time() - self._solve_wall_time)
        self.solve_constraint_cpu_time.append(time.process_time() - self._solve_cpu_time)


    def start(self):
        self._start_wall_time = time.time()
        self._start_cpu_time = time.process_time()

    def end(self, constraint_complexity=None, *, completed: bool = True):
        self.total_wall_time = time.time() - self._start_wall_time
        self.total_cpu_time = time.process_time() - self._start_cpu_time
        self.is_finish = completed
        
        self.save_stats_dict(constraint_complexity=constraint_complexity)


    def total_timeout(self):
        if self.attack_label is None:
            self.is_timeout = True

    def no_ctr_to_solve(self):
        self.solve_all_ctr = True

    def mark_error(self, error_type, reason, *, phase=None, child_pid=None, event_type=None):
        self.extra_meta["status"] = "error"
        self.extra_meta["error_type"] = error_type
        self.extra_meta["error_reason"] = reason
        if phase is not None:
            self.extra_meta["error_phase"] = phase
        if child_pid is not None:
            self.extra_meta["child_pid"] = child_pid
        if event_type is not None:
            self.extra_meta["child_event_type"] = event_type

    def mark_child_event(self, event_type, message, *, phase=None, child_pid=None):
        self.extra_meta["child_event_type"] = event_type
        self.extra_meta["child_event_message"] = message
        if phase is not None:
            self.extra_meta["child_event_phase"] = phase
        if child_pid is not None:
            self.extra_meta["child_pid"] = child_pid

        
    def first_execution_end(self):
        # the iteration 0 has no constraint to solve
        # because iteration 0 only run self._one_execution to generate constrains
        # and at the beginning of iteration 1, we solve constraint first,
        # and then run self._one_execution again to generate new constrains.
        self.solve_constraint_wall_time.append(0)
        self.solve_constraint_cpu_time.append(0)

    def find_adversarial_input(self, input_dict, attack_label):
        adv_input = self._build_input_from_dict(input_dict)
        if adv_input is not None:
            self.adversarial_input = adv_input
        self.attack_label = attack_label
        
        
    def save_sat_input(self, input_dict):
        # 儲存solver找到的滿足條件的input
        sat_input = self._build_input_from_dict(input_dict)
        if sat_input is not None:
            self.sat_inputs.append(sat_input)
    
    def save_original_input(self, input_dict):
        if self.original_input is not None:
            return
        ori_input = self._build_input_from_dict(input_dict)
        if ori_input is not None:
            self.original_input = ori_input

    def _build_input_from_dict(self, input_dict):
        if self.input_shape is None:
            return None
        built = np.zeros(self.input_shape, dtype=np.float32)
        dims = len(self.input_shape)
        for key, value in input_dict.items():
            if not isinstance(key, str) or not key.startswith("v_"):
                continue
            try:
                idx = tuple(int(i) for i in key.split("_")[1:])
            except ValueError:
                continue
            if len(idx) != dims:
                continue
            out_of_bounds = any(axis < 0 or axis >= self.input_shape[d] for d, axis in enumerate(idx))
            if out_of_bounds:
                continue
            built[idx] = value
        return built
        
    
    def save_adversarial_input_as_image(self, save_path):
        if self.adversarial_input is not None:
            img_0_255 = self.adversarial_input.copy()
            img_0_255 = (img_0_255*255).astype(int)
            cv2.imwrite(save_path, img_0_255)

    @staticmethod
    def _summarize_numeric(values):
        if not values:
            return None
        total = sum(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": total / len(values),
            "sum": total,
        }

    @staticmethod
    def _count_values(values):
        counts = {}
        for val in values or []:
            counts[val] = counts.get(val, 0) + 1
        return counts or None

    def _build_progress(self):
        progress = self.extra_meta.get("progress")
        if progress is not None:
            return progress
        ton_current = self.extra_meta.get("ton")
        ton_next = self.extra_meta.get("ton_next")
        if ton_current is None and ton_next is None:
            return None
        return {
            "ton_current": ton_current,
            "ton_next": ton_next,
            "stop_at": None,
            "reason": None,
        }

    def output_stats_dict(self, constraint_complexity=None):
        status = "incomplete"
        if self.attack_label is not None:
            status = "success"
        elif self.is_timeout:
            status = "timeout"
        elif self.solve_all_ctr:
            status = "exhausted"

        meta = {
            "input_name": self.input_name,
            "original_label": self.original_label,
            "attack_label": self.attack_label,
            "is_finish": self.is_finish,
            "is_timeout": self.is_timeout,
            "solve_all_ctr": self.solve_all_ctr,
            "status": status,
        }
        if isinstance(self.extra_meta, dict):
            meta.update(self.extra_meta)

        summary = {
            "total_wall_time": self.total_wall_time,
            "total_cpu_time": self.total_cpu_time,
            "total_iter": self.total_iter,
        }
        summary["execute_wall_time_total"] = (
            sum(self.execute_wall_time) if self.execute_wall_time else None
        )
        summary["execute_cpu_time_total"] = (
            sum(self.execute_cpu_time) if self.execute_cpu_time else None
        )
        summary["solve_constraint_wall_time_total"] = (
            sum(self.solve_constraint_wall_time) if self.solve_constraint_wall_time else None
        )
        summary["solve_constraint_cpu_time_total"] = (
            sum(self.solve_constraint_cpu_time) if self.solve_constraint_cpu_time else None
        )
        summary["iter_wall_time_total"] = (
            sum(self.iter_wall_time) if self.iter_wall_time else None
        )
        summary["iter_cpu_time_total"] = (
            sum(self.iter_cpu_time) if self.iter_cpu_time else None
        )

        solver = {
            "sat": sum(self.sat),
            "unsat": sum(self.unsat),
            "unknown": sum(self.unknown),
            "solver_time_total": sum(self.solve_constraint_wall_time),
        }

        constraints = {
            "generated_total": sum(self.gen_constraint),
            "solved_total": sum(self.solve_constraint),
            "queue_max": self.queue_max,
        }

        iters_summary = {
            "sat": self._summarize_numeric(self.sat),
            "unsat": self._summarize_numeric(self.unsat),
            "unknown": self._summarize_numeric(self.unknown),
            "solve_constraint": self._summarize_numeric(self.solve_constraint),
            "gen_constraint": self._summarize_numeric(self.gen_constraint),
            "iter_wall_time": self._summarize_numeric(self.iter_wall_time),
            "iter_cpu_time": self._summarize_numeric(self.iter_cpu_time),
            "execute_wall_time": self._summarize_numeric(self.execute_wall_time),
            "execute_cpu_time": self._summarize_numeric(self.execute_cpu_time),
            "solve_constraint_wall_time": self._summarize_numeric(self.solve_constraint_wall_time),
            "solve_constraint_cpu_time": self._summarize_numeric(self.solve_constraint_cpu_time),
        }

        complexity_summary = None
        if isinstance(constraint_complexity, dict):
            complexity_summary = {}
            type_counts = self._count_values(constraint_complexity.get("type"))
            if type_counts is not None:
                complexity_summary["type_counts"] = type_counts
            for key in ("assert_num", "byte", "time", "path_len", "build_time", "total_time"):
                summary_stats = self._summarize_numeric(constraint_complexity.get(key, []))
                if summary_stats is not None:
                    complexity_summary[key] = summary_stats
            entries = constraint_complexity.get("detail")
            if isinstance(entries, list) and entries:
                complexity_summary["entries"] = entries

        res = {
            "meta": meta,
            "summary": summary,
            "solver": solver,
            "constraints": constraints,
            "constraint_complexity": complexity_summary,
            "iters_summary": iters_summary,
        }
        return res


    def save_stats_dict(self, constraint_complexity=None):
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            stats_dict = self.output_stats_dict(constraint_complexity=constraint_complexity)
            with open(os.path.join(self.save_dir, "stats.json"), 'w') as f:
                # json.dump(stats_dict, f, indent="\t") # 較容易讀懂但浪費儲存空間
                json.dump(stats_dict, f, default=_json_default) # 最節省儲存空間但不容易讀懂
            
            img_name = f"adv_{self.original_label}_to_{self.attack_label}.jpg"
            self.save_adversarial_input_as_image(os.path.join(self.save_dir, img_name))

            if self.original_input is not None:
                np.save(
                    os.path.join(self.save_dir, "ori_input.npy"),
                    self.original_input.astype(np.float32, copy=False),
                )
            if self.adversarial_input is not None:
                np.save(
                    os.path.join(self.save_dir, "adv_input.npy"),
                    self.adversarial_input.astype(np.float32, copy=False),
                )
                        
            # 取代原本的 np.save(..., np.array(self.sat_inputs))
            if len(self.sat_inputs) == 0:
                np.save(os.path.join(self.save_dir, "sat_inputs.npy"),
                        np.empty((0, *self.input_shape), dtype=np.float32))
            else:
                np.save(os.path.join(self.save_dir, "sat_inputs.npy"),
                        np.stack(self.sat_inputs).astype(np.float32))
