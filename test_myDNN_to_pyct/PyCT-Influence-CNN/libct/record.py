import time
import numpy as np
import cv2
import os
import json

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
        self.is_finish = False # finish all iteration or generate an adversarial input
        self.is_timeout = False
        self.solve_all_ctr = False # solve all constraints

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

    def end(self, constraint_complexity=None):
        self.total_wall_time = time.time() - self._start_wall_time
        self.total_cpu_time = time.process_time() - self._start_cpu_time
        self.is_finish = True
        
        self.save_stats_dict(constraint_complexity=constraint_complexity)


    def total_timeout(self):
        self.is_timeout = True
        
    def no_ctr_to_solve(self):
        self.solve_all_ctr = True
     
        
    def first_execution_end(self):
        # the iteration 0 has no constraint to solve
        # because iteration 0 only run self._one_execution to generate constrains
        # and at the beginning of iteration 1, we solve constraint first,
        # and then run self._one_execution again to generate new constrains.
        self.solve_constraint_wall_time.append(0)
        self.solve_constraint_cpu_time.append(0)

    def find_adversarial_input(self, input_dict, attack_label):
        adv_input = np.zeros(self.input_shape).astype(np.float32)

        for k, v in input_dict.items():
            idx = k.split('_')[1:]
            
            if len(self.input_shape) == 2:
                i, j = (int(i) for i in idx)
                adv_input[i, j] = v
            elif len(self.input_shape) == 3:
                i, j, k = (int(i) for i in idx)
                adv_input[i, j, k] = v
            elif len(self.input_shape) == 4:
                i, j, k, l = (int(i) for i in idx)
                adv_input[i, j, k, l] = v
        
        self.adversarial_input = adv_input
        self.attack_label = attack_label
        
        
    def save_sat_input(self, input_dict):
        # 儲存solver找到的滿足條件的input
        sat_input = np.zeros(self.input_shape).astype(np.float32)

        for k, v in input_dict.items():
            idx = k.split('_')[1:]
            
            if len(self.input_shape) == 2:
                i, j = (int(i) for i in idx)
                sat_input[i, j] = v
            elif len(self.input_shape) == 3:
                i, j, k = (int(i) for i in idx)
                sat_input[i, j, k] = v
            elif len(self.input_shape) == 4:
                i, j, k, l = (int(i) for i in idx)
                sat_input[i, j, k, l] = v
        
        self.sat_inputs.append(sat_input)
        
    
    def save_adversarial_input_as_image(self, save_path):
        if self.adversarial_input is not None:
            img_0_255 = self.adversarial_input.copy()
            img_0_255 = (img_0_255*255).astype(int)
            cv2.imwrite(save_path, img_0_255)
        

    def output_stats_dict(self):
        res = {
            "meta": dict(),
            "total": dict(),
            "iters": dict(),
            "constraint_complexity": None,
        }
        res['meta']['input_name'] = self.input_name
        res['meta']['original_label'] = self.original_label
        res['meta']['attack_label'] = self.attack_label
        res['meta']['is_finish'] = self.is_finish
        res['meta']['is_timeout'] = self.is_timeout
        res['meta']['solve_all_ctr'] = self.solve_all_ctr
        
        
        res['total']['total_wall_time'] = self.total_wall_time
        res['total']['total_cpu_time'] = self.total_cpu_time
        res['total']['total_iter'] = self.total_iter


        res['iters']['sat'] = self.sat
        res['iters']['unsat'] = self.unsat
        res['iters']['unknown'] = self.unknown
        res['iters']['gen_constraint'] = self.gen_constraint
        res['iters']['solve_constraint'] = self.solve_constraint
        res['iters']['iter_wall_time'] = self.iter_wall_time
        res['iters']['iter_cpu_time'] = self.iter_cpu_time
        res['iters']['execute_wall_time'] = self.execute_wall_time
        res['iters']['execute_cpu_time'] = self.execute_cpu_time
        res['iters']['solve_constraint_wall_time'] = self.solve_constraint_wall_time
        res['iters']['solve_constraint_cpu_time'] = self.solve_constraint_cpu_time


        return res

    def save_stats_dict(self, constraint_complexity=None):
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            with open(os.path.join(self.save_dir, "stats.json"), 'w') as f:
                stats_dict = self.output_stats_dict()
                stats_dict['constraint_complexity'] = constraint_complexity
                serializable_stats = json_serializable(stats_dict)
                
                try:
                    json.dump(serializable_stats, f)
                except TypeError as e:
                    print(f"Error during JSON serialization: {e}")
                    print("Problematic data:")
                    for key, value in serializable_stats.items():
                        print(f"{key}: {type(value)}")
            
            img_name = f"adv_{self.original_label}_to_{self.attack_label}.jpg"
            self.save_adversarial_input_as_image(os.path.join(self.save_dir, img_name))
                        
            np.save(os.path.join(self.save_dir, "sat_inputs.npy"), np.array(self.sat_inputs))

def json_serializable(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return json_serializable(obj.__dict__)
    else:
        return str(obj)
