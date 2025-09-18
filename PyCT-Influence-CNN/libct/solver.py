import logging, os, re, subprocess, sys, time
from libct.concolic import Concolic
from libct.predicate import Predicate
from libct.utils import py2smt

log = logging.getLogger("ct.solver")

class Solver:
    # options = {"lan": "smt.string_solver=z3str3", "stdin": "-in"}
    cnt = 1 # for store
    # limit the percentage of variable, if x=100, percentage is 0.1, means that the range of new x is [100*0.9, 100*1.1]
    limit_change_range = None
    norm = None
    iter = None # for the filename of saved smt constraint
    iter_count = 1 # for the filename of saved smt constraint
    

    @classmethod # similar to our constructor
    def set_basic_configurations(cls, solver, timeout, safety, store, smtdir):
        cls.safety = safety; cls.smtdir = smtdir
        cls.stats = {'sat_number': 0, 'sat_time': 0, 'unsat_number': 0, 'unsat_time': 0, 'otherwise_number': 0, 'otherwise_time': 0}
        
        # assert_len 是一個二維的list，第一個維度是每個iteration，第二個維度是該iteration的每個assert的長度
        cls.ctr_size = {'type':[], 'time': [], 'byte': [], 'assert_num': [], 'assert_len':[]}
        
        if cls.smtdir:            
            os.makedirs(os.path.join(cls.smtdir, 'formula'))
            
        if store is not None:
            if not os.path.isdir(store):
                if not re.compile(r"^\d+$").match(store):
                    raise IOError(f"Query folder {store} not found")
        cls.store = store
        ##########################################################################################
        # Build the command from the solver type
        if solver == "cvc4":
            #cls.cmd = ["cvc4"] + ["--produce-models", "--lang", "smt", "--strings-exp"]
             cls.cmd = ["cvc4"] + ["--produce-models", "--lang", "smt", "--quiet", "--strings-exp"]
        # elif solver == "z3seq":
        #     cls.cmd = "z3 -in".split(' ')
        # elif solver == "z3str":
        #     cls.cmd = ["z3"] + self.options.values()
        # elif solver == "trauc":
        #     cls.cmd = ["trauc"] + self.options.values()
        else:
            raise NotImplementedError
        ##########################################################################################
        # Build the command from the timeout parameter
        assert isinstance(timeout, int)
        if "z3" in solver or  "trauc" in solver:
            cls.cmd += ["-T:" + str(timeout)]
        else:
            cls.cmd += ["--tlimit=" + str(1000 * timeout)]

    @classmethod
    def find_model_from_constraint(cls, engine, constraint, ori_args):
        print("[DEBUG]Finding model ... ")
        formulas = Solver._build_formulas_from_constraint(engine, constraint, ori_args); log.smtlib2(f"Solving To: {constraint}")
        start = time.time()
        try: completed_process = subprocess.run(cls.cmd, input=formulas.encode(), capture_output=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            with open("smt_error.txt", 'a') as f:
                f.writelines(e.output)

        elapsed = time.time() - start
        
        
        output = completed_process.stdout.decode()
        model = None
        if output is None or len(output) == 0:
            status = "UNKNOWN"
        else:
            outputs = output.splitlines()
            status = outputs[0].lower()
            if "error" in status:
                print('solver error:', status)
                print(f"at SMT-id: {Solver.cnt}")
                print(formulas)
                sys.exit(1)
            if "sat" == status:
                cls.stats['sat_number'] += 1; cls.stats['sat_time'] += elapsed
                model = Solver._get_model(engine, outputs[1:])
                # FIXME make the value of non-concolic argument unchanged
            else:
                if "unsat" == status: cls.stats['unsat_number'] += 1; cls.stats['unsat_time'] += elapsed
                else: status = "UNKNOWN"; cls.stats['otherwise_number'] += 1; cls.stats['otherwise_time'] += elapsed
        
        
        # {'type':[], 'time': [], 'byte': [], 'assert_num': [], 'assert_len':[]}
        def save_constraint_complexity():
            file_byte = len(formulas.encode('utf-8'))
            
            assert_count = 0  # 计数满足条件的行数
            assert_lens = []  # 用于存储每行字符串的长度
            pattern = r'\(assert.*'
            for match in re.finditer(pattern, formulas):
                line = match.group()
                assert_count += 1
                assert_lens.append(len(line))
            
            cls.ctr_size['type'].append(status)
            cls.ctr_size['time'].append(elapsed)
            cls.ctr_size['byte'].append(file_byte)            
            cls.ctr_size['assert_num'].append(assert_count)
            cls.ctr_size['assert_len'].append(assert_lens)
            
        
        save_constraint_complexity()
        
        ##########################################################################################
        if cls.store is not None:
            if re.compile(r"^\d+$").match(cls.store):
                if int(cls.store) == Solver.cnt:
                    with open(cls.store + f"_{status}.smt2", 'w') as f:
                        f.write(formulas)
            else:
                with open(os.path.join(cls.store, f"{Solver.cnt}_{status}.smt2"), 'w') as f:
                    f.write(formulas)
        
        if cls.smtdir:
            save_smt_filename = f"{Solver.iter}_{Solver.iter_count}_{status}.smt2"
            with open(os.path.join(cls.smtdir, "formula", save_smt_filename), 'w') as f:
                f.write(formulas)
            
        print(status)
        ##########################################################################################
        # log.smtlib2(f"SMT-id: {Solver.cnt}／Status: {status}／Model: {model}")
        Solver.cnt += 1
        Solver.iter_count += 1
        return model

    @staticmethod
    def _get_model(engine, models):
        model = {}
        for line in models:
            #print(line)
            assert line.startswith('((') and line.endswith('))')
            name, value = line[2:-2].split(" ", 1)
            if engine.var_to_types[name] == "Bool":
                if value == 'true': value = True
                elif value == 'false': value = False
                else: raise NotImplementedError
            elif engine.var_to_types[name] == "Real":
                if "(" in value:
                    tmp = value.replace("(", "").replace(")", "").replace("-", "").split()
                    if value.count('-') % 2 == 1:
                        value = - ( float(tmp[1])/float(tmp[2]) )
                    else:
                        value = ( float(tmp[1])/float(tmp[2]) ) 
                else:
                    value = float(value)
            elif engine.var_to_types[name] == "Int":
                if "(" in value:
                    if "-" in value:
                        value = -int(value.replace("(", "").replace(")", "").split(" ")[2])
                    else:
                        value = int(value.replace("(", "").replace(")", "").split(" ")[1])
                else:
                    value = int(value)
            elif engine.var_to_types[name] == "String":
                assert value.startswith('"') and value.endswith('"')
                value = value[1:-1]
                value = value.replace('""', '"').replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\")
                # Note the decoding order above must be in reverse with its encoding method (line 41 in libct/utils.py)
            else:
                raise NotImplementedError
            assert name.endswith('_VAR') # '_VAR' is used to avoid name collision
            model[name[:-len('_VAR')]] = value
        return model

    @staticmethod
    def _build_formulas_from_constraint(engine, constraint, ori_args):
        # declare_vars = "\n".join(f"(declare-const {name} {_type})" 
        #                for (name, _type) in engine.var_to_types.items()) #if engine.concolic_dict.get(name, 1))
        #NOTE DNN
        declare_vars = "\n".join(f"(declare-const {name} {engine.var_to_types[name]})"         ##問題在隻鱷        
                                 for (name) in engine.concolic_name_list)
        queries = "\n".join(assertion.get_formula() for assertion in constraint.get_all_asserts())
        
        norm_queries = ""        
        if Solver.norm: # limit solve range [0,1]
            norm_queries = "\n".join(f"(assert (and (<= {name} 1) (>= {name} 0)))"
                            for (name) in engine.concolic_name_list)
            
        if Solver.limit_change_range is not None:
            # limit solve range x +- p%, e.g. p=0.1, [100 * (1-p), 100 * (1+p)]
            limit_queries = []
            for name in engine.concolic_name_list:
                x = ori_args[name[:-4]] # not including _VAR
                lb = x * (1-Solver.limit_change_range) 
                ub = x * (1+Solver.limit_change_range)
                if lb < 0:
                    new_ub = abs(lb)
                    new_lb = abs(ub)
                    limit_queries.append(f"(assert (and (<= {name} (- {new_ub})) (>= {name} (- {new_lb}))))")
                else:
                    limit_queries.append(f"(assert (and (<= {name} {ub}) (>= {name} {lb})))")
            
            norm_queries += "\n".join(limit_queries)

        
        # get_vars = "\n".join(f"(get-value ({name}))" for name in engine.var_to_types.keys())
        #NOTE DNN
        get_vars = "\n".join(f"(get-value ({name}))" for name in engine.concolic_name_list)
        return f"(set-logic ALL)\n{declare_vars}\n{queries}\n{norm_queries}\n(check-sat)\n{get_vars}\n"

    @classmethod
    def _expr_has_engines_and_equals_value(cls, expr, value):
        if e:=Concolic.find_engine_in_expr(expr):
            if cls.safety <= 0: return e # This line is used to disable the value validation feature temporarily.
            if isinstance(value, float): # TODO: Floating point operations may cause subtle errors.
                formulas = f"(assert (and (<= (- (/ 1 1000000000000000)) (- {Predicate.get_formula_shallow(expr)} {py2smt(value)})) (<= (- {Predicate.get_formula_shallow(expr)} {py2smt(value)}) (/ 1 1000000000000000))))\n(check-sat)"
            else:
                formulas = f"(assert (= {Predicate.get_formula_shallow(expr)} {py2smt(value)}))\n(check-sat)"
            try: completed_process = subprocess.run(cls.cmd, input=formulas.encode(), capture_output=True)
            except subprocess.CalledProcessError as e: print(e.output)
            try:
                if completed_process.stdout.decode().splitlines()[0] == 'sat': return e
                raise Exception # move to the following block
            except:
                print(formulas); print(completed_process.stdout.decode().splitlines()); print()
                import traceback; traceback.print_stack()
                if cls.safety >= 2: sys.exit(1)
        return None
