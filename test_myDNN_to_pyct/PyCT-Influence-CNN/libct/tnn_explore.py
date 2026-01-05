import builtins, coverage, func_timeout, inspect, logging, multiprocessing, os, pickle, sys, traceback
from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.solver import Solver
from libct.utils import ConcolicObject, unwrap, get_in_dict_shape
from libct.record import ConcolicTestRecorder
import cProfile
import numpy as np 
import hashlib
import pandas as pd
from tensorflow.keras.models import load_model


log = logging.getLogger("ct.explore")
sys.setrecursionlimit(1000000) # The original limit is not enough in some special cases.
module = None
execute = None
recorder = None

def prepare():
    #################################################################
    # Since the source code in https://github.com/python/cpython/blob/e822e37946f27c09953bb5733acf3b07c2db690f/Modules/socketmodule.c#L6485
    # only accepts "unwrapped" input arguments, we simply do it here.
    #################################################################
    import socket
    _socket_getaddrinfo = socket.getaddrinfo
    def socket_getaddrinfo(*args, **kwargs):
        return _socket_getaddrinfo(*map(unwrap, args), **{k: unwrap(v) for (k, v) in kwargs.items()})
    socket.getaddrinfo = socket_getaddrinfo
    #####################################################################
    # The builtin len(...) function will automatically unwrap our result,
    # so we want to avoid this by doing the following line.
    #####################################################################
    builtins.len = lambda x: x.__len__()

class ExplorationEngine:
    class Exception(metaclass=type('', (type,), {"__repr__": lambda self: '<EXCEPTION>'})): pass # indicate occurrence of Exception during execution
    class Timeout(metaclass=type('', (type,), {"__repr__": lambda self: '<TIMEOUT>'})): pass # indicate timeout after either a concolic or a primitive execution
    class Unpicklable(metaclass=type('', (type,), {"__repr__": lambda self: '<UNPICKLABLE>'})): pass # indicate that an object is unpicklable
    class LazyLoading(metaclass=type('', (type,), {"__repr__": lambda self: '<DEFAULT>'})): pass # lazily loading default values of primitive arguments

    def __init__(self, *, solver='cvc4', timeout=20, safety=0, store=None,
                 verbose=1, logfile=None, statsdir=None, smtdir=None, save_dir=None, input_name=None,
                 module_, execute_, only_first_forward, model_name, matrix_shape):
        global module, execute

        module = module_
        execute = execute_
        
        
        self.save_dir = save_dir
        self.input_name = input_name
        self.only_first_forward = only_first_forward
        self.model_name = model_name
        self.matrix_shape = matrix_shape

        self.normalize = None
        self.__init2__(); self.statsdir = statsdir
        if self.statsdir: os.system(f"rm -rf '{statsdir}'"); os.system(f"mkdir -p '{statsdir}'")
        Solver.set_basic_configurations(solver, timeout, safety, store, smtdir)
        ############################################################
        # This section mainly deals with the logging settings.
        log_level = 25 - 5 * verbose
        if logfile:
            logging.basicConfig(filename=logfile, level=log_level,
                                format='%(asctime)s  %(name)s\t%(levelname)s\t %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
        elif logfile == '':
            logging.basicConfig(level=logging.CRITICAL+1)
        else:
            logging.basicConfig(level=log_level,# stream=sys.stdout,
                                format='  %(name)s\t%(levelname)s\t %(message)s')
        ############################################################
        # We add our new logging level called "SMTLIB2" to print out
        # messages related to the solving process.
        ############################################################
        logging.SMTLIB2 = (logging.DEBUG + logging.INFO) // 2
        logging.addLevelName(logging.SMTLIB2, "SMTLIB2")
        def smtlib2(self, message, *args, **kwargs): # https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/13638084#13638084
            if self.isEnabledFor(logging.SMTLIB2): # Yes, logger takes its '*args' as 'args'.
                self._log(logging.SMTLIB2, message, args, **kwargs)
        logging.Logger.smtlib2 = smtlib2

    def __init2__(self):
        global recorder
        recorder = ConcolicTestRecorder(self.save_dir, self.input_name)

        self.constraints_to_solve = [] # consists of the constraints that are going to be solved by the solver
        self.path = PathToConstraint()
        self.in_out = []
        self.coverage_data = coverage.CoverageData()
        self.coverage_accumulated_missing_lines = {}
        self.var_to_types = {}
        self.concolic_name_list = [] ##NOTE for DNN testing
        self.concolic_flag_dict = {} ##NOTE for DNN testing        
        self.previous_result = None
        self.original_args = None # used to limit variable range
        

    def _execution_loop(self, max_iterations, all_args, concolic_dict):
        recorder.start()
        Solver.norm = self.normalize
        Solver.limit_change_range = self.limit_change_range
        
        tried_input_args = [all_args.copy()] # .copy() is important!!
        iterations = 0
        cont = True

        # this execution only for generating constraints
        log.info(f"=== Iterations: {iterations} ===")
        recorder.iter_start(Solver)
        recorder.execution_start()
        self._one_execution(all_args, concolic_dict)
        recorder.execution_end()
        recorder.iter_end(Solver.stats, 0)
        recorder.gen_constraint.append(len(self.constraints_to_solve))
        recorder.first_execution_end()

        # first self.previous_result is the original label
        recorder.original_label = self.previous_result
        recorder.save_stats_dict()
        
        # After First execution, no constr to solve
        if len(self.constraints_to_solve) == 0:
            print('[FIRST_NO_CONSTR]: After First execution, no constr to solve')
            return 0


        while cont and (max_iterations==0 or iterations<max_iterations):
            ##############################################################
            # In each iteration, we take one constraint out of the queue
            # and try to solve for it. After that we'll obtain a model as
            # a list of arguments and continue the next iteration with it.            
            log.info(f"=== Iterations: {iterations+1} ===")
            recorder.iter_start(Solver)
            

            recorder.solve_constr_start()
            solve_constr_num = len(self.constraints_to_solve)
            while len(self.constraints_to_solve) > 0:
                
                if self.solve_order_stack:
                    # NOTE stack is used instead of queue for DNN
                    constraint = self.constraints_to_solve.pop() # stack
                else:
                    constraint = self.constraints_to_solve.pop(0) # queue
                
                model = Solver.find_model_from_constraint(self, constraint, self.original_args)
            
                if model is not None and not self.only_first_forward:
                    # sat
                    all_args.update(model) # from model to argument
                    recorder.save_sat_input(all_args)
                    if all_args not in tried_input_args:
                        # sat and this input args have not used
                        tried_input_args.append(all_args.copy()) # .copy() is important!!
                        break

            recorder.solve_constr_end()
            solve_constr_num = solve_constr_num - len(self.constraints_to_solve)

            # solve new input and use it to execute
            if not self.only_first_forward:
                gen_constr_num = len(self.constraints_to_solve)
                recorder.execution_start()
                cont = self._one_execution(all_args, concolic_dict)
                recorder.execution_end()
                gen_constr_num = len(self.constraints_to_solve) - gen_constr_num
                recorder.gen_constraint.append(gen_constr_num)
            
            iterations += 1
            recorder.iter_end(Solver.stats, solve_constr_num)
            recorder.save_stats_dict()
            ##############################################################

            if len(self.constraints_to_solve) == 0:
                recorder.no_ctr_to_solve()
                print('[SOLVED_ALL_CONSTR]: There is no constr to solve')
                break


    def explore(
        self, modpath, all_args={}, /, *, root='.', funcname=None,
        max_iterations=0, single_timeout=15, total_timeout=900,
        deadcode=set(), include_exception=False, lib=None, single_coverage=True,
        file_as_total=False, concolic_dict={}, solve_order_stack=False,
        norm=False, limit_change_range=None):
        
        self.modpath = modpath; self.funcname = funcname
        self.single_timeout = single_timeout; self.total_timeout = total_timeout
        self.include_exception = include_exception; self.deadcode = deadcode
        self.lib = lib; self.file_as_total = file_as_total; self.normalize = norm
        self.solve_order_stack = solve_order_stack
        self.limit_change_range = limit_change_range

        if self.funcname is None: self.funcname = self.modpath.split('.')[-1]
        
        self.__init2__()
                
        recorder.input_shape = get_in_dict_shape(all_args)
        self.original_args = all_args.copy()
        
        self.root = os.path.abspath(root)
        self.target_file = os.path.join(self.root, self.modpath.replace('./', ''))
        
        

        self.single_coverage = single_coverage
        if self.single_coverage:
            self.coverage = coverage.Coverage(data_file=None, include=[self.target_file])
        else:
            self.coverage = coverage.Coverage(data_file=None, source=[self.root], omit=['**/__pycache__/**', '**/.venv/**'])
        if self.lib: sys.path.insert(0, os.path.abspath(self.lib))
        sys.path.insert(0, self.root)#; sys.path.insert(0, file_dir)
        # file_dir = os.path.abspath(os.path.dirname(os.path.join(self.root, self.modpath.replace('.', '/') + '.py')))
        file_dir = os.path.abspath(os.path.dirname(os.path.join(self.root, self.modpath.replace('./', '') + '.py')))
        now_dir = os.getcwd(); os.chdir(file_dir)
        self.can_use_concolic_wrapper = self._can_use_concolic_wrapper(self.root, self.modpath)
        
        try:
            func_timeout.func_timeout(
                self.total_timeout, self._execution_loop, args=(max_iterations, all_args, concolic_dict)
            )

        except func_timeout.exceptions.FunctionTimedOut:            
            recorder.total_timeout()
            log.info('[TOTAL TIMEOUT]: Total Timeout happened')
            
        except BaseException as e: # importantly note that func_timeout.FunctionTimedOut is NOT inherited from the (general) Exception class.
            print(type(e))
            # traceback.print_exc()
        
        # 結束後，將所有的統計資料存起來
        recorder.end(constraint_complexity=Solver.ctr_size)
        
        
        # After finishing self._execution_loop, we can get total iteration from recorder
        iteration = recorder.total_iter
        

        os.chdir(now_dir); del sys.path[0]
        if self.lib: del sys.path[0]
        if self.statsdir:
            with open(self.statsdir + '/inputs.pkl', 'wb') as f:
                pickle.dump([e[0] for e in self.in_out], f) # store only inputs
            if self.single_coverage:
                with open(self.statsdir + '/missing_lines.txt', 'w') as f:
                    if self.file_as_total:
                        f.write(str(sorted(self.module_lines_range & self.coverage_accumulated_missing_lines[self.target_file])) + '\n')
                        f.write(str(sorted(self.module_lines_range)) + '\n')
                    else:
                        f.write(str(sorted(self.function_lines_range & self.coverage_accumulated_missing_lines[self.target_file])) + '\n')
                        f.write(str(sorted(self.function_lines_range)) + '\n')
            with open(self.statsdir + '/smt.csv', 'w') as f:
                f.write(',number,time\n')
                f.write(f'sat,{Solver.stats["sat_number"]},{Solver.stats["sat_time"]}\n')
                f.write(f'unsat,{Solver.stats["unsat_number"]},{Solver.stats["unsat_time"]}\n')
                f.write(f'otherwise,{Solver.stats["otherwise_number"]},{Solver.stats["otherwise_time"]}\n')
        
        return iteration, recorder


    def _one_execution(self, all_args, concolic_dict):

        result = self._one_execution_concolic(all_args, concolic_dict) # primitive input arguments "all_args" may be modified here.   
        origin_result= self._one_execution_predicted(all_args)
        print("origin_result :",origin_result)
        if not self.single_coverage: # We don't measure coverage in the primitive mode under the non-single coverage setting.
            self.in_out.append((all_args.copy(), result)) # .copy() is important! Think why.
            return True # continue iteration
        answer = self._one_execution_primitive(all_args) # we must measure the coverage in the primitive mode since self.constraints_to_solve would become unpicklable if measured in the concolic mode
        
        if self.Timeout not in (result, answer):
            if result != answer: print('Input:', all_args, '／My result:', result, '／Correct answer:', answer)
            assert result == answer
        else:
            print('[DEBUG] Single Timeout happened')
        # Note only in the self.single_coverage mode does the program go here.
        if self.file_as_total:
            s = (self.module_lines_range - self.deadcode) & self.coverage_accumulated_missing_lines[self.target_file]
        else:
            s = (self.function_lines_range - self.deadcode) & self.coverage_accumulated_missing_lines[self.target_file]
        if self.previous_result != None and self.previous_result != origin_result:
            print('#'*60)
            print(f'[Result Change]: self.previous_result : {self.previous_result}!= result : {origin_result}')
            print('#'*60)
            recorder.find_adversarial_input(all_args, origin_result)
            return False
        self.previous_result = origin_result
        print(f'[Result Not Change]: self.previous_result : {self.previous_result}== result : {origin_result}')

        return True
        # return s # continue iteration only if the target file / function coverage is not full yet.
    # def _one_execution_predicted(self, all_args):
    #     def convert_dict_to_matrix(all_args):
    #         matrix = np.zeros((28, 28, 1))
    #         for key, value in all_args.items():
    #             indices = key.split('_')[1:]  # ['0', '0', '0']
    #             indices = [int(index) for index in indices]
    #             matrix[indices[0], indices[1], indices[2]] = value
    #         return matrix
    #     input = convert_dict_to_matrix(all_args)
    #     input = np.expand_dims(input, axis=0)
    #     model_name="transformer_fashion_mnist"
    #     cnn_model = load_model('model/'+model_name+'.h5')
    #     adversarial_predictions = cnn_model.predict(input)
    #     adversarial_predicted_class = np.argmax(adversarial_predictions, axis=-1)
    #     return adversarial_predicted_class.tolist()[0]
    # def _one_execution_predicted(self, all_args):
    #     def convert_dict_to_matrix(all_args):
    #         matrix = np.zeros((500, 32,1))
    #         for key, value in all_args.items():
    #             indices = key.split('_')[1:]  # ['0', '0', '0']
    #             indices = [int(index) for index in indices]
    #             matrix[indices[0], indices[1]] = value
    #         return matrix
    #     input = convert_dict_to_matrix(all_args)
    #     input = np.expand_dims(input, axis=0)
    #     model_name="imdb_LSTM_08509"
    #     cnn_model = load_model('model/'+model_name+'.h5')
    #     adversarial_predictions = cnn_model.predict(input)
    #     adversarial_binary_result = (adversarial_predictions >= 0.5).astype(int)
    #     return adversarial_binary_result.tolist()[0][0]
    def _one_execution_predicted(self, all_args):
        def convert_dict_to_matrix(all_args):
            matrix = np.zeros(self.matrix_shape)
            for key, value in all_args.items():
                indices = key.split('_')[1:]
                indices = [int(index) for index in indices]
                matrix[tuple(indices[:len(self.matrix_shape)])] = value
            return matrix
        
        input = convert_dict_to_matrix(all_args)
        input = np.expand_dims(input, axis=0)
        cnn_model = load_model(f'model/{self.model_name}.h5')
        if "mnist" in self.model_name.lower():
            adversarial_predictions = cnn_model.predict(input)
            adversarial_predicted_class = np.argmax(adversarial_predictions, axis=-1)
            return adversarial_predicted_class.tolist()[0]
        elif "imdb" in self.model_name.lower():
            adversarial_predictions = cnn_model.predict(input)
            adversarial_binary_result = (adversarial_predictions >= 0.5).astype(int)
            return adversarial_binary_result.tolist()[0][0]

    def _one_execution_concolic(self, all_args, concolic_dict):
        r1, s1 = multiprocessing.Pipe(); r2, s2 = multiprocessing.Pipe(); r3, s3 = multiprocessing.Pipe(); r0, s0 = multiprocessing.Pipe()
        def child_process():
            sys.dont_write_bytecode = True # very important to prevent the later primitive mode from using concolic objects imported here...
            prepare(); self.path.__init__()
            # log.info("Inputs: " + str(all_args))
            if self.can_use_concolic_wrapper:
                import libct.wrapper
            else:
                import libct

            # module = get_module_from_rootdir_and_modpath(self.root, self.modpath)
            # execute = get_function_from_module_and_funcname(module, self.funcname)

            ccc_args, ccc_kwargs = self._get_concolic_arguments(execute, all_args, concolic_dict) # primitive input arguments "all_args" may be modified here.
                        
            s1.send((all_args, self.var_to_types, self.concolic_name_list, self.concolic_flag_dict))
            result = self.Exception
            try:
                result = libct.utils.unwrap(func_timeout.func_timeout(self.single_timeout, execute, args=ccc_args, kwargs=ccc_kwargs))
                log.info(f"Return: {result}")
            except func_timeout.FunctionTimedOut:
                result = self.Timeout
                log.error(f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception")#; traceback.print_exc()
                if self.statsdir:
                    with open(self.statsdir + '/exception.txt', 'a') as f:
                        print(f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception", file=f)
            except Exception as e:
                # log.error(f"Exception for:  >> ./pyct '{self.root}' '{self.modpath}' -s {self.funcname} {{}} -m 20 --lib '{self.lib}' --include_exception"); #log.error(e); traceback.print_exc()
                if self.statsdir:
                    with open(self.statsdir + '/exception.txt', 'a') as f:
                        #print(f"Exception for: {all_args} >> ./pyct '{self.root}' '{self.modpath}' -s {self.funcname} {{}} -m 20 --lib '{self.lib}' --include_exception", file=f); print(e, file=f)
                        pass
            ###################################### Communication Section ######################################
            s0.send(0) # just a notification to the parent process that we're going to send data
            try: s2.send(result)
            except: s2.send(self.Unpicklable)

            try:
                s3.send((Constraint.global_constraints, self.constraints_to_solve, self.path))                
            except Exception as e:
                traceback.print_exc()
                s3.send(self.Unpicklable) # may fail if they contain some unpicklable objects

        process = multiprocessing.Process(target=child_process); process.start()
        (all_args2, self.var_to_types, self.concolic_name_list, self.concolic_flag_dict) =\
            r1.recv(); r1.close(); s1.close(); all_args.clear()            
        all_args.update(all_args2) # update the parameter directly

        if not r0.poll(self.single_timeout + 5):
            result = self.Timeout
            log.error(f"Timeout (hard) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception")
            if self.statsdir:
                with open(self.statsdir + '/exception.txt', 'a') as f:
                    print(f"Timeout (hard) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception", file=f)
        else:
            result = r2.recv()

            if (t:=r3.recv()) is not self.Unpicklable:
                Constraint.global_constraints, self.constraints_to_solve, self.path = t
            else:
                assert t is not self.Unpicklable

                
        r2.close(); s2.close(); r3.close(); s3.close(); r0.close(); s0.close()
        if process.is_alive(): process.kill()
        return result

    def _one_execution_primitive(self, all_args):
        r1, s1 = multiprocessing.Pipe(); r2, s2 = multiprocessing.Pipe(); r0, s0 = multiprocessing.Pipe()
        def child_process():
            sys.dont_write_bytecode = True # same reason mentioned in the concolic mode
            self.coverage.start()
            # module = get_module_from_rootdir_and_modpath(self.root, self.modpath)
            # execute = get_function_from_module_and_funcname(module, self.funcname)
            s1.send(set(self.coverage.analysis(self.target_file)[1]) & set(range(1, 1+len(inspect.getsourcelines(module)[0])))) # Note inspect.getsourcelines(module)[1] always returns 0, which is not the fact.
            s1.send(set(self.coverage.analysis(self.target_file)[1]) & set(range(inspect.getsourcelines(execute)[1], inspect.getsourcelines(execute)[1] + len(inspect.getsourcelines(execute)[0]))))
            pri_args, pri_kwargs = self._complete_primitive_arguments(execute, all_args)
            answer = self.Exception
            try:
                answer = func_timeout.func_timeout(self.single_timeout, execute, args=pri_args, kwargs=pri_kwargs)
            except func_timeout.FunctionTimedOut: answer = self.Timeout
            except: pass
            self.coverage.stop(); self.coverage_data.update(self.coverage.get_data())
            for file in self.coverage_data.measured_files(): # "file" is absolute here.
                _, _, missing_lines, _ = self.coverage.analysis(file)
                if file not in self.coverage_accumulated_missing_lines:
                    self.coverage_accumulated_missing_lines[file] = set(missing_lines)
                else:
                    self.coverage_accumulated_missing_lines[file] &= set(missing_lines)
            ###################################### Communication Section ######################################
            s0.send(0) # just a notification to the parent process that we're going to send data
            try: s1.send(answer)
            except: answer = self.Unpicklable; s1.send(answer)
            if self.include_exception or (answer is not self.Exception):
                s2.send((self.coverage_data, self.coverage_accumulated_missing_lines))
            else:
                s2.send(self.Exception)
        process = multiprocessing.Process(target=child_process); process.start()
        self.module_lines_range = r1.recv(); self.function_lines_range = r1.recv()
        if not r0.poll(self.single_timeout + 5): answer = self.Timeout
        else:
            answer = r1.recv()
            if (t:=r2.recv()) is not self.Exception: (self.coverage_data, self.coverage_accumulated_missing_lines) = t
        if self.target_file not in self.coverage_accumulated_missing_lines: self.coverage_accumulated_missing_lines[self.target_file] = self.module_lines_range
        self.in_out.append((all_args.copy(), answer)); r1.close(); s1.close(); r2.close(); s2.close(); r0.close(); s0.close()
        if process.is_alive(): process.kill()
        return answer

    @classmethod
    def _complete_primitive_arguments(cls, func, all_args):
        prim_args = []; prim_kwargs = {}
        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL, ):
                continue # ignore *args
            elif v.kind in (inspect.Parameter.VAR_KEYWORD, ):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1
                prim_kwargs = all_args.copy()
                break                 
            else:
                value = v.default if (t:=all_args[v.name]) is cls.LazyLoading else t
                if v.kind is inspect.Parameter.KEYWORD_ONLY:
                    prim_kwargs[v.name] = value
                else: # v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    prim_args.append(value)


        return prim_args, prim_kwargs

    def _get_concolic_arguments(self, func, prim_args, concolic_dict):
        ccc_args = []; ccc_kwargs = {}
        self.concolic_name_list = []

        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL, ):
                # do not support *args currently
                prim_args.pop(v.name, None); continue
            
            elif v.kind in (inspect.Parameter.VAR_KEYWORD, ):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1

                for name, value in prim_args.items():
                    ccc_obj_name = name + '_VAR'  # '_VAR' is used to avoid name collision
                    self.concolic_flag_dict[ccc_obj_name] = 0
                    if type(value) in (bool, float, int, str) and concolic_dict[name]:
                        value = ConcolicObject(value, ccc_obj_name, self)
                        self.concolic_name_list.append(ccc_obj_name)
                        self.concolic_flag_dict[ccc_obj_name] = 1
                    
                    ccc_kwargs[name] = value

                break                
            else:
                if v.name in prim_args:
                    value = prim_args[v.name]
                else:
                    has_value = False
                    if (t:=v.annotation) is not inspect._empty:
                        try: value = t(); has_value = True # may raise TypeError: Cannot instantiate ...
                        except: pass
                    if not has_value:
                        if (t:=v.default) is not inspect._empty: value = unwrap(t) # default values may also be wrapped
                        else: value = ''
                    prim_args[v.name] = value if type(value) in (bool, float, int, str) else self.LazyLoading

                self.concolic_flag_dict[v.name+'_VAR'] = 0
                if type(value) in (bool, float, int, str) and concolic_dict.get(v.name, 1): 
                    #print(v.name + " set to ConcolicObj")
                    value = ConcolicObject(value, v.name + '_VAR', self) # '_VAR' is used to avoid name collision
                    self.concolic_name_list.append(v.name + '_VAR')
                    self.concolic_flag_dict[v.name+'_VAR'] = 1
                
                if v.kind is inspect.Parameter.KEYWORD_ONLY:
                    ccc_kwargs[v.name] = value
                else: # v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    ccc_args.append(value)


        if not self.var_to_types: # remain unchanged once determined
            for (k, v) in prim_args.items():
                k += '_VAR' # '_VAR' is used to avoid name collision
                if type(v) is bool: self.var_to_types[k] = 'Bool'
                elif type(v) is float: self.var_to_types[k] = 'Real'
                elif type(v) is int: self.var_to_types[k] = 'Int'
                elif type(v) is str: self.var_to_types[k] = 'String'
                else: pass # for some default values that cannot be concolic-ized

        return ccc_args, ccc_kwargs

    def _can_use_concolic_wrapper(self, root, modpath):
        r, s = multiprocessing.Pipe()
        if os.fork() == 0: # child process
            try:
                import libct.wrapper
                # module = get_module_from_rootdir_and_modpath(root, modpath)
                s.send(1)
            except:
                s.send(0)
            os._exit(os.EX_OK)
        os.wait()
        ans = r.recv(); r.close(); s.close()
        return ans

    def coverage_statistics(self):
        total_lines = 0
        executed_lines = 0
        missing_lines = {}
        for file in self.coverage_data.measured_files():
            executable_lines = set(self.coverage.analysis(file)[1])
            if file == self.target_file and not self.file_as_total:
                executable_lines &= self.function_lines_range
            m_lines = self.coverage_accumulated_missing_lines[file]
            if file == self.target_file and not self.file_as_total:
                m_lines &= self.function_lines_range
            total_lines += len(set(executable_lines))
            executed_lines += len(set(executable_lines)) - len(m_lines) # Do not use "len(set(self.coverage_data.lines(file)))" here!!!
            if m_lines: missing_lines[file] = m_lines
            # print(file, executed_lines, total_lines)
        if self.statsdir:
            with open(self.statsdir + '/coverage.txt', 'w') as f:
                f.write("{}/{} ({:.2%})\n".format(executed_lines, total_lines, (executed_lines/total_lines) if total_lines > 0 else 0))
        return total_lines, executed_lines, missing_lines

    def print_coverage(self):
        total_lines, executed_lines, missing_lines = self.coverage_statistics()
        print("\nLine coverage {}/{} ({:.2%})".format(executed_lines, total_lines, (executed_lines/total_lines) if total_lines > 0 else 0))
        if missing_lines and self.single_coverage:
            print("Missing lines:") # only print this info when the scope of coverage is a single file.
            for file, lines in missing_lines.items():
                print(f"  {file}: {sorted(lines)}")
        print("")
