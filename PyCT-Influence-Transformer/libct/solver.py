import logging, math, os, re, subprocess, sys, time, traceback, func_timeout, unittest
from fractions import Fraction
from typing import Optional
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from libct.concolic import Concolic
from libct.position import summarize_position
from libct.predicate import Predicate
from libct.utils import py2smt
from tasks.paths import get_repo_output_subdir


log = logging.getLogger("ct.solver")
_SMTLIB2_REGISTERED = False


class InvalidSolverModelError(ValueError):
    """Raised when a SAT model value cannot be decoded into a finite primitive."""


def _parse_smt_expression(value: str):
    tokens = re.findall(r"\(|\)|[^\s()]+", value)
    if not tokens:
        raise InvalidSolverModelError("empty SMT value")

    def parse_at(index):
        token = tokens[index]
        if token != "(":
            return token, index + 1

        items = []
        index += 1
        while index < len(tokens) and tokens[index] != ")":
            item, index = parse_at(index)
            items.append(item)
        if index >= len(tokens):
            raise InvalidSolverModelError("unbalanced SMT value")
        return items, index + 1

    expression, next_index = parse_at(0)
    if next_index != len(tokens):
        raise InvalidSolverModelError("unexpected trailing tokens in SMT value")
    return expression


def _parse_real_fraction(expression) -> Fraction:
    if isinstance(expression, str):
        try:
            return Fraction(expression)
        except (ValueError, ZeroDivisionError) as exc:
            raise InvalidSolverModelError("unsupported SMT Real atom") from exc

    if not isinstance(expression, list) or not expression:
        raise InvalidSolverModelError("unsupported SMT Real expression")

    operator = expression[0]
    operands = expression[1:]
    if operator == "-" and len(operands) == 1:
        return -_parse_real_fraction(operands[0])
    if operator == "/" and len(operands) == 2:
        denominator = _parse_real_fraction(operands[1])
        if denominator == 0:
            raise InvalidSolverModelError("zero denominator in SMT Real")
        return _parse_real_fraction(operands[0]) / denominator
    raise InvalidSolverModelError("unsupported SMT Real operator")


def _parse_real_model_value(value: str) -> float:
    fraction = _parse_real_fraction(_parse_smt_expression(value.strip()))
    try:
        parsed = float(fraction)
    except OverflowError as exc:
        raise InvalidSolverModelError("SMT Real is outside the float range") from exc
    if not math.isfinite(parsed):
        raise InvalidSolverModelError("SMT Real decoded to a non-finite float")
    return parsed


def _float_debug(value: str):
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        return {"finite": False, "class": type(exc).__name__}
    if math.isfinite(parsed):
        return {"finite": True, "class": "finite", "value": parsed}
    if math.isnan(parsed):
        return {"finite": False, "class": "nan"}
    if parsed > 0:
        return {"finite": False, "class": "positive_infinity"}
    return {"finite": False, "class": "negative_infinity"}


def _float_result_debug(value: float):
    if math.isfinite(value):
        return {"finite": True, "class": "finite", "value": value}
    if math.isnan(value):
        return {"finite": False, "class": "nan"}
    if value > 0:
        return {"finite": False, "class": "positive_infinity"}
    return {"finite": False, "class": "negative_infinity"}


def _integer_atom_digits(value: str):
    atom = value
    if atom.startswith("-"):
        atom = atom[1:]
    if atom.isdigit():
        return len(atom)
    return None


def _unwrap_negative_integer(expression):
    if isinstance(expression, str):
        return expression
    if (
        isinstance(expression, list)
        and len(expression) == 2
        and expression[0] == "-"
        and isinstance(expression[1], str)
    ):
        return "-" + expression[1]
    return None


def _unwrap_rational_expression(expression):
    sign = 1
    if (
        isinstance(expression, list)
        and len(expression) == 2
        and expression[0] == "-"
    ):
        sign = -1
        expression = expression[1]
    if not (
        isinstance(expression, list)
        and len(expression) == 3
        and expression[0] == "/"
    ):
        return None

    numerator = _unwrap_negative_integer(expression[1])
    denominator = _unwrap_negative_integer(expression[2])
    if numerator is None or denominator is None:
        return None
    if numerator.startswith("-"):
        sign *= -1
        numerator = numerator[1:]
    if denominator.startswith("-"):
        sign *= -1
        denominator = denominator[1:]
    return sign, numerator, denominator


def _describe_real_model_value(value: str):
    raw_value = value.strip()
    diagnostic = {
        "raw_value": raw_value,
        "raw_value_length": len(raw_value),
        "is_rational": False,
    }
    try:
        expression = _parse_smt_expression(raw_value)
        fraction = _parse_real_fraction(expression)
    except InvalidSolverModelError as exc:
        diagnostic["parse_error"] = str(exc)
        return diagnostic

    rational = _unwrap_rational_expression(expression)
    if rational is not None:
        sign, numerator, denominator = rational
        diagnostic.update(
            {
                "is_rational": True,
                "sign": sign,
                "numerator_digits": _integer_atom_digits(numerator),
                "denominator_digits": _integer_atom_digits(denominator),
            }
        )
        legacy_numerator = _float_debug(numerator)
        legacy_denominator = _float_debug(denominator)
        diagnostic["legacy_numerator_float"] = legacy_numerator
        diagnostic["legacy_denominator_float"] = legacy_denominator
        try:
            legacy_division = (
                sign
                * legacy_numerator.get("value", math.copysign(math.inf, sign))
                / legacy_denominator.get("value", math.inf)
            )
            diagnostic["legacy_division"] = _float_result_debug(legacy_division)
        except (OverflowError, ZeroDivisionError) as exc:
            diagnostic["legacy_division"] = {
                "finite": False,
                "class": type(exc).__name__,
            }
    else:
        atom_digits = _integer_atom_digits(raw_value)
        if atom_digits is not None:
            diagnostic["atom_digits"] = atom_digits

    try:
        exact_float = float(fraction)
    except OverflowError as exc:
        diagnostic["exact_float"] = {"finite": False, "class": type(exc).__name__}
    else:
        diagnostic["exact_float"] = _float_result_debug(exact_float)
    diagnostic["exact_in_norm_range"] = bool(0 <= fraction <= 1)
    return diagnostic


def _build_model_diagnostics(engine, model_lines):
    diagnostics = []
    for line in model_lines:
        entry = {"line": line}
        if not line.startswith('((') or not line.endswith('))'):
            entry["parse_error"] = "malformed SMT model line"
            diagnostics.append(entry)
            continue
        try:
            name, value = line[2:-2].split(" ", 1)
        except ValueError:
            entry["parse_error"] = "malformed SMT model binding"
            diagnostics.append(entry)
            continue
        value_type = getattr(engine, "var_to_types", {}).get(name)
        entry.update(
            {
                "name": name,
                "value_type": value_type,
                "raw_value": value,
                "raw_value_length": len(value),
            }
        )
        if value_type == "Real":
            entry["real"] = _describe_real_model_value(value)
        diagnostics.append(entry)
    return diagnostics


def _ensure_smtlib2_logger() -> None:
    """Register the custom SMTLIB2 logging level once per process."""
    global _SMTLIB2_REGISTERED
    if _SMTLIB2_REGISTERED:
        return

    level = getattr(logging, "SMTLIB2", (logging.DEBUG + logging.INFO) // 2)
    logging.SMTLIB2 = level
    logging.addLevelName(level, "SMTLIB2")

    if not hasattr(logging.Logger, "smtlib2"):
        def smtlib2(self, message, *args, **kwargs):
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)
        logging.Logger.smtlib2 = smtlib2  # type: ignore[attr-defined]

    _SMTLIB2_REGISTERED = True


_ensure_smtlib2_logger()

class Solver:
    # options = {"lan": "smt.string_solver=z3str3", "stdin": "-in"}
    cnt = 1 # for store
    # limit the percentage of variable, if x=100, percentage is 0.1, means that the range of new x is [100*0.9, 100*1.1]
    limit_change_range = None
    norm = None
    iter = None # for the filename of saved smt constraint
    iter_count = 1 # for the filename of saved smt constraint
    build_timeout_enabled = True
    build_timeout_seconds: Optional[int] = 30
    run_timeout: Optional[int] = None
    

    @classmethod # similar to our constructor
    def set_basic_configurations(
        cls,
        solver,
        timeout,
        safety,
        store,
        smtdir,
        constraint_build_timeout=True,
        constraint_build_timeout_seconds: Optional[int] = 30,
        solver_run_timeout: Optional[int] = None,
    ):
        cls.safety = safety; cls.smtdir = smtdir
        cls.stats = {
            'sat_number': 0,
            'sat_time': 0,
            'unsat_number': 0,
            'unsat_time': 0,
            'otherwise_number': 0,
            'otherwise_time': 0,
            'invalid_model_number': 0,
        }
        cls.build_timeout_enabled = bool(constraint_build_timeout)
        if cls.build_timeout_enabled:
            timeout_seconds = 30 if constraint_build_timeout_seconds is None else int(constraint_build_timeout_seconds)
            if timeout_seconds < 1:
                raise ValueError(
                    f"constraint_build_timeout_seconds must be >= 1 when enabled, got {timeout_seconds}"
                )
            cls.build_timeout_seconds = timeout_seconds
        else:
            cls.build_timeout_seconds = None
        cls.run_timeout = solver_run_timeout
        
        # assert_len 是一個二維的list，第一個維度是每個iteration，第二個維度是該iteration的每個assert的長度
        cls.ctr_size = {
            'type': [],
            'time': [],
            'byte': [],
            'assert_num': [],
            'assert_len': [],
            'path_len': [],
            'build_time': [],
            'total_time': [],
            'detail': [],
        }
        
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
        elif solver == "cvc5":
            cls.cmd = ["cvc5"] + ["--produce-models", "--lang", "smt", "--quiet", "--strings-exp"]
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
    def _derive_attack_mode(cls, engine) -> str:
        """Return attack mode label derived from engine metadata."""
        override = getattr(engine, "popped_log_attack_mode", None)
        if isinstance(override, str):
            normalized = override.strip().lower()
            if normalized and normalized != "unknown":
                return normalized
        shap_flag = getattr(engine, "shap_value_pre_calculated", None)
        if shap_flag is True:
            return "shap"
        if shap_flag is False:
            return "random"
        return "unknown"

    @classmethod
    def _derive_attack_ton(cls, engine) -> str:
        """Estimate ton value from concolic flags; fall back to 'unknown'."""
        flags = getattr(engine, "concolic_flag_dict", None)
        if not flags:
            return "unknown"
        ton = sum(1 for flag in flags.values() if flag)
        return str(ton if ton > 0 else 0)

    @classmethod
    def _resolve_constraint_log_path(cls, engine, idx: int) -> Path:
        """Compute constraint log path grouped by model and attack parameters."""
        model_path = getattr(engine, "model_path", None)
        model_name = Path(model_path).stem if model_path else "unknown_model"
        attack_mode = cls._derive_attack_mode(engine)
        ton_label = cls._derive_attack_ton(engine)
        output_dir = get_repo_output_subdir(
            "popped_constraint_position",
            model_name,
            f"{attack_mode}_{ton_label}",
        )
        return output_dir / f"constraint_{idx}.txt"

    @staticmethod
    def _append_constraint_log(log_path: Path, position, shap_value, message: str) -> None:
        """Append a constraint entry to the resolved log file."""
        with log_path.open("a", encoding="utf-8") as file:
            file.write("\n")
            file.write(f"popped constraint with position: {summarize_position(position)}\n")
            file.write(f"popped constraint with shap value: {shap_value}\n")
            file.write(f"{message}")

    @classmethod
    def find_model_from_constraint(cls, engine,constraint,shap_value, position, idx, ori_args):
        log.debug(
            "Finding model (idx=%s, position=%s, shap_value=%s)",
            idx,
            summarize_position(position),
            shap_value,
        )
        log_path = cls._resolve_constraint_log_path(engine, idx)
        path_len = getattr(constraint, "height", None)
        current_iter = cls.iter
        current_attempt_index = cls.iter_count
        model_error = None
        solver_stdout = None
        solver_stderr = None
        solver_returncode = None
        solver_raw_model_lines = None
        solver_model_diagnostics = None

        def _record_constraint_complexity(status, formulas, build_elapsed, solver_elapsed):
            for key in (
                "type",
                "time",
                "byte",
                "assert_num",
                "assert_len",
                "path_len",
                "build_time",
                "total_time",
                "detail",
            ):
                cls.ctr_size.setdefault(key, [])
            file_byte = None
            assert_count = None
            assert_lens = []
            if formulas is not None:
                file_byte = len(formulas.encode("utf-8"))
                assert_count = 0
                pattern = r'\(assert.*'
                for match in re.finditer(pattern, formulas):
                    line = match.group()
                    assert_count += 1
                    assert_lens.append(len(line))

            total_time = None
            if build_elapsed is not None and solver_elapsed is not None:
                total_time = build_elapsed + solver_elapsed
            elif build_elapsed is not None:
                total_time = build_elapsed
            elif solver_elapsed is not None:
                total_time = solver_elapsed

            if status is not None:
                cls.ctr_size['type'].append(status)
            if solver_elapsed is not None:
                cls.ctr_size['time'].append(solver_elapsed)
            if file_byte is not None:
                cls.ctr_size['byte'].append(file_byte)
            if assert_count is not None:
                cls.ctr_size['assert_num'].append(assert_count)
            cls.ctr_size['assert_len'].append(assert_lens)
            if path_len is not None:
                cls.ctr_size['path_len'].append(path_len)
            if build_elapsed is not None:
                cls.ctr_size['build_time'].append(build_elapsed)
            if total_time is not None:
                cls.ctr_size['total_time'].append(total_time)

            detail = {
                "iter": current_iter,
                "attempt_index": current_attempt_index,
                "status": status,
                "path_len": path_len,
                "assert_num": assert_count,
                "byte": file_byte,
                "assert_len": assert_lens,
                "total_time": total_time,
                "formula_build_time_s": build_elapsed,
                "solver_subprocess_time_s": solver_elapsed,
                "solve_total_time_s": total_time,
            }
            if build_elapsed is not None:
                detail["build_time"] = build_elapsed
            if solver_elapsed is not None:
                detail["solver_time"] = solver_elapsed
            if formulas is not None:
                detail["smt_formula"] = formulas
            if model_error is not None:
                detail["model_error"] = model_error
            if solver_returncode is not None:
                detail["solver_returncode"] = solver_returncode
            if solver_stdout is not None:
                detail["solver_stdout"] = solver_stdout
            if solver_stderr is not None:
                detail["solver_stderr"] = solver_stderr
            if solver_raw_model_lines is not None:
                detail["solver_raw_model_lines"] = solver_raw_model_lines
            if solver_model_diagnostics is not None:
                detail["solver_model_diagnostics"] = solver_model_diagnostics
            cls.ctr_size['detail'].append(detail)
        #limit_constraint_time_start
        build_elapsed = None
        try:
            build_formula_start = time.time()
            if cls.build_timeout_enabled:
                timeout_seconds = cls.build_timeout_seconds or 30
                formulas = func_timeout.func_timeout(
                    timeout_seconds,
                    Solver._build_formulas_from_constraint,
                    args=(engine, constraint, ori_args),
                )
            else:
                formulas = Solver._build_formulas_from_constraint(engine, constraint, ori_args)
            build_formula_end = time.time()
            build_elapsed = build_formula_end - build_formula_start
            cls._append_constraint_log(
                log_path,
                position,
                shap_value,
                f"formulas built time:{build_elapsed}",
            )
        except func_timeout.exceptions.FunctionTimedOut:
            if build_elapsed is None:
                build_elapsed = time.time() - build_formula_start
            cls._append_constraint_log(
                log_path,
                position,
                shap_value,
                "Solver timeout",
            )
            log.warning(
                "SMT formula construction timed out (idx=%s, position=%s)",
                idx,
                summarize_position(position),
            )
            log.info(
                "[SOLVER] idx=%s attack=%s ton=%s position=%s status=timeout sat=%d unsat=%d unknown=%d",
                idx,
                cls._derive_attack_mode(engine),
                cls._derive_attack_ton(engine),
                summarize_position(position),
                cls.stats["sat_number"],
                cls.stats["unsat_number"],
                cls.stats["otherwise_number"],
            )
            _record_constraint_complexity(
                "build_timeout",
                None,
                build_elapsed,
                None,
            )
            return None
        #limit_constraint_time_end

        #skip_last_start
        # with open(f"./popped_constraint_position/transformer_fashion_mnist_two_mha/skip_last/shap-constraint-{idx}.txt", "a") as file:
        #         file.write("\n")
        #         file.write(f"popped constraint with position: {position}\n")
        #         file.write(f"popped constraint with shap value: {shap_value}\n")
        # build_formula_start = time.time()
        # formulas = Solver._build_formulas_from_constraint(engine, constraint, ori_args)
        # build_formula_end = time.time()
        # with open(f"./popped_constraint_position/transformer_fashion_mnist_two_mha/skip_last/shap-constraint-{idx}.txt", "a") as file:
        #         file.write("\n")
        #         file.write(f"popped constraint with position: {position}\n")
        #         file.write(f"popped constraint with shap value: {shap_value}\n")
        #         file.write(f"formulas built time:{build_formula_end - build_formula_start}")
        #skip_last_end
        #original_start
        # formulas = Solver._build_formulas_from_constraint(engine, constraint, ori_args)
        #original_end
        start = time.time()
        try:
            completed_process = subprocess.run(
                cls.cmd,
                input=formulas.encode(),
                capture_output=True,
                timeout=cls.run_timeout if cls.run_timeout else None,
            )
        except subprocess.CalledProcessError as e:
            solver_elapsed = time.time() - start
            log.error("SMT solver process failed (idx=%s)", idx, exc_info=e)
            _record_constraint_complexity(
                "error",
                formulas,
                build_elapsed,
                solver_elapsed,
            )
            with open("smt_error.txt", 'a') as f:
                f.writelines(e.output)
            return None
        except subprocess.TimeoutExpired:
            solver_elapsed = time.time() - start
            log.warning("SMT solver subprocess timed out (idx=%s)", idx)
            _record_constraint_complexity(
                "timeout",
                formulas,
                build_elapsed,
                solver_elapsed,
            )
            return None

        solver_elapsed = time.time() - start
        
        
        solver_returncode = getattr(completed_process, "returncode", None)
        solver_stdout = (getattr(completed_process, "stdout", b"") or b"").decode(
            errors="replace"
        )
        solver_stderr = (getattr(completed_process, "stderr", b"") or b"").decode(
            errors="replace"
        )
        output = solver_stdout
        model = None
        if output is None or len(output) == 0:
            status = "UNKNOWN"
            log.warning("SMT solver returned empty output (idx=%s)", idx)
        else:
            outputs = output.splitlines()
            status = outputs[0].lower()
            solver_raw_model_lines = outputs[1:]
            solver_model_diagnostics = _build_model_diagnostics(
                engine,
                solver_raw_model_lines,
            )
            if "error" in status:
                log.error(
                    "Solver error '%s' at SMT-id=%s. See smt_error.txt for formula",
                    status,
                    Solver.cnt,
                )
                log.error("Failing formula:\n%s", formulas)
                sys.exit(1)
            if "sat" == status:
                cls.stats['sat_number'] += 1; cls.stats['sat_time'] += solver_elapsed
                try:
                    model = Solver._get_model(engine, outputs[1:])
                except InvalidSolverModelError as exc:
                    cls.stats['invalid_model_number'] = cls.stats.get('invalid_model_number', 0) + 1
                    status = "invalid_model"
                    model_error = str(exc)
                    log.warning(
                        "Discarding invalid SAT model (idx=%s, position=%s): %s",
                        idx,
                        summarize_position(position),
                        exc,
                    )
                # FIXME make the value of non-concolic argument unchanged
            else:
                if "unsat" == status: cls.stats['unsat_number'] += 1; cls.stats['unsat_time'] += solver_elapsed
                else: status = "UNKNOWN"; cls.stats['otherwise_number'] += 1; cls.stats['otherwise_time'] += solver_elapsed
        log.info(
            "[SOLVER] idx=%s attack=%s ton=%s position=%s status=%s sat=%d unsat=%d unknown=%d invalid_model=%d",
            idx,
            cls._derive_attack_mode(engine),
            cls._derive_attack_ton(engine),
            summarize_position(position),
            status,
            cls.stats["sat_number"],
            cls.stats["unsat_number"],
            cls.stats["otherwise_number"],
            cls.stats.get("invalid_model_number", 0),
        )
        _record_constraint_complexity(
            status,
            formulas,
            build_elapsed,
            solver_elapsed,
        )
        
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
            
        log.info("SMT solver status for idx=%s: %s", idx, status)
        ##########################################################################################
        log.smtlib2(f"SMT-id: {Solver.cnt}／Status: {status}／Model: {model}")
        Solver.cnt += 1
        Solver.iter_count += 1
        return model

    @staticmethod
    def _get_model(engine, models):
        model = {}
        for line in models:
            if not line.startswith('((') or not line.endswith('))'):
                raise InvalidSolverModelError("malformed SMT model line")
            try:
                name, value = line[2:-2].split(" ", 1)
                value_type = engine.var_to_types[name]
            except (KeyError, ValueError) as exc:
                raise InvalidSolverModelError("malformed SMT model binding") from exc
            if value_type == "Bool":
                if value == 'true': value = True
                elif value == 'false': value = False
                else: raise InvalidSolverModelError("invalid SMT Bool value")
            elif value_type == "Real":
                value = _parse_real_model_value(value)
                if Solver.norm and not 0.0 <= value <= 1.0:
                    raise InvalidSolverModelError("SMT Real is outside the normalized [0, 1] range")
            elif value_type == "Int":
                try:
                    if "(" in value:
                        if "-" in value:
                            value = -int(value.replace("(", "").replace(")", "").split(" ")[2])
                        else:
                            value = int(value.replace("(", "").replace(")", "").split(" ")[1])
                    else:
                        value = int(value)
                except (IndexError, ValueError) as exc:
                    raise InvalidSolverModelError("invalid SMT Int value") from exc
            elif value_type == "String":
                assert value.startswith('"') and value.endswith('"')
                value = value[1:-1]
                value = value.replace('""', '"').replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\")
                # Note the decoding order above must be in reverse with its encoding method (line 41 in libct/utils.py)
            else:
                raise InvalidSolverModelError("unsupported SMT model value type")
            if not name.endswith('_VAR'):
                raise InvalidSolverModelError("SMT model name is missing the _VAR suffix")
            model[name[:-len('_VAR')]] = value
        return model

    @staticmethod
    def _build_formulas_from_constraint(engine, constraint, ori_args):
        # declare_vars = "\n".join(f"(declare-const {name} {_type})" 
        #                for (name, _type) in engine.var_to_types.items()) #if engine.concolic_dict.get(name, 1))
        #NOTE DNN
        declare_vars = "\n".join(f"(declare-const {name} {engine.var_to_types[name]})"                 
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
            try:
                completed_process = subprocess.run(cls.cmd, input=formulas.encode(), capture_output=True)
            except subprocess.CalledProcessError as exc:
                log.error("Safety solver invocation failed", exc_info=exc)
                return None
            output_lines = completed_process.stdout.decode().splitlines()
            if output_lines and output_lines[0] == 'sat':
                return e
            log.error(
                "Safety validation mismatch. Formulas: %s Output: %s",
                formulas,
                output_lines,
            )
            log.debug("Safety validation stack:\n%s", "".join(traceback.format_stack()))
            if cls.safety >= 2: sys.exit(1)
        return None
