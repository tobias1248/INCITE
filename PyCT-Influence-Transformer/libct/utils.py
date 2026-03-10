import functools, importlib, inspect, os

_MODULE_CACHE = {}

def _int(obj):
    from libct.concolic import Concolic
    if isinstance(obj, Concolic) and hasattr(obj, '__int2__'): return obj.__int2__()
    return int(obj)

def _str(obj):
    from libct.concolic import Concolic
    if isinstance(obj, Concolic) and hasattr(obj, '__str2__'): return obj.__str2__()
    return str(obj)

def _is(obj1, obj2):
    from libct.concolic import Concolic
    from libct.utils import unwrap
    if obj1 is obj2: return True
    if isinstance(obj1, Concolic): obj1 = unwrap(obj1)
    if isinstance(obj2, Concolic): obj2 = unwrap(obj2)
    return obj1 is obj2

def ConcolicObject(value, expr=None, engine=None):
    from libct.concolic import Concolic
    from libct.concolic.bool import ConcolicBool
    from libct.concolic.float import ConcolicFloat
    from libct.concolic.int import ConcolicInt
    from libct.concolic.str import ConcolicStr
    resolved_engine = engine
    if resolved_engine is None and expr is not None:
        resolved_engine = Concolic.find_engine_in_expr(expr)
    if resolved_engine is not None and getattr(resolved_engine, "symbolic_enabled", True) is False:
        return unwrap(value)
    engine = resolved_engine
    if type(value) is bool: return ConcolicBool(value, expr, engine)
    if type(value) is float: return ConcolicFloat(value, expr, engine)
    if type(value) is int: return ConcolicInt(value, expr, engine)
    if type(value) is str: return ConcolicStr(value, expr, engine)
    if isinstance(value, list): # TODO: Are there other types of mutable sequences? What about "slice"?
        return list(map(ConcolicObject, value))
    return value

def unwrap(x): # call primitive's casting function to avoid getting stuck when the concolic's function is modified.
    from libct.concolic.bool import ConcolicBool
    from libct.concolic.float import ConcolicFloat
    from libct.concolic.int import ConcolicInt
    from libct.concolic.str import ConcolicStr
    if type(x) is ConcolicBool: 
        return bool.__bool__(x)
    if type(x) is ConcolicFloat: return float.__float__(x)
    if type(x) is ConcolicInt: return int.__int__(x)
    if type(x) is ConcolicStr: return str.__str__(x)
    if isinstance(x, list): # TODO: Are there other types of mutable sequences? What about "slice"?
        return list(map(unwrap, x))
    return x

def py2smt(x): # convert the Python object into the smtlib2 string constant
    if type(x) is bool: return 'true' if x else 'false'
    if type(x) is (int): return '(- ' + str(-x) + ')' if x < 0 else str(x)
    if type(x) is (float): return '(- ' + f"{-x:.15f}"  + ')' if x < 0 else f"{x:.15f}"
    if type(x) is str:
        x = x.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t").replace('"', '""')
        x_new = "" # this kind of encoding is just a workaround but incorrect since it changes the length of "\r", "\n", "\t".
        for ch in x:
            if ord(ch) > 127: # unicode characters
                x_new += '\\u{' + str(hex(ord(ch)))[2:] + '}'
            else:
                x_new += ch
        x = '"' + x_new + '"' # all string constants must be enclosed by double quotes in smtlib2.
        return x
    raise NotImplementedError

def get_module_from_rootdir_and_modpath(rootdir, modpath):
    # filepath = os.path.join(rootdir, modpath.replace('.', '/') + '.py')
    filepath = os.path.abspath(os.path.join(rootdir, modpath.replace('./', '')))
    mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else None
    cache_key = (filepath, mtime)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    # drop stale entries for same filepath but older mtime
    for key in list(_MODULE_CACHE.keys()):
        if key[0] == filepath and key != cache_key:
            _MODULE_CACHE.pop(key, None)
    # print(filepath)
    # print(modpath)
    spec = importlib.util.spec_from_file_location(modpath, filepath)
    module = importlib.util.module_from_spec(spec)
    now_dir = os.getcwd(); os.chdir(os.path.dirname(filepath))
    spec.loader.exec_module(module)
    os.chdir(now_dir)
    _MODULE_CACHE[cache_key] = module
    return module

def get_function_from_module_and_funcname(module, funcname, enforce=True):
    try:
        while '.' in funcname:
            module = getattr(module, funcname.split('.')[0])
            funcname = funcname.split('.')[1]
        func = getattr(module, funcname)
        if enforce: return func
        ###########################################################################
        if len(list(inspect.signature(func).parameters)) > 0:
            for v in inspect.signature(func).parameters.values():
                if v.annotation not in (int, str):
                    return None
            return func
        return None
        ###########################################################################
        # if len(list(inspect.signature(func).parameters)) > 0:
        #     if list(inspect.signature(func).parameters)[0] == 'cls':
        #         func = functools.partial(func, module)
        #     elif list(inspect.signature(func).parameters)[0] == 'self':
        #         try: func = functools.partial(func, module())
        #         except: pass # module() requires some arguments we don't know
        # return func
    except Exception as e:
        print(e); import traceback; traceback.print_exc(); return None


def get_in_dict_shape(in_dict):
    max_indices: dict[int, int] = {}
    for key in in_dict.keys():
        if not isinstance(key, str) or not key.startswith("v_"):
            continue
        try:
            indices = [int(i) for i in key.split("_")[1:]]
        except ValueError:
            continue
        for axis, value in enumerate(indices):
            prev = max_indices.get(axis, -1)
            if value > prev:
                max_indices[axis] = value

    if not max_indices:
        return tuple()

    axis_count = max(max_indices.keys()) + 1
    shape = []
    for axis in range(axis_count):
        if axis not in max_indices:
            return tuple()
        shape.append(max_indices[axis] + 1)
    return tuple(shape)
