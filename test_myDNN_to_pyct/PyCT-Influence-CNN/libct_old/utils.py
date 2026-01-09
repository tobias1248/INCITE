import functools, importlib, inspect, os

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
    from libct.concolic.bool import ConcolicBool
    from libct.concolic.float import ConcolicFloat
    from libct.concolic.int import ConcolicInt
    from libct.concolic.str import ConcolicStr
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
    if type(x) is ConcolicBool: return bool.__bool__(x)
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
    filepath = os.path.join(rootdir, modpath.replace('./', ''))
    print(filepath)
    print(modpath)
    spec = importlib.util.spec_from_file_location(modpath, os.path.abspath(filepath))
    module = importlib.util.module_from_spec(spec)
    now_dir = os.getcwd(); os.chdir(os.path.dirname(filepath))
    spec.loader.exec_module(module)
    os.chdir(now_dir)
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
    max_idx = None
    max_sum_idx = 0
    for k in in_dict.keys():
        idx = [int(i) for i in k.split('_')[1:]]
        if sum(idx) > max_sum_idx:
            max_idx = idx
            max_sum_idx = sum(idx)
    
    max_idx = [i+1 for i in max_idx]
    return tuple(max_idx)
