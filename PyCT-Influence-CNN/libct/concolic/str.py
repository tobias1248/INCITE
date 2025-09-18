# Copyright: see copyright.txt

import copy, logging, re
from libct.concolic import Concolic, MetaFinal
from libct.utils import ConcolicObject, py2smt, unwrap

log = logging.getLogger("ct.con.str")

class ConcolicStr(str, Concolic, metaclass=MetaFinal):
    def __new__(cls, value, expr=None, engine=None):
        assert type(value) is str
        obj = super().__new__(cls, value)
        Concolic.__init2__(obj, value, expr, engine)
        log.debug(f"ConStr, value: {value}, expr: {obj.expr}")
        return obj

    def __add__(self, value, /): # <slot wrapper '__add__' of 'str' objects>
        """Return self+value."""
        log.debug("ConStr, __add__ is called")
        return self._bin_op('__add__', value)

    # __class__, <class 'type'>

    def __contains__(self, key, /): # <slot wrapper '__contains__' of 'str' objects>
        """Return key in self."""
        log.debug("ConStr, __contains__ is called")
        return self._bin_op('__contains__', key)

    # __delattr__, <slot wrapper '__delattr__' of 'object' objects>

    # __dir__, <method '__dir__' of 'object' objects>

    def __eq__(self, value, /): # <slot wrapper '__eq__' of 'str' objects>
        """Return self==value."""
        log.debug("ConStr, __eq__ is called")
        return self._bin_op('__eq__', value)

    def __format__(self, format_spec, /): # <method '__format__' of 'str' objects> TODO
        """Return a formatted version of the string as described by format_spec."""
        log.debug("ConStr, __format__ is called")
        return ConcolicObject(super().__format__(unwrap(format_spec)))

    def __ge__(self, value, /): # <slot wrapper '__ge__' of 'str' objects>
        """Return self>=value."""
        log.debug("ConStr, __ge__ is called")
        return self._bin_op('__ge__', value)

    # def __getattribute__(self, name, /): # <slot wrapper '__getattribute__' of 'str' objects>
    #     """Return getattr(self, name)."""

    def __getitem__(self, key, /): # <slot wrapper '__getitem__' of 'str' objects>
        """Return self[key]."""
        log.debug("ConStr, __getitem__ is called")
        if isinstance(key, int): (self.__len__() > key).__bool__() # insert a handmade branch since we cannot access an element whose index is beyond the scope of this string.
        value = super().__getitem__(unwrap(key))
        if isinstance(key, int):
            if not isinstance(key, Concolic): key = ConcolicObject(int(key)) # ConcolicInt
            if key < 0:
                key += self.__len__()
                if key < 0:
                    key = ConcolicObject(0)
            # key_ = ["+", key, ["str.len", self]]
            # key = ["ite", ["<", key, "0"], ["ite", ["<", key_, "0"], "0", key_], key]
            expr = ["str.at", self, key]
            return ConcolicObject(value, expr)
        if isinstance(key, slice): # TODO: From here we simply assume key is a slice object, and we don't consider the "step" attribute.
            start = key.start
            if not isinstance(start, Concolic):
                try: start = int(start)
                except: start = 0
                start = ConcolicObject(start) # ConcolicInt
            stop = key.stop
            if not isinstance(stop, Concolic):
                try: stop = int(stop)
                except: stop = self.__len__()
                stop = ConcolicObject(stop) # ConcolicInt
            if value == unwrap(co:=self._substr(start, stop)): return co
        return ConcolicObject(value)

    # def __getnewargs__(...): # <method '__getnewargs__' of 'str' objects>

    def __gt__(self, value, /): # <slot wrapper '__gt__' of 'str' objects>
        """Return self>value."""
        log.debug("ConStr, __gt__ is called")
        return self._bin_op('__gt__', value)

    def __hash__(self, /): # <slot wrapper '__hash__' of 'str' objects>
        """Return hash(self)."""
        log.debug("ConStr, __hash__ is called")
        return super().__hash__() # may cause errors if we return a concolic object.

    # __init__, <slot wrapper '__init__' of 'object' objects>

    # __init_subclass__, <built-in method __init_subclass__ of type object at 0x9036c0>

    def __iter__(self, /): # <slot wrapper '__iter__' of 'str' objects>
        """Implement iter(self)."""
        log.debug("ConStr, __iter__ is called")
        index = ConcolicObject(0)
        while index < self.__len__(): # Note this branch must be added in order to make this iterator work!
            result = self.__getitem__(index)
            index += ConcolicObject(1)
            yield result

    def __le__(self, value, /): # <slot wrapper '__le__' of 'str' objects>
        """Return self<=value."""
        log.debug("ConStr, __le__ is called")
        return self._bin_op('__le__', value)

    def __len__(self, /): # <slot wrapper '__len__' of 'str' objects>
        """Return len(self)."""
        log.debug("ConStr, __len__ is called")
        value = super().__len__()
        expr = ["str.len", self]
        return ConcolicObject(value, expr)

    def __lt__(self, value, /): # <slot wrapper '__lt__' of 'str' objects>
        """Return self<value."""
        log.debug("ConStr, __lt__ is called")
        return self._bin_op('__lt__', value)

    def __mod__(self, other, /): # <slot wrapper '__mod__' of 'str' objects> TODO: move from approximation to exact computation
        # Meaning of modifiers is explained here: https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting
        """Return self%value.""" # I renamed the argument to "other" for convenience.
        log.debug("ConStr, __mod__ is called")
        value = super().__mod__(unwrap(other))
        if not isinstance(other, tuple): return ConcolicObject(value) # TODO: hide one's head in the sand
        # One split example: ['abcd%def%dga%ib%dc%idefg']
        # -> [['abcd', 'ef', 'ga%ib', 'c%idefg']]
        # -> [['abcd', '%d', 'ef', '%d', 'ga%ib', '%d', 'c%idefg']]
        # -> ['abcd', '%d', 'ef', '%d', 'ga%ib', '%d', 'c%idefg'] (*1)
        # -> [['abcd'], ['%d'], ['ef'], ['%d'], ['ga', 'b'], ['%d'], ['c', 'defg']] (*2)
        # -> [['abcd'], ['%d'], ['ef'], ['%d'], ['ga', '%i', 'b'], ['%d'], ['c', '%i', 'defg']] (*3)
        # -> ['abcd', '%d', 'ef', '%d', 'ga', '%i', 'b', '%d', 'c', '%i', 'defg'] (*4)
        res_list = [unwrap(self)]
        ds = ['%d', '%i', '%o', '%u', '%x', '%X', '%e', '%E', '%f', '%F', '%g', '%G', '%c', '%r', '%s', '%a', '%%']
        for d in ds:
            res_list = list(map(lambda s: s.split(d), res_list)) # (*1 -> *2)
            def insert_delimiter(lst, d): res = [d] * (len(lst) * 2 - 1); res[0::2] = lst; return res
            res_list = list(map(lambda sublist: insert_delimiter(sublist, d), res_list)) # (*2 -> *3)
            res_list = [item for sublist in res_list for item in sublist] # (*3 -> *4)
        other = list(other) # convert from tuple to list
        res_str = self.__class__('')
        for s in res_list:
            if s not in ds:
                res_str += s
            else: # currently only supports '%d', '%i', '%r', '%s', '%%'. For other delimiters we just call str(e) for simplicity.
                if s == '%%':
                    res_str += '%'
                    continue
                if not other: # element list already empty
                    break
                e = other.pop(0)
                if s in ('%d', '%i'):
                    if isinstance(e, Concolic) and hasattr(e, '__int2__'):
                        e = e.__int2__().__str2__()
                    else:
                        try: e = int(e)
                        except: e = 0
                        e = str(e)
                elif s == '%r':
                    if isinstance(e, Concolic): # in concolic environment, we don't distinguish str() from repr()
                        if hasattr(e, '__str2__'): e = e.__str2__()
                    else:
                        try: e = repr(e)
                        except: e = ''
                else: # include the case '%s'
                    if isinstance(e, Concolic):
                        if hasattr(e, '__str2__'): e = e.__str2__()
                    else:
                        try: e = str(e)
                        except: e = ''
                res_str += e
        return res_str if unwrap(res_str) == value else value # TODO: difficult to pick all engines of elements in "other" here, so we omit truncation of expression here.

    def __mul__(self, value, /): # <slot wrapper '__mul__' of 'str' objects>
        """Return self*value."""
        log.debug("ConStr, __mul__ is called")
        return self._bin_op('__mul__', value)

    def __ne__(self, value, /): # <slot wrapper '__ne__' of 'str' objects>
        """Return self!=value."""
        log.debug("ConStr, __ne__ is called")
        return self._bin_op('__ne__', value)

    # __reduce__, <method '__reduce__' of 'object' objects>

    # __reduce_ex__, <method '__reduce_ex__' of 'object' objects>

    # def __repr__(self, /): # <slot wrapper '__repr__' of 'str' objects>
    #     """Return repr(self)."""

    def __rmod__(self, value, /): # <slot wrapper '__rmod__' of 'str' objects> TODO
        """Return value%self."""
        log.debug("ConStr, __rmod__ is called")
        return super().__rmod__(unwrap(value))

    def __rmul__(self, value, /): # <slot wrapper '__rmul__' of 'str' objects>
        """Return value*self."""
        log.debug("ConStr, __rmul__ is called")
        return self._bin_op('__rmul__', value)

    # __setattr__, <slot wrapper '__setattr__' of 'object' objects>

    # def __sizeof__(self, /): # <method '__sizeof__' of 'str' objects>
    #     """Return the size of the string in memory, in bytes."""

    # def __str__(self, /): # <slot wrapper '__str__' of 'str' objects>
    #     """Return str(self)."""

    # __subclasshook__, <built-in method __subclasshook__ of type object at 0x9036c0>

    def capitalize(self, /): # <method 'capitalize' of 'str' objects> TODO
        """Return a capitalized version of the string.\n\nMore specifically, make the first character have upper case and the rest lower case."""
        log.debug("ConStr, capitalize is called")
        return ConcolicObject(super().capitalize())

    def casefold(self, /): # <method 'casefold' of 'str' objects> TODO
        """Return a version of the string suitable for caseless comparisons."""
        log.debug("ConStr, casefold is called")
        return ConcolicObject(super().casefold())

    def center(self, width, fillchar=' ', /): # <method 'center' of 'str' objects> TODO
        """Return a centered string of length width.\n\nPadding is done using the specified fill character (default is a space)."""
        log.debug("ConStr, center is called")
        return ConcolicObject(super().center(unwrap(width), unwrap(fillchar)))

    def count(self, *args): # <method 'count' of 'str' objects>
        """S.count(sub[, start[, end]]) -> int\n\nReturn the number of non-overlapping occurrences of substring sub in\nstring S[start:end].  Optional arguments start and end are\ninterpreted as in slice notation."""
        log.debug("ConStr, count is called"); args = copy.copy(args) # disconnect the input arguments
        if isinstance(args, tuple): args = list(args)
        value = super().count(*map(unwrap, args))
        if not isinstance(args[0], Concolic):
            try: args[0] = str(args[0])
            except: args[0] = ''
            args[0] = ConcolicObject(args[0])
        if len(args) < 2: args.append(0)
        if not isinstance(args[1], Concolic):
            try: args[1] = int(args[1])
            except: args[1] = 0
            args[1] = ConcolicObject(args[1]) # ConcolicInt
        if len(args) < 3: args.append(self.__len__())
        if not isinstance(args[2], Concolic):
            try: args[2] = int(args[2])
            except: args[2] = self.__len__() # Default values are also ok to have expressions.
            args[2] = ConcolicObject(args[2]) # ConcolicInt
        main = self._substr(args[1], args[2])
        (args[0] in main).__bool__() # our handmade branch in order to produce more successful testcases
        expr = ["ite", ["<=", ["str.len", args[0]], "0"], ["+", "1", ["str.len", main]], ["div", ["-", ["str.len", main], ["str.len", ["str.replaceall", main, args[0], py2smt("")]]], ["str.len", args[0]]]]
        return ConcolicObject(value, expr)

    def encode(self, /, encoding='utf-8', errors='strict'): # <method 'encode' of 'str' objects> TODO
        """Encode the string using the codec registered for encoding."""
        log.debug("ConStr, encode is called")
        return ConcolicObject(super().encode(unwrap(encoding), unwrap(errors)))

    def endswith(self, *args): # <method 'endswith' of 'str' objects>
        """S.endswith(suffix[, start[, end]]) -> bool\n\nReturn True if S ends with the specified suffix, False otherwise.\nWith optional start, test S beginning at that position.\nWith optional end, stop comparing S at that position.\nsuffix can also be a tuple of strings to try."""
        log.debug("ConStr, endswith is called"); args = copy.copy(args) # disconnect the input arguments
        if isinstance(args, tuple): args = list(args)
        value = super().endswith(*map(unwrap, args)) # TODO: still not deal with the case when suffix is a tuple!!!
        if not isinstance(args[0], Concolic):
            try: args[0] = str(args[0])
            except: args[0] = ''
            args[0] = ConcolicObject(args[0])
        if len(args) < 2: args.append(0)
        if not isinstance(args[1], Concolic):
            try: args[1] = int(args[1])
            except: args[1] = 0
            args[1] = ConcolicObject(args[1]) # ConcolicInt
        if len(args) < 3: args.append(self.__len__())
        if not isinstance(args[2], Concolic):
            try: args[2] = int(args[2])
            except: args[2] = self.__len__() # Default values are also ok to have expressions.
            args[2] = ConcolicObject(args[2]) # ConcolicInt
        expr = ["str.suffixof", args[0], self._substr(args[1], args[2])]
        return ConcolicObject(value, expr)

    def expandtabs(self, /, tabsize=8): # <method 'expandtabs' of 'str' objects> TODO
        """Return a copy where all tab characters are expanded using spaces.\n\nIf tabsize is not given, a tab size of 8 characters is assumed."""
        log.debug("ConStr, expandtabs is called")
        return ConcolicObject(super().expandtabs(unwrap(tabsize)))

    def find(self, *args): # <method 'find' of 'str' objects>
        """S.find(sub[, start[, end]]) -> int\n\nReturn the lowest index in S where substring sub is found,\nsuch that sub is contained within S[start:end].  Optional\narguments start and end are interpreted as in slice notation.\n\nReturn -1 on failure."""
        log.debug("ConStr, find is called"); args = copy.copy(args) # disconnect the input arguments
        if isinstance(args, tuple): args = list(args)
        value = super().find(*map(unwrap, args))
        if not isinstance(args[0], Concolic):
            try: args[0] = str(args[0])
            except: args[0] = ''
            args[0] = ConcolicObject(args[0])
        if len(args) < 2: args.append(0)
        if not isinstance(args[1], Concolic):
            try: args[1] = int(args[1])
            except: args[1] = 0
            args[1] = ConcolicObject(args[1]) # ConcolicInt
        if len(args) < 3: args.append(self.__len__())
        if not isinstance(args[2], Concolic):
            try: args[2] = int(args[2])
            except: args[2] = self.__len__() # Default values are also ok to have expressions.
            args[2] = ConcolicObject(args[2]) # ConcolicInt
        (args[0] in self._substr(args[1], args[2])).__bool__() # our handmade branch in order to produce more successful testcases
        expr = ["+", args[1], ["str.indexof", self._substr(args[1], args[2]), args[0], "0"]]
        return ConcolicObject(value, expr)

    def format(self, *args, **kwargs): # <method 'format' of 'str' objects> TODO
        """S.format(*args, **kwargs) -> str\n\nReturn a formatted version of S, using substitutions from args and kwargs.\nThe substitutions are identified by braces ('{' and '}')."""
        log.debug("ConStr, format is called"); args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        return ConcolicObject(super().format(*args, **kwargs))

    def format_map(self, *args, **kwargs): # <method 'format_map' of 'str' objects> TODO
        """S.format_map(mapping) -> str\n\nReturn a formatted version of S, using substitutions from mapping.\nThe substitutions are identified by braces ('{' and '}')."""
        log.debug("ConStr, format_map is called"); args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        return ConcolicObject(super().format_map(*args, **kwargs))

    def index(self, *args): # <method 'index' of 'str' objects>
        """S.index(sub[, start[, end]]) -> int\n\nReturn the lowest index in S where substring sub is found,\nsuch that sub is contained within S[start:end].  Optional\narguments start and end are interpreted as in slice notation.\n\nRaises ValueError when the substring is not found."""
        log.debug("ConStr, index is called"); args = copy.copy(args) # disconnect the input arguments
        if isinstance(args, tuple): args = list(args)
        value = super().index(*map(unwrap, args))
        if not isinstance(args[0], Concolic):
            try: args[0] = str(args[0])
            except: args[0] = ''
            args[0] = ConcolicObject(args[0])
        if len(args) < 2: args.append(0)
        if not isinstance(args[1], Concolic):
            try: args[1] = int(args[1])
            except: args[1] = 0
            args[1] = ConcolicObject(args[1]) # ConcolicInt
        if len(args) < 3: args.append(self.__len__())
        if not isinstance(args[2], Concolic):
            try: args[2] = int(args[2])
            except: args[2] = self.__len__() # Default values are also ok to have expressions.
            args[2] = ConcolicObject(args[2]) # ConcolicInt
        expr = ["str.indexof", self._substr(args[1], args[2]), args[0], "0"]
        return ConcolicObject(value, expr)

    def isalnum(self, /): # <method 'isalnum' of 'str' objects> (also appears in 04_Python)
        """Return True if the string is an alpha-numeric string, False otherwise.\n\nA string is alpha-numeric if all characters in the string are alpha-numeric and\nthere is at least one character in the string."""
        log.debug("ConStr, isalnum is called")
        value = super().isalnum()
        expr = ["str.in.re", self, ["re.+", ["re.union", ["re.union", ["re.range", py2smt("A"), py2smt("Z")], ["re.range", py2smt("a"), py2smt("z")]], ["re.range", py2smt("0"), py2smt("9")]]]]
        return ConcolicObject(value, expr)

    def isalpha(self, /): # <method 'isalpha' of 'str' objects> (also appears in 04_Python)
        """Return True if the string is an alphabetic string, False otherwise.\n\nA string is alphabetic if all characters in the string are alphabetic and there\nis at least one character in the string."""
        log.debug("ConStr, isalpha is called")
        value = super().isalpha()
        expr = ["str.in.re", self, ["re.+", ["re.union", ["re.range", py2smt("A"), py2smt("Z")], ["re.range", py2smt("a"), py2smt("z")]]]]
        return ConcolicObject(value, expr)

    def isascii(self, /): # <method 'isascii' of 'str' objects> TODO
        """Return True if all characters in the string are ASCII, False otherwise.\n\nASCII characters have code points in the range U+0000-U+007F.\nEmpty string is ASCII too."""
        log.debug("ConStr, isascii is called")
        return ConcolicObject(super().isascii())

    def isdecimal(self, /): # <method 'isdecimal' of 'str' objects> TODO
        """Return True if the string is a decimal string, False otherwise.\n\nA string is a decimal string if all characters in the string are decimal and\nthere is at least one character in the string."""
        log.debug("ConStr, isdecimal is called")
        return ConcolicObject(super().isdecimal())

    def isdigit(self, /): # <method 'isdigit' of 'str' objects> (also appears in 04_Python)
        """Return True if the string is a digit string, False otherwise.\n\nA string is a digit string if all characters in the string are digits and there\nis at least one character in the string."""
        log.debug("ConStr, isdigit is called")
        value = super().isdigit()
        expr = ["str.in.re", self, ["re.+", ["re.range", py2smt('0'), py2smt('9')]]]
        # Question: Is this expression equivalent to ["not", ["=", "-1", ["str.to.int", self]]]?
        return ConcolicObject(value, expr)

    def isidentifier(self, /): # <method 'isidentifier' of 'str' objects> TODO
        """Return True if the string is a valid Python identifier, False otherwise.\n\nCall keyword.iskeyword(s) to test whether string s is a reserved identifier,\nsuch as "def" or "class"."""
        log.debug("ConStr, isidentifier is called")
        return ConcolicObject(super().isidentifier())

    def islower(self, /): # <method 'islower' of 'str' objects> (also appears in 04_Python)
        """Return True if the string is a lowercase string, False otherwise.\n\nA string is lowercase if all cased characters in the string are lowercase and\n there is at least one cased character in the string."""
        log.debug("ConStr, islower is called")
        value = super().islower()
        expr = ["str.in.re", self, ["re.+", ["re.range", py2smt('a'), py2smt('z')]]]
        return ConcolicObject(value, expr)

    def isnumeric(self, /): # <method 'isnumeric' of 'str' objects> (also appears in 04_Python)
        """Return True if the string is a numeric string, False otherwise.\n\nA string is numeric if all characters in the string are numeric and there is at\nleast one character in the string."""
        log.debug("ConStr, isnumeric is called")
        value = super().isnumeric()
        expr = ["str.in.re", self, ["re.+", ["re.range", py2smt('0'), py2smt('9')]]]
        return ConcolicObject(value, expr)

    def isprintable(self, /): # <method 'isprintable' of 'str' objects> TODO
        """Return True if the string is printable, False otherwise.\n\nA string is printable if all of its characters are considered printable in\nrepr() or if it is empty."""
        log.debug("ConStr, isprintable is called")
        return ConcolicObject(super().isprintable())

    def isspace(self, /): # <method 'isspace' of 'str' objects> TODO
        """Return True if the string is a whitespace string, False otherwise.\n\nA string is whitespace if all characters in the string are whitespace and there\nis at least one character in the string."""
        log.debug("ConStr, isspace is called")
        return ConcolicObject(super().isspace())

    def istitle(self, /): # <method 'istitle' of 'str' objects> TODO
        """Return True if the string is a title-cased string, False otherwise.\n\nIn a title-cased string, upper- and title-case characters may only\nfollow uncased characters and lowercase characters only cased ones."""
        log.debug("ConStr, istitle is called")
        return ConcolicObject(super().istitle())

    def isupper(self, /): # <method 'isupper' of 'str' objects> (also appears in 04_Python)
        """Return True if the string is an uppercase string, False otherwise.\n\nA string is uppercase if all cased characters in the string are uppercase and\nthere is at least one cased character in the string."""
        log.debug("ConStr, isupper is called")
        value = super().isupper()
        expr = ["str.in.re", self, ["re.+", ["re.range", py2smt('A'), py2smt('Z')]]]
        return ConcolicObject(value, expr)

    def join(self, iterable, /): # <method 'join' of 'str' objects> TODO
        """Concatenate any number of strings.\n\nThe string whose method is called is inserted in between each given string.\nThe result is returned as a new string.\n\nExample: '.'.join(['ab', 'pq', 'rs']) -> 'ab.pq.rs'"""
        log.debug("ConStr, join is called")
        return ConcolicObject(super().join(unwrap(iterable)))

    def ljust(self, width, fillchar=' ', /): # <method 'ljust' of 'str' objects> TODO
        """Return a left-justified string of length width.\n\nPadding is done using the specified fill character (default is a space)."""
        log.debug("ConStr, ljust is called")
        return ConcolicObject(super().ljust(unwrap(width), unwrap(fillchar)))

    def lower(self, /): # <method 'lower' of 'str' objects>
        """Return a copy of the string converted to lowercase."""
        log.debug("ConStr, lower is called")
        value = super().lower()
        expr = ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", self, py2smt("A"), py2smt("a")], py2smt("B"), py2smt("b")], py2smt("C"), py2smt("c")], py2smt("D"), py2smt("d")], py2smt("E"), py2smt("e")], py2smt("F"), py2smt("f")], py2smt("G"), py2smt("g")], py2smt("H"), py2smt("h")], py2smt("I"), py2smt("i")], py2smt("J"), py2smt("j")], py2smt("K"), py2smt("k")], py2smt("L"), py2smt("l")], py2smt("M"), py2smt("m")], py2smt("N"), py2smt("n")], py2smt("O"), py2smt("o")], py2smt("P"), py2smt("p")], py2smt("Q"), py2smt("q")], py2smt("R"), py2smt("r")], py2smt("S"), py2smt("s")], py2smt("T"), py2smt("t")], py2smt("U"), py2smt("u")], py2smt("V"), py2smt("v")], py2smt("W"), py2smt("w")], py2smt("X"), py2smt("x")], py2smt("Y"), py2smt("y")], py2smt("Z"), py2smt("z")]
        # Note that the above expr can be generated by the following code snippet.
        # def f(x):
        #     if x == 96: return 'self'
        #     return f'["str.replaceall", {f(x-1)}, py2smt("{chr(x-32)}"), py2smt("{chr(x)}")]'
        # print(f(122))
        return ConcolicObject(value, expr)

    def lstrip(self, chars=None, /): # <method 'lstrip' of 'str' objects> TODO: move from approximation to exact semantics
        """Return a copy of the string with leading whitespace removed.\n\nIf chars is given and not None, remove characters in chars instead."""
        log.debug("ConStr, lstrip is called")
        value = super().lstrip(unwrap(chars))
        if not isinstance(chars, Concolic):
            if not isinstance(chars, str): chars = ' ' # TODO: Only this kind of whitespace?
            chars = ConcolicObject(chars)
        expr = self
        s = unwrap(self)
        while any(map(lambda ch: s.startswith(unwrap(ch)), chars)):
            s = s[1:]
            expr = ["ite", ['or'] + list(map(lambda ch: ["str.prefixof", ch, expr], chars)), ["str.substr", expr, "1", ["str.len", expr]], expr]
        return ConcolicObject(value, expr)

    # maketrans, <built-in method maketrans of type object at 0x9036c0> (@staticmethod)

    def partition(self, sep, /): # <method 'partition' of 'str' objects> TODO
        """Partition the string into three parts using the given separator."""
        log.debug("ConStr, partition is called")
        return ConcolicObject(super().partition(unwrap(sep)))

    def replace(self, old, new, count=-1, /): # <method 'replace' of 'str' objects> TODO: move from approximation to exact semantics
        """Return a copy with all occurrences of substring old replaced by new.\n\n  count\n    Maximum number of occurrences to replace.\n    -1 (the default value) means replace all occurrences.\n\nIf the optional argument count is given, only the first count occurrences are\nreplaced."""
        log.debug("ConStr, replace is called")
        value = super().replace(unwrap(old), unwrap(new), unwrap(count))
        if not isinstance(old, Concolic):
            try: old = str(old)
            except: old = ''
            old = ConcolicObject(old)
        if not isinstance(new, Concolic):
            try: new = str(new)
            except: new = ''
            new = ConcolicObject(new)
        if not isinstance(count, Concolic):
            try: count = int(count)
            except: count = -1
            count = ConcolicObject(count) # ConcolicInt
        old_len = old.__len__() # a cached concolic constant for speeding up
        result = self.__class__('')
        current = self # current substring after several rounds of "replace"
        while True:
            if count == 0 or (split_index := current.find(old)) == -1:
                result += current
                break
            elif count < 0:
                v = unwrap(current).replace(unwrap(old), unwrap(new), unwrap(count))
                expr = ["str.replaceall", current, old, new]
                result += ConcolicObject(v, expr)
                break
            else: # if count > 0:
                # <--- result ---><------------- current ---------------------->
                # <--- result ---><------><-- old --><------------------------->
                # <------- result -------><------ new ------><-------- current -------->
                result += current._substr(end=split_index)
                result += new
                current = current._substr(start=split_index + old_len)
                count -= 1
        # If our concolic result is different from the standard one, we can only return the standard result.
        return result if unwrap(result) == value else ConcolicObject(value)

    def rfind(self, *args, **kwargs): # <method 'rfind' of 'str' objects> TODO
        """S.rfind(sub[, start[, end]]) -> int"""
        log.debug("ConStr, rfind is called"); args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        return ConcolicObject(super().rfind(*args, **kwargs))

    def rindex(self, *args, **kwargs): # <method 'rindex' of 'str' objects> TODO
        """S.rindex(sub[, start[, end]]) -> int"""
        log.debug("ConStr, rindex is called"); args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        return ConcolicObject(super().rindex(*args, **kwargs))

    def rjust(self, width, fillchar=' ', /): # <method 'rjust' of 'str' objects> TODO
        """Return a right-justified string of length width.\n\nPadding is done using the specified fill character (default is a space)."""
        log.debug("ConStr, rjust is called")
        return ConcolicObject(super().rjust(unwrap(width), unwrap(fillchar)))

    def rpartition(self, sep, /): # <method 'rpartition' of 'str' objects> TODO
        """Partition the string into three parts using the given separator."""
        log.debug("ConStr, rpartition is called")
        return ConcolicObject(super().rpartition(unwrap(sep)))

    def rsplit(self, /, sep=None, maxsplit=-1): # <method 'rsplit' of 'str' objects> TODO
        """Return a list of the words in the string, using sep as the delimiter string."""
        log.debug("ConStr, rsplit is called")
        return ConcolicObject(super().rsplit(unwrap(sep), unwrap(maxsplit)))

    # TODO: Temp
    def rstrip(self, chars=None, /): # <method 'rstrip' of 'str' objects> TODO: move from approximation to exact semantics
        """Return a copy of the string with trailing whitespace removed.\n\nIf chars is given and not None, remove characters in chars instead."""
        log.debug("ConStr, rstrip is called")
        value = super().rstrip(unwrap(chars))
        if not isinstance(chars, Concolic):
            if not isinstance(chars, str): chars = ' ' # TODO: Only this kind of whitespace?
            chars = ConcolicObject(chars)
        expr = self
        s = unwrap(self)
        while any(map(lambda ch: s.endswith(unwrap(ch)), chars)):
            s = s[:-1]
            expr = ["ite", ['or'] + list(map(lambda ch: ["str.suffixof", ch, expr], chars)), ["str.substr", expr, "0", ["-", ["str.len", expr], "1"]], expr]
        return ConcolicObject(value, expr)

    def split(self, /, sep=None, maxsplit=-1): # <method 'split' of 'str' objects> TODO: move from approximation to exact semantics
        """Return a list of the words in the string, using sep as the delimiter string.\n\n  sep\n    The delimiter according which to split the string.\n    None (the default value) means split according to any whitespace,\n    and discard empty strings from the result.\n  maxsplit\n    Maximum number of splits to do.\n    -1 (the default value) means no limit."""
        log.debug("ConStr, split is called")
        sepori = sep
        if sep is None: sep = " "
        sep_idx = self.find(sep)
        if maxsplit == 0 or sep_idx == -1:
            if sepori is not None or self:
                return [self]
            return []
        if maxsplit > 0: maxsplit -= 1
        if sepori is not None or self[0:sep_idx]:
            return [self[0:sep_idx]] + self[sep_idx+len(sep):].split(sepori, maxsplit)
        else:
            return self[sep_idx+len(sep):].split(sepori, maxsplit)

    def splitlines(self, /, keepends=False): # <method 'splitlines' of 'str' objects> TODO: move from approximation to exact computation
        """Return a list of the lines in the string, breaking at line boundaries.\n\nLine breaks are not included in the resulting list unless keepends is given and\ntrue."""
        log.debug("ConStr, splitlines is called")
        value = super().splitlines(unwrap(keepends))
        result = self.split("\r\n") if super().__contains__("\r\n") else self.split("\n")
        return result if list(map(unwrap, result)) == value else ConcolicObject(value)

    def startswith(self, *args): # <method 'startswith' of 'str' objects>
        """S.startswith(prefix[, start[, end]]) -> bool\n\nReturn True if S starts with the specified prefix, False otherwise.\nWith optional start, test S beginning at that position.\nWith optional end, stop comparing S at that position.\nprefix can also be a tuple of strings to try."""
        log.debug("ConStr, startswith is called"); args = copy.copy(args) # disconnect the input arguments
        if isinstance(args, tuple): args = list(args)
        value = super().startswith(*map(unwrap, args)) # TODO: still not deal with the case when prefix is a tuple!!!
        if not isinstance(args[0], Concolic):
            try: args[0] = str(args[0])
            except: args[0] = ''
            args[0] = ConcolicObject(args[0])
        if len(args) < 2: args.append(0)
        if not isinstance(args[1], Concolic):
            try: args[1] = int(args[1])
            except: args[1] = 0
            args[1] = ConcolicObject(args[1]) # ConcolicInt
        if len(args) < 3: args.append(self.__len__())
        if not isinstance(args[2], Concolic):
            try: args[2] = int(args[2])
            except: args[2] = self.__len__() # Default values are also ok to have expressions.
            args[2] = ConcolicObject(args[2]) # ConcolicInt
        expr = ["str.prefixof", args[0], self._substr(args[1], args[2])]
        return ConcolicObject(value, expr)

    def strip(self, chars=None, /): # <method 'strip' of 'str' objects> TODO: not sure whether ".lstrip(chars).rstrip(chars)" is the exact definition or not.
        """Return a copy of the string with leading and trailing whitespace removed.\n\nIf chars is given and not None, remove characters in chars instead."""
        log.debug("ConStr, strip is called")
        value = super().strip(unwrap(chars))
        result = self.lstrip(chars).rstrip(chars)
        return result if unwrap(result) == value else ConcolicObject(value)

    def swapcase(self, /): # <method 'swapcase' of 'str' objects> TODO
        """Convert uppercase characters to lowercase and lowercase characters to uppercase."""
        log.debug("ConStr, swapcase is called")
        return ConcolicObject(super().swapcase())

    def title(self, /): # <method 'title' of 'str' objects> TODO
        """Return a version of the string where each word is titlecased.\n\nMore specifically, words start with uppercased characters and all remaining\ncased characters have lower case."""
        log.debug("ConStr, title is called")
        return ConcolicObject(super().title())

    def translate(self, table, /): # <method 'translate' of 'str' objects> TODO
        """Replace each character in the string using the given translation table."""
        log.debug("ConStr, translate is called")
        return ConcolicObject(super().translate(unwrap(table)))

    def upper(self, /): # <method 'upper' of 'str' objects>
        """Return a copy of the string converted to uppercase."""
        log.debug("ConStr, upper is called")
        value = super().upper()
        expr = ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", ["str.replaceall", self, py2smt("a"), py2smt("A")], py2smt("b"), py2smt("B")], py2smt("c"), py2smt("C")], py2smt("d"), py2smt("D")], py2smt("e"), py2smt("E")], py2smt("f"), py2smt("F")], py2smt("g"), py2smt("G")], py2smt("h"), py2smt("H")], py2smt("i"), py2smt("I")], py2smt("j"), py2smt("J")], py2smt("k"), py2smt("K")], py2smt("l"), py2smt("L")], py2smt("m"), py2smt("M")], py2smt("n"), py2smt("N")], py2smt("o"), py2smt("O")], py2smt("p"), py2smt("P")], py2smt("q"), py2smt("Q")], py2smt("r"), py2smt("R")], py2smt("s"), py2smt("S")], py2smt("t"), py2smt("T")], py2smt("u"), py2smt("U")], py2smt("v"), py2smt("V")], py2smt("w"), py2smt("W")], py2smt("x"), py2smt("X")], py2smt("y"), py2smt("Y")], py2smt("z"), py2smt("Z")]
        # Note that the above expr can be generated by the following code snippet.
        # def f(x):
        #     if x == 96: return 'self'
        #     return f'["str.replaceall", {f(x-1)}, py2smt("{chr(x)}"), py2smt("{chr(x-32)}")]'
        # print(f(122))
        return ConcolicObject(value, expr)

    def zfill(self, width, /): # <method 'zfill' of 'str' objects> TODO
        """Pad a numeric string with zeros on the left, to fill a field of the given width.\n\nThe string is never truncated."""
        log.debug("ConStr, zfill is called")
        return ConcolicObject(super().zfill(unwrap(width)))

    ################################################################
    # Other helper methods are implemented in the following section.
    ################################################################

    def _bin_op(self, op, other):
        if op == '__add__':
            try:
                if (value := super().__add__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__radd__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            expr = ['str.++', self, other]
            return ConcolicObject(value, expr)
        if op == '__contains__':
            value = super().__contains__(unwrap(other))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            expr = ['str.contains', self, other]
            return ConcolicObject(value, expr)
        # Note that string comparisons in smtlib2 may cause incorrect results if
        # the strings contain non-alnum characters. For instance, ord('~')==126
        # and ord('.')==46 but (str.< "~" ".") evaluates to true in smtlib2.
        if op == '__eq__':
            try:
                if (value := super().__eq__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__eq__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            expr = ['=', self, other]
            return ConcolicObject(value, expr)
        if op == '__ge__':
            try:
                if (value := super().__ge__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__le__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            if self.isalnum() and other.isalnum():
                expr = ['str.<=', other, self]
                return ConcolicObject(value, expr)
            else:
                return ConcolicObject(value)
        if op == '__gt__':
            try:
                if (value := super().__gt__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__lt__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            if self.isalnum() and other.isalnum():
                expr = ['str.<', other, self]
                return ConcolicObject(value, expr)
            else:
                return ConcolicObject(value)
        if op == '__le__':
            try:
                if (value := super().__le__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__ge__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            if self.isalnum() and other.isalnum():
                expr = ['str.<=', self, other]
                return ConcolicObject(value, expr)
            else:
                return ConcolicObject(value)
        if op == '__lt__':
            try:
                if (value := super().__lt__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__gt__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            if self.isalnum() and other.isalnum():
                expr = ['str.<', self, other]
                return ConcolicObject(value, expr)
            else:
                return ConcolicObject(value)
        if op == '__mul__':
            try:
                if (value := super().__mul__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__rmul__(unwrap(self))
            assert isinstance(other, (bool, int))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__int2__()
            else:
                if type(other) is bool: other = int(other)
                other = ConcolicObject(other)
            result = ConcolicObject("")
            while other > 0:
                result += self
                other -= 1
            return ConcolicObject(value, result)
        if op == '__ne__':
            try:
                if (value := super().__ne__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__ne__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            expr = ['not', ['=', self, other]]
            return ConcolicObject(value, expr)
        if op == '__radd__':
            value = unwrap(other).__add__(unwrap(self))
            assert isinstance(other, str)
            if not isinstance(other, Concolic): other = ConcolicObject(other)
            expr = ['str.++', other, self]
            return ConcolicObject(value, expr)
        if op == '__rmul__':
            value = super().__rmul__(unwrap(other))
            assert isinstance(other, (bool, int))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__int2__()
            else:
                if type(other) is bool: other = int(other)
                other = ConcolicObject(other)
            result = ConcolicObject("")
            while other > 0:
                result += self
                other -= 1
            return ConcolicObject(value, result)
        raise NotImplementedError

    def __bool__(self):
        log.debug("ConStr, __bool__ is called")
        value = bool(unwrap(self))
        expr = ["not", ["=", self, py2smt('')]]
        ConcolicObject(value, expr).__bool__() # insert handmade branch, since
        return value # "__bool__" can only return primitive objects...

    def __bool2__(self):
        log.debug("ConStr, __bool2__ is called")
        value = bool(unwrap(self))
        expr = ["not", ["=", self, py2smt('')]]
        return ConcolicObject(value, expr)

    def _is_int(self):
        log.debug("ConStr, _is_int is called")
        value = re.compile(r"^[-]?\d+$").match(unwrap(self)) is not None
        expr = ["str.in.re", ["ite", ["str.prefixof", py2smt('-'), self],
                                     ["str.substr", self, "1", ["str.len", self]],
                                     self], # For better results, avoid using ["str.replace", self, "-", ""] in the above line.
                             ["re.+", ["re.range", py2smt('0'), py2smt('9')]]]
        return ConcolicObject(value, expr)

    def _substr(self, start=None, end=None): # end is exclusive...
        log.debug("ConStr, _substr is called")
        if start is None: start = ConcolicObject(0) # ConcolicInt
        if end is None: end = self.__len__() # ConcolicInt
        value = super().__getitem__(slice(unwrap(start), unwrap(end)))
        if start < 0:
            start += self.__len__()
            if start < 0:
                start = ConcolicObject(0)
        if end < 0:
            end += self.__len__()
            if end < 0:
                end = ConcolicObject(0)
        # start_ = ["+", start, ["str.len", self]]
        # start = ["ite", ["<", start, "0"], ["ite", ["<", start_, "0"], "0", start_], start]
        # end_ = ["+", end, ["str.len", self]]
        # end = ["ite", ["<", end, "0"], ["ite", ["<", end_, "0"], "0", end_], end]
        if unwrap(start) == 0 and unwrap(end) == unwrap(self.__len__()): return self # important acceleration!!!
        expr = ["str.substr", self, start, ["-", end, start]]
        return ConcolicObject(value, expr)

    def __int2__(self):
        log.debug("ConStr, __int2__ is called")
        self._is_int().__bool__() # our handmade branch in order to produce more successful testcases
        value = int(self)
        expr = ["ite", ["str.prefixof", py2smt('-'), self],
                       ["-", ["str.to.int", ["str.substr", self, "1", ["str.len", self]]]],
                       ["str.to.int", self]] # For better results, avoid using ["str.replace", self, "-", ""] in the above line.
        return ConcolicObject(value, expr)

    def __radd__(self, value, /):
        """Return value+self."""
        log.debug("ConStr, __radd__ is called")
        return self._bin_op('__radd__', value)

    def __str2__(self):
        log.debug("ConStr, __str2__ is called")
        return self
