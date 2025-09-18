# Copyright: see copyright.txt

import logging
from libct.concolic import Concolic, MetaFinal
from libct.utils import ConcolicObject, unwrap

log = logging.getLogger("ct.con.float")

class ConcolicFloat(float, Concolic, metaclass=MetaFinal):
    def __new__(cls, value, expr=None, engine=None):
        assert type(value) is float
        obj = super().__new__(cls, value)
        Concolic.__init2__(obj, value, expr, engine)
        log.debug(f"ConFloat, value: {value}, expr: {obj.expr}")
        return obj

    ####### Unary Operation, only value modified #######

    def __abs__(self, /): # <slot wrapper '__abs__' of 'float' objects>
        """abs(self)"""
        log.debug("ConFloat, __abs__ is called")
        value = super().__abs__()
        expr = ["abs", self]
        return ConcolicObject(value, expr)

    def __ge__(self, value, /):
        """Return self>=value."""
        log.debug("ConFloat, __ge__ is called")
        return self._bin_op('__ge__', value)

    def __ceil__(self, *args, **kwargs): # <method '__ceil__' of 'float' objects>
        log.debug("ConFloat, __ceil__ is called")
        args = [unwrap(arg) for arg in args] 
        kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        value = super().__ceil__(*args, **kwargs)
        return ConcolicObject(value, self)

    def __floor__(self, *args, **kwargs): # <method '__floor__' of 'float' objects>
        log.debug("ConFloat, __floor__ is called")
        args = [unwrap(arg) for arg in args]
        kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        value = super().__floor__(*args, **kwargs)
        return ConcolicObject(value, self)

    def __bool__(self, /): # <slot wrapper '__bool__' of 'float' objects>
        """self != 0"""
        log.debug("ConFloat, __bool__ is called")
        value = super().__bool__()
        expr = ["not", ["=", self, "0"]]
        ConcolicObject(value, expr).__bool__() # insert handmade branch, since
        return value # "__bool__" can only return primitive objects...

    ############# Binary Operation ################

    def __rtruediv__(self, value, /): # <slot wrapper '__rtruediv__' of 'float' objects>
        """Return value/self."""
        log.debug("ConFloat, __rtruediv__ is called")
        return self._bin_op('__rtruediv__', value)

    def __truediv__(self, value): # <slot wrapper '__truediv__' of 'float' objects>
        """Return self/value."""
        log.debug("ConFloat, __truediv__ is called")
        return self._bin_op('__truediv__', value)

    def __rmul__(self, value, /): # <slot wrapper '__rmul__' of 'float' objects>
        """Return value*self."""
        log.debug("ConFloat, __rmul__ is called")
        return self._bin_op('__rmul__', value)

    def __mul__(self, value, /): # <slot wrapper '__mul__' of 'float' objects>
        """Return self*value."""
        log.debug("ConFloat, __mul__ is called")
        return self._bin_op('__mul__', value)
    
    def __radd__(self, value, /): # <slot wrapper '__radd__' of 'float' objects>
        """Return value+self."""
        log.debug("ConFloat, __radd__ is called")
        return self._bin_op('__radd__', value)
    def __add__(self, value, /): # <slot wrapper '__add__' of 'float' objects>
        """Return self+value."""
        log.debug("ConFloat, __add__ is called")
        return self._bin_op('__add__', value)

    def __le__(self, value, /): # <slot wrapper '__le__' of 'float' objects>
        """Return self<=value."""
        log.debug("ConFloat, __le__ is called")
        return self._bin_op('__le__', value)

    def __lt__(self, value, /): # <slot wrapper '__lt__' of 'float' objects>
        """Return self<value."""
        log.debug("ConFloat, __lt__ is called")
        return self._bin_op('__lt__', value)
    
    def __eq__(self, value, /): # <slot wrapper '__eq__' of 'float' objects>
        """Return self==value."""
        log.debug("ConFloat, __eq__ is called")
        return self._bin_op('__eq__', value)

    def __ge__(self, value, /): # <slot wrapper '__ge__' of 'float' objects>
        """Return self>=value."""
        log.debug("Confloat, __ge__ is called")
        return self._bin_op('__ge__', value)

    def __gt__(self, value, /): # <slot wrapper '__gt__' of 'float' objects>
        """Return self>value."""
        log.debug("ConFloat, __gt__ is called")
        return self._bin_op('__gt__', value)

    ################################################################
    # Other helper methods are implemented in the following section.
    ################################################################

    def __float2__(self):
        log.debug("ConFloat, __float2__ is called")
        return self

    def __int2__(self):
        log.debug("ConFloat, __int2__ is called")
        value = super().__int__()
        expr = ['+', ['to_int', self], ['ite', ['and', ['<', self, '0'], ['not', ['is_int', self]]], '1', '0']]
        # Please note that ['to_int', -2.5] evaluates to -3 in smtlib2, but int(-2.5) evaluates to -2 in Python!
        return ConcolicObject(value, expr)

    def _bin_op(self, op, other):
        if op == '__rtruediv__':
            # TODO: not sure in the following statement what if "other" does not support the "!=" operator.
            try: (self != 0).__bool__() # insert a handmade branch since a number cannot be divided by zero.
            except: pass
            value = super().__rtruediv__(unwrap(other))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that int / float -> float instead of int, so we cannot convert float to int here!
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: super().__rtruediv__(unwrap(other))
            else:
                if type(other) is bool: other = float(other)
                # FIXME currently "int" version implementation
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                other = ConcolicObject(other)
            expr = ['/', other, self]
            return ConcolicObject(value, expr)
        if op == '__truediv__':
            try: (other != 0).__bool__() # insert a handmade branch since a number cannot be divided by zero.
            except: pass
            try:
                if (value := super().__truediv__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__rtruediv__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, '__float2__'): other = other.__float2__()
            else:
                try: other = float(other)
                except: other = 1.0
                other = self.__class__(other)
            expr = ['/', self, other]
            return ConcolicObject(value, expr)
        if op == '__rmul__':
            value = super().__rmul__(unwrap(other))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that float * int -> float instead of int, so we cannot convert float to int here!
                # FIXME convert int to float ???
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                # other cannot be of type "range" here, since Exception should be thrown at the statement: value = unwrap(other).__rmul__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                # FIXME
                if type(other) is float and other==0.0: return ConcolicObject(0.0)
                other = ConcolicObject(other)
            expr = ['*', other, self]
            return ConcolicObject(value, expr)
        if op == '__mul__':
            try:
                if (value := super().__mul__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: 
                value = unwrap(other).__rmul__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that int * float -> float instead of int, so we cannot convert float to int here!
                # FIXME convert int to float ???
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                # other cannot be of type "range" here, since Exception should be thrown at the statement: value = unwrap(other).__rmul__(unwrap(self))
            else:
                if type(other) is bool: other = int(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                # FIXME
                if type(other) is float and other==0.0: return ConcolicObject(0.0)
                other = ConcolicObject(other)
            expr = ['*', self, other]
            return ConcolicObject(value, expr)
        if op == '__gt__':
            try:
                if (value := super().__gt__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__lt__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that (int > float) will convert int to float, so we cannot convert float to int here!
                # FIXME convert int to float ???
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: value = unwrap(other).__lt__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                other = ConcolicObject(other)
            expr = ['>', self, other]
            return ConcolicObject(value, expr)
        if op == '__ge__':
            try:
                if (value := super().__ge__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__le__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: value = unwrap(other).__ge__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                other = ConcolicObject(other)
            expr = ['>=', self, other]
            return ConcolicObject(value, expr)
        if op == '__lt__':
            try:
                if (value := super().__lt__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__gt__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that (int < float) will convert int to float, so we cannot convert float to int here!
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: value = unwrap(other).__gt__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                other = ConcolicObject(other)
            expr = ['<', self, other]
            return ConcolicObject(value, expr)
        if op == '__le__':
            try:
                if (value := super().__le__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__ge__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that (int <= float) will convert int to float, so we cannot convert float to int here!
                # FIXME convert int to float ???
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: value = unwrap(other).__ge__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                other = ConcolicObject(other)
            expr = ['<=', self, other]
            return ConcolicObject(value, expr)
        if op == '__eq__':
            try:
                if (value := super().__eq__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__eq__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that (int == float) will convert int to float, so we cannot convert float to int here!
                # FIXME convert int to float ???
                elif (hasattr(other, 'super') and isinstance(other.super, range)) or isinstance(other, str):
                    return ConcolicObject(value) # smtlib2 cannot compare int with range and str.
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                other = ConcolicObject(other)
            expr = ['=', self, other]
            return ConcolicObject(value, expr)
        if op == '__radd__':
            value = super().__radd__(unwrap(other))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that float + int -> float instead of int, so we cannot convert float to int here!
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: value = unwrap(other).__radd__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                if type(other) is float and other==0.0: return self
                other = ConcolicObject(other)
            expr = ['+', other, self]
            return ConcolicObject(value, expr)
        if op == '__add__':
            try:
                if (value := super().__add__(unwrap(other))) is NotImplemented: raise NotImplementedError
            except: value = unwrap(other).__radd__(unwrap(self))
            if isinstance(other, Concolic):
                if hasattr(other, 'isBool'): other = other.__float2__()
                # Please note that int + float -> float instead of int, so we cannot convert float to int here!
                assert not (hasattr(other, 'super') and isinstance(other.super, range))
                assert not isinstance(other, str)
                # other cannot be of type "range" and "str" here, since Exception should be thrown at the statement: value = unwrap(other).__radd__(unwrap(self))
            else:
                if type(other) is bool: other = float(other)
                if type(other) is not int and type(other) is not float: return ConcolicObject(value) # discard the symbolic expression if it cannot match the concrete value
                #FIXME
                if type(other) is float and other==0.0: return self
                other = ConcolicObject(other)
            expr = ['+', self, other]
            return ConcolicObject(value, expr)
        raise NotImplementedError
