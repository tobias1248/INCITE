# Copyright: see copyright.txt

import logging
from libct.concolic import Concolic, MetaFinal
from libct.utils import ConcolicObject, unwrap

log = logging.getLogger("ct.con.bool")

class ConcolicBool(int, Concolic, metaclass=MetaFinal):
    def __new__(cls, value, expr=None, engine=None):
        if value == 0: value = False # special handling for pickling
        if value == 1: value = True # special handling for pickling
        assert type(value) is bool
        obj = super().__new__(cls, value); obj.isBool = True
        Concolic.__init2__(obj, value, expr, engine)
        log.debug(f"ConBool, value: {value}, expr: {obj.expr}")
        return obj

    def __bool__(self): # <slot wrapper '__bool__' of 'int' objects>
        log.debug("ConBool, __bool__ is called")
        if self.engine: # Please note that this is the only place where branches are generated.
            self.engine.path.add_branch(self)
        return super().__bool__()

    def __xor__(self, other): # <slot wrapper '__xor__' of 'bool' objects>
        log.debug("ConBool, __xor__ is called")
        value = unwrap(self).__xor__(unwrap(other))
        # Note that our concolic bool in fact inherits from 'int' class, so calling super().__xor__(...)
        # is actually calling int's implementation and may not be what we want.
        if isinstance(other, Concolic):
            if hasattr(other, '__bool2__'): other = other.__bool2__()
            # TODO: We currently don't support bitwise operations between integers.
        else:
            try: other = bool(other)
            except: other = False
            other = self.__class__(other)
        expr = ['xor', self, other]
        return ConcolicObject(value, expr)

    def __add__(self, value, /):
        if isinstance(value, Concolic):
            return value.__radd__(self)
        return super().__add__(value)

    ################################################################
    # Other helper methods are implemented in the following section.
    ################################################################

    def __bool2__(self):
        log.debug("ConBool, __bool2__ is called")
        return self

    def __float2__(self):
        log.debug("ConBool, __float2__ is called")
        value = super().__float__()
        expr = ["ite", self, "1.0", "0.0"]
        return ConcolicObject(value, expr)

    def __int2__(self):
        log.debug("ConBool, __int2__ is called")
        value = super().__int__()
        expr = ["ite", self, "1", "0"]
        return ConcolicObject(value, expr)
