# Copyright: see copyright.txt

import logging
from libct.concolic import Concolic, MetaFinal
from libct.utils import ConcolicObject, unwrap

log = logging.getLogger("ct.con.range")

class ConcolicRange(metaclass=MetaFinal): # "range" class cannot be inherited.
    def __init__(self, *args):
        self.super = range(*[unwrap(arg) for arg in args]) # emulate inheritance
        if len(args) < 2:
            self.start = 0
            self.stop = args[0]
        else:
            self.start = args[0]
            self.stop = args[1]
        if len(args) < 3:
            self.step = 1
        else:
            self.step = args[2]
        # We must ensure that all attributes are concolic integers in order to
        # make the overridden __getitem__ and __iter__ work!
        if not (isinstance(self.start, int) and isinstance(self.start, Concolic)): # not ConcolicInt
            if isinstance(self.start, Concolic) and hasattr(self.start, '__int2__'):
                self.start = self.start.__int2__()
                if self.start != self.super.start: self.start = ConcolicObject(self.super.start) # ConcolicInt
            else:
                self.start = ConcolicObject(self.super.start) # ConcolicInt
        if not (isinstance(self.stop, int) and isinstance(self.stop, Concolic)): # not ConcolicInt
            if isinstance(self.stop, Concolic) and hasattr(self.stop, '__int2__'):
                self.stop = self.stop.__int2__()
                if self.stop != self.super.stop: self.stop = ConcolicObject(self.super.stop) # ConcolicInt
            else:
                self.stop = ConcolicObject(self.super.stop) # ConcolicInt
        if not (isinstance(self.step, int) and isinstance(self.step, Concolic)): # not ConcolicInt
            if isinstance(self.step, Concolic) and hasattr(self.step, '__int2__'):
                self.step = self.step.__int2__()
                if self.step != self.super.step: self.step = ConcolicObject(self.super.step) # ConcolicInt
            else:
                self.step = ConcolicObject(self.super.step) # ConcolicInt

    def __bool__(self, /): # <slot wrapper '__bool__' of 'range' objects>
        """self != 0"""
        log.debug("ConRange, __bool__ is called")
        return ConcolicObject(self.super.__bool__())

    def __contains__(self, key, /): # <slot wrapper '__contains__' of 'range' objects>
        """Return key in self."""
        log.debug("ConRange, __contains__ is called")
        value = self.super.__contains__(unwrap(key))
        if isinstance(key, Concolic) and hasattr(key, '__int2__'):
            key = key.__int2__()
        else:
            try: key = int(key)
            except: key = 0
            key = ConcolicObject(key)
        if self.start < self.stop:
            return ConcolicObject(value, self.start <= key < self.stop and \
                                         (key - self.start) % self.step == 0)
        if self.start > self.stop:
            return ConcolicObject(value, self.start >= key > self.stop and \
                                         (key - self.start) % self.step == 0)
        return ConcolicObject(value)

    def __eq__(self, value, /): # <slot wrapper '__eq__' of 'range' objects>
        """Return self==value."""
        log.debug("ConRange, __eq__ is called")
        return ConcolicObject(self.super.__eq__(unwrap(value)))

    def __ge__(self, value, /): # <slot wrapper '__ge__' of 'range' objects>
        """Return self>=value."""
        log.debug("ConRange, __ge__ is called")
        return ConcolicObject((self.super.__ge__(unwrap(value))))

    # def __getattribute__(self, name, /): # <slot wrapper '__getattribute__' of 'range' objects>
    #     """Return getattr(self, name)."""

    def __getitem__(self, key, /): # <slot wrapper '__getitem__' of 'range' objects>
        value = self.value.__getitem__(unwrap(key))
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step == -1:
            if self.step > 0:
                k = (self.stop - self.start) // self.step - int((self.stop - self.start) % self.step == 0)
                start = self.start + k * self.step
                stop = self.start - self.step
                step = -self.step
                if start == value.start and stop == value.stop and step == value.step:
                    return self.__class__(start, stop, step)
        return ConcolicObject(value) # TODO

    def __gt__(self, value, /): # <slot wrapper '__gt__' of 'range' objects>
        """Return self>value."""
        log.debug("ConRange, __gt__ is called")
        return ConcolicObject(self.super.__gt__(unwrap(value)))

    def __hash__(self, /): # <slot wrapper '__hash__' of 'range' objects>
        """Return hash(self)."""
        log.debug("ConRange, __hash__ is called")
        return ConcolicObject(self.super.__hash__())

    def __iter__(self, /): # <slot wrapper '__iter__' of 'range' objects>
        """Implement iter(self).""" # Since it's difficult to verify equality of two iterators, we omit the action in this function.
        log.debug("ConRange, __iter__ is called")
        current = self.start
        while True:
            if self.step > 0:
                if current < self.stop:
                    result = current
                    current += self.step
                    yield result
                else:
                    break
            else: # self.step < 0
                if current > self.stop:
                    result = current
                    current += self.step
                    yield result
                else:
                    break

    def __le__(self, value, /): # <slot wrapper '__le__' of 'range' objects>
        """Return self<=value."""
        log.debug("ConRange, __le__ is called")
        return ConcolicObject(self.super.__le__(unwrap(value)))

    def __len__(self, /): # <slot wrapper '__len__' of 'range' objects>
        """Return len(self)."""
        log.debug("ConRange, __len__ is called")
        value = self.super.__len__()
        expr = ((self.stop - self.start) // self.step) + ((self.stop - self.start) % self.step != 0).__int2__()
        return ConcolicObject(value, expr)

    def __lt__(self, value, /): # <slot wrapper '__lt__' of 'range' objects>
        """Return self<value."""
        log.debug("ConRange, __lt__ is called")
        return ConcolicObject(self.super.__lt__(unwrap(value)))

    def __ne__(self, value, /): # <slot wrapper '__ne__' of 'range' objects>
        """Return self!=value."""
        log.debug("ConRange, __ne__ is called")
        return ConcolicObject(self.super.__ne__(unwrap(value)))

    # def __reduce__(self, *args, **kwargs): # <method '__reduce__' of 'range' objects>
    #     """Helper for pickle."""

    # def __repr__(self, /): # <slot wrapper '__repr__' of 'range' objects>
    #     """Return repr(self)."""

    def __reversed__(self, *args, **kwargs): # <method '__reversed__' of 'range' objects>
        """Return a reverse iterator."""
        log.debug("ConRange, __reversed__ is called"); args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        return ConcolicObject(self.super.__reversed__(*args, **kwargs))

    def count(self, key): # <method 'count' of 'range' objects>
        """rangeobject.count(value) -> integer -- return number of occurrences of value"""
        log.debug("ConRange, count is called") #; args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        value = self.super.count(unwrap(key))
        if isinstance(key, Concolic) and hasattr(key, '__int2__'):
            key = key.__int2__()
        else:
            try: key = int(key)
            except: key = 0
            key = ConcolicObject(key)
        if self.start < self.stop:
            return ConcolicObject(value, (self.start <= key < self.stop and \
                                         (key - self.start) % self.step == 0).__int2__())
        if self.start > self.stop:
            return ConcolicObject(value, (self.start >= key > self.stop and \
                                         (key - self.start) % self.step == 0).__int2__())
        return ConcolicObject(value)

    def index(self, key): # <method 'index' of 'range' objects>
        """rangeobject.index(value) -> integer -- return index of value.\nRaise ValueError if the value is not present."""
        log.debug("ConRange, index is called") #; args = [unwrap(arg) for arg in args]; kwargs = {k: unwrap(v) for (k, v) in kwargs.items()}
        value = self.super.index(unwrap(key)) # raise Exception when "key" is not in the range object
        if isinstance(key, Concolic) and hasattr(key, '__int2__'):
            key = key.__int2__()
        else:
            try: key = int(key)
            except: key = 0
            key = ConcolicObject(key)
        return ConcolicObject(value, (key - self.start) // self.step)
