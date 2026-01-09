# Copyright: see copyright.txt
from __future__ import annotations
import logging
from libct.constraint import Constraint
from libct.predicate import Predicate
from libct.utils import unwrap
from libct.position import get_current_position

log = logging.getLogger("ct.path")


class PathToConstraint:
    root_constraint: Constraint | None = None

    def __init__(self):
        if self.root_constraint is None:
            self.root_constraint = Constraint(None, None)
        self.current_constraint: Constraint | None = self.root_constraint

    def add_branch(self, conbool):
        p = Predicate(conbool.expr, unwrap(conbool))
        c = self.current_constraint.find_child(p)
        pneg = Predicate(conbool.expr, not unwrap(conbool))
        cneg = self.current_constraint.find_child(pneg)
        if c is None and cneg is None:
            c = self.current_constraint.add_child(p)
            c.processed = True  # for debugging purposes
            cneg = self.current_constraint.add_child(pneg)
            # add the negated constraint to the queue for later traversal
            conbool.engine.push_constraint(cneg, get_current_position())
            # log.smtlib2(f"Now constraint: {c}")
            # log.smtlib2(f"Add constraint: {cneg}")
        else:
            assert c is not None and cneg is not None
        self.current_constraint = c  # move the current constraint to the child we want now
