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
        engine = getattr(conbool, "engine", None)
        if engine is not None and getattr(engine, "symbolic_enabled", True) is False:
            return
        concrete_value = bool(unwrap(conbool))
        trivial_value = Predicate.trivial_truth_value(conbool.expr)
        if trivial_value is not None and concrete_value == trivial_value:
            if engine is not None:
                engine.trivial_branch_pruned_count = (
                    getattr(engine, "trivial_branch_pruned_count", 0) + 1
                )
            log.debug("Pruned reflexive branch with fixed value=%s", trivial_value)
            return
        if trivial_value is not None:
            log.warning(
                "Keeping reflexive branch because concrete value=%s differs from "
                "SMT Real truth=%s",
                concrete_value,
                trivial_value,
            )
        p = Predicate(conbool.expr, concrete_value)
        c = self.current_constraint.find_child(p)
        pneg = Predicate(conbool.expr, not concrete_value)
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
