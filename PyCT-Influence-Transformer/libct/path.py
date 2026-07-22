# Copyright: see copyright.txt
from __future__ import annotations
import logging
from libct.constraint import Constraint
from libct.predicate import Predicate
from libct.utils import unwrap
from libct.position import get_current_position
from libct.branch_trace import BranchTraceEvent, branch_site_digest, canonical_position

log = logging.getLogger("ct.path")


class PathToConstraint:
    root_constraint: Constraint | None = None

    def __init__(self):
        if self.root_constraint is None:
            self.root_constraint = Constraint(None, None)
        self.current_constraint: Constraint | None = self.root_constraint
        self.branch_trace: list[BranchTraceEvent] = []

    def add_branch(self, conbool):
        engine = getattr(conbool, "engine", None)
        if engine is not None and getattr(engine, "symbolic_enabled", True) is False:
            return
        observed_outcome = bool(unwrap(conbool))
        position = get_current_position()
        p = Predicate(conbool.expr, observed_outcome)
        if engine is not None and getattr(engine, "branch_trace_enabled", False):
            self.branch_trace.append(
                BranchTraceEvent(
                    site_digest=branch_site_digest(
                        conbool.expr,
                        position,
                        getattr(engine, "branch_model_sha256", ""),
                    ),
                    observed_outcome=observed_outcome,
                    depth=int(getattr(self.current_constraint, "height", 0) or 0) + 1,
                    position=canonical_position(position),
                )
            )
        c = self.current_constraint.find_child(p)
        pneg = Predicate(conbool.expr, not observed_outcome)
        cneg = self.current_constraint.find_child(pneg)
        if c is None and cneg is None:
            c = self.current_constraint.add_child(p)
            c.processed = True  # for debugging purposes
            cneg = self.current_constraint.add_child(pneg)
            # add the negated constraint to the queue for later traversal
            conbool.engine.push_constraint(cneg, position)
            # log.smtlib2(f"Now constraint: {c}")
            # log.smtlib2(f"Add constraint: {cneg}")
        else:
            assert c is not None and cneg is not None
        self.current_constraint = c  # move the current constraint to the child we want now
