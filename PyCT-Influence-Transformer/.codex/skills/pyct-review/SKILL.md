---
name: pyct-review
description: Use when reviewing PyCT-Influence-Transformer changes for bugs, edge cases, maintainability problems, repo-convention mismatches, or missing validation, with findings-first output and precise patch suggestions when helpful.
---

# PyCT Review

Use this skill when the request is to inspect change quality rather than to implement the change.

## Review Focus

- Behavioral regressions in `pyct/`, `orchestration/`, `engine/`, `datasets/`, `explainability/`, `reporting/`, `libct/`, and `dnnct/`
- Missing validation for CLI entrypoints and task orchestration
- Weak abstractions, confusing ownership, and repo-boundary violations
- Naming or structure changes that make future maintenance harder

## Workflow

1. Read the diff or touched files.
2. Identify concrete findings before any summary.
3. Cite file and line references whenever possible.
4. If there are no findings, say that directly and mention remaining risk or test gaps.

## Output Style

- Findings first
- Ordered by severity
- Short evidence-backed explanations
- Patch suggestions only when they reduce ambiguity

## Avoid

- Re-implementing the feature as the main deliverable
- Turning the review into a broad changelog
- Relying on style-only nits when correctness or maintainability issues exist
