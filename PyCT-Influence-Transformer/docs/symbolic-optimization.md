# Shared symbolic expressions

PyCT expressions already share intermediate objects in memory. The default
SMT writer expands each reference into text, so a small expression graph can
produce a very large formula. Shared serialization names reused expressions
with SMT-LIB `let` bindings, including expressions reused by different
predicates in one query.

## Modes

`PYCT_SMT_SERIALIZATION=raw|shared` selects the output representation. The
default is `raw`, preserving the existing writer. This setting is independent
of `PYCT_SMT_EXPERIMENT_MODE=raw|exact_affine`, whose default remains
`exact_affine`, and `PYCT_SMT_PATH_MODE=full|last`, whose default remains `full`.

| Normalization | Serialization | Behavior |
| --- | --- | --- |
| `raw` | `raw` | Existing expanded expressions |
| `raw` | `shared` | Existing expressions with common subexpressions named |
| `exact_affine` | `raw` | Exact affine normalization, then existing text fallback |
| `exact_affine` | `shared` | Exact affine normalization, then shared fallback |

For example, using the existing local MNIST model and dataset cache:

```bash
PYCT_ALLOW_DATASET_DOWNLOAD=0 PYCT_SMT_SERIALIZATION=shared \
  .venv/bin/python -m pyct \
  --dataset mnist --model-name simple_mnist_m6_09585 \
  --attack-mode queue --pixel-search 1 --first-n 1 --num-process 1 \
  --timeout 15 --solver-run-timeout 2 \
  --constraint-build-timeout-seconds 5 --symbolic-path-threshold 128
```

The model, dataset and any cache required by the selected attack must already
exist locally. Shared runs add `_smtshared` to the experiment directory name,
so they do not resume from or overwrite raw runs. To compare the same sample,
run the same command with `PYCT_SMT_SERIALIZATION=raw`. Existing resume behavior
still applies within each mode; do not overwrite an existing experiment merely
to obtain a timing comparison.

Queue mode requires a target-class input SHAP cache. If it is absent, prepare
one case offline with the same local model and cached dataset:

```bash
PYCT_ALLOW_DATASET_DOWNLOAD=0 .venv/bin/python -m pyct.shap \
  --dataset mnist --model-name simple_mnist_m6_09585 --first-n 1 \
  --sleep-seconds 0
```

Auxiliary constraint logs also add `_smtshared` to their attack-mode directory.

## Semantics and cache lifetime

Shared serialization preserves operators, constants, predicate directions,
and variable bounds. It combines the selected predicates into one conjunction
under dependency-ordered lets; independent bindings at one level share a let.
Generated names avoid identifiers already present in the query. For `last`,
only the final selected predicate is serialized; bounds are still included.

Clipping, PWL segment conditions, and nonlinear expressions stay in the
formula. Affine normalization can reject these expressions; shared mode then
serializes the original graph without first materializing expanded text. It
does not replace a PWL function with its current segment or freeze additional
model values.

Graph size estimation, affine normalization, and structural equality use
iterative traversal. Exact affine forms are memoized per normalization, and
shared queries reuse a normalization context across predicates. Structural
equality memoizes node pairs within one comparison. None of these tables is
a process-global cache. A recorded predicate may retain its small immutable
normalization result, matching the existing predicate text-cache contract:
do not mutate recorded expressions after constructing predicates.

The affine node budget counts newly visited nodes rather than repeated
references. Existing variable and rational-digit budgets still apply. The
shared serializer rejects cyclic graphs, unsupported node types, and graphs
larger than one million unique nodes explicitly; it does not fall back to
potentially enormous expanded text after those errors.

This stage does not change `ConcolicFloat`, implement runtime `a*t+b` values,
or repair Transformer approximations and concrete dependencies. It does not
change branch matching to position-based deduplication. SMT Real equivalence
also does not imply exact equivalence to floating-point model execution.

## Diagnostics and validation

Case metadata records `smt_serialization`. In shared mode, each recorded SMT
transform includes `serialization`, `serialization_time_s`,
`subexpression_cache_hits`, and `shared_serialization`:

- `unique_nodes`: distinct objects in the serialized graph, including atoms.
- `shared_nodes`: named subexpressions.
- `input_bytes`: estimated expanded size of the selected expressions after
  normalization, without constructing the expanded text.
- `input_bytes_capped`: the size estimate reached `2**63 - 1`; it is a capped
  value rather than an exact size. Saturation keeps huge DAG estimates cheap
  and JSON-safe without changing any SMT expression.
- `output_bytes`: actual shared assertion size.

The enclosing transform's `input_bytes` estimates the original expressions
before normalization; its `output_bytes` is the shared assertion size. Those
sizes exclude declarations, bounds, and solver commands. `assertion_count`
counts selected logical predicates, while the emitted conjunction is one
physical SMT assertion. Predicate and subexpression cache hits are separate.

Compare total forward time, formula construction, solver subprocess time,
query size and candidate outcomes independently. A let binding can make small
queries slightly larger, and sharing does not eliminate nonlinear solving or
path growth. Different but equivalent formulas can produce different SAT
models; compare satisfiability and replayed outcomes rather than requiring the
solver to return the identical input.

Regression tests use cvc5 to check formula equivalence and cover clip/PWL,
nonlinear fallback, name collisions, dependent lets, bounds, both path modes,
deep graphs, and cache isolation. Bounded real CLI runs complement these tests;
a timed-out model run is a compatibility observation, not a successful complete
benchmark or evidence of attack completeness.
