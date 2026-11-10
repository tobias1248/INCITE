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

Shared serialization does not repair Transformer approximations or concrete
dependencies, or change branch matching to position-based deduplication. SMT
Real equivalence does not imply exact equivalence to floating-point execution.

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

## Compact affine runtime (opt-in)

`PYCT_SYMBOLIC_REPRESENTATION=legacy|compact-affine` selects the numeric symbolic
representation independently of SMT normalization and serialization. The default
is `legacy`. Compact mode keeps the existing `ConcolicFloat` class and stores an
immutable `a*t+b` form alongside its concrete float. Coefficients are exact
`Fraction` values based on the legacy SMT decimal constants, not binary floats.
The representation does not retain historical operand objects. Its ordinary
expression AST is generated lazily when inspected or serialized.

Concrete values still use native float operations and legacy zero shortcuts;
they are not recomputed from the symbolic coefficients. Comparisons still use
the original dispatcher and `ConcolicBool` branch recording. Pickle transfers
the coefficients and drops engine references using the existing protocol; old
states containing an `expr` field remain readable. No new float subclass or
child-process protocol is introduced.

The first version compresses single-variable float inputs, addition, scalar
multiplication, and division by constants. Unsupported expressions, multiple
variables, input clipping/PWL, symbolic products or denominators, mixed owners,
and rational-budget overflow retain a regular symbolic AST. Existing unsupported
Python operations are not repaired by this feature. Safety-validation mode
uses the legacy arithmetic dispatcher. The limit is 4096 combined numerator
and denominator digits per coefficient; exceeding it falls back without
rounding. A lower active Python integer-to-string limit also triggers fallback
before retaining an unprintable coefficient. Exact rational arithmetic has a
cost and can be slower on some models.

Because PWL/clip inputs remain complete piecewise expressions, they mainly
benefit from shared serialization in this version. This mode does not select
one PWL segment and extend it outside its valid region. It also does not repair
attention softmax, LayerNorm or other Transformer concrete dependencies.

After preparing the local cache described above, run a bounded case with:

```bash
PYCT_ALLOW_DATASET_DOWNLOAD=0 \
PYCT_SYMBOLIC_REPRESENTATION=compact-affine PYCT_SMT_SERIALIZATION=shared \
  .venv/bin/python -m pyct \
  --dataset mnist --model-name simple_mnist_m6_09585 \
  --attack-mode queue --pixel-search 1 --first-n 1 --num-process 1 \
  --timeout 17 --solver-run-timeout 2 \
  --constraint-build-timeout-seconds 5 --symbolic-path-threshold 128
```

Compact runs append `_sym-affine` to case and auxiliary constraint-log directory
names, following `_smtshared` when both are enabled. Existing legacy directories
remain unchanged. Case metadata records `symbolic_representation`. Compact SMT
transform statistics add `compact_affine`, counting live compact values and
fallback reasons visible in the selected query, not every forward operation.
The diagnostic traversal is limited to 100,000 objects and reports `truncated`
when it reaches that limit.

Validate with all combinations of legacy/compact runtime and raw/shared output.
Use concrete bit patterns, branch directions, cvc5 formula equivalence, and
candidate replay as correctness checks. Compact canonical expressions may
change structural matching and equivalent solver formulas may yield different
SAT inputs, so identical exploration order or final attack success is not a
guarantee. Measure graph size/memory and forward time separately; exact rational
coefficients do not guarantee the speedups of a floating-point affine method.
