# All-family translation-tangent objective

## Decision

The paper-dataset training path applies the localized translation-tangent
objective to every nonflat training and validation row. It no longer selects
Tanaka rows or assigns a family-specific weight. The configured coefficient
remains `10`.

This is a deliberate change from C27, where the same objective was evaluated
only on legacy Tanaka sources. A run with this implementation must therefore
be described as an all-family translation-tangent experiment, not an exact
C27 reproduction.

## Rationale

Spatial translation is a symmetry of the periodic Dirichlet--Neumann operator
for every represented physical family. The field `eta_x` is the infinitesimal
tangent to that group orbit, so projecting the prediction error onto `eta_x`
emphasizes a phase-sensitive component of operator error without assuming that
the complete state is a rigid traveling wave. Because the objective acts on
prediction minus truth, it does not penalize the reference dynamics; it only
changes the relative importance assigned to one error direction.

The loss retains its numerical low-energy gate. A row contributes only when
its smoothed `eta_x` energy is above the existing threshold. No physical-family
identifier is consumed by the trainer or regularizer.

## Evidence status

The repository contains Tanaka-only versus tangent-deletion evidence, but no
completed Tanaka-only versus all-family ablation. Before promoting the changed
scope, report the unweighted and weighted tangent contribution on the new
dataset and evaluate matched long rollouts for all four families. In particular,
check that Benjamin--Feir and JONSWAP/TMA losses do not dominate solely because
their slope-energy distributions differ from Tanaka.
