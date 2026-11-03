# Project Spec

## Hypothesis
If brain-derived embeddings encode task-relevant abstract structure, models aligned
to real brain embeddings generalize differently (better OOD / novel-variant transfer)
than models aligned to noise-matched surrogates.

## Conditions
Real brain | MaxEnt surrogate | Shuffled brain | Random noise

## Primary metric
delta_causal = metric(Real) - max(metric(MaxEnt), metric(Shuffled), metric(Random))
Measured in-distribution (ID) and out-of-distribution (OOD).

## Success criteria (fixed BEFORE running experiments)
Brain-specific transfer is supported if:
1. delta_causal > 0 on OOD, with 95% bootstrap CI excluding 0
2. Paired test across >= 5 seeds significant after multiple-comparison correction
3. Effect not explained by alignment weight alone (lambda sweep)
Otherwise the result is reported as "consistent with regularization".

## Research questions
RQ1 Does brain similarity causally predict generalization beyond noise-matched controls?
RQ2 Which brain regions contribute most?
RQ3 Does the effect hold across RNN, Transformer, RL?
(RQ4 emotion: dropped.)

## Dataset
OpenNeuro ds004752 (primary). UPenn RAM (replication, later).