# ADR 0001: Separate risk scoring, investigation, and language generation

- Status: Accepted
- Date: 2026-08-23

## Context

Sentinel AI will help identify and investigate potentially fraudulent
transactions. The system needs quantitative decisions that are reproducible,
structured investigation steps that are auditable, and clear explanations for
human reviewers. A language model alone is not an acceptable primary source of
fraud classification because it can produce unsupported statements and is not a
deterministic quantitative risk engine.

## Decision

Sentinel AI will maintain three distinct responsibilities:

1. Machine Learning and statistical components calculate quantitative risk from
   defined, versioned inputs.
2. Investigation agents conduct structured evidence collection and investigation
   workflows using controlled tools.
3. The LLM plans only within controlled constraints and produces explanations
   based on supplied evidence; it does not invent indicators or evidence and it
   does not replace the quantitative risk decision.

The guiding principle is: **ML detects, agents investigate, LLM explains.**

## Consequences

- Important outcomes can be traced to model inputs, model versions, evidence,
  and investigation steps.
- Interfaces between these responsibilities must preserve provenance and enable
  reproducibility.
- LLM outputs must be bounded by validated data and presented as explanations,
  not as the primary fraud score.
- This separation introduces integration work later, but prevents a single
  opaque component from owning classification, evidence, and narrative.
