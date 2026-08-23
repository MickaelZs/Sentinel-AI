# Product

## Problem

Reviewing potentially fraudulent transactions is difficult when quantitative
risk signals, supporting evidence, and reviewer-facing explanations are spread
across disconnected processes. Sentinel AI is intended to make that review more
structured and traceable.

## Hypothetical users

- Fraud analysts who triage and investigate flagged transactions.
- Risk operations teams that need consistent review workflows.
- Technical and compliance stakeholders who need reproducible rationale for
  important outcomes.

## Intended use cases

- Calculate a quantitative risk signal for an incoming transaction.
- Investigate a transaction using controlled, attributable evidence sources.
- Present a grounded explanation and investigation report to a human reviewer.

## Main flow

1. A transaction enters the feature and risk-assessment flow.
2. A quantitative risk decision is produced from defined inputs.
3. Potentially relevant cases proceed to structured evidence collection and
   investigation.
4. The system prepares an evidence-grounded explanation for human review.
5. The reviewer evaluates the report and remains responsible for the outcome.

## Out of scope

- Autonomous enforcement, blocking, or account closure.
- Using private customer data or real customer information for development.
- Treating LLM output as the primary fraud classification.
- Replacing human review for material decisions.

## Limitations

- Risk signals can be incomplete, biased, or affected by changing fraud
  patterns; they require monitoring and periodic review.
- Evidence availability constrains the quality of an investigation.
- LLM explanations can only be trusted when grounded in validated evidence and
  bounded by system controls.

## Initial success criteria

- Quantitative risk results can be reproduced from versioned inputs and model
  metadata.
- Material evidence in an investigation can be traced to its source.
- Reviewer-facing explanations distinguish facts, model outputs, and inference.
- Development uses only public, anonymized, or synthetic data.
