# Architecture

## Conceptual flow

```text
Transaction
  -> Feature Pipeline
  -> ML Risk Engine
  -> Risk Decision
  -> Investigation Agent
  -> Evidence Collection
  -> LLM Explanation
  -> Reviewer
  -> Investigation Report
```

## Governing principle

**ML detects, agents investigate, LLM explains.**

The quantitative risk decision belongs to statistical and Machine Learning
components using defined inputs. Investigation agents collect and organize
evidence through controlled tools and workflows. The LLM supports constrained
planning and produces explanations from that evidence; it is not the primary
fraud classifier and must not fabricate indicators or evidence.

## Initial architectural boundaries

- The Feature Pipeline will produce versioned, reproducible inputs for the ML
  Risk Engine.
- The ML Risk Engine will emit a quantitative risk result and supporting
  metadata suitable for audit.
- The Investigation Agent will not silently alter the risk result; it will
  perform a separate, structured investigation.
- Evidence Collection will preserve source and provenance for material facts.
- LLM Explanation will be grounded in the risk result and collected evidence.
- A human Reviewer remains responsible for reviewing material investigation
  results.

These are conceptual boundaries only. No runtime components are implemented in
this foundation stage.
