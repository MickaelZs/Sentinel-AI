# Deterministic Investigation Agent

This experimental agent uses LangGraph only to orchestrate deterministic local
capabilities. It has no LLM, prompts, external tools, database access, internet
access, persistence, or autonomous claims.

```text
START → validate → score → explain → collect evidence → assess
      → decision route → finalize → END
```

ML produces the risk probability. Deterministic explainability supplies reason
codes. LangGraph coordinates state and the bounded flow. The current policy is
exactly `risk_prediction=True → manual_review`; otherwise `no_review`. It adds
no threshold and may recommend review, never declare fraud as fact.

Evidence is a direct one-to-one conversion of existing reason codes, ordered as
increasing then decreasing contribution, with deterministic `evidence-001`
identifiers. Future controlled tools may include customer/device history and
behavior comparison, but are not implemented here.
