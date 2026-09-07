# Sentinel AI

Sentinel AI is a portfolio project for the detection and structured
investigation of potentially fraudulent transactions.

## Problem

Fraud-review workflows need both reliable quantitative risk assessment and a
clear, auditable way to assemble supporting evidence for human review.

## Proposal

The project will combine classical Machine Learning for quantitative risk,
controlled investigation agents for structured evidence gathering, and an LLM
for grounded planning and explanations. The LLM will not be the primary fraud
classifier and must not invent indicators or evidence.

## Objectives

- Produce reproducible, traceable quantitative risk decisions.
- Support structured investigations with attributable evidence.
- Provide clear, evidence-grounded explanations to human reviewers.
- Maintain clear boundaries between risk scoring, investigation, and language
  generation.

## Conceptual architecture

```text
Transaction -> Feature Pipeline -> ML Risk Engine -> Risk Decision
    -> Investigation Agent -> Evidence Collection -> LLM Explanation
    -> Reviewer -> Investigation Report
```

**ML detects, agents investigate, LLM explains.**

## Planned technologies

- Python, statistics, and classical Machine Learning
- FastAPI
- PostgreSQL and SQL
- Generative AI, LLMs, LangGraph, and controlled tools for agents
- Automated tests, observability, Docker, CI/CD, and later cloud deployment

## Application

The FastAPI application foundation is available locally with:

```powershell
.venv\Scripts\uvicorn.exe sentinel_ai.app:app --reload
```

The liveness endpoint is available at `GET /health` and returns the service
status and package version.

## Persistence

The persistence foundation uses synchronous SQLAlchemy with PostgreSQL and
Alembic-managed schema migrations. Set `DATABASE_URL` in the environment; copy
the safe local example in `.env.example` if needed. Apply migrations only to a
database you explicitly control:

```powershell
.venv\Scripts\alembic.exe upgrade head
```

## Synthetic data

The project provides a 100% synthetic, reproducible transaction dataset for
future EDA and ML research. Generate the default 10,000 rows with:

```powershell
.venv\Scripts\python.exe scripts\generate_dataset.py --rows 10000 --seed 42
```

The output is `data/synthetic_transactions.csv`; `/data/` is ignored by Git.
Its `is_fraud` column is synthetic research-only ground truth and is not part of
the operational database model.

## Exploratory analysis

Reproducible EDA for the canonical synthetic dataset can be run with:

```powershell
.venv\Scripts\python.exe scripts\run_eda.py --rows 10000 --seed 42
```

The documented findings and future feature/split recommendations are in
[`docs/eda.md`](docs/eda.md). No Machine Learning model is implemented.

## Data policy

Development uses **only public, anonymized, or synthetic data**. Real customer
data, private information, credentials, and sensitive datasets must not be
committed to this repository.

## Status

- Implemented: Git repository, Python package layout, documentation,
  architecture boundaries, FastAPI application foundation, `GET /health`, and
  PostgreSQL persistence foundation, synthetic dataset pipeline, and
  reproducible exploratory analysis.
- Planned: Data contracts, risk modelling, investigation workflows, application
  interfaces, testing, observability, delivery automation, and deployment.

No business functionality, models, agent frameworks, external services, or
datasets are implemented at this stage.

## Roadmap

1. **Implemented — Stage 0:** Foundation and specification.
2. **Planned:** Define data contracts and reproducibility practices.
3. **Planned:** Build and evaluate the quantitative ML risk engine.
4. **Planned:** Add controlled investigation workflows and grounded LLM
   explanations.
5. **Planned:** Deliver APIs, automated tests, observability, containerization,
   CI/CD, and cloud deployment.

## Repository layout

```text
src/sentinel_ai/  # Python package; intentionally empty of business features
tests/            # Reserved for automated tests
docs/             # Product, architecture, and decision records
```
