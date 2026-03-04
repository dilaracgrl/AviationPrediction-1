# Aviation Flight Delay Prediction
Predictive Analytics Coursework Project

---

## Project Overview

This project develops a predictive model to estimate whether a flight will arrive delayed (≥15 minutes) using the U.S. Bureau of Transportation Statistics Airline On-Time Performance dataset.

Predictions are made at **scheduled departure time**, meaning only information available before departure is allowed as model input.

The project emphasises:

- statistical validity
- reproducibility
- auditability
- responsible AI-assisted development

---

## Project Rules (AI Usage)

### 1. Logging is Mandatory
After every meaningful AI-assisted action:

- A new entry must be added to `agent_logs/agent_log.md`
- The prompt used must be recorded
- Files changed must be listed
- Risks must be identified
- Human verification must be completed later

---

### 2. Decision Register
AI suggestions requiring judgement (features, models, metrics, validation strategy) must be proposed but NOT automatically accepted.

The human decides whether to record them in: agent_logs/decision_register.csv



---

### 3. No Silent Changes
All AI-generated changes must include:
- explanation of what changed
- explanation of why it changed

---

## Reproducibility Principle

All results must be reproducible through:
- documented data access
- version-controlled code
- logged AI interactions
- deterministic sampling

---

## Author
Predictive Analytics Coursework – Aviation Delay Modelling by Dilara Cigerli
