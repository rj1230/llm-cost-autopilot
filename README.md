# LLM Cost Autopilot

**Production-oriented, cost-aware LLM routing with confidence-aware tier promotion, provider fallback, quality verification, escalation, audit logging, and observability.**

LLM Cost Autopilot is an intelligent routing layer that sits in front of multiple LLM providers. It classifies incoming requests by complexity, routes them to an appropriate cost/quality tier, verifies lower-cost responses when required, and escalates when quality is insufficient.

The system is designed around a practical production constraint:

> **Use the least expensive model that is likely to satisfy the request, while maintaining explicit safety, quality, failure, and accounting paths.**

---

## Architecture

```text
                         Incoming Request
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Complexity Classifier │
                    │   Random Forest V4   │
                    └──────────┬───────────┘
                               │
                     Raw tier + confidence
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Confidence Safety Policy │
                 │ <85% → promote one tier │
                 └────────────┬─────────────┘
                              │
                              ▼
                    Final Routing Tier
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
          Tier 1           Tier 2           Tier 3
        Mistral Small   Groq GPT-OSS-20B    Higher tier
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                    Provider / Circuit
                     Breaker / Retry
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  Success             Failure
                    │                   │
                    │              Fallback model
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Response to caller
                              │
                              ▼
                    Async Quality Verification
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  Passed             Failed
                    │                   │
                    ▼                   ▼
                  Keep              Escalate
                                      │
                                      ▼
                              Stronger reference
                                  response
                              │
                              ▼
                         Audit + Metrics
```

---

## Current Results

The current deterministic benchmark contains **30 fixed requests** and uses mocked provider responses for reproducibility.

| Metric                         |       Current result |
| ------------------------------ | -------------------: |
| Benchmark requests             |               **30** |
| Routing accuracy               |          **100.00%** |
| Success rate                   |          **100.00%** |
| Fallback rate                  |            **0.00%** |
| Average latency                |         **0.9333 s** |
| P50 latency                    |         **0.9000 s** |
| P95 latency                    |         **1.3500 s** |
| P99 latency                    |         **1.3500 s** |
| Total tokens                   |            **5,800** |
| Actual benchmark cost          |        **$0.001425** |
| Average cost/request           |       **$0.0000475** |
| Configured all-GPT-4o baseline |        **$0.041125** |
| Estimated cost reduction       | **$0.0397 / 96.53%** |

The cost-reduction figure is an **estimated comparison against the project's configured all-GPT-4o price baseline**, not a claim of universal production savings.

### Frozen V4 baseline

The classifier's frozen V4 evaluation remains preserved independently from the current benchmark artifact:

* 30-request independent benchmark
* **93.33% routing accuracy**
* 100% success
* 0% fallback
* Frozen artifact: `benchmark/results/v4_final.json`

The current `benchmark/results/latest.json` represents the latest benchmark run and should not be confused with the frozen V4 evaluation.

---

## Why This Project Is More Than Basic LLM Routing

The project combines several production-oriented mechanisms rather than simply selecting a model from a static list.

### 1. Confidence-aware routing

The classifier produces:

* predicted complexity tier
* classification confidence
* probability distribution

A conservative routing policy is applied:

```text
confidence >= 85%
    → retain predicted tier

confidence < 85%
    → promote T1/T2 by one tier

T3
    → remains T3
```

This separates the **machine-learning prediction** from the **operational safety policy**.

For example:

```text
Classifier:
    Tier 1
    Confidence: 62%

Safety policy:
    Low confidence

Final routing:
    Tier 2
```

The API exposes both the raw classifier decision and the final routing decision, making the promotion auditable.

---

## 2. V4 Complexity Classifier

The current classifier uses a Random Forest model trained on a curated labeled dataset.

Current V4 dataset:

* **335 labeled examples**
* Tier 1: 121
* Tier 2: 106
* Tier 3: 108
* **19 structural features**
* Selected Random Forest classifier
* Frozen production artifact: `data/classifier_v4_final.joblib`

The V4 held-out model evaluation reached approximately **97% accuracy**.

The independent benchmark is kept separate from the training/held-out evaluation to provide a more realistic routing check.

### Validate

```powershell
uv run python -m scripts.check_routing_offline
uv run pytest -q
```

The production service loads the frozen V4 artifact `data/classifier_v4_final.joblib` directly. The application does not retrain the classifier during startup or request handling.

---

## 3. Provider-Agnostic Model Registry

Models are represented through a common registry and normalized response interface.

Current routing configuration includes:

| Tier   | Model                          | Provider              |
| ------ | ------------------------------ | --------------------- |
| Tier 1 | `mistral-small`                | Mistral               |
| Tier 2 | `groq-gpt-oss-20b`             | Groq                  |
| Tier 3 | Configured higher-quality tier | Provider-configurable |

Provider-specific failures are normalized into categories such as:

* rate limit
* timeout
* authentication
* server error
* circuit open
* provider failure

This keeps routing and accounting logic independent of provider-specific SDK behavior.

---

## 4. Retry and Circuit-Breaker Resilience

Provider calls are protected by bounded retry and circuit-breaker behavior.

The system can:

1. Attempt the primary provider.
2. Retry transient failures with bounded backoff.
3. Open a circuit after repeated failures.
4. Route to a configured fallback.
5. Record the primary failure and fallback decision.
6. Return a standardized response to the caller.

The audit trail distinguishes the primary model from the model that actually served the request.

Example:

```text
Primary model:
    mistral-small

Primary failure:
    rate_limit

Fallback:
    groq-gpt-oss-20b

used_fallback:
    true
```

This makes operational failure behavior observable rather than hiding it behind a generic success response.

---

## 5. Async Quality Verification

Lower-cost responses can be verified against a stronger reference model.

The verification pipeline records:

* task type
* quality score
* verification threshold
* original model
* reference model
* pass/fail result
* escalation status
* quality gap
* additional cost
* verification latency

Conceptually:

```text
Cheap response
      │
      ▼
Quality verification
      │
 ┌────┴────┐
 │         │
Pass      Fail
 │         │
 ▼         ▼
Keep    Escalate
           │
           ▼
      Reference response
```

Tier-3 requests can skip automatic verification because they are already routed directly to the highest configured quality tier.

Verification can operate asynchronously so the normal routing path does not have to wait for the quality audit.

---

## 6. Full Audit and Cost Accounting

Every routed request receives a request ID and an audit record.

The SQLite audit layer tracks information including:

* request ID
* classifier tier
* final routing tier
* classifier confidence
* selected model
* primary model
* fallback model
* provider error type
* token usage
* cost
* latency
* verification status
* escalation
* quality information

Raw prompts are not stored in the audit database; the system stores a hash for request-level identification.

This enables operational questions such as:

```text
How often is the classifier uncertain?

How often does uncertainty cause promotion?

Which provider is failing?

How often does fallback occur?

How much does verification cost?

How much would the configured GPT-4o baseline have cost?

Which routing tiers receive the most traffic?
```

---

## 7. FastAPI Service

The routing engine is exposed through an OpenAI-style completion API.

### Endpoints

| Method | Endpoint             | Purpose                      |
| ------ | -------------------- | ---------------------------- |
| `POST` | `/v1/completions`    | Route and complete a request |
| `GET`  | `/v1/models`         | List available models        |
| `GET`  | `/v1/stats`          | Routing and cost statistics  |
| `PUT`  | `/v1/routing-config` | Update routing configuration |

Interactive API documentation is available through FastAPI's generated `/docs` interface when the service is running.

### Start locally

```powershell
uv run uvicorn src.api.main:app --reload
```

---

## 8. Streamlit Command Center

The Streamlit dashboard provides an operational view of the system.

It includes:

* request playground
* routing decision
* raw classifier tier
* classifier confidence
* confidence safety policy
* final routing tier
* promotion transitions
* model distribution
* provider health
* fallback statistics
* cost trends
* verification metrics
* escalation metrics
* benchmark results
* audit records
* architecture visualization
* routing observability

The dashboard can operate locally against the FastAPI service or use **direct in-process routing** for Streamlit Cloud deployment.

### Start locally

```powershell
uv run streamlit run dashboard/app.py
```

---

## 9. Streamlit Cloud Deployment

The dashboard supports a Cloud direct mode that runs the routing engine in-process instead of requiring a separately hosted FastAPI server.

Enable it with:

```text
AUTOPILOT_CLOUD_MODE=true
```

Provider credentials should be supplied through Streamlit's secrets configuration rather than committed to the repository.

Typical secrets include:

```text
AUTOPILOT_CLOUD_MODE=true
MISTRAL_API_KEY=...
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
```

Secrets are intentionally excluded from Git.

### Important deployment limitation

The current audit database is SQLite.

Streamlit Cloud's local filesystem should not be treated as durable production storage. The application can therefore demonstrate the routing system and its observability layer in Cloud, but durable multi-instance production analytics would require an external persistent database.

---

## 10. API Response Contract

A successful completion exposes routing metadata alongside the generated response.

Conceptually:

```json
{
  "id": "request-id",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "routing": {
    "tier": 2,
    "classifier_tier": 1,
    "classification_confidence": 0.62,
    "low_confidence": true,
    "selected_model": "groq-gpt-oss-20b",
    "reasoning": "classified as simple - routed to the cheapest model (low-confidence classifier prediction was promoted to a safer routing tier)",
    "used_fallback": false,
    "cost_usd": 0.00005,
    "latency_s": 0.91,
    "verification": "queued"
  }
}
```

This makes the routing decision explainable and auditable instead of exposing only the final generated text.

---

## 11. Testing

The project currently has:

```text
135 passed
1 dependency warning
```

Full regression:

```powershell
uv run pytest -q
```

The remaining warning originates from the installed Starlette/AnyIO dependency stack rather than application code.

The test suite covers routing, classification, API behavior, authentication edge cases, provider behavior, fallback handling, verification, logging, statistics, and resilience paths.

---

## 12. Benchmarking

The deterministic benchmark runner is:

```powershell
uv run python -m benchmark.benchmark
```

Do **not** execute `benchmark\benchmark.py` directly because the project imports the repository's `src` package.

The benchmark writes the current result to:

```text
benchmark/results/latest.json
```

The frozen V4 result is preserved at:

```text
benchmark/results/v4_final.json
```

The benchmark uses mocked provider responses so routing evaluation is deterministic and does not require live provider calls.

---

## 13. Operational Validation

The project also validates live provider failure behavior.

For example, when the Mistral provider is rate-limited, the system records the provider failure and successfully exercises the configured fallback path rather than treating the failure as an unexplained application error.

This distinction is important:

```text
Model selection
      +
Provider reliability
      +
Fallback behavior
      +
Quality verification
      +
Cost accounting
```

are evaluated as separate system concerns.

---

## Repository Structure

```text
llm-cost-autopilot/
│
├── benchmark/
│   ├── benchmark.py
│   └── results/
│       ├── latest.json
│       └── v4_final.json
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── classifier_v4_final.joblib
│   ├── labeled_dataset_v4.csv
│   └── ...
│
├── src/
│   ├── api/
│   ├── classifier/
│   ├── models/
│   ├── providers/
│   ├── verification/
│   ├── routing.py
│   ├── logging_db.py
│   └── config.py
│
├── tests/
│
├── .streamlit/
│   └── config.toml
│
├── CASE_STUDY.md
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Design Principles

### Cost-aware, not cost-only

The cheapest model is not always the correct model. The router therefore combines complexity classification with confidence-aware promotion and post-response verification.

### ML decision ≠ production decision

The classifier produces a prediction. A separate safety policy can modify that prediction when confidence is low.

### Failures are first-class events

Provider failures, retries, circuit states, fallback decisions, verification failures, and escalations are recorded rather than hidden.

### Reproducibility over impressive-looking numbers

The benchmark separates:

* held-out classifier evaluation
* independent routing benchmark
* frozen V4 results
* current benchmark results
* live provider smoke tests

This prevents unrelated measurements from being presented as the same metric.

### Explainability at the API boundary

The caller can inspect:

```text
raw classifier tier
confidence
final routing tier
promotion
selected model
fallback
cost
latency
verification status
```

rather than receiving only a model response.

---

## Engineering Highlights

* Confidence-aware ML routing
* Three-tier model architecture
* Provider abstraction and normalized responses
* Bounded retry logic
* Circuit breakers
* Automatic fallback
* Async quality verification
* Quality-based escalation
* SQLite audit logging
* Cost accounting
* Routing observability
* API authentication hardening
* Streamlit operational dashboard
* Streamlit Cloud direct execution mode
* Deterministic routing benchmark
* Frozen model artifact
* Comprehensive regression suite

---

## Reproducibility

Create/activate the environment and install dependencies according to the project's package configuration.

Run the complete regression suite:

```powershell
uv run pytest -q
```

Run the deterministic routing benchmark:

```powershell
uv run python -m benchmark.benchmark
```

Start the API:

```powershell
uv run uvicorn src.api.main:app --reload
```

Start the dashboard:

```powershell
uv run streamlit run dashboard/app.py
```

---

## Further Reading

The detailed engineering decisions, evaluation methodology, trade-offs, and project evolution are documented in:

**[CASE_STUDY.md](./CASE_STUDY.md)**

---

## Project Status

**Engineering implementation: complete**

The current repository represents a production-oriented portfolio implementation of cost-aware LLM routing with explicit ML, reliability, verification, accounting, API, and observability layers.

The remaining limitations are primarily infrastructure-related, including durable cloud persistence and the need for larger real-world evaluation datasets before making production-scale performance claims.