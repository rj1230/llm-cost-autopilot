



LLM Cost Autopilot
Production-oriented, cost-aware LLM routing with confidence-aware tier promotion, provider fallback, quality verification, escalation, audit logging, and observability.

LLM Cost Autopilot is an intelligent routing layer that sits in front of multiple LLM providers. It classifies incoming requests by complexity, routes them to an appropriate cost/quality tier, verifies lower-cost responses when required, and escalates when quality is insufficient.

Core principle: Use the least expensive model that is likely to satisfy the request, while maintaining explicit safety, quality, failure, and accounting paths.

Validation Snapshot
Gate	Result
Automated regression	135 passed
Deterministic benchmark	100% routing accuracy
Benchmark success rate	100%
Benchmark fallback rate	0%
Average benchmark latency	0.9333 s
Benchmark P95/P99	1.3500 s
Benchmark actual cost	$0.001425
Estimated reduction vs configured all-GPT-4o baseline	96.53%
Live confidence-promotion smoke test	Passed
Live provider-fallback smoke test	Passed
SQLite audit reconstruction	Passed
Streamlit direct-mode smoke test	Passed
The 96.53% figure is an estimated comparison against the project's configured all-GPT-4o pricing baseline. It is not a claim of universal production savings.

Architecture
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
│ <85% → promote T1/T2     │
└────────────┬─────────────┘
             │
             ▼
      Final Routing Tier
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
     T1      T2     T3
   Mistral   Groq   Highest
    Small   GPT-OSS configured
             20B      tier
       │     │       │
       └─────┼───────┘
             ▼
    Provider / Retry /
      Circuit Breaker
             │
       ┌─────┴─────┐
       │           │
    Success      Failure
       │           │
       │      Configured fallback
       │           │
       └─────┬─────┘
             ▼
       Response to caller
             │
             ▼
   Async Quality Verification
             │
       ┌─────┴─────┐
       │           │
     Passed       Failed
       │           │
      Keep      Escalate
                   │
                   ▼
           Stronger reference
                response
                   │
                   ▼
             Audit + Metrics
Current routing configuration
Tier 1 → mistral-small
Tier 2 → groq-gpt-oss-20b
Tier 3 → groq-gpt-oss-20b
Tier 3 is retained as the highest complexity tier in the routing policy. In the current deployment, Tier 2 and Tier 3 map to the same Groq model rather than separate live providers.

Current Benchmark
The current deterministic benchmark contains 30 fixed requests and uses mocked provider responses for reproducibility.

Metric	Current result
Requests	30
Routing accuracy	100.00%
Success rate	100.00%
Fallback rate	0.00%
Average latency	0.9333 s
P50 latency	0.9000 s
P95 latency	1.3500 s
P99 latency	1.3500 s
Input tokens	2,250
Output tokens	3,550
Total tokens	5,800
Actual benchmark cost	$0.001425
Average cost/request	$0.0000475
Configured all-GPT-4o baseline	$0.041125
Estimated cost reduction	$0.039700 / 96.53%
Run it with:

uv run python -m benchmark.benchmark
The result is written to:

benchmark/results/latest.json
The benchmark is deterministic and does not require live provider calls.

Frozen V4 baseline
The frozen V4 evaluation is preserved separately from the current benchmark:

30-request independent benchmark

93.33% routing accuracy

100% success

0% fallback

Frozen result: benchmark/results/v4_final.json

The 93.33% figure is the historical frozen V4 evaluation. The 100% figure is the current deterministic benchmark after subsequent routing-policy and implementation work. They should not be presented as the same measurement.

Why This Is More Than Basic LLM Routing
The project combines model selection with explicit reliability, quality, and accounting mechanisms.

1. Confidence-aware routing
The classifier produces:

predicted complexity tier

classification confidence

probability distribution

The operational policy is:

confidence >= 85%
    → retain predicted tier

confidence < 85%
    → promote T1/T2 by one tier

T3
    → remains T3
This deliberately separates the ML prediction from the production routing policy.

Example:

Classifier:
    Tier 1
    Confidence: 80.5%

Safety policy:
    Low confidence

Final routing:
    Tier 2
The API and audit database expose both decisions.

2. V4 Complexity Classifier
The current classifier is a frozen Random Forest model trained on a curated labeled dataset.

V4 dataset
335 labeled examples

Tier 1: 121

Tier 2: 106

Tier 3: 108

19 structural features

Random Forest classifier

Frozen artifact: data/classifier_v4_final.joblib

The V4 held-out evaluation reached approximately 97% accuracy.

The independent benchmark is kept separate from the training/held-out evaluation to provide a more realistic routing check.

Validate
uv run python -m scripts.check_routing_offline
uv run pytest -q
The production service loads the frozen V4 artifact directly. The application does not retrain the classifier during startup or request handling.

3. Provider-Agnostic Model Registry
Models are represented through a common registry and normalized response interface.

Current live routing:

Tier	Model	Provider
Tier 1	mistral-small	Mistral
Tier 2	groq-gpt-oss-20b	Groq
Tier 3	groq-gpt-oss-20b	Groq
The repository also retains GPT-4o pricing information as a configured baseline for cost comparisons; it is not the current live routing target.

Provider failures are normalized into categories such as:

rate limit

timeout

authentication

server error

network/provider failure

circuit open

This keeps routing and accounting logic independent of provider-specific SDK behavior.

4. Retry, Circuit Breaker, and Fallback
Provider calls are protected by bounded retry and circuit-breaker behavior.

The system can:

Attempt the primary provider.

Retry transient failures with bounded backoff.

Open a circuit after repeated failures.

Route to a configured fallback.

Record the primary failure and fallback decision.

Return a standardized response.

The audit trail distinguishes the model selected as primary from the model that actually served the request.

Verified live failure path
A Streamlit smoke test exercised the real fallback path:

Primary:
    mistral-small

Primary failure:
    rate_limit

Fallback:
    groq-gpt-oss-20b

Result:
    successful response

Quality score:
    1.0
The corresponding SQLite audit record confirmed:

tier = 1
classifier_tier = 1
classification_confidence = 1.0
primary_model = mistral-small
primary_error_type = rate_limit
routed_model = groq-gpt-oss-20b
used_fallback = 1
verified = 1
quality_score = 1.0
circuit_state = closed
This demonstrates that fallback is an observable production path rather than only a unit-tested branch.

5. Async Quality Verification
Lower-cost responses can be verified against a stronger reference model.

The verification pipeline records:

task type

quality score

verification threshold

original model

reference model

pass/fail result

escalation status

quality gap

additional cost

verification latency

Conceptually:

Cheap response
      │
      ▼
Quality verification
      │
   ┌──┴──┐
   │     │
 Pass   Fail
   │     │
 Keep  Escalate
          │
          ▼
     Reference response
Tier-3 requests can skip automatic verification because they are already routed to the highest configured quality tier.

Verification can operate asynchronously so the normal routing path does not have to wait for the quality audit.

6. Full Audit and Cost Accounting
Every routed request receives a request ID and an audit record.

The SQLite audit layer tracks information including:

request ID

classifier tier

final routing tier

classifier confidence

selected/routed model

primary model

provider error type

token usage

cost

latency

verification status

escalation

quality information

Raw prompts are not stored in the audit database; the system stores a hash for request-level identification.

This enables operational questions such as:

How often is the classifier uncertain?

How often does uncertainty cause promotion?

Which provider is failing?

How often does fallback occur?

How much does verification cost?

How much would the configured GPT-4o baseline have cost?

Which routing tiers receive the most traffic?
7. FastAPI Service
The routing engine is exposed through an OpenAI-style completion API.

Method	Endpoint	Purpose
GET	/healthz	Liveness
GET	/readyz	Dependency/readiness checks
POST	/v1/completions	Route and complete a request
GET	/v1/models	List available models
GET	/v1/stats	Routing and cost statistics
PUT	/v1/routing-config	Update routing configuration
Interactive API documentation is available through FastAPI's generated /docs interface when the service is running.

Start locally
uv run uvicorn src.api.main:app --reload
8. Streamlit Command Center
The Streamlit dashboard provides an operational view of the system.

It includes:

request playground

routing decision

raw classifier tier

classifier confidence

confidence safety policy

final routing tier

promotion transitions

model distribution

provider health

fallback statistics

cost trends

verification metrics

escalation metrics

benchmark results

audit records

routing observability

The dashboard can operate locally against FastAPI or use direct in-process routing for Streamlit Cloud deployment.

Start locally
uv run streamlit run dashboard/app.py
9. Streamlit Cloud Deployment
The dashboard supports a Cloud direct mode that runs the routing engine in-process instead of requiring a separately hosted FastAPI server.

Enable it with:

AUTOPILOT_CLOUD_MODE=true
Provider credentials should be supplied through Streamlit's secrets configuration rather than committed to Git.

Typical configuration:

AUTOPILOT_CLOUD_MODE=true
MISTRAL_API_KEY=...
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
Secrets are intentionally excluded from Git.

Deployment limitation
The current audit database is SQLite.

Streamlit Cloud's local filesystem should not be treated as durable production storage. The application can therefore demonstrate the routing system and its observability layer in Cloud, but durable multi-instance production analytics would require an external persistent database.

10. API Response Contract
A successful completion exposes routing metadata alongside the generated response.

The following is an illustrative response shape; values are examples:

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
This makes the routing decision explainable and auditable instead of exposing only the generated text.

11. Testing
The current regression suite reports:

135 passed
1 dependency warning
Run:

uv run pytest -q
The remaining warning originates from the installed Starlette/AnyIO dependency stack rather than application code.

The test suite covers routing, classification, API behavior, authentication edge cases, provider behavior, fallback handling, verification, logging, statistics, and resilience paths.

12. Benchmarking
Use the module invocation:

uv run python -m benchmark.benchmark
Do not execute the file directly with:

uv run python .\benchmark\benchmark.py
The module invocation is the supported project command because the benchmark imports the repository's src package.

Results:

benchmark/results/latest.json
Frozen V4 result:

benchmark/results/v4_final.json
The benchmark uses mocked provider responses so routing evaluation is deterministic and does not require live provider calls.

13. Operational Validation
The project validates more than deterministic unit-test behavior.

The final local validation sequence included:

FastAPI /healthz                  PASS
FastAPI /readyz                   PASS
Authenticated /v1/models         PASS
Authenticated /v1/completions    PASS
Confidence promotion             PASS
SQLite request audit             PASS
/v1/stats aggregation            PASS
Deterministic benchmark          PASS
Streamlit direct mode            PASS
Live fallback path               PASS
Full regression                  PASS
Verified confidence-promotion path
A live completion produced:

Classifier tier:       1
Confidence:            80.5%
Low confidence:        true
Final tier:            2
Selected model:        groq-gpt-oss-20b
Fallback:              false
Response:              4
The corresponding SQLite audit record confirmed the same routing transition.

Verified provider-fallback path
A separate Streamlit request produced:

Classifier tier:       1
Confidence:            100%
Primary:               mistral-small
Primary error:         rate_limit
Fallback:              groq-gpt-oss-20b
Response:              4
Quality score:         1.0
The fallback request was persisted to the audit database.

Repository Structure
llm-cost-autopilot/
│
├── benchmark/
│   ├── benchmark.py
│   └── results/
│       ├── latest.json
│       └── v4_final.json
│
├── config/
│   └── routing.yaml
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── classifier_v4_final.joblib
│   ├── labeled_dataset_v4.csv
│   └── ...
│
├── scripts/
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
├── README.md
├── requirements.txt
└── pyproject.toml
Runtime secrets, local environments, databases, logs, and other generated artifacts are intentionally excluded from Git where appropriate.

Design Principles
Cost-aware, not cost-only
The cheapest model is not always the correct model. The router combines complexity classification with confidence-aware promotion and post-response verification.

ML decision ≠ production decision
The classifier produces a prediction. A separate safety policy can modify that prediction when confidence is low.

Failures are first-class events
Provider failures, retries, circuit states, fallback decisions, verification failures, and escalations are recorded rather than hidden.

Reproducibility over impressive-looking numbers
The evaluation separates:

held-out classifier evaluation

independent routing benchmark

frozen V4 results

current benchmark results

live provider smoke tests

This prevents unrelated measurements from being presented as the same metric.

Explainability at the API boundary
The caller can inspect:

raw classifier tier
confidence
final routing tier
promotion
selected model
fallback
cost
latency
verification status
rather than receiving only a model response.

Engineering Highlights
Confidence-aware ML routing

Three-tier routing policy

Provider abstraction and normalized responses

Bounded retry logic

Circuit breakers

Automatic fallback

Async quality verification

Quality-based escalation

SQLite audit logging

Cost accounting

Routing observability

API authentication

Streamlit operational dashboard

Streamlit Cloud direct execution mode

Deterministic routing benchmark

Frozen V4 model artifact

Comprehensive regression suite

Reproducibility
Create and activate the environment according to the project's package configuration.

Run the complete regression suite:

uv run pytest -q
Run offline routing validation:

uv run python -m scripts.check_routing_offline
Run the deterministic benchmark:

uv run python -m benchmark.benchmark
Start the API:

uv run uvicorn src.api.main:app --reload
Start the dashboard:

uv run streamlit run dashboard/app.p