
# LLM Cost Autopilot

**Cost-aware LLM routing with async quality verification, escalation, and observability.**

LLM Cost Autopilot is an intelligent routing layer that sits in front of multiple LLM providers.
It classifies each request's complexity, routes it to the cheapest model likely to handle it well,
verifies lower-cost responses in the background, and escalates or learns from disagreements.

**Result on a 400-request benchmark:** **52.6% lower cost** than routing every request to GPT-4o,
with **86.9% average agreement** on self-verified responses.

The full engineering narrative, benchmark methodology, and trade-offs are in  
[`CASE_STUDY.md`](https://github.com/rj1230/llm-cost-autopilot/blob/main/CASE_STUDY.md).

## How It Works

```text
Incoming request
      ↓
Complexity classifier
      ↓
Tier → model routing (YAML-configurable)
      ↓
Selected provider responds
      ↓
Cheap-tier response sent to caller
      ↓
Background verification against a stronger model
      ↓
Agreement / disagreement
      ├── Agreement → logged for quality tracking
      └── Disagreement → escalation + feedback example for future retraining
```

The system optimizes for a practical production constraint: use expensive frontier models only
where their additional quality is actually needed, while retaining a verification path to detect
when cheaper routing was wrong.

## Key Results

| Metric | Result |
|---|---|
| Cost reduction vs. GPT-4o baseline | **52.6%** |
| Verification agreement rate | **86.9%** |
| Benchmark size | 400 requests |
| Classifier held-out accuracy | 90.7% |
| Tests | 17 offline / mocked tests |

Results are reproducible:

```bash
python -m scripts.generate_report
```

## Features

### 1. Unified Model Interface

A single `send_request(prompt, model_config)` interface works consistently across:

- OpenAI
- Anthropic
- Local Ollama

Every provider returns the same normalized `Response` object, so routing, logging, verification,
and billing logic do not need provider-specific branches.

The model registry includes real per-token pricing for:

- GPT-4o
- GPT-4o-mini
- Claude Sonnet 4.5
- Claude Haiku 4.5
- Local Llama

Run provider tests:

```bash
python -m tests.test_providers
```

### 2. Complexity Classifier

Incoming prompts are scored using nine heuristic features, including prompt length, keyword signals,
and structural indicators. A Random Forest classifier assigns each request to a routing tier.

- 215-prompt labeled dataset
- 9 heuristic features
- 90.7% held-out classification accuracy
- YAML-based tier-to-model mapping
- Routing rules editable without redeployment

Rebuild or validate the classifier:

```bash
python -m data.generate_dataset
python -m src.classifier.train
python -m scripts.check_routing_offline
```

The trained classifier ships as `data/classifier.joblib`, so the project runs without needing to
retrain immediately.

### 3. Async Quality Verification

Cheap-tier responses are returned immediately. In the background, the same prompt is sent to a
stronger tier-3 model and the two responses are compared using a task-aware scoring strategy:

| Task type | Verification method |
|---|---|
| Classification | Exact label match |
| Extraction | Word overlap |
| Summarization / general generation | LLM-as-judge |

When disagreement exceeds the configured threshold, the system:

1. Logs the disagreement.
2. Escalates the request.
3. Stores the example as feedback for future classifier retraining.

Weekly retraining is supported through:

```bash
python -m src.verification.retrain_weekly
```

Run verification tests:

```bash
python -m unittest tests.test_verification -v
```

### 4. Logging and Cost Dashboard

Every request is written to a SQLite audit trail. For privacy, the system stores a prompt hash,
not the raw prompt.

The Streamlit dashboard surfaces:

- Total estimated cost savings
- Cost over time
- Routing distribution by model tier
- Quality-score distribution
- Escalation rate over time

Seed demo data and launch the dashboard:

```bash
python -m scripts.seed_demo_data
streamlit run dashboard/app.py
```

Run logging and analytics tests:

```bash
python -m unittest tests.test_logging_and_stats -v
```

### 5. FastAPI Service

The service exposes an OpenAI-style completion endpoint while keeping model selection internal.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/v1/completions` | Route and complete a request |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/stats` | Return routing and cost metrics |
| `PUT` | `/v1/routing-config` | Update routing configuration live |

Start the API:

```bash
uvicorn src.api.main:app --reload
```

### 6. Portfolio Polish and Reproducibility

The repository includes a full evaluation workflow:

- `scripts/load_test.py` — runs 500–1,000 fresh prompts through the pipeline
- `scripts/generate_report.py` — turns `data/requests.db` into a Markdown report and charts
- `CASE_STUDY.md` — documents architecture decisions, benchmark setup, and trade-offs

Generated artifacts include:

- `data/cost_savings_report.md`
- Four report charts in `data/report_charts/`
