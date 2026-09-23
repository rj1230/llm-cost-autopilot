# LLM Cost Autopilot

An intelligent routing layer that sits in front of multiple LLM providers,
scores each incoming request's complexity, routes it to the cheapest model
capable of handling it, verifies that decision against the best model in the
background, and escalates + learns from its own mistakes.

**Result on a 400-request test run: 52.6% lower cost than sending everything
to GPT-4o, with 86.9% average agreement on the requests it double-checked
itself.** Full narrative + engineering trade-offs: [`CASE_STUDY.md`](CASE_STUDY.md).
Current numbers regenerate any time via `python -m scripts.generate_report`.

## Status — all 6 phases built

- [x] Phase 1 — Unified Model Interface
- [x] Phase 2 — Complexity Classifier
- [x] Phase 3 — Async Quality Verification Loop
- [x] Phase 4 — Logging + Cost Dashboard
- [x] Phase 5 — FastAPI Service
- [x] Phase 6 — Portfolio Polish (load test, cost report, case study)

## Quick start

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env         # add your OpenAI / Anthropic keys

# optional: for the local model
ollama pull llama3.1 && ollama serve

# build the classifier (or skip - data/classifier.joblib already ships trained)
python -m data.generate_dataset
python -m src.classifier.train

# try it
python -m scripts.run_single_request "Summarize the following: ..." --sync

# run the API
uvicorn src.api.main:app --reload

# run the dashboard (seed synthetic demo data first if you want it populated)
python -m scripts.seed_demo_data
streamlit run dashboard/app.py

# or run everything in Docker
docker compose up
```

## What's built, phase by phase

**Phase 1 — Unified Model Interface** (`src/models/`, `src/providers/`,
`src/client.py`) - one `send_request(prompt, model_config)` call that works
identically across OpenAI, Anthropic, and local Ollama, always returning the
same `Response` shape. Registry ships with real per-token pricing for GPT-4o,
GPT-4o-mini, Claude Sonnet 4.5, Claude Haiku 4.5, and a free local Llama.
Test: `python -m tests.test_providers`.

**Phase 2 — Complexity Classifier** (`src/classifier/`, `data/
generate_dataset.py`, `config/routing.yaml`) - a 215-prompt labeled dataset
(templated, meant to be hand-reviewed before training), 9 heuristic features,
a random forest hitting **90.7% held-out accuracy**, and a YAML tier→model
map editable with no redeploy. Test: `python -m src.classifier.train`,
`python -m scripts.check_routing_offline`.

**Phase 3 — Async Quality Verification Loop** (`src/verification/`) - after
a cheap-tier response goes out, a background thread re-sends the prompt to
the tier-3 model and scores agreement with a task-matched method (exact
label match for classification, word-overlap for extraction, LLM-as-judge
for everything else). A real disagreement gets logged, escalated, and fed
back as a new training example; `src/verification/retrain_weekly.py`
retrains on accumulated feedback. 15 unit tests + a real mocked integration
run: `python -m unittest tests.test_verification -v`.

**Phase 4 — Logging + Cost Dashboard** (`src/logging_db.py`, `src/stats.py`,
`dashboard/app.py`) - every routed request lands a row in SQLite (prompt
*hash* only, never the prompt itself); the Streamlit dashboard shows the
savings-% headline metric, cost per day, routing distribution, quality score
distribution, and escalation rate over time. Test: `python -m unittest
tests.test_logging_and_stats -v`, then `streamlit run dashboard/app.py`.

**Phase 5 — FastAPI Service** (`src/api/`, `Dockerfile`,
`docker-compose.yml`) - `POST /v1/completions` (router picks the model, the
caller doesn't), `GET /v1/models`, `GET /v1/stats`, `PUT /v1/routing-config`
(live-editable, no redeploy). `docker-compose.yml` runs the API, a Streamlit
dashboard, and a `worker` service — see "Why no separate verification
worker?" below for why that container handles weekly retraining rather than
per-request verification.

**Phase 6 — Portfolio Polish** (`scripts/load_test.py`, `scripts/
generate_report.py`, `CASE_STUDY.md`) - sends 500-1,000 fresh prompts through
the full pipeline, then turns `data/requests.db` into
`data/cost_savings_report.md` + 4 chart PNGs in `data/report_charts/`. The
case study write-up is [`CASE_STUDY.md`](CASE_STUDY.md).

## Why no separate verification worker?

The build guide's docker-compose step calls for "a background worker for
async verification." Per-request verification here runs on a
`ThreadPoolExecutor` *inside* the API process - it's already non-blocking and
doesn't need its own container at this scale (a broker like Celery/Redis
would be the honest way to split it out, and a solo project doesn't need that
yet). What genuinely is a separate, independent, scheduled job is the weekly
classifier retrain, so that's what the `worker` service in `docker-compose.yml`
actually runs (`scripts/retrain_worker_loop.py`). More in `CASE_STUDY.md`.

## Testing

```bash
python -m unittest discover -s tests -v   # 17 tests, all offline/mocked
```

None of the tests call a real provider - `tests/test_verification.py` and
`tests/test_logging_and_stats.py` mock `send_request` and use a temp SQLite
DB respectively, so this runs in CI with no API keys.

## Project layout

```
llm-cost-autopilot/
├── src/
│   ├── config.py                  # loads .env
│   ├── client.py                  # send_request(prompt, model_config)
│   ├── routing.py                 # route_request[_with_verification]()
│   ├── logging_db.py              # SQLite audit trail (Phase 4)
│   ├── stats.py                   # dashboard/API query layer (Phase 4)
│   ├── models/
│   │   ├── registry.py            # ModelConfig + MODEL_REGISTRY
│   │   └── response.py            # Response dataclass
│   ├── providers/                 # openai / anthropic / ollama + base ABC
│   ├── classifier/                # features, train, predict (Phase 2)
│   ├── verification/               # task_type, thresholds, scoring,
│   │                               # verifier, queue, feedback, retrain (Phase 3)
│   └── api/
│       ├── main.py                # FastAPI app
│       └── schemas.py             # pydantic request/response models
├── dashboard/
│   └── app.py                     # Streamlit cost dashboard (Phase 4)
├── config/
│   └── routing.yaml               # tier -> model map
├── scripts/
│   ├── check_routing_offline.py   # Phase 2: no-API-call routing sanity check
│   ├── run_single_request.py      # Phase 3: one real request through the pipeline
│   ├── seed_demo_data.py          # Phase 4: synthetic data for an instant dashboard
│   ├── retrain_worker_loop.py     # Phase 5: docker-compose worker entrypoint
│   ├── load_test.py               # Phase 6: 500-1,000 real requests
│   └── generate_report.py         # Phase 6: DB -> report + charts
├── tests/
│   ├── prompts.py, test_providers.py         # Phase 1
│   ├── test_verification.py                  # Phase 3 (15 tests)
│   └── test_logging_and_stats.py              # Phase 4 (2 tests)
├── data/
│   ├── generate_dataset.py        # builds labeled_dataset.csv
│   ├── labeled_dataset.csv        # 215 labeled prompts (generated, ships pre-built)
│   ├── classifier.joblib          # trained model (generated, ships pre-built)
│   ├── requests.db                # SQLite audit trail (generated)
│   ├── cost_savings_report.md     # Phase 6 output (generated)
│   └── report_charts/             # Phase 6 chart PNGs (generated)
├── Dockerfile
├── docker-compose.yml             # api + worker + dashboard services
├── requirements.txt
├── .env.example
├── CASE_STUDY.md                  # the portfolio write-up
└── README.md                      # this file
```

## Known limitations / next steps if you keep going

- The classifier's features are heuristic (word/keyword counts), not a real
  tokenizer or embedding - fine for V1's 80%+ bar, but an embedding-based
  classifier would likely push accuracy higher on genuinely ambiguous prompts.
- LLM-as-judge scoring (summarization/other) costs one extra API call per
  verified request - fine at demo volume, worth watching at real scale.
- No auth on the FastAPI service - add an API key dependency before exposing
  this beyond localhost.
- `data/requests.db` and the JSONL logs grow unbounded - add a retention/
  archival job before running this for real for months at a time.
# llm-cost-autopilot
