# Case Study: LLM Cost Autopilot

**I built a routing layer that cut LLM API costs by 52.6% compared to sending
every request to GPT-4o, while keeping 86.9% average agreement with GPT-4o's
own answers on the requests it double-checked itself — across a 400-request
test run.** (Numbers below regenerate from real data any time via
`python -m scripts.load_test` + `python -m scripts.generate_report` - see
`data/cost_savings_report.md` for the current run.)

## The problem

Every team running LLM features at scale is paying flagship-model prices for
requests a much cheaper model could have handled just as well - extracting a
field from text, classifying sentiment, reformatting a name. That's not a
model quality problem, it's a routing problem: nothing is deciding, per
request, whether it actually needs GPT-4o-level reasoning.

## The architecture

Four pieces, built and delivered in phases:

1. **A unified model interface** (`src/client.py`, `src/providers/`) - one
   `send_request(prompt, model_config)` call that works identically across
   OpenAI, Anthropic, and a local Ollama model, always returning the same
   `Response` shape (text, tokens, cost, latency). Everything downstream is
   provider-agnostic.

2. **A complexity classifier** (`src/classifier/`) - a random forest over 9
   heuristic features (word count, reasoning-keyword hits, constraint
   markers, output-format markers, etc.), trained on a 215-prompt labeled
   dataset, hitting **90.7% held-out accuracy**. It sorts each request into
   simple / moderate / complex, and `config/routing.yaml` maps each tier to a
   model - editable with no redeploy.

3. **An async quality verification loop** (`src/verification/`) - this is
   the part that makes routing to a cheap model safe instead of reckless.
   After a cheap-tier response goes out, a background job re-sends the same
   prompt to the top-tier model and scores agreement with a method matched to
   the task: exact label match for classification, word-overlap for
   extraction, LLM-as-judge for summarization/open-ended requests. A
   meaningful disagreement gets logged as an escalation *and* fed back into
   the training set - the next weekly retrain (`src/verification/
   retrain_weekly.py`) learns from every routing mistake the system caught on
   itself.

4. **Logging, an API, and a dashboard** (`src/logging_db.py`,
   `src/api/`, `dashboard/app.py`) - every request lands a row in SQLite
   (prompt hash only, never the prompt itself); a FastAPI service exposes
   `/v1/completions` (the router picks the model, the caller doesn't) plus
   `/v1/stats` and a live-editable `/v1/routing-config`; a Streamlit
   dashboard turns that table into the cost/quality picture a stakeholder
   actually wants to see.

## The feedback loop, concretely

This is the part that turns "a router" into "a system that gets smarter":

```
request -> classify -> route to cheap model -> respond
                              |
                              v  (background, non-blocking)
                    verify against top-tier model
                              |
                    disagreement significant? --no--> done
                              |
                             yes
                              v
              log escalation + relabel this prompt "complex"
                              |
                              v
        weekly: retrain classifier on base data + every relabeled prompt
```

Every routing mistake the system catches on itself becomes training data. The
classifier doesn't just route - it corrects its own blind spots over time
without anyone manually re-labeling anything.

## Engineering trade-offs I made, and why

A portfolio project is a chance to show judgment, not just guide-following -
a few decisions worth calling out:

- **`ThreadPoolExecutor`, not Celery/Redis, for the async verification
  queue.** A solo project doesn't need a broker to get "non-blocking
  background job" - a thread pool inside the same process does the job with
  zero extra infrastructure. Documented in `src/verification/queue.py` as an
  explicit choice, with the swap-out point noted for when it'd stop being
  enough (multiple processes/machines).
- **Task-specific verification scoring, not one-size-fits-all LLM-as-judge.**
  Classification has a ground-truth label - asking a model to "judge" it
  would be slower, costlier, and less reliable than just comparing strings.
  LLM-as-judge is reserved for the requests that actually need judgment
  (summarization, open-ended reasoning).
- **A templated, reviewable dataset generator instead of 215 hand-typed
  prompts.** `data/generate_dataset.py` authors ~10 templates per tier by
  hand (the label is a deliberate per-template decision) and crosses them
  with varied entities for volume - documented as needing a human read-
  through before training, not presented as free of that step.
- **The docker-compose `worker` service handles weekly retraining, not
  per-request verification.** Per-request verification already runs
  async in-process; what genuinely benefits from its own container is the
  independent, scheduled retrain job, which is what `worker` actually does.
  Calling it a "verification worker" would have been guide-compliance
  theater over an honest architecture description.

## Try it yourself

```bash
cp .env.example .env              # add your OpenAI/Anthropic keys
python -m data.generate_dataset
python -m src.classifier.train
python -m scripts.seed_demo_data  # or scripts.load_test for real traffic
streamlit run dashboard/app.py
```

Full setup, phase-by-phase build notes, and the complete architecture are in
[`README.md`](README.md).
##(llm-cost-autopilot) PS C:\Users\HP\OneDrive\Desktop\llm-cost-autopilot> $apiKey = ((Get-Content .env | Where-Object { $_ -match '^API_KEY=' }) -replace '^API_KEY=', '').Trim().Trim('"')
>> 
>> $apiKey.Length
43
(llm-cost-autopilot) PS C:\Users\HP\OneDrive\Desktop\llm-cost-autopilot> $headers = @{
>>     Authorization = "Bearer $apiKey"
>>     "Content-Type" = "application/json"
>> }
>> 
>> $body = @{
>>     messages = @(
>>         @{
>>             role = "user"
>>             content = "Explain how a circuit breaker improves reliability in an LLM multi-provider system."
>>         }
>>     )
>> } | ConvertTo-Json -Depth 10
>> 
>> $r = Invoke-RestMethod -Uri "http://localhost:8000/v1/completions" -Method POST -Headers $headers -Body $body
>> 
>> $r.routing | ConvertTo-Json -Depth 10
{
  "request_id": "3569bf8c-3470-4823-a301-482399491c51",
  "tier": 2,
  "selected_model": "groq-gpt-oss-20b",
  "reasoning": "classified as moderate complexity - routed to a mid-tier model",
  "used_fallback": false,
  "cost_usd": 0.00044655,
  "latency_s": 3.513163899999199,
  "verification": "queued"
}
(llm-cost-autopilot) PS C:\Users\HP\OneDrive\Desktop\llm-cost-autopilot>