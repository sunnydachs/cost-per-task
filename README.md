# cost-per-task

Measure what a task actually costs to complete with an LLM — attempts, tokens and wall-clock — instead of the price per request.

[English](README.md) | [日本語](README.ja.md)

Token price tells you the price of one request. It does not tell you the price of **one finished piece of work**: the failed attempts, the retries, and the time you spent waiting for the machine to answer. This tool measures that, with a deterministic verifier so the PASS/FAIL decision never depends on a human.

## Why

Most teams already log tokens per call. What nobody logs is the denominator: how many *successful tasks* those tokens produced. That ratio ("cost per successful task") is the number that decides whether a cheap model with retries beats an expensive model that gets it right first try.

The point of this repository is the *procedure*, not a verdict about any model:

1. pick tasks whose output can be judged by code,
2. write the verifier,
3. run each model arm on the same tasks with the same prompt, up to N attempts,
4. record every attempt (tokens, wall-clock, verdict),
5. aggregate: tokens per solved task, retry tax (attempts ÷ solved tasks), machine time.

## Requirements

- Python 3.11+
- An API key for any OpenAI-compatible chat-completions endpoint
- No runtime dependencies (standard library only)

## Usage

```bash
export NVIDIA_API_KEY=...            # or: --api-key-env MY_KEY

# one arm
python cost_per_task.py --model A=vendor/model-one

# two arms, same tasks, same prompts
python cost_per_task.py --model A=vendor/model-one --model B=vendor/model-two

# only some tasks, more attempts, custom endpoint
python cost_per_task.py --model A=vendor/model-one \
    --tasks csv_convert doc_extract \
    --max-attempts 5 \
    --base-url https://my-endpoint.example/v1 \
    --out-dir ./run1
```

Output:

- `results.json` — one record per attempt (tokens, latency, verdict, raw response)
- `report.md` — per-task and per-arm aggregates

The run is **resumable**: re-running the same command skips pairs that already passed and does not repeat recorded attempts. Delete `results.json` to start over.

## The tasks and their verifiers

| task | what it asks for | how PASS is decided |
|---|---|---|
| `csv_convert` | normalize a messy CSV (rename columns, lowercase emails, map countries, drop rows without an email) | header and row set compared against the expected table |
| `py_bugfix` | fix an off-by-one bug in `chunk()` | the returned file is written to disk and a test file is **executed** |
| `json_build` | build an order JSON from prose | keys, values and the computed total compared |
| `doc_extract` | extract 5 fields from a noisy invoice | values compared; the prior-balance line must not leak into the total |

No task is judged by an LLM, and no task is judged by a human. If the verifier is wrong, the measurement is wrong — which is why the verifiers have their own tests (`tests/test_verifiers.py`).

## Reading the numbers

- **tokens per solved task** — total tokens spent on the tasks that ended up solved. This is the headline number.
- **retry tax** — attempts ÷ solved tasks. 1.0 means everything passed first try; 2.0 means every task needed two tries on average.
- **tokens per success incl. failed-task waste** — the same ratio but counting tokens burned on tasks that never succeeded. This is the number that shows up on the bill.
- **total seconds (machine time)** — wall-clock of the whole run. On free tiers this is often the axis that actually binds.

Transport failures (HTTP 429/503, timeouts) are retried *inside* one attempt and are not counted as task attempts; only verifier failures count. They are still recorded in `transport_errors`.

## Limits

- Four tasks, one language, a couple of arms: too small to generalise about any model.
- The cash cost is unknown unless your provider tells you. This tool reports tokens and wall-clock; it never invents a price.
- Retries repeat the same prompt. The verifier's error is not fed back to the model, so this measures naive retries.
- Human review minutes are not measured. No observability product records them either.

## Privacy

The API key is read from an environment variable and never written to disk. Prompts and responses go only to the endpoint you configure. `results.json` contains the raw model responses — do not commit it (it is git-ignored).

## License

MIT — see [LICENSE](LICENSE).
