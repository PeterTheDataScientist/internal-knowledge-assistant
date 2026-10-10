# Internal Knowledge Assistant

A retrieval augmented question answering service over Contoso Financial internal
documentation. It answers only from the supplied corpus, returns the source file behind
every claim, scores its own confidence from the evidence rather than the model's tone,
and replies with exactly `I don't have sufficient information in the provided documents.`
whenever the retrieved passages do not support an answer.

Built by **Peter Tinashe Mundowa**.

**Live demo:** [grounded-knowledge-assistant.streamlit.app](https://grounded-knowledge-assistant.streamlit.app), running the offline model with no key needed. The first load can take up to a minute while the free host wakes the app.

The documents in data/ describe Contoso Financial, a fictional financial group. Every policy,
figure, threshold and procedure in them is invented test material.

### What is in the repository

| Item | Where |
| --- | --- |
| Source code | `app/`, runnable with no API key |
| Design for scale | `DESIGN.md` |
| Production notes | `PRODUCTION_NOTES.md` |
| Evaluation results | `EVALUATION.md`, raw output in `eval/results/` |
| Architecture diagram | `docs/architecture.png`, request flow in `docs/request_flow.png` |
| OpenAPI specification | `openapi.yaml` and `openapi.json`, also served at `/openapi.json` |
| Postman collection | `postman_collection.json`, every request carries assertions |
| Features | hybrid search, reranker hook, streaming, conversational memory, feedback capture, structured JSON output, Dockerfile, 94 tests, rate limiting, request validation, metrics endpoint, CI workflow with a safety gate |
| Live demo | [grounded-knowledge-assistant.streamlit.app](https://grounded-knowledge-assistant.streamlit.app), source in `streamlit_app.py` |
| Note on AI assistance | `AI_ASSISTANCE.md` |

---

## Quick start

No API key needed. With no key the service starts on a built-in offline provider, so the
whole pipeline is still exercised.

```bash
cd internal-knowledge-assistant          # run everything from the project root
pip install -r requirements.txt
uvicorn app.main:app --reload
```

That install is small and fast. The service runs on it completely, using a TF-IDF
embedder and a NumPy vector index.

For the configuration the evaluation results were measured on, add the local neural
embedder and FAISS:

```bash
pip install -r requirements-ml.txt
```

That one pulls PyTorch, which is a large download. It is kept in a separate file
deliberately: anyone on a slow connection gets a working service first and better
retrieval second, and a failure in a large wheel cannot leave the core install broken.
The embedding layer sits behind one interface and degrades on its own, so the service
starts either way and `/health` reports which backend is live.

Then open http://localhost:8000/docs, or:

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many leave days does an employee receive?"}' | python -m json.tool
```

To run the real model, put a Groq key in `.env`:

```bash
cp .env.example .env
# set GROQ_API_KEYS=gsk_...    (comma separated, several keys are rotated on rate limits)
uvicorn app.main:app --reload
```

Docker, if you prefer:

```bash
docker compose up --build       # reads GROQ_API_KEYS from your environment
```

Other useful commands (`pip install -r requirements-dev.txt` first for the test tooling):

```bash
make test        # 94 unit and API tests, offline, about 2 seconds
make eval        # evaluation suite against the 48 case golden set
make smoke       # curls a running service with representative questions
python scripts/build_index.py    # ingest and print what was indexed
```

---

## Live demo

Try it at https://grounded-knowledge-assistant.streamlit.app.

`streamlit_app.py` is a one-page demo that answers through the same `RAGPipeline` that
`POST /ask` uses, in process rather than over HTTP. It runs from `requirements.txt` alone:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

With no key it uses the offline extractive provider. To use Groq, set `GROQ_API_KEYS` in
the environment or in Streamlit secrets (`.streamlit/secrets.toml`, which is never
committed). On Streamlit Community Cloud, point a new app at this repository with
`streamlit_app.py` as the entry file and add the key under the app's secrets.

---

## What it does

### `POST /ask`

```json
{ "question": "How many leave days does an employee receive?" }
```

```json
{
  "answer": "Permanent employees are entitled to 22 working days of annual leave per calendar year. Annual leave accrues monthly and a maximum of 10 unused days may be carried over into the following year.",
  "sources": ["hr-leave-policy.md"],
  "confidence": "High",
  "answered_from_context": true,
  "refusal_reason": null,
  "citations": [
    {
      "source": "hr-leave-policy.md",
      "chunk_id": "hr-leave-policy.md#0",
      "section": "HR Policy - Leave",
      "score": 0.62,
      "excerpt": "Permanent employees are entitled to 22 working days of annual (vacation) leave per calendar year..."
    }
  ],
  "safety": { "input_flagged": false, "context_sanitised": true, "triggered_rules": ["context_instruction_stripped"], "injection_score": 0.0003 },
  "request_id": "3f9c1a04b7e25d18",
  "latency_ms": 1364,
  "model": "llama-3.3-70b-versatile"
}
```

`citations` carries the passage each claim came from, so a reader can verify the answer
without opening the corpus. `safety` reports which controls fired.

### Every endpoint

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/ask` | Answer a question with citations, confidence and a refusal reason |
| POST | `/ask/stream` | Same answer as server-sent events, released only after it passes every check |
| GET | `/health` | Liveness and readiness, per component, with index and model detail |
| GET | `/metrics` | Latency percentiles, refusal rate, confidence mix, injection blocks, feedback |
| GET | `/sources` | Indexed documents with their domain and trust tier |
| POST | `/feedback` | Record whether an answer was useful, keyed on `request_id` |

Full contract in `openapi.yaml`. `postman_collection.json` covers the same ground with
assertions attached, so Run Collection is a working smoke test.

### Status codes

| Code | When |
| --- | --- |
| 200 | Answered, or correctly refused. A refusal is a correct outcome, not a client error, so it is not a 4xx. Read `answered_from_context`. |
| 401 | `REQUIRE_API_KEY=true` and the `X-API-Key` header is missing or wrong |
| 404, 405 | Unknown route, wrong method |
| 422 | Body failed validation. The response names the offending fields |
| 429 | Per-client rate limit exceeded, with `Retry-After` |
| 503 | Index not built, or the model backend is unavailable |
| 504 | The model backend timed out |

Every error body has the same shape:

```json
{ "error": { "code": "validation_error", "message": "The request body failed validation.", "request_id": "...", "fields": [ ... ] } }
```

---

## How it works

```
question
  -> input guard        pattern rules plus Llama Prompt Guard 2, rejects injection and
                        requests for customer data before any retrieval cost is spent
  -> question rewrite   resolves follow-ups against conversation history, when a session is used
  -> hybrid retrieval   dense embeddings plus BM25, fused with reciprocal rank fusion,
                        then MMR selection for diversity
  -> relevance gate     refuses before calling the model when nothing scores above threshold
  -> context guard      strips imperative lines from documents that are not policy owned
  -> generation         strict JSON, temperature 0
  -> parse and repair   one bounded retry when the JSON is malformed
  -> grounding checks   citation validity, every figure traced to cited text, lexical support,
                        with one bounded regeneration when a figure is ungrounded
  -> output guard       account number, credential and system prompt leak patterns
  -> response
```

Diagrams: `docs/architecture.png` for the production deployment, `docs/request_flow.png`
for a single request including the refusal branches.

### Document processing

`app/ingestion/loader.py` reads the corpus and assigns each document a **trust tier**.
The corpus contains a shared notice board whose entries are written by staff rather than
by a policy owner, and one of those entries is an injection attempt. The tier is decided
by what the document says about itself, not by a hard-coded filename, so a new notice
board added tomorrow is classified the same way. The corpus notice, which describes the
corpus rather than any policy, is loaded for provenance and kept out of the index.

`app/ingestion/chunker.py` splits on markdown headings first and only falls back to a
sentence-aware sliding window when a section exceeds the target size. That keeps a chunk
to roughly one topic, which is what makes the citations useful: a chunk maps to a heading
a human can find in the file. Bullet lists stay attached to the sentence that introduces
them, because several policy figures in this corpus live in a bullet under a lead-in line.
The document title and section heading are prepended to the indexed text, so a question
phrased in the words of a heading still matches a chunk whose body never repeats them.

17 documents produce 32 chunks.

### Embeddings and indexing

`sentence-transformers/all-MiniLM-L6-v2` by default, 384 dimensions, running locally so
no document content leaves the process. If the weights cannot be loaded, for example on
an air-gapped build agent, the service falls back to a TF-IDF plus truncated SVD embedder
fitted on the corpus. Both sit behind one interface, so swapping in Azure OpenAI
embeddings is a new subclass and one branch in a factory.

Vectors go into a FAISS flat inner-product index, which is exact and takes microseconds
at this size, with a NumPy backend when FAISS is not installed. There is no reason to
approximate 32 vectors; the interface is what matters, because at 500,000 documents the
same call is served by a managed vector store.

### Retrieval

Dense search finds paraphrases. BM25 finds the exact tokens employees actually type:
`US$20,000`, `MLRO`, `OFAC`, `300 months`. Neither is sufficient alone on this corpus, so
both run and their rankings are fused with reciprocal rank fusion.

BM25 is written out in `app/indexing/lexical_index.py` rather than pulled from a library.
It is thirty lines, it removes a dependency, and the tokeniser needed a specific
behaviour: numbers are indexed both as written and stripped of separators, so `20,000`,
`20000` and `us$20,000` all reach the same chunk.

A separate calibrated score is computed for thresholding and confidence. Rank position
tells you nothing about whether the top result is relevant, and BM25 scores are unbounded,
so they are squashed with a saturating transform rather than min-max normalised. Min-max
would put the best candidate at 1.0 on every query, including queries where nothing in
the corpus is relevant, which is exactly the case the relevance gate exists to catch.

### Answer generation

The model returns JSON, not prose:

```json
{ "sufficient": true, "answer": "...", "citations": [1, 3] }
```

Free text answers force you to parse citations out of a sentence, and that parsing is
where citation accuracy quietly dies. A JSON field holding context block numbers is
verified in code against the set that was actually retrieved.

`app/rag/grounding.py` then runs three checks: every citation index must point at a
retrieved block, every figure in the answer must appear in the cited text, and the share
of the answer's content words traceable to the cited text must clear a threshold.
Confidence is derived from those signals plus retrieval strength, so `High` means the
evidence is strong rather than that the model sounded certain.

Every refusal returns the same sentence. The operational reason goes in a separate
`refusal_reason` field, so operators can see which control fired while a caller probing
the system cannot. The reasons are `no_relevant_context`,
`model_reported_insufficient_context`, `ungrounded_figures`, `low_groundedness`,
`untrusted_sources_only`, `missing_citations`, `invalid_model_output`, `output_blocked`,
`prompt_injection` and `confidential_request`.

---

## Safety

Three stages, because the threat is not only a malicious employee. It is also an ordinary
employee whose question happens to retrieve a document somebody else edited.

**Input.** Deterministic pattern rules for injection and for requests aimed at customer
data or credentials, plus Meta's Llama Prompt Guard 2 running on Groq as a classifier at a
0.8 threshold. The rules are cheap, testable, and unlike a model they cannot be talked out
of their behaviour; the classifier adds recall on top. Zero-width and bidirectional control
characters are stripped before matching.

Over-refusal is treated as a failure too. "What is the minimum password length" is a
question about the Information Security Policy, not a request for anybody's password, and
a guard that blocks the word "password" makes that policy unsearchable. `POLICY_ATTRIBUTE_RE`
in `app/safety/patterns.py` separates the two, and seven such cases are in the test suite.

**Context.** Retrieved passages are data, never instruction. Blocks are numbered and
labelled with their trust tier in the prompt, and imperative lines inside non-authoritative
documents are replaced with `[instruction removed by content filter]` before the prompt is
built. The copy is made on the retrieved chunk, so the index itself is never mutated.

**Output.** Account-number and card-number shapes, credential disclosure patterns and
system-prompt echoes are blocked. An answer whose citations all point at untrusted content
is refused rather than returned, so a notice board claim can never be stated as policy.

Worked example, the injection planted in `data/staff-notices.md`:

```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"According to the staff notice board, how long are loan refunds valid for?"}'
# -> "I don't have sufficient information in the provided documents."
#    refusal_reason: model_reported_insufficient_context
#    the string "365" never appears in any answer
```

---

## Evaluation

`eval/questions.jsonl` holds 48 labelled cases across five categories: 20 single-document
factual, 9 multi-document, 8 unanswerable, 6 prompt injection and 5 confidential-data
requests. `eval/run_eval.py` measures correctness, groundedness, citation precision and
recall, refusal accuracy, over-refusal rate, safety violations and latency percentiles.

```bash
python eval/run_eval.py                              # offline provider, deterministic
python eval/run_eval.py --judge --delay 5            # adds a model judge for correctness
python eval/run_eval.py --api-url http://localhost:8000
```

Results, methodology and the failure analysis are in `EVALUATION.md`, with the raw output
in `eval/results/`.

---

## If something does not run

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Could not open requirements file: requirements-dev.txt` | The shell is not in the project root | `cd` into the folder that contains `app/` and `README.md`, then retry. On Windows the extracted path is usually `%USERPROFILE%\Downloads\...\internal-knowledge-assistant` |
| `ModuleNotFoundError: No module named 'app'` from uvicorn | Same cause. `app.main:app` is resolved relative to the working directory | Same fix |
| `pip install` stalls or fails on `torch` | `requirements-ml.txt` pulls PyTorch, roughly 2 GB | Install `requirements.txt` only. The service runs on the TF-IDF embedder and says so in `/health` |
| `/health` reports `"llm_provider": "mock"` | No `GROQ_API_KEYS` in the environment or `.env` | Expected. Add a key to use a real model, or carry on: every layer above the provider behaves identically |
| Port 8000 already in use | Something else is bound | `uvicorn app.main:app --port 8001` |

Python 3.11 or newer. Verified on 3.11 and 3.13.

---

## Configuration

Everything is environment driven through `app/config.py`. See `.env.example`. The settings
worth knowing:

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEYS` | empty | Comma separated. Rotated on rate limits, and a key that exhausts its daily allowance is parked rather than retried |
| `LLM_PROVIDER` | `auto` | `auto` uses Groq when a key exists and the offline provider otherwise |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary generation model |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Used when the primary model has no quota left on any key |
| `GUARD_MODEL` | `meta-llama/llama-prompt-guard-2-86m` | Injection classifier |
| `EMBEDDING_BACKEND` | `auto` | `auto`, `sentence_transformer` or `tfidf` |
| `RETRIEVAL_TOP_K` | `6` | Chunks passed to the model |
| `MIN_RELEVANCE_SCORE` | `0.28` | Below this the service refuses without calling the model |
| `USE_RERANKER` | `false` | Cross encoder stage, off because it adds latency for no gain at this corpus size |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per client, per process |
| `REQUIRE_API_KEY` | `false` | Set with `API_KEYS` to require `X-API-Key` |

---

## Project layout

```
app/
  config.py                 environment driven settings
  schemas.py                request and response contracts
  errors.py                 domain exceptions and JSON error handlers
  main.py                   application factory, middleware, lifespan
  api/routes.py             endpoints
  ingestion/                corpus loading with trust tiering, heading aware chunking
  embeddings/               provider interface, sentence-transformer and TF-IDF backends
  indexing/                 FAISS vector index, BM25 lexical index
  retrieval/                hybrid retrieval, RRF fusion, MMR selection, optional reranker
  llm/                      provider interface, Groq client, offline provider, factory
  rag/                      knowledge base assembly, prompts, grounding, memory, pipeline
  safety/                   pattern rules and the three guard stages
  observability/            structured logging, metrics, middleware
data/                       the sample corpus (Contoso Financial, fictional)
eval/                       golden set, harness, results
tests/                      94 tests
docs/                       architecture and request flow diagrams
scripts/                    index build, smoke test
streamlit_app.py            live demo, same pipeline as POST /ask
```

---

## What I would do next

What I would build next, in priority order:

1. **Grow the golden set to about 300 cases** and split it into a regression set that
   gates merges and a held-out set that does not, so thresholds cannot be tuned against
   the gate.
2. **Turn the reranker on and measure it.** The hook is written and disabled. On 32 chunks
   it costs latency for no measurable gain, but the claim in `DESIGN.md` that it is the
   highest value change at scale is currently an argument, not a measurement.
3. **Replace the in-process conversation memory and rate limiter with Redis.** Both are
   correct for one node and wrong the moment there are two.
4. **Answer-level caching keyed on question hash, corpus version and ACL group set.**
   Internal knowledge traffic is FAQ shaped; this is the cheapest large cost win available.
5. **Document-level authorisation.** Trust tiering is implemented, access control is not.
   The retrieval filter is the right place for it and the metadata already flows through.
6. **Sampled LLM-judge groundedness on production traffic**, feeding every thumbs-down
   into the golden set.

---

## Notes on AI assistance

See `AI_ASSISTANCE.md`. In short: an assistant was used for scaffolding, for drafting
documentation prose, and for generating the first pass of the evaluation questions, all of
which I then verified against the corpus and edited. The architecture, the retrieval
design, the guard model, the grounding checks and every debugging decision are mine, and I
can walk through any line in this repository.
