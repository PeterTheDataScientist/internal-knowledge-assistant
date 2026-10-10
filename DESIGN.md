# DESIGN: scaling the Internal Knowledge Assistant

Target: 50 requests per second, 500,000 documents, daily updates. The prototype already does
hybrid dense plus BM25 retrieval with RRF fusion, a three stage guard chain, citation and
numeric grounding validation, and JSON logs holding only a hash of the question. This covers
what changes at 30,000 times the corpus size. Diagrams: `docs/architecture.png` and
`docs/request_flow.png`.

## 1. Architecture

Components and flow are in `docs/architecture.png`. The AI service on AKS runs the stages
already in `app/rag/pipeline.py` against the vector store, document repository, cache, model
endpoint and Key Vault.

Sizing by Little's law: 50 rps at a 1.5 s mean is 75 requests in flight, almost all of it
waiting on the model, so the orchestrator is IO bound. Async workers hold about 25 in flight
per pod, so three pods carry the load. Run eight across three zones and autoscale 8 to 24 on
an in-flight gauge, not on CPU, which stays flat while the queue grows. The reranker is
compute heavy and gets its own deployment: 2,500 pairs per second, so two GPU pods or about
ten ONNX INT8 cores.

Azure South Africa North as primary and South Africa West as secondary, because there is no
Azure region in Zimbabwe. If the RBZ or POTRAZ, the authority under the Cyber and Data
Protection Act [Chapter 12:07], refuses offshore processing, the same containers run on
Kubernetes in the Contoso Financial data centre with a self-hosted model.

## 2. Retrieval and re-ranking

Keep hybrid: dense finds paraphrases, BM25 finds the exact tokens employees type, such as
`US$20,000`, `MLRO` and `OFAC`. RRF stays because it fuses rankings without needing the two
score distributions to be comparable. Two changes: widen the pool from 20 to 50 per channel
and enable the cross encoder that `CrossEncoderReranker` already wraps, cutting 50 candidates
to the 8 that reach the prompt; and filter on `domain`, `doc_type`, `effective_date` and the
employee's ACL groups inside both queries. Post-filtering shrinks top-k to zero for restricted
users and leaks document existence through result counts.

| Quantity | Value |
| --- | --- |
| Chunks at 4 to 8 per document | 2M to 4M vectors |
| 768 dimensions, float32 | 3,072 bytes each |
| Raw index | 6 GB to 12 GB, plus about 10% for the HNSW graph at M=32 |
| int8 scalar quantisation | 1.5 GB to 3 GB, costing about one point of recall@10 |

So HNSW with scalar quantisation on Azure AI Search or self-hosted Qdrant, memory resident on
one node. Not product quantisation: it fits under 1 GB, but the recall it costs turns a missed
clause into a wrong answer.

## 3. Chunking

The heading-aware chunker assumes structure that disappears once the corpus is PDFs, DOCX and
scans. Four changes: layout-aware parsing with Azure Document Intelligence to recover heading
hierarchy and page numbers, with an OCR confidence floor below which a document is quarantined
rather than indexed; tables extracted whole, because splitting a fees table mid-row makes a
correct retrieval return a wrong figure; parent document retrieval, embedding 300 token
children for precision and returning the parent section for context; and versioned chunk ids
`{doc_id}:{version}:{position}`, so superseded clauses stay distinguishable and old citations
still resolve.

## 4. Index refresh and daily updates

Change data capture from SharePoint or Blob Storage through Event Grid into a queue. Per
document, recompute the SHA-256 checksum `loader.py` already stores; unchanged means no work.
Changed means re-chunk, compare per-chunk hashes, embed and upsert only the chunks that moved.
Deletions write tombstones, so a withdrawn policy stops being retrievable in the same cycle,
ahead of compaction.

At 0.5% daily churn that is 2,500 documents, about 15,000 chunks, under a minute of GPU
embedding. A full rebuild of 3M chunks at roughly 1,500 chunks per second per GPU takes 33
minutes on one GPU and 9 on four. Rebuilds run monthly, or on an embedding model change, into
a shadow index, then an atomic alias swap gated on the golden set passing against that shadow.
Rollback is swapping the alias back; the previous index is kept 7 days.

## 5. Latency

| Stage | Now | Target | Change |
| --- | --- | --- | --- |
| Input guard | 250 ms | 15 ms | Prompt Guard 86M as local ONNX, one less network hop |
| Question rewrite | 600 ms | 0 to 200 ms | Skip with no history, small model when there is |
| Dense plus lexical | 120 ms | 60 ms | Run concurrently, take the max rather than the sum |
| Rerank 50 to 8 | absent | 90 ms | Batched ONNX INT8 cross encoder, or GPU |
| Generation | 3,500 ms | 700 to 1,600 ms | Prompt cut 2,000 to 1,200 tokens, small model fast path with escalation |
| Guards and grounding | 40 ms | 20 ms | Already pure Python |

p95 falls from about 6 s to 1.9 s, p50 near 0.9 s. Keep-alive pools remove a TLS handshake per
call. What I would not do is stream raw model tokens: `stream_answer` validates first, because
publishing an unverified figure to a bank employee is worse than 300 ms of waiting.

## 6. Caching

| Layer | Key | Invalidated by |
| --- | --- | --- |
| Exact response | Hash of normalised question, corpus version, ACL group set | Corpus version bump on index swap |
| Semantic | Query embedding, cosine at or above 0.97 | Corpus version bump, plus a 1 hour TTL |
| Embedding | Chunk content hash | Never, the hash is the identity |
| Provider prompt cache | Static `SYSTEM_PROMPT` prefix | Prompt version change |

The ACL group set has to be part of the response key, or an answer built from a restricted
document reaches someone who cannot see it. The semantic layer is the risky one: at 0.97,
"mortgage approval limit" and "personal loan approval limit" can collide, and those are
different figures in different documents. It ships behind a flag with the false hit rate
measured on the golden set.

## 7. Cost

Assumption: 50 rps is peak, not the 24 hour mean. A 5,000 employee bank over a 10 hour day at
a peak to mean ratio of 4 gives 12.5 rps mean and 9.9M requests a month. Budget per request:
2,000 prompt and 200 completion tokens.

| Path | Share | Model class | Cost per request |
| --- | --- | --- | --- |
| Easy, single document lookup | 70% | 8B, US$0.10 / US$0.30 per 1M | US$0.00026 |
| Hard, multi document or numeric | 30% | 70B, US$0.60 / US$0.80 per 1M | US$0.00136 |
| Blended | | | US$0.00059 |
| After 35% cache hits | | | US$0.00038 |

That is US$3,800 a month of inference plus roughly US$4,000 for AKS, the reranker and the
vector store, so about US$8,000 a month. At a true 24/7 50 rps it is 130M requests and about
US$49,000 a month of inference, which is the ceiling figure.

## 8. Model selection

Criteria in order: grounded instruction following, meaning does it set `sufficient: false`
when it should; JSON reliability under `response_format`; p95 latency; data residency; cost.
A small Azure OpenAI model on the easy path with escalation to the frontier model keeps data
inside the Azure tenancy under Microsoft processing terms. An open weight 70B on a serverless
endpoint is cheaper but sits in a third party tenancy. Self-hosting the same weights on two
A100s costs roughly US$6,000 a month, worse than the cloud figure, and is the only option that
survives a ruling that no data leaves Zimbabwe. The embedder is self-hosted either way:
embedding 3M chunks through a hosted API ships the entire policy corpus offshore.

## 9. Monitoring and logging

KPIs, most already emitted by `app/observability/metrics.py`: latency p50, p95 and p99;
refusal rate by reason; retrieval hit rate on the golden set; mean groundedness; cost per
request; cache hit rate; injection block rate; feedback ratio. Logs stay one JSON object per
line with `request_id` propagated from the gateway. Question text is never written, only
`hash_text(question)`: under the Cyber and Data Protection Act a question such as "am I still
on a final written warning" is personal data, and a hash still groups repeats. OpenTelemetry
spans per stage feed Application Insights so a slow request is attributed rather than guessed
at. Page on: p95 above 2.5 s for 10 minutes; refusal rate more than 10 points off the 7 day
baseline in either direction, since a fall means the grounding checks stopped firing;
`invalid_model_output` above 1%; index freshness lag above 26 hours; error rate above 2%.

## 10. Security

Entra ID OIDC at the gateway, with the employee's group claims passed into the retrieval
filter so document-level authorisation is enforced inside the search at query time. Mutual TLS
between mesh services. Secrets in Key Vault through workload identity, so no key reaches an
image or an environment file, rotated every 90 days. Encryption at rest with customer managed
keys, TLS 1.3 in transit. Injection controls are the three stages in `app/safety/guards.py`.
Every question hash, source list, refusal reason and answer hash goes to an append-only audit
store with 7 year retention for compliance review. Tenant isolation by index namespace per
subsidiary. Egress restricted to the model endpoint and Microsoft 365 through NSG rules and
Private Link.

## 11. Disaster recovery

The vector index is a derived artefact and the document repository is the system of record,
which sets the targets: RPO 24 hours for the index, since it rebuilds from source in 33
minutes; RPO 15 minutes for the repository, covered by geo-redundant storage; RTO 1 hour for
the service. Active across three zones with a warm standby in the secondary region at one pod,
failed over by a Front Door origin switch. Restore drills are quarterly and timed, because an
untested restore is an assumption.

## 12. CI/CD and evaluation gates

Every pull request runs lint, unit tests, then the golden set, 48 cases today and 300 as the
target, against a pinned model. Gates that block a merge, each set just under the current measurement rather than
aspirationally: zero safety violations, refusal accuracy at or above 0.90, citation
precision at or above 0.80, groundedness at or above 0.75, correctness no more than 2
points below the previous baseline, harness p95 at or below 2,000 ms. Merge deploys to staging, then a 5% canary
for 30 minutes with automatic rollback on a p95, error rate or refusal rate breach.
`.github/workflows/ci.yml` implements the first half of this today.

## 13. Model upgrades

Model version and prompt version are pinned together, because a prompt tuned for one model is
not tuned for the next. A candidate goes through offline replay of the golden set, then shadow
traffic where it answers in parallel and is scored but never returned, then side by side
scoring on groundedness, refusal accuracy and cost, then a rollout at 5%, 25% and 100% with
the KPI dashboard as the gate. Rollback is a config flag rather than a redeploy, and the
previous version stays available for 30 days.
