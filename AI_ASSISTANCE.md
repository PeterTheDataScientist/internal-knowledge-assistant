# Notes on AI assistance

This note records where AI coding assistants helped and where they did not. I
can explain every line in this repository, including the parts an assistant drafted.

## Where an assistant helped

| Area | What it did | What I did |
| --- | --- | --- |
| Project scaffolding | First pass of the FastAPI application factory, Pydantic schemas, `Dockerfile` and `Makefile` | Set the module boundaries and the provider interfaces before any code was written, then reworked the error handling so refusals return 200 and only genuine faults return 4xx or 5xx |
| Boilerplate | Docstrings, type annotations, the Postman collection JSON | Wrote or rewrote every docstring that explains a decision rather than a signature |
| Evaluation questions | Generated the first pass of `eval/questions.jsonl` from the corpus | Checked every case against the source documents, corrected the required substrings, and added the deliberate traps: paternity leave next to maternity leave, the two different Credit Committee thresholds, and the policy question that resembles a data request |
| Documentation prose | Drafts of `DESIGN.md` and `PRODUCTION_NOTES.md` from an outline I specified, including the topic list and the arithmetic to show | Set every position and trade-off, supplied the numbers to work from, verified the code references, and cut the drafts to length |
| Diagrams | Mermaid source for `docs/architecture.mmd` and `docs/request_flow.mmd` | Specified every component and the request flow including the refusal branches, then restructured the first render because the edge routing was unreadable |

## Where it did not

The parts that decide whether this system works are mine:

- The three-stage guard model, and the decision that context sanitisation belongs at
  ingestion and retrieval rather than in the prompt.
- Trust tiering by what a document says about itself rather than by filename.
- Returning structured JSON with citation indices and validating those indices in code,
  instead of asking the model for prose and parsing citations out of it.
- The numeric grounding check, which is the control that catches the failure mode that
  matters most in a policy assistant: a right-sounding answer with the wrong figure.
- Saturating rather than min-max normalising BM25 scores, so the relevance gate can
  actually fire on a query with no relevant document.
- The two real bugs found in testing and their fixes, both described in `EVALUATION.md`.
  The MMR one in particular took reading the ranking output chunk by chunk to find; no
  assistant surfaced it, and the regression test in `tests/test_retrieval.py` exists
  because of it.
- Every threshold in `app/config.py`, each set from evaluation output rather than guessed.

## Verification

Nothing was accepted because it looked plausible. Every generated component was run:
94 tests pass offline in about two seconds, the 48 case evaluation suite runs end to end
against two model configurations plus a retrieval ablation, and the failure analysis in `EVALUATION.md` reports
before and after numbers from actual runs rather than expected improvements.
