# medial-retrieval

Multilingual biomedical information retrieval for the ViBioMIR competition.
Vietnamese queries, answers retrieved from Vietnamese, English and Chinese sources.

## The problem

Biomedical knowledge is unevenly distributed across languages.
Vietnamese sources are growing but remain shallow in clinical depth, while English and Chinese hold vastly larger bodies of evidence.
That gap is what this system exists to close: given a Vietnamese query, find the relevant material regardless of the language it was published in.

For each Vietnamese query the system must return both the relevant **documents** and the specific **chunks** of content that carry the answer.

## Scoring

Both levels are scored with macro-averaged **F2**:

```
F2 = 5PR / (4P + R)
```

F2 weights recall twice as heavily as precision.
Tuning should favour recall.

**Document level** - precision and recall over the returned set of document ids.

**Chunk level** - a submitted chunk counts as correct when, against at least one reference chunk of the same query and same `doc_id`:

```
LCS_tokens(submitted, reference) / len(reference_tokens) >= 0.40
```

Both sides are normalised first: NFKC, HTML entity decode, lowercase, whitespace collapse, trim.

Note the denominator is the *reference* length.
A submitted chunk that fully contains the reference still scores 1.0, so over-long chunks are not punished by this formula directly - only indirectly, through chunk-level precision (the count of chunks submitted).
This is why chunk content must be preserved verbatim through the pipeline: any dropped or rewritten token lowers the LCS ratio against ground truth.

## Data

The official corpus is the HuggingFace dataset `AIGuruTinix/ViBioMIR`, loaded through `src/ingestion/utils.py`:

| Config | Schema | Rows |
|---|---|---|
| `corpus` | `id: int64`, `url: string` | 1,220,598 |
| `query` | `id: int64`, `query: string` | 1,200 |

The `corpus` config is the `links_corpus` named in the competition rules.
Its `id` is the official document identifier used in submissions.

Crawled documents are stored at `data/raw_documents/{domain}/{doc_id}.json`.
**The filename `doc_id` is the official corpus `id`** - verified against the corpus URL on a 60-document sample, 60/60 exact matches.
No id mapping table is needed when building a submission.

### Crawl coverage

| Language | Crawled | In corpus | Coverage |
|---|---|---|---|
| vi | 303,683 | 740,598 | 41.0% |
| en | 239,574 | 240,000 | 99.8% |
| zh | 9,790 | 240,000 | 4.1% |
| **Total** | **553,047** | **1,220,598** | **45.3%** |

The Chinese branch is thin, so low zh recall is expected and is a data-coverage limit rather than a retrieval defect.
PubMed (the English source) is effectively complete, but the crawler IP is currently blocked by NCBI.

### Language identification

Each document carries two language fields.
Use `source.language_category` - the curated per-domain ground truth from `src/ingestion/categories.py`.
Do **not** use `source.language`, which comes from per-page extraction and is frequently wrong (Chinese pages labelled `en`, Vietnamese pages labelled `en-US`).

## Pipeline

```
query
  -> [seat: router]                            (future, picks branches/strategies per query)
  -> language-aware retrieval                  vi top-50 | en top-30 | zh top-20
  -> fusion (RRF)                              -> top 50
  -> reranker                                  -> top 20 chunks
  -> submission
```

Retrieval runs one filtered search per language rather than a single blended search, so each language contributes a guaranteed share of candidates instead of competing on uncalibrated cosine scores.
Fusion is rank-based (Reciprocal Rank Fusion) for the same reason: scores from three separate searches are not comparable to each other, but ranks are.

The router stage is deliberately not implemented yet.
It has a reserved seat in the pipeline so it can be dropped in without restructuring.

A Vietnamese spelling-correction stage was considered and **dropped** from the pipeline plan.
Retrieval runs dense-only for now; `bge-m3` can also emit a lexical sparse vector from the same forward pass, so a sparse slot is reserved in the collection schema and hybrid retrieval stays available to switch on once there is a dev set to measure whether it actually helps.

### Chunking

Per-language hierarchical chunkers live under `src/rag/chunker/`, dispatched by `language_category`.
Each language has its own cleaner, heading promoter, sentence splitter and chunker.

| Language | Target | Max |
|---|---|---|
| vi | 800 chars | 1000 |
| en | 800 chars | 1000 |
| zh | 600 chars | 800 |

Chinese uses smaller thresholds because Han characters carry far more information per character than Latin or Vietnamese script.

Chunk content fidelity is treated as a correctness property, not a nicety, because of the LCS scoring rule above.
It is covered by `tests/test_chunking.py` (66 tests).

## Submission format

```json
[
  {
    "id": 1,
    "relevant_docs": ["584"],
    "relevant_chunks": [
      { "doc_id": "584", "chunk_text": "verbatim text extracted from the source document" }
    ]
  }
]
```

`chunk_text` must be extracted verbatim from the source document, never rewritten or generated.
All 1,200 queries must be present in every submission; the grader picks the public and private subsets itself.
Submit as a ZIP containing exactly one JSON file, no subdirectory.

## Rules that constrain design

- Open-weights models only. No closed models (GPT-4o, Gemini, and similar).
- Models must have been released before 2026-08-01 and be 15B parameters or smaller.
- External data is allowed but must be cited with verifiable provenance.
- 10 submissions per day; only 5 total in the private phase.
- Final results are not official until a working-notes paper describing the method is submitted.

`BAAI/bge-m3` (embeddings) and `BAAI/bge-reranker-v2-m3` (reranking) both satisfy these constraints and are the multilingual defaults here.

## Deadlines

| Date | Milestone |
|---|---|
| 2026-10-01 | Public test released |
| 2026-10-31 | Public test submission deadline |
| 2026-11-01 | Private phase opens |
| 2026-11-04 | System submission deadline |
| 2026-11-11 | Final results |

All deadlines are 23:59 Vietnam time (UTC+07:00).

## Layout

```
src/ingestion/      crawling, extraction, per-domain language mapping
src/rag/chunker/    per-language hierarchical chunkers (vi/zh/en) + dispatcher
src/rag/            vector store, ingestion into Qdrant, retrieval pipeline
data/raw_documents/ crawled corpus, {domain}/{doc_id}.json
data/testset/       curated per-language domain samples for inspection
tmp/test_chunking/  before/after chunking dumps for manual review
tests/              unit tests
```

## Analyze spelling corrections

Run the spelling corrector over the query dataset's `train` split:

```powershell
python -m scripts.analyze_spelling_corrections
```

The script writes `changed_queries.txt` (original, corrected, and changed
words) and `word_map.txt` (old-to-corrected word mappings) under
`data/spelling_correction_analysis`. To test it on a small sample, pass
`--limit 20`; use `--output-dir` to choose another output directory.
