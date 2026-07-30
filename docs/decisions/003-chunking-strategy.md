# ADR 003: Three chunking strategies, and relevance defined on text spans

Status: accepted. The measured comparison is in [../evaluation.md](../evaluation.md) and
`eval_results/`.

## Context

Retrieval operates on chunks, not documents, so where a document is cut decides what can be
retrieved. Cut too small and a claim is separated from the subject it is about. Cut too large
and the embedding averages several topics into one vector that is close to nothing.

There is no single right answer, which is why this is an ablation row rather than a constant.

## Decision

Implement three strategies behind one async protocol, selectable by config.

| Strategy | Cuts on | Good at | Bad at |
|---|---|---|---|
| `fixed_token` | Token windows with overlap | Predictable size and cost, no assumptions about the document | Cuts mid-sentence, strands a claim from its subject |
| `recursive_structural` (default) | Headings, then paragraphs, then sentences | Keeps a heading with the text beneath it, so `section_path` is true | Produces sections of very uneven size |
| `semantic` | Cosine distance between consecutive sentences | Finds topic boundaries no heading marks | Embeds every sentence at ingest time, and needs a percentile threshold that depends on document length |

Three things are shared rather than reimplemented per strategy, and that sharing is what makes
the comparison meaningful:

1. **Every strategy produces candidate character ranges**, and one finaliser turns those into
   chunks. The hard token cap is enforced there, so no strategy can emit an oversized chunk by
   forgetting to check, and ids, page numbers, section paths, and inherited metadata are
   attached there, so those cannot drift between strategies.
2. **Ranges are always slices of the original text.** Nothing is reconstructed from token ids,
   which would lose whitespace and subword detail, so `text`, `start_char`, and `end_char`
   always agree and a citation resolves by offset.
3. **Token counts come from the embedder's own tokenizer**, via character offsets rather than
   by decoding ids back to text.

The strategies therefore differ only in where they cut, which is the property an ablation over
chunking needs.

### Structural chunking never merges across a heading

Merging a short section into its neighbour would produce a chunk whose `section_path` is true
of only part of its text. Since `section_path` is what a citation shows the reader, a short
section is left short instead. That is a deliberate trade of chunk-size uniformity for
citation honesty.

### Relevance is defined on text spans, not chunk ids

This is the most consequential decision in the whole evaluation, and it exists because of this
ADR.

Chunk ids are a function of `(doc_id, start_char)`, so they change when the chunking strategy
changes. If the golden set recorded relevance as chunk ids, each chunking row would be scored
against a different ground truth, and the comparison would be meaningless while producing a
table that looks perfectly reasonable.

So the golden set records verbatim text spans, and `eval/metrics.py` defines the predicate
once: a retrieved chunk is relevant if it contains a golden span in full, or shares at least
50 consecutive characters with one. The 50-character rule is what makes a span split across
two chunks count for both halves, which is precisely the case a chunking comparison hinges on.
There is a test that splits a span deliberately and asserts both halves qualify.

## Consequences

**Semantic chunking costs an embedding pass at ingest time.** Every sentence is embedded to
find the boundaries, on top of embedding the resulting chunks. On a large corpus that roughly
doubles ingestion cost.

**The semantic threshold is a percentile, so it is document-relative.** On a very short
document the 95th percentile of a handful of distances is essentially the maximum, which
yields no boundary at all. That is correct behaviour rather than a bug, and the shared token
cap then splits the document if it is too large, but it means the strategy does little on
short inputs.

**Changing strategy invalidates the index.** Chunk ids change, so the store must be rebuilt.
The evaluation harness handles this by caching one index per chunking strategy rather than
re-ingesting per ablation row.

**Fixed-token overlap duplicates text.** With the default 512 token size and 64 overlap,
roughly an eighth of the corpus is stored twice. That is the price of not stranding a claim
that happens to sit on a boundary.

## Alternatives considered

**One strategy only.** Simpler, and most systems do this. Rejected because the spec asks the
project to defend a chunking choice, and defending it without measuring it is exactly the kind
of claim this repository is trying not to make.

**Sentence windows with a sliding stride.** A reasonable middle ground between fixed and
structural. Not implemented, because it occupies roughly the same position as structural
chunking with sentence-level splitting of oversized blocks, which is already the fallback path
inside `RecursiveStructuralChunker`.

**Late chunking (embed the whole document, then pool per chunk).** Genuinely interesting,
because it gives each chunk vector document-level context. Rejected here because it requires a
model with a long context window and changes the ingestion path substantially, which is a poor
trade against a 384-dimensional small model chosen for running on a laptop with no API key.

**Propositional chunking (decompose into standalone facts with a model).** Strong for
precision, and it would make citations very tight. Rejected because it needs a language model
at ingest time for the whole corpus, which conflicts directly with the constraint that
ingestion works with no model server running.
