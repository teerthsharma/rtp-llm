# Topology KV Compression And Inference Sparsity Design

## Context

PR #1160 adds an opt-in benchmark for topology-derived KV candidate schedules under
`benchmark/`. This follow-up branch starts from the #1160 branch head and is valid
only after #1160 merges. The follow-up PR must be opened as a draft and must state
that it is blocked by #1160 until the benchmark branch lands in `main`.

The benchmark proves that topology-derived candidate token sets can be constructed
and timed in a standalone SDPA screening harness. It does not yet prove serving-path
speedup, model quality, KV memory reduction, or token movement reduction. This
design moves the topology idea into the existing runtime sparse MLA path while
keeping the production default unchanged.

NVIDIA/NeMo-Relay#322 is the systems-design reference for the follow-up: its useful
claim is not only faster wall time, but avoided repeated prefix token movement via
stable topology fingerprints and cache reuse. RTP-LLM should use the same discipline:
measure compressed/cached KV tokens avoided, not only attention kernel speedup.

## Goal

Add a disabled-by-default topology-aware KV compression and inference sparsity path.
The path should compress stable or repetitive KV regions into reusable summaries,
then sparsely select raw and compressed candidates for sparse MLA. The primary
success metric is token efficiency for repeated/agentic workflows: fewer raw KV
tokens moved, read, written, and attended, while preserving quality and controlling
tail latency.

## Non-Goals

- Do not replace FlashMLA kernels in this PR.
- Do not claim end-to-end serving speedup until measured in RTP-LLM runtime.
- Do not make topology sparsity the default path.
- Do not remove the existing learned sparse indexer or dense fallback.
- Do not rely on synthetic-only evidence for PR-ready performance claims.
- Do not treat raw sparse top-k as sufficient; compression and token movement
  counters are part of the design goal.

## Existing Runtime Path

Sparse MLA currently flows through:

1. `rtp_llm/models_py/modules/hybrid/mla_attention.py`
2. `rtp_llm/models_py/modules/hybrid/indexer.py`
3. `rtp_llm/models_py/modules/base/cuda/indexer_op.py`
4. `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py`
5. `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_cp_impl.py`

`MlaAttention` calls `Indexer` when `attn_config.is_sparse` is enabled. `Indexer`
computes request-local `topk_indices`. Sparse MLA then converts those local token
indices through the page table into global cache indices and runs the existing
FlashMLA sparse path. The topology implementation should preserve this contract:
produce request-local token indices with the same shape and dtype expectations as
the existing top-k indexer.

## Topological Model

Topological object: per-request KV block point cloud in indexer key space, plus a
workflow-level stable-prefix/cache-topology fingerprint for repeated agentic
actions.

Construction: each KV cache block is represented by a compact block summary derived
from the indexer key representation. Adjacent blocks form the causal time backbone.
Additional edges or rankings come from centroid drift and query-to-block affinity.
Repeated workflow scaffolds, tool schemas, system policy, and output contracts form
stable cache regions whose topology can be fingerprinted and reused across task
suffixes.

Invariant or summary: stable coverage by sink blocks, newest local blocks,
high-drift witness blocks, and compressed stable regions. High drift acts as a
witness proxy for regions where the key-space trajectory changes enough that a
purely local or uniform selector may miss attention mass. Low-drift, repeatedly
matched regions are compression candidates: they can be represented by summaries
or cache fingerprints instead of repeatedly moving every raw KV token.

Runtime rule: topology first decides whether a KV region is raw, compressed, or
reused, then produces a candidate guarantee over request-local token indices and
compressed region handles. The existing learned indexer remains the scoring
baseline and safety source. Topology reserves part of the budget for structural
candidates, but the higher-value path is reducing raw KV movement before sparse
attention runs.

## Proposed Approach

Use a two-stage topology policy around the existing indexer output.

When the topology path is disabled, `Indexer.forward` returns exactly the current
learned top-k result.

When enabled, the path:

1. Computes or reuses per-request block summaries from the indexer key-space data.
2. Detects stable KV regions using a deterministic topology fingerprint over block
   summaries, scaffold identity, and output-contract identity where available.
3. Emits compressed-region metadata for low-drift stable regions that can be reused
   or summarized.
4. Selects mandatory raw structural blocks:
   - sink blocks for global anchors,
   - newest local blocks for causal locality,
   - high-drift witness blocks for topology coverage.
5. Merges raw structural candidates, compressed-region representatives, and learned
   indexer candidates.
6. Trims raw candidates to the same `indexer_topk` budget and separately reports
   compressed tokens avoided.
7. Returns the same raw-token shape consumed by `SparseMlaImpl` and `SparseMlaCpImpl`
   for the first implementation, while keeping the compressed metadata available
   for validation and follow-up kernel/runtime work.

This avoids changing sparse MLA kernels in the first inference PR and isolates the
experimental variable to candidate selection.

## Configuration

The feature must be opt-in. Acceptable gates are:

- a model or attention config field if the repository already has an established
  sparse-attention configuration surface for experimental selectors, or
- an environment-controlled experiment flag if adding a config field would widen
  the public API too early.

The chosen gate must make default serving behavior byte-for-byte equivalent for
the current sparse indexer path when disabled.

Recommended initial policy names:

- `disabled`: existing behavior.
- `topology_sparse_merge`: learned top-k plus mandatory structural candidates.
- `topology_compress_sparse`: stable-region compression metadata plus sparse raw
  candidate merge.
- `topology_only`: test-only mode for ablations, not a production default.

## Candidate Budget

Budget units:

- raw request-local selected tokens, equal to `attn_config.indexer_topk`;
- compressed KV tokens represented or avoided;
- KV bytes read and written;
- schedule construction time;
- sparse attention execution time.

All comparisons must use the same budget:

- existing learned indexer,
- topology sparse merge,
- topology compress sparse,
- locality-only same budget,
- random same budget where practical in benchmark or test harnesses.

The topology path must preserve causality. A query row must not select future KV
tokens. For decode, each request has a single new query row and candidates must be
within that request's current KV length. For prefill, rows must respect the row's
ragged causal window and any existing `topk_indices_offset` semantics.

For repeated agentic actions, the cache/compression comparison must also report:

- repeated stable KV tokens avoided,
- compressed KV tokens read,
- compressed KV tokens written,
- raw uncached KV tokens still required for suffix-specific work,
- cache/compression hit rate,
- aggregate and tail latency.

## Data Flow

The intended insertion point is in `rtp_llm/models_py/modules/hybrid/indexer.py`.
The policy should live near the hybrid indexer code rather than inside FlashMLA
operators. The FlashMLA sparse implementations should continue to receive
`topk_indices` and remain unaware of how candidates were chosen.

Decode path:

1. Existing indexer computes logits and learned top-k through `_get_topk_paged`.
2. Topology policy fingerprints stable KV block regions and classifies them as raw,
   compressed, or reused.
3. Policy builds request-local structural raw candidates from available request KV
   metadata.
4. Policy merges raw structural candidates and compressed-region representatives
   into the learned top-k row.
5. Sparse MLA converts local raw indices to global slots as it does today.
6. Validation records compressed/reused KV token counters alongside runtime timing.

Prefill path:

1. Existing indexer computes ragged learned top-k through `_get_topk_ragged`.
2. Topology policy respects `ks`, `ke`, `expanded_seq_lens`, and
   `topk_indices_offset`.
3. Policy merges only candidates that are valid for each row's causal span.
4. CP prefill support is deferred unless the same local-index contract can be
   preserved without extra collectives.

Agentic repeated-workflow path:

1. Identify a stable scaffold fingerprint from reusable prefix material when the
   application provides it: system policy, tool schema, workflow scaffold, and
   output contract.
2. Reuse compressed topology state only when the stable fingerprint and output
   contract still match.
3. Reopen learning/compression when scaffold topology, tool contracts, or output
   contract changes.
4. Keep task suffix KV raw or lightly compressed until quality evidence supports
   stronger compression.

## Error Handling

The topology path should fail closed. If topology metadata is missing, malformed,
or unsupported for a mode, the implementation should use the existing learned
indexer result and emit a low-noise debug or warning signal appropriate to the
repository's logging style.

Invalid candidates must be rejected before sparse MLA receives them:

- negative indices except existing sentinel semantics where supported,
- indices outside the request-local KV length,
- future indices for causal rows,
- duplicate candidates after final merge if the downstream kernel expects unique
  selected tokens.

Compressed/reused KV metadata must be invalidated when stable-prefix fingerprints,
tool schemas, output contracts, model/layer identity, dtype, block size, or cache
layout change.

## Research And Falsification Gates

The research hypothesis is:

Topology-aware KV compression plus sparse selection improves agentic inference
efficiency by avoiding repeated raw KV token movement for stable workflow regions,
while preserving global anchors, recent causal context, and high-drift witness
regions that the learned indexer can underselect.

Null models:

- existing learned sparse indexer,
- locality-only same-budget selector,
- random same-budget selector in offline harnesses,
- sparse-only topology selector without KV compression,
- compression-only stable-prefix reuse without topology sparse reranking,
- dense or fastest available non-sparse attention path for runtime comparison.

Falsification gates:

- Accuracy: topology merge must not regress sparse MLA output quality beyond the
  existing learned sparse indexer tolerance on representative long-context cases.
- Token efficiency: repeated raw KV tokens avoided must be reported and must exceed
  the compressed metadata overhead on repeated agentic workflows.
- Speed: topology compression and sparse merge must not add enough schedule
  construction overhead to erase sparse/runtime savings.
- Memory: candidate and compression metadata must remain bounded by request length,
  block count, and stable-prefix count.
- Coverage: results must include at least two long sequence lengths and the target
  head dimensions used by sparse MLA configurations.
- Stability: topology should not pass only on synthetic random tensors; real model
  Q/K/V or end-to-end generation captures are required before non-draft claims.
- Tail latency: p90, p95, and p99 must be reported. A run with better aggregate
  latency but worse tail latency cannot be marked ready without a mitigation or a
  clear workload-specific tradeoff.

## Tests

Focused tests should cover:

- disabled gate returns the exact existing learned top-k result;
- topology merge preserves output shape, dtype, and device;
- compression metadata is disabled by default and does not alter raw top-k shape;
- stable fingerprints match only when scaffold and output contract match;
- fingerprint drift invalidates compressed/reused KV state;
- local newest block is retained under partial budgets;
- sink and witness candidates are included when budget allows;
- future tokens are not selected in prefill rows;
- duplicate candidates are removed or resolved deterministically;
- same-budget learned, topology, locality, and random selectors are comparable in
  benchmark or validation tooling;
- token counters report raw selected tokens, compressed tokens represented, raw KV
  tokens avoided, cache/compression hits, and schedule cost;
- sparse MLA still accepts the topology-produced top-k indices.

CUDA-specific tests should stay manual if they require CUDA 12.9, FlashMLA, or H20
class hardware. CPU-deterministic helper tests should run without GPU when possible.

## Documentation

The draft PR should document:

- it is blocked by #1160 until the benchmark PR merges;
- topology compression and sparse merge are disabled by default;
- microbenchmark speedup from #1160 is not a serving speedup claim;
- token/KV movement reduction is a first-class metric alongside latency;
- schedule construction time must be reported separately from sparse attention
  execution time;
- tail latency must be reported, following the NeMo-Relay #322 caveat that aggregate
  E2E wins can hide p90/p95/p99 regressions;
- real model validation remains required before marking the PR ready for review.

## Draft PR Strategy

Open the PR as a draft from `feat/topology-inference-sparisty`.

Base strategy:

- While #1160 is unmerged, the branch remains stacked on top of
  `codex/topology-kv-candidate-schedule`.
- The PR targets `main` so GitHub shows the full dependency stack.
- The PR body states that it is blocked by #1160 and must be rebased or retargeted
  after #1160 merges.

The title should use a feature prefix, for example:

`feat: add topology-aware sparse MLA candidate policy`

## Acceptance Criteria

- Default behavior is unchanged when the topology gate is disabled.
- The topology policy feeds the existing sparse MLA path through the current
  request-local `topk_indices` contract.
- The compression policy produces stable-region metadata and token-efficiency
  counters without requiring FlashMLA kernel changes in the first implementation.
- Tests cover candidate validity, shape compatibility, causality, fallback,
  fingerprint drift, and token counters.
- Benchmarks or validation scripts report schedule cost separately from attention
  execution cost.
- Benchmarks report raw KV tokens avoided, compressed tokens represented, cache hit
  rate, aggregate latency, and tail latency.
- PR copy clearly labels the work as draft, dependent on #1160, and not yet a
  production speedup claim.
