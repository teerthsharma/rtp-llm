# Topology Inference Sparsity Design

## Context

PR #1160 adds an opt-in benchmark for topology-derived KV candidate schedules under
`benchmark/`. This follow-up branch starts from the #1160 branch head and is valid
only after #1160 merges. The follow-up PR must be opened as a draft and must state
that it is blocked by #1160 until the benchmark branch lands in `main`.

The benchmark proves that topology-derived candidate token sets can be constructed
and timed in a standalone SDPA screening harness. It does not yet prove serving-path
speedup or model quality. This design moves the topology idea into the existing
runtime sparse MLA path while keeping the production default unchanged.

## Goal

Add a disabled-by-default topology-aware inference sparsity path that can feed
request-local candidate token indices into the existing sparse MLA runtime, then
measure whether topology-derived candidates improve the latency, memory, and
quality tradeoff against the current learned indexer at the same budget.

## Non-Goals

- Do not replace FlashMLA kernels in this PR.
- Do not claim end-to-end serving speedup until measured in RTP-LLM runtime.
- Do not make topology sparsity the default path.
- Do not remove the existing learned sparse indexer or dense fallback.
- Do not rely on synthetic-only evidence for PR-ready performance claims.

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

Topological object: per-request KV block point cloud in indexer key space.

Construction: each KV cache block is represented by a compact block summary derived
from the indexer key representation. Adjacent blocks form the causal time backbone.
Additional edges or rankings come from centroid drift and query-to-block affinity.

Invariant or summary: stable coverage by sink blocks, newest local blocks, and
high-drift witness blocks. High drift acts as a witness proxy for regions where the
key-space trajectory changes enough that a purely local or uniform selector may
miss attention mass.

Runtime rule: topology produces a candidate guarantee over request-local token
indices. The existing learned indexer remains the scoring baseline and safety
source. Topology either reserves part of the budget for structural candidates or
reranks/merges learned candidates with sink, local, and witness candidates under
the same `indexer_topk` budget.

## Proposed Approach

Use a decorator-style topology candidate policy around the existing indexer output.

When the topology path is disabled, `Indexer.forward` returns exactly the current
learned top-k result.

When enabled, the path:

1. Computes or reuses per-request block summaries from the indexer key-space data.
2. Selects mandatory structural blocks:
   - sink blocks for global anchors,
   - newest local blocks for causal locality,
   - high-drift witness blocks for topology coverage.
3. Converts selected blocks to request-local token candidates.
4. Merges those candidates with learned indexer candidates.
5. Trims to the same `indexer_topk` budget.
6. Returns the same shape consumed by `SparseMlaImpl` and `SparseMlaCpImpl`.

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
- `topology_merge`: learned top-k plus mandatory structural candidates.
- `topology_only`: test-only mode for ablations, not a production default.

## Candidate Budget

Budget unit: request-local selected tokens, equal to `attn_config.indexer_topk`.

All comparisons must use the same budget:

- existing learned indexer,
- topology merge,
- locality-only same budget,
- random same budget where practical in benchmark or test harnesses.

The topology path must preserve causality. A query row must not select future KV
tokens. For decode, each request has a single new query row and candidates must be
within that request's current KV length. For prefill, rows must respect the row's
ragged causal window and any existing `topk_indices_offset` semantics.

## Data Flow

The intended insertion point is in `rtp_llm/models_py/modules/hybrid/indexer.py`.
The policy should live near the hybrid indexer code rather than inside FlashMLA
operators. The FlashMLA sparse implementations should continue to receive
`topk_indices` and remain unaware of how candidates were chosen.

Decode path:

1. Existing indexer computes logits and learned top-k through `_get_topk_paged`.
2. Topology policy builds request-local structural candidates from available
   request KV metadata.
3. Policy merges structural candidates into the learned top-k row.
4. Sparse MLA converts local indices to global slots as it does today.

Prefill path:

1. Existing indexer computes ragged learned top-k through `_get_topk_ragged`.
2. Topology policy respects `ks`, `ke`, `expanded_seq_lens`, and
   `topk_indices_offset`.
3. Policy merges only candidates that are valid for each row's causal span.
4. CP prefill support is deferred unless the same local-index contract can be
   preserved without extra collectives.

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

## Research And Falsification Gates

The research hypothesis is:

Topology-aware structural candidates improve sparse inference robustness at the
same selected-token budget by preserving global anchors, recent causal context,
and high-drift witness regions that the learned indexer can underselect.

Null models:

- existing learned sparse indexer,
- locality-only same-budget selector,
- random same-budget selector in offline harnesses,
- dense or fastest available non-sparse attention path for runtime comparison.

Falsification gates:

- Accuracy: topology merge must not regress sparse MLA output quality beyond the
  existing learned sparse indexer tolerance on representative long-context cases.
- Speed: topology merge must not add enough schedule construction overhead to erase
  sparse runtime savings.
- Memory: candidate metadata must remain bounded by request length and block count.
- Coverage: results must include at least two long sequence lengths and the target
  head dimensions used by sparse MLA configurations.
- Stability: topology should not pass only on synthetic random tensors; real model
  Q/K/V or end-to-end generation captures are required before non-draft claims.

## Tests

Focused tests should cover:

- disabled gate returns the exact existing learned top-k result;
- topology merge preserves output shape, dtype, and device;
- local newest block is retained under partial budgets;
- sink and witness candidates are included when budget allows;
- future tokens are not selected in prefill rows;
- duplicate candidates are removed or resolved deterministically;
- same-budget learned, topology, locality, and random selectors are comparable in
  benchmark or validation tooling;
- sparse MLA still accepts the topology-produced top-k indices.

CUDA-specific tests should stay manual if they require CUDA 12.9, FlashMLA, or H20
class hardware. CPU-deterministic helper tests should run without GPU when possible.

## Documentation

The draft PR should document:

- it is blocked by #1160 until the benchmark PR merges;
- topology merge is disabled by default;
- microbenchmark speedup from #1160 is not a serving speedup claim;
- schedule construction time must be reported separately from sparse attention
  execution time;
- real model validation remains required before marking the PR ready for review.

## Draft PR Strategy

Open the PR as a draft from `feat/topology-inference-sparsity`.

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
- Tests cover candidate validity, shape compatibility, causality, and fallback.
- Benchmarks or validation scripts report schedule cost separately from attention
  execution cost.
- PR copy clearly labels the work as draft, dependent on #1160, and not yet a
  production speedup claim.
