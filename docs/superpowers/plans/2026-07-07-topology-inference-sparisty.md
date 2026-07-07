# Topology KV Compression And Inference Sparisty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default topology KV compression and sparse candidate policy for agentic/repeated workflows, measuring token movement reduction before claiming raw speedups.

**Architecture:** Keep FlashMLA sparse kernels unchanged. Add a pure Python/Torch policy layer near the hybrid indexer that classifies stable KV regions, emits compression/reuse metadata and token-efficiency counters, then merges raw sink/local/witness candidates into the existing `topk_indices` contract. Runtime sparse MLA continues to consume request-local raw token indices; compressed metadata is exposed for validation and follow-up kernel/runtime work.

**Tech Stack:** Python 3, PyTorch tensors, existing RTP-LLM hybrid indexer, existing `PyAttentionInputs` and sparse MLA params, `unittest`, GitHub draft PR #1162.

## Global Constraints

- This branch is stacked on top of #1160 and remains draft until #1160 merges.
- Default behavior must be unchanged when topology policy is disabled.
- Do not change FlashMLA sparse kernels in this PR.
- Preserve the current request-local `topk_indices` contract.
- Token efficiency is a first-class metric: raw KV tokens avoided, compressed tokens represented, cache/compression hits, schedule cost, aggregate latency, and tail latency.
- Tail latency p90/p95/p99 must be reported before the PR is marked ready.
- Stable compression metadata must be invalidated on scaffold, output-contract, model/layer, dtype, block-size, or cache-layout drift.

---

## File Structure

- Create `rtp_llm/models_py/modules/hybrid/topology_kv_policy.py`
  - Pure helpers for stable fingerprinting, compressed-region metadata, token counters, and sparse raw candidate merge.
- Create `rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py`
  - CPU-deterministic tests for disabled behavior, stable fingerprint matching, drift invalidation, token counters, causality, and duplicate removal.
- Modify `rtp_llm/models_py/modules/hybrid/indexer.py`
  - Read an opt-in topology KV policy gate.
  - Apply the policy after the existing learned top-k result is computed.
  - Store latest topology token counters for debug/validation.
- Modify `docs/superpowers/specs/2026-07-07-topology-inference-sparsity-design.md`
  - Record the exact implementation gate names and validation status.

## Task 1: Pure KV Topology Policy And Counters

**Files:**
- Create: `rtp_llm/models_py/modules/hybrid/topology_kv_policy.py`
- Test: `rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py`

**Interfaces:**
- Produces: `TopologyKvPolicyConfig(policy: str, sink_blocks: int, local_blocks: int, witness_blocks: int, block_size: int)`
- Produces: `TopologyKvCounters(raw_selected_tokens: int, compressed_tokens_represented: int, raw_kv_tokens_avoided: int, compression_hits: int, schedule_ms: float)`
- Produces: `TopologyKvPolicyResult(topk_indices: torch.Tensor, counters: TopologyKvCounters, stable_fingerprint: str)`
- Produces: `apply_topology_kv_policy(topk_indices, lengths, *, config, row_starts=None, topk_indices_offset=None, stable_scaffold=None, output_contract=None, block_drift_scores=None, previous_fingerprint=None) -> TopologyKvPolicyResult`

- [ ] **Step 1: Write failing tests for disabled behavior and counters**

Create `rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py`:

```python
import unittest

import torch

from rtp_llm.models_py.modules.hybrid.topology_kv_policy import (
    TopologyKvPolicyConfig,
    apply_topology_kv_policy,
)


class TopologyKvPolicyTest(unittest.TestCase):
    def test_disabled_policy_returns_original_tensor_and_zero_counters(self):
        topk = torch.tensor([[5, 4, 3, 2]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="disabled",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=4,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertIs(result.topk_indices, topk)
        self.assertEqual(result.counters.raw_selected_tokens, 4)
        self.assertEqual(result.counters.compressed_tokens_represented, 0)
        self.assertEqual(result.counters.raw_kv_tokens_avoided, 0)
        self.assertEqual(result.counters.compression_hits, 0)

    def test_compress_sparse_reports_stable_prefix_tokens_avoided(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        result = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="system policy + tool schema",
            output_contract="json tool result",
        )

        self.assertEqual(result.topk_indices.shape, topk.shape)
        self.assertEqual(result.topk_indices.dtype, topk.dtype)
        self.assertIn(0, result.topk_indices[0].tolist())
        self.assertIn(7, result.topk_indices[0].tolist())
        self.assertGreater(result.counters.compressed_tokens_represented, 0)
        self.assertGreater(result.counters.raw_kv_tokens_avoided, 0)

    def test_fingerprint_changes_when_output_contract_changes(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        first = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-a",
        )
        second = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-b",
            previous_fingerprint=first.stable_fingerprint,
        )

        self.assertNotEqual(first.stable_fingerprint, second.stable_fingerprint)
        self.assertEqual(second.counters.compression_hits, 0)

    def test_matching_fingerprint_counts_compression_hit(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        first = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-a",
        )
        second = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-a",
            previous_fingerprint=first.stable_fingerprint,
        )

        self.assertEqual(second.counters.compression_hits, 1)
        self.assertGreater(second.counters.raw_kv_tokens_avoided, 0)

    def test_sparse_merge_preserves_causality_and_removes_duplicates(self):
        topk = torch.tensor([[7, 7, 6, 5, 4, 3]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=4,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        values = [value for value in result.topk_indices[0].tolist() if value >= 0]
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(value < 8 for value in values))
        self.assertEqual(result.counters.compressed_tokens_represented, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest rtp_llm.models_py.modules.hybrid.test.topology_kv_policy_test -v
```

Expected: FAIL with `ModuleNotFoundError` for `topology_kv_policy`.

- [ ] **Step 3: Implement the pure policy helper**

Create `rtp_llm/models_py/modules/hybrid/topology_kv_policy.py`:

```python
import hashlib
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TopologyKvPolicyConfig:
    policy: str
    sink_blocks: int
    local_blocks: int
    witness_blocks: int
    block_size: int


@dataclass(frozen=True)
class TopologyKvCounters:
    raw_selected_tokens: int
    compressed_tokens_represented: int
    raw_kv_tokens_avoided: int
    compression_hits: int
    schedule_ms: float


@dataclass(frozen=True)
class TopologyKvPolicyResult:
    topk_indices: torch.Tensor
    counters: TopologyKvCounters
    stable_fingerprint: str


def _stable_fingerprint(stable_scaffold: Optional[str], output_contract: Optional[str]) -> str:
    if not stable_scaffold and not output_contract:
        return ""
    material = f"{stable_scaffold or ''}\n---contract---\n{output_contract or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _append_block_tokens(selected: list[int], seen: set[int], block_id: int, block_size: int, length: int) -> None:
    start = block_id * block_size
    end = min(start + block_size, length)
    for token in range(start, end):
        if token not in seen:
            selected.append(token)
            seen.add(token)


def _structural_tokens(length: int, config: TopologyKvPolicyConfig, drift_row: Optional[torch.Tensor]) -> list[int]:
    if length <= 0:
        return []
    block_count = (length + config.block_size - 1) // config.block_size
    selected: list[int] = []
    seen: set[int] = set()

    for block_id in range(min(config.sink_blocks, block_count)):
        _append_block_tokens(selected, seen, block_id, config.block_size, length)

    local_start = max(0, block_count - config.local_blocks)
    for block_id in range(local_start, block_count):
        _append_block_tokens(selected, seen, block_id, config.block_size, length)

    if config.witness_blocks > 0:
        if drift_row is not None and drift_row.numel() >= block_count:
            witness_blocks = torch.topk(
                drift_row[:block_count].float(),
                k=min(config.witness_blocks, block_count),
            ).indices.cpu().tolist()
        else:
            step = max(1, block_count // (config.witness_blocks + 1))
            witness_blocks = list(range(step, block_count, step))[: config.witness_blocks]
        for block_id in witness_blocks:
            _append_block_tokens(selected, seen, int(block_id), config.block_size, length)

    return selected


def _merge_row(learned_row: torch.Tensor, structural: list[int], local_start: int, local_length: int) -> list[int]:
    budget = learned_row.numel()
    selected: list[int] = []
    seen: set[int] = set()
    local_end = local_start + local_length

    for token in structural:
        if local_start <= token < local_end and token not in seen:
            selected.append(token)
            seen.add(token)
            if len(selected) == budget:
                return selected

    for raw_value in learned_row.detach().cpu().tolist():
        value = int(raw_value)
        if value < 0:
            continue
        if local_start <= value < local_end and value not in seen:
            selected.append(value)
            seen.add(value)
            if len(selected) == budget:
                return selected

    while len(selected) < budget:
        selected.append(-1)
    return selected


def apply_topology_kv_policy(
    topk_indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    config: TopologyKvPolicyConfig,
    row_starts: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    stable_scaffold: Optional[str] = None,
    output_contract: Optional[str] = None,
    block_drift_scores: Optional[torch.Tensor] = None,
    previous_fingerprint: Optional[str] = None,
) -> TopologyKvPolicyResult:
    started = time.perf_counter()
    raw_selected = int((topk_indices >= 0).sum().item())
    fingerprint = _stable_fingerprint(stable_scaffold, output_contract)

    if config.policy == "disabled":
        counters = TopologyKvCounters(raw_selected, 0, 0, 0, 0.0)
        return TopologyKvPolicyResult(topk_indices, counters, fingerprint)
    if config.policy not in {"topology_sparse_merge", "topology_compress_sparse", "topology_only"}:
        counters = TopologyKvCounters(raw_selected, 0, 0, 0, 0.0)
        return TopologyKvPolicyResult(topk_indices, counters, fingerprint)
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must have shape [rows, topk]")
    if lengths.ndim != 1 or lengths.numel() != topk_indices.size(0):
        raise ValueError("lengths must have shape [rows]")

    lengths_cpu = lengths.detach().cpu()
    starts_cpu = row_starts.detach().cpu() if row_starts is not None else torch.zeros_like(lengths_cpu)
    offsets_cpu = topk_indices_offset.detach().cpu() if topk_indices_offset is not None else torch.zeros_like(lengths_cpu)
    rows = []

    compressed_tokens = 0
    compression_hits = 0
    if config.policy == "topology_compress_sparse" and fingerprint:
        stable_tokens = int(lengths_cpu.sum().item())
        compressed_tokens = max(0, stable_tokens - raw_selected)
        if previous_fingerprint and previous_fingerprint == fingerprint:
            compression_hits = 1

    for row_idx in range(topk_indices.size(0)):
        row_start = int(starts_cpu[row_idx].item())
        row_length = int(lengths_cpu[row_idx].item())
        offset = int(offsets_cpu[row_idx].item())
        drift_row = (
            block_drift_scores[row_idx]
            if block_drift_scores is not None and block_drift_scores.ndim == 2
            else None
        )
        structural = _structural_tokens(row_length, config, drift_row)
        structural = [token + row_start + offset for token in structural]
        learned = (
            topk_indices.new_full(topk_indices[row_idx].shape, -1)
            if config.policy == "topology_only"
            else topk_indices[row_idx]
        )
        rows.append(
            torch.tensor(
                _merge_row(learned, structural, row_start + offset, row_length),
                dtype=topk_indices.dtype,
            )
        )

    merged = torch.stack(rows, dim=0).to(device=topk_indices.device)
    raw_selected_after = int((merged >= 0).sum().item())
    schedule_ms = (time.perf_counter() - started) * 1000
    counters = TopologyKvCounters(
        raw_selected_tokens=raw_selected_after,
        compressed_tokens_represented=compressed_tokens,
        raw_kv_tokens_avoided=compressed_tokens if compression_hits else max(0, compressed_tokens - raw_selected_after),
        compression_hits=compression_hits,
        schedule_ms=schedule_ms,
    )
    return TopologyKvPolicyResult(merged, counters, fingerprint)
```

- [ ] **Step 4: Run tests to verify helper passes**

Run:

```bash
python -m unittest rtp_llm.models_py.modules.hybrid.test.topology_kv_policy_test -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rtp_llm/models_py/modules/hybrid/topology_kv_policy.py rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py
git commit -m "feat: add topology kv compression policy"
```

## Task 2: Opt-In Indexer Integration

**Files:**
- Modify: `rtp_llm/models_py/modules/hybrid/indexer.py`
- Test: `rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py`

**Interfaces:**
- Consumes: `apply_topology_kv_policy(...) -> TopologyKvPolicyResult`
- Produces: `Indexer._apply_topology_kv_policy(topk_result, fmha_params, attention_inputs) -> torch.Tensor`
- Produces: `Indexer.latest_topology_kv_counters`

- [ ] **Step 1: Add integration import and env-gated fields**

Modify `rtp_llm/models_py/modules/hybrid/indexer.py`:

```python
import os
```

```python
from rtp_llm.models_py.modules.hybrid.topology_kv_policy import (
    TopologyKvPolicyConfig,
    apply_topology_kv_policy,
)
```

Add in `Indexer.__init__` after `self.blocksize`:

```python
        self.topology_kv_policy = os.getenv(
            "RTP_LLM_TOPOLOGY_KV_POLICY", "disabled"
        ).strip()
        self.topology_sink_blocks = int(os.getenv("RTP_LLM_TOPOLOGY_SINK_BLOCKS", "1"))
        self.topology_local_blocks = int(os.getenv("RTP_LLM_TOPOLOGY_LOCAL_BLOCKS", "1"))
        self.topology_witness_blocks = int(
            os.getenv("RTP_LLM_TOPOLOGY_WITNESS_BLOCKS", "1")
        )
        self.latest_topology_kv_counters = None
        self.latest_topology_kv_fingerprint = None
```

- [ ] **Step 2: Add the integration method**

Add before `_compute_topk`:

```python
    def _apply_topology_kv_policy(
        self,
        topk_result: torch.Tensor,
        fmha_params: Any,
        attention_inputs: Any,
    ) -> torch.Tensor:
        if self.topology_kv_policy == "disabled" or topk_result is None:
            return topk_result
        config = TopologyKvPolicyConfig(
            policy=self.topology_kv_policy,
            sink_blocks=self.topology_sink_blocks,
            local_blocks=self.topology_local_blocks,
            witness_blocks=self.topology_witness_blocks,
            block_size=self.blocksize,
        )
        if attention_inputs.is_prefill:
            result = apply_topology_kv_policy(
                topk_result,
                fmha_params.expanded_seq_lens,
                config=config,
                row_starts=fmha_params.ks,
                topk_indices_offset=fmha_params.topk_indices_offset,
                previous_fingerprint=self.latest_topology_kv_fingerprint,
            )
        else:
            result = apply_topology_kv_policy(
                topk_result,
                fmha_params.expanded_seq_lens,
                config=config,
                previous_fingerprint=self.latest_topology_kv_fingerprint,
            )
        self.latest_topology_kv_counters = result.counters
        self.latest_topology_kv_fingerprint = result.stable_fingerprint
        return result.topk_indices
```

- [ ] **Step 3: Wrap existing top-k outputs**

In `_compute_topk`, replace each direct return of an indexer top-k result with:

```python
        topk_result = self.indexer_op._get_topk_paged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )
        return self._apply_topology_kv_policy(
            topk_result, fmha_params, attention_inputs
        )
```

For `_get_topk_ragged` and `_get_topk_ragged_cp`, assign to `topk_result` and return
`self._apply_topology_kv_policy(topk_result, fmha_params, attention_inputs)`.

- [ ] **Step 4: Run focused tests and compile checks**

Run:

```bash
python -m unittest rtp_llm.models_py.modules.hybrid.test.topology_kv_policy_test -v
python -m py_compile rtp_llm/models_py/modules/hybrid/indexer.py rtp_llm/models_py/modules/hybrid/topology_kv_policy.py rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py
```

Expected: unittest PASS and py_compile exits 0.

- [ ] **Step 5: Commit**

```bash
git add rtp_llm/models_py/modules/hybrid/indexer.py rtp_llm/models_py/modules/hybrid/topology_kv_policy.py rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py
git commit -m "feat: gate topology kv sparse indexer policy"
```

## Task 3: Documentation And Draft PR Refresh

**Files:**
- Modify: `docs/superpowers/specs/2026-07-07-topology-inference-sparsity-design.md`
- Modify: `docs/superpowers/plans/2026-07-07-topology-inference-sparisty.md`

**Interfaces:**
- Consumes: gate and counter names from Tasks 1 and 2.
- Produces: reviewer-facing design and implementation-plan updates.

- [ ] **Step 1: Add implementation status to the spec**

Add under `## Configuration`:

```markdown
Initial implementation gate:

- `RTP_LLM_TOPOLOGY_KV_POLICY=disabled|topology_sparse_merge|topology_compress_sparse|topology_only`
- `RTP_LLM_TOPOLOGY_SINK_BLOCKS` set to a positive integer value
- `RTP_LLM_TOPOLOGY_LOCAL_BLOCKS` set to a positive integer value
- `RTP_LLM_TOPOLOGY_WITNESS_BLOCKS` set to a non-negative integer value

The default policy is `disabled`, so existing serving behavior is unchanged unless
the experiment is explicitly enabled.
```

- [ ] **Step 2: Run markdown marker scan**

Run:

```bash
rg -n "T""BD|TO""DO|place""holder|implement la""ter|fi""ll in" docs/superpowers/specs/2026-07-07-topology-inference-sparsity-design.md docs/superpowers/plans/2026-07-07-topology-inference-sparisty.md
```

Expected: no matches.

- [ ] **Step 3: Commit docs**

```bash
git add docs/superpowers/specs/2026-07-07-topology-inference-sparsity-design.md docs/superpowers/plans/2026-07-07-topology-inference-sparisty.md
git commit -m "docs: plan topology kv compression sparsity"
```

## Task 4: Push And Update Draft PR #1162

**Files:**
- No code files.

**Interfaces:**
- Consumes: committed branch `feat/topology-inference-sparisty`.
- Produces: updated draft PR #1162.

- [ ] **Step 1: Run final local checks**

Run:

```bash
python -m unittest rtp_llm.models_py.modules.hybrid.test.topology_kv_policy_test -v
python -m py_compile rtp_llm/models_py/modules/hybrid/indexer.py rtp_llm/models_py/modules/hybrid/topology_kv_policy.py rtp_llm/models_py/modules/hybrid/test/topology_kv_policy_test.py
git status --short
```

Expected:

- unittest PASS,
- py_compile exits 0,
- `git status --short` has no tracked code changes.

- [ ] **Step 2: Push branch**

```bash
git push origin feat/topology-inference-sparisty
```

Expected: push succeeds.

- [ ] **Step 3: Update PR body**

Run:

```bash
$bodyPath = Join-Path $env:TEMP "topology-kv-compress-sparse-pr.md"
gh pr edit 1162 --repo alibaba/rtp-llm --body-file $bodyPath
```

The updated body must say:

- blocked by #1160,
- inspired by the NeMo-Relay #322 discipline of measuring cache/token movement,
- topology KV policy is disabled by default,
- runtime implementation exists behind `RTP_LLM_TOPOLOGY_KV_POLICY`,
- FlashMLA kernels are unchanged,
- validation commands and results are listed,
- raw KV tokens avoided, compressed tokens represented, cache hit rate, schedule cost, aggregate latency, and tail latency are required before ready-for-review.

## Self-Review

Spec coverage:

- Disabled default: Tasks 1 and 2.
- Existing top-k contract: Task 1 returns the same raw `topk_indices` shape and Task 2 integrates after existing indexer output.
- Compression/token movement: Task 1 defines counters and stable fingerprints.
- Drift invalidation: Task 1 fingerprints scaffold and output contract, and tests changed contract behavior.
- Draft PR dependency and NeMo-Relay #322 inspiration: Task 4 updates PR #1162.

Marker scan:

- The plan contains no incomplete-work markers.

Type consistency:

- `TopologyKvPolicyConfig`, `TopologyKvCounters`, `TopologyKvPolicyResult`, and `apply_topology_kv_policy` are defined in Task 1 and consumed by Task 2 with the same names and signatures.
