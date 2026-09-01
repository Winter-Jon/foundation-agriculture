# Checkpoint Cleanup Review

Status: review only — **no checkpoint has been deleted**.

## Scope and safety rule

This review covers the 60 local checkpoints under `outputs/vlm_sft/` (497GB).
Each checkpoint is about 8.28GiB. A path is listed as a deletion candidate
only when it is not one of the current Formal-618 checkpoints and has no
active exact-path reference in the checked docs, configs, scripts, source, or
tools. This is an audit signal, not a deletion decision.

Retain without review:

- Corrected M1: `qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/.../checkpoint-165`
- Current RAG: `qwen3_vl_4b_hcv_manual_json_v12_direct_anchor_mix_base_8gpu_standard_lr2e6/.../checkpoint-232`
- v4, restored training system prompt: `qwen3_vl_4b_m1_latest_ms_swift_v4_m1_system_format_8gpu/.../checkpoint-320`

Also retain configured or historical reference checkpoints until their own
configs/docs are retired: M3 `checkpoint-72`, allocator v12 `checkpoint-232`,
direct-only v12 `checkpoint-210`, Hermes M2 `checkpoint-175`, M1-like
`checkpoint-128`, M1 v3 `checkpoint-320`, and Direct reproduction
`checkpoint-165`.

## Candidate groups for approval

| Review group | Candidate checkpoints | Count | Estimated space | Recommendation |
|---|---|---:|---:|---|
| Current RAG LR-2e-6 intermediate epochs | `v12_direct_anchor_mix_base_8gpu_standard_lr2e6`: 464, 696 | 2 | 16.6GB | Delete after confirming checkpoint-232 remains the selected RAG result. |
| Direct-only v12 intermediate epochs | `v12_direct_only_base_8gpu_lr2e6`: 70, 140 | 2 | 16.6GB | Delete; checkpoint-210 is retained. |
| Interim Direct-only rejected run | `m1_direct_current_hcv_interim_8gpu`: 51, 102, 153 | 3 | 24.8GB | Delete; all promotion gates rejected. |
| Interim comparison-chain ablation | `m1_direct_current_hcv_interim_comparison_m1like_8gpu`: 26, 52, 78, 104, 130 | 5 | 41.4GB | Needs confirmation: checkpoint-130 was an ablation selection in an older record, but is not an active evaluation dependency. |
| Terminal-v5 base run | `m1_direct_current_hcv_terminal_v5_8gpu`: 64, 128, 192 | 3 | 24.8GB | Needs confirmation: no active exact checkpoint reference. |
| Terminal-v5 M1-like run | `m1_direct_current_hcv_terminal_v5_m1like_8gpu`: 32, 64, 96, 160 | 4 | 33.1GB | Delete; checkpoint-128 is retained. |
| M1-like tail LR 1e-6 | `...m1like_tail_e2_lr1e6_8gpu`: 32, 64 | 2 | 16.6GB | Needs confirmation: no active reference, but preserve if tail comparison is still needed. |
| M1-like tail LR 5e-7 | `...m1like_tail_e2_lr5e7_8gpu`: 32, 64 | 2 | 16.6GB | Needs confirmation: no active exact reference, but recent notes discuss this route. |
| User-guidance v3 M1-exact intermediates | `...user_guidance_v3_m1exact_4gpu`: 64, 128, 192, 256 | 4 | 33.1GB | Delete; checkpoint-320 is retained. |
| User-guidance v3 open-only branch | `...user_guidance_v3_open_only_m1exact_4gpu`: 32, 64, 96, 128, 160 | 5 | 41.4GB | Needs confirmation: all checkpoints lack active exact references. |
| Direct reproduction intermediates | `m1_direct_reproduction_4gpu`: 33, 66, 99, 132 | 4 | 33.1GB | Delete; checkpoint-165 is retained. |
| Latest ms-swift reproduction | `m1_latest_ms_swift_reproduction_4gpu`: 33, 66, 99, 132, 165 | 5 | 41.4GB | Needs confirmation: all are unreferenced, but this is a completed reproduction branch. |
| Latest ms-swift v3 | `m1_latest_ms_swift_v3_4gpu`: 64, 128, 192, 256, 320 | 5 | 41.4GB | Needs confirmation: the derived scoring evidence remains in `outputs/runs/`; keep 320 only if model replay is required. |
| v4 intermediates | `m1_latest_ms_swift_v4_m1_system_format_8gpu`: 64, 128, 192, 256 | 4 | 33.1GB | Delete; the current v4 comparison retains checkpoint-320. |

Total candidate capacity: **50 checkpoints / 413.8GB**.

## Suggested deletion batches

Approve separately so a review decision remains reversible until execution:

1. **Batch A — unambiguous intermediates:** 23 checkpoints / 190.4GB.
   Includes the two v12 groups, rejected interim run, M1-like non-selected
   checkpoints, v3 M1-exact intermediates, Direct reproduction intermediates,
   and v4 intermediates.
2. **Batch B — historical ablation branches:** 12 checkpoints / 99.4GB.
   Includes comparison-chain, terminal-v5 base, and both tail branches.
3. **Batch C — completed reproduction/alternative branches:** 15 checkpoints
   / 124.2GB. Includes open-only, latest ms-swift reproduction, and latest
   ms-swift v3.

Before any batch is removed, verify each explicit target again against the
current `docs/`, `configs/`, `scripts/`, `src/`, and `tools/` trees, then
remove only those exact checkpoint directories. Keep associated result
summaries and scored predictions under `outputs/runs/` unless separately
reviewed.
