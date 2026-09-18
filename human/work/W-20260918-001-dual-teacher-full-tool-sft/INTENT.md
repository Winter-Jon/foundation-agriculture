---
id: "W-20260918-001"
state: "active"
created_at: "2026-09-18T18:00:00+08:00"
---
# Dual-teacher full-tool SFT

## Objective
Implement the approved 1881-row faithful conversion, native multimodal full-tool evaluation, gated eight-GPU Qwen3-VL-4B full SFT for six epochs, and test-only base/epoch3/epoch6 reports. No test-driven selection.

## Plan
Freeze data and assets; audit actual Hermes targets; test shared public protocol and multimodal rendering; run three-step training and inference smoke; train six epochs from base; evaluate three predetermined models under one frozen protocol.

## Constraints
Preserve unrelated changes and collection artifacts. Use registered local agrinet workflows, .venv_test for SFT, 16K without truncation, LR2e-6, batch1/GA8, seed42. Do not terminate unidentified GPU users.
