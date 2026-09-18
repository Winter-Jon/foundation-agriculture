# Plan

## Goal

Finish the authorized E3.42 collection and independently audit its 82 sample-local outcomes.

## Current Hypothesis

The managed campaign can recover from upstream ambiguity through its persisted R1/R2 lineage without replaying already-ambiguous intents.

## Current Strategy

Monitor the existing managed process, inspect bounded ledger and artifact metadata, and aggregate only after it reaches a terminal state.

## Decision Logic

Do not relaunch while the recorded process is alive. If it exits, inspect status, outcomes, ledgers, and reports before considering a resume.

## Current Milestones

- Prepared 82-row source and two immutable scheduling shards.
- Live campaign launched once; collection and audit remain.
