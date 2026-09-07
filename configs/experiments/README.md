# Experiment Definitions

This directory contains reusable, reviewed experiment definitions.  A definition
belongs here only when its identifier, inputs, and policy are useful to another
checkout or a future run.

Per-attempt retry and restart snapshots are local operational state.  Keep them
under `outputs/volatile/configs/` (which is ignored) beside their run evidence;
do not add them here or modify a versioned definition to record a retry.

Run outputs, checkpoints, logs, service state, downloaded datasets, and models
also remain outside version control under their respective ignored roots.
