"""ms-swift plugin for Hermes-weighted per-token Liger cross entropy.

ms-swift bypasses Liger CE when ``loss_scale`` is present. Hermes uses that
path to weight tool-call spans, so an unweighted replacement changes training
semantics. This plugin replaces only the unreduced per-token helper, retaining
ms-swift's existing shift, mask, weighting, and denominator logic.
"""
from __future__ import annotations

import os
from importlib.metadata import version

EXPECTED_SWIFT = "4.5.2"
EXPECTED_LIGER = "0.8.2"


def _require_versions() -> None:
    actual_swift = version("ms-swift")
    actual_liger = version("liger-kernel")
    if actual_swift != EXPECTED_SWIFT or actual_liger != EXPECTED_LIGER:
        raise RuntimeError(
            "weighted_liger_plugin supports only "
            f"ms-swift=={EXPECTED_SWIFT} and liger-kernel=={EXPECTED_LIGER}; "
            f"found ms-swift=={actual_swift}, liger-kernel=={actual_liger}"
        )


def liger_per_token_loss(outputs, labels, enable_dft_loss: bool = False, **_: object):
    """Return shifted unreduced CE, matching ms-swift ``per_token_loss_func``."""
    if enable_dft_loss:
        raise RuntimeError("weighted Liger CE does not support enable_dft_loss")
    import torch
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

    logits = outputs.logits.float()
    shifted_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1).to(logits.device)
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    return LigerCrossEntropyLoss(ignore_index=-100, reduction="none")(flattened_logits, shifted_labels)


def install() -> None:
    """Install the narrow patch before ms-swift constructs its trainer."""
    _require_versions()
    import swift.trainers.seq2seq_trainer as seq2seq_trainer
    import swift.trainers.utils as trainer_utils

    if getattr(trainer_utils, "_agrinet_weighted_liger_installed", False):
        return
    trainer_utils.per_token_loss_func = liger_per_token_loss
    seq2seq_trainer.per_token_loss_func = liger_per_token_loss
    trainer_utils._agrinet_weighted_liger_installed = True
    print("[agrinet] weighted_liger_ce=active mode=unreduced_logits version_guard=passed", flush=True)


if os.environ.get("AGRINET_WEIGHTED_LIGER_PLUGIN_NO_INSTALL") != "1":
    install()
