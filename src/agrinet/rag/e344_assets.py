"""Verified classifier bindings and formal encoder features for fresh selection."""
from __future__ import annotations

import json
from pathlib import Path

from agrinet.rag.e344_prepare import digest, rows


def verify_binding(checkpoint, label_map, training_manifest, heldout=None):
    """Hash the actual entities and retain exclusion sets, never trust old cards."""
    paths = {"checkpoint": Path(checkpoint), "label_map": Path(label_map),
             "training_manifest": Path(training_manifest)}
    labels = json.loads(paths["label_map"].read_text())
    codes = {r["canonical_class_code"] for r in labels}
    training = list(rows(paths["training_manifest"]))
    bound = {key: str(path) for key, path in paths.items()}
    bound.update({key + "_sha256": digest(path) for key, path in paths.items()})
    bound["label_codes"] = sorted(codes)
    bound["training_shas"] = {r["image_sha256"] for r in training}
    bound["training_codes"] = {r["canonical_class_code"] for r in training}
    if heldout is not None:
        bound["heldout_shas"] = {r["image_sha256"] for r in rows(heldout)}
        if bound["heldout_shas"] & bound["training_shas"]:
            raise ValueError("oof_train_heldout_overlap")
    if not bound["training_codes"].issubset(codes):
        raise ValueError("training_labels_not_in_checkpoint_label_map")
    return bound


def eligible_binding(row, binding, arm):
    sha, code = row["image_sha256"], row["canonical_class_code"]
    if sha in binding["training_shas"]:
        return False
    if arm == "known":
        return sha in binding.get("heldout_shas", set()) and code in binding["label_codes"]
    if arm == "simulated_unknown":
        return code not in binding["label_codes"] and code not in binding["training_codes"]
    raise ValueError("unknown_arm")


def load_classifier(binding, device="cuda:0"):
    import torch
    import timm
    checkpoint = torch.load(binding["checkpoint"], map_location="cpu", weights_only=False)
    labels = json.loads(Path(binding["label_map"]).read_text())
    if checkpoint.get("label_map") != labels:
        raise ValueError("checkpoint_embedded_label_map_mismatch")
    model = timm.create_model(checkpoint["arch"], pretrained=False, num_classes=len(labels))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device), labels


def extract_features(candidates, checkpoint, output, device="cuda:0", batch_size=64):
    """Resume by immutable batch; freeze checkpoint hash, row order and transform."""
    import numpy as np
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "image_shas": [r["image_sha256"] for r in candidates], "batch_size": batch_size,
                "transform": "resize256-centercrop224-imagenet-normalize", "l2_normalized": True}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("feature_cache_binding_changed")
    manifest.write_text(json.dumps(metadata))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = timm.create_model(payload["arch"], pretrained=False, num_classes=len(payload["label_map"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.reset_classifier(0)
    model.eval().to(device)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    chunks = []
    for start in range(0, len(candidates), batch_size):
        path = output / f"batch-{start:08d}.npy"
        batch = candidates[start:start + batch_size]
        if not path.exists():
            images = []
            for row in batch:
                if digest(row["image_path"]) != row["image_sha256"]:
                    raise ValueError("candidate_image_hash_changed")
                with Image.open(row["image_path"]) as image:
                    images.append(transform(image.convert("RGB")))
            with torch.inference_mode():
                features = model(torch.stack(images).to(device))
                features = torch.nn.functional.normalize(features.float(), dim=1).cpu().numpy()
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, features)
            temporary.replace(path)
        chunk = np.load(path, allow_pickle=False)
        if len(chunk) != len(batch) or not np.isfinite(chunk).all():
            raise ValueError("invalid_feature_cache")
        chunks.append(chunk)
    return np.concatenate(chunks)


def predict_top3(row, binding, model, labels, device="cuda:1"):
    import torch
    from PIL import Image
    from torchvision import transforms
    if digest(row["image_path"]) != row["image_sha256"]:
        raise ValueError("classifier_image_hash_changed")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    with Image.open(row["image_path"]) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores, indices = model(tensor).softmax(dim=-1)[0].topk(3)
    return {"top3": [{"code": labels[i]["canonical_class_code"], "name": labels[i]["english_name"], "score": score}
                     for score, i in zip(scores.cpu().tolist(), indices.cpu().tolist())]}
