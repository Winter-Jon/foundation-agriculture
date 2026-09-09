"""Reproducible OpenAgri v3 vision workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifests(dataset_root: Path, artifact_root: Path) -> None:
    image_rows = _read_jsonl(dataset_root / "manifests" / "images.jsonl")
    class_rows = _read_jsonl(dataset_root / "manifests" / "class_split.jsonl")
    known = sorted((row for row in class_rows if row["class_role"] == "known"), key=lambda row: row["canonical_class_code"])
    if not known:
        raise ValueError("no Known classes in OpenAgri v3")
    labels = [{"index": idx, "canonical_class_code": row["canonical_class_code"], "english_name": row["canonical_english_name"], "chinese_name": row["canonical_chinese_name"], "domain": row["domain"]} for idx, row in enumerate(known)]
    index = {row["canonical_class_code"]: row["index"] for row in labels}
    eval_rows = [row for row in image_rows if row["image_split"] in {"dev", "test"}]
    excluded_sha = {row["image_sha256"] for row in eval_rows}
    excluded_paths = {str(Path(row["image_path"]).resolve()) for row in eval_rows}
    splits: dict[str, list[dict]] = {"train": [], "dev_known": [], "test_known": []}
    for row in image_rows:
        code = row["canonical_class_code"]
        if code not in index:
            continue
        if row["image_split"] == "train_candidate": split = "train"
        elif row["image_split"] == "dev": split = "dev_known"
        elif row["image_split"] == "test": split = "test_known"
        else: continue
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        splits[split].append({"image_id": row["image_name"], "image_path": str(image_path.resolve()), "image_sha256": row["image_sha256"], "label": index[code], "canonical_class_code": code, "domain": row["domain"]})
    for name, rows in splits.items():
        rows.sort(key=lambda row: (row["canonical_class_code"], row["image_id"]))
        if not rows: raise ValueError(f"empty {name} manifest")
        _write_jsonl(artifact_root / "manifests" / f"{name}.jsonl", rows)
    if {row["image_sha256"] for row in splits["train"]} & excluded_sha:
        raise ValueError("classification train overlaps v3 dev/test")
    _write_json(artifact_root / "label_map.json", labels)
    _write_json(artifact_root / "mae_excluded_sha256.json", sorted(excluded_sha))
    mae_rows = []
    for path in sorted((dataset_root.parent / "all").rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved not in excluded_paths:
            mae_rows.append({"image_path": resolved})
    if not mae_rows:
        raise ValueError("MAE corpus is empty after v3 evaluation exclusion")
    _write_jsonl(artifact_root / "manifests" / "mae_train.jsonl", mae_rows)
    _write_json(artifact_root / "build_report.json", {"known_classes": len(labels), "split_counts": {name: len(rows) for name, rows in splits.items()}, "mae_corpus_images": len(mae_rows), "mae_excluded_images": len(excluded_sha), "mae_exclusion": "resolved source paths derived from v3 dev/test SHA-256 manifest", "dataset_root": str(dataset_root.resolve()), "policy": "MAE excludes all OpenAgri v3 dev/test SHA-256; classifier uses Known only"})


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, transform: transforms.Compose):
        self.rows = _read_jsonl(manifest)
        self.transform = transform

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            return self.transform(image.convert("RGB")), int(row["label"]), row


class MAEDataset(Dataset):
    def __init__(self, manifest: Path, transform: transforms.Compose, limit: int | None = None):
        self.transform = transform
        rows = _read_jsonl(manifest)
        self.paths = [Path(row["image_path"]) for row in (rows if limit is None else rows[:limit])]
        if not self.paths:
            raise ValueError("MAE corpus is empty after v3 evaluation exclusion")

    def __len__(self) -> int: return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def _distributed() -> tuple[int, int, int]:
    if "RANK" not in __import__("os").environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, world, local


def _primary(rank: int) -> bool: return rank == 0


def _reduce(value: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1: dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


_ARCHITECTURES = {
    "mae_vit_base_patch16_224": "vit_base_patch16_224",
    "mae_vit_large_patch16_224": "vit_large_patch16_224",
}


def _mae_model(architecture: str):
    from vision.pretrain import model as mae_models

    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unsupported MAE architecture: {architecture}")
    return getattr(mae_models, architecture)()


def _save_encoder(model: nn.Module, target: Path, epoch: int, args: argparse.Namespace) -> None:
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save({"arch": f"{args.architecture}_encoder", "state_dict": raw.encoder_state_dict(), "epoch": epoch, "args": vars(args), "version": 1}, target)


def run_mae(args: argparse.Namespace) -> None:
    rank, world, local = _distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    root = Path(args.artifact_root)
    transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = MAEDataset(root / "manifests" / "mae_train.jsonl", transform, args.corpus_limit)
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.workers, pin_memory=True, drop_last=True)
    model = _mae_model(args.architecture).to(device)
    model = DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    base_lr = 1.5e-4 * (args.batch_size * world / 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.05)
    warmup_epochs = min(5, max(0, args.epochs - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / warmup_epochs if warmup_epochs and epoch < warmup_epochs else 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    history = [] ; step = 0
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train(); total = torch.zeros(2, device=device)
        for images in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _, _ = model(images, mask_ratio=0.75)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += torch.tensor([loss.detach(), 1], device=device); step += 1
            if args.max_steps and step >= args.max_steps: break
        total = _reduce(total, world)
        if _primary(rank):
            row = {"epoch": epoch + 1, "step": step, "loss": (total[0] / total[1]).item(), "corpus_images": len(dataset)}; history.append(row); _write_json(output / "mae_metrics.json", history)
            if (epoch + 1) % args.checkpoint_interval == 0 or epoch + 1 == args.epochs or args.max_steps:
                checkpoint = {"epoch": epoch + 1, "state_dict": (model.module if isinstance(model, DistributedDataParallel) else model).state_dict(), "optimizer": optimizer.state_dict()}
                _save_encoder(model, output / "encoder_last.pth.tar", epoch + 1, args)
                _save_encoder(model, output / f"encoder_epoch_{epoch + 1:03d}.pth.tar", epoch + 1, args)
                torch.save(checkpoint, output / "checkpoint_last.pth.tar")
                torch.save(checkpoint, output / f"checkpoint_epoch_{epoch + 1:03d}.pth.tar")
        scheduler.step()
        if args.max_steps and step >= args.max_steps: break
    if world > 1: dist.destroy_process_group()


def _classifier_model(num_classes: int, encoder_checkpoint: Path, architecture: str) -> nn.Module:
    import timm
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unsupported MAE architecture: {architecture}")
    model = timm.create_model(_ARCHITECTURES[architecture], pretrained=False, num_classes=num_classes)
    payload = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"invalid MAE encoder transfer; missing={missing}, unexpected={unexpected}")
    return model


def _metrics(logits: torch.Tensor, labels: torch.Tensor, labels_meta: list[dict]) -> dict:
    pred = logits.argmax(dim=1)
    count = len(labels_meta)
    matrix = torch.bincount(labels * count + pred, minlength=count * count).reshape(count, count).float()
    true = matrix.sum(1); predicted = matrix.sum(0); diagonal = matrix.diag()
    recall = diagonal / true.clamp_min(1); precision = diagonal / predicted.clamp_min(1)
    f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).nan_to_num(0)
    domains = {}
    for domain in sorted({row["domain"] for row in labels_meta}):
        ids = torch.tensor([row["index"] for row in labels_meta if row["domain"] == domain])
        domains[f"{domain}_macro_f1"] = f1[ids].mean().item()
    return {"top1": diagonal.sum().div(matrix.sum().clamp_min(1)).item(), "balanced_accuracy": recall.mean().item(), "macro_f1": f1.mean().item(), **domains, "confusion_matrix": matrix.to(dtype=torch.int64).tolist(), "per_class": [{**row, "precision": precision[row["index"]].item(), "recall": recall[row["index"]].item(), "f1": f1[row["index"]].item(), "support": int(true[row["index"]].item())} for row in labels_meta]}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels_meta: list[dict], write_predictions: Path | None = None) -> dict:
    model.eval(); logits_all = []; labels_all = []; predictions = []
    for images, labels, rows in loader:
        logits = model(images.to(device, non_blocking=True)); logits_all.append(logits.cpu()); labels_all.append(labels)
        if write_predictions is not None:
            probabilities = logits.softmax(dim=1).cpu()
            for probability, row in zip(probabilities, rows):
                top = probability.topk(min(5, len(labels_meta)))
                predictions.append({"image_id": row["image_id"], "image_sha256": row["image_sha256"], "class_index": int(top.indices[0]), "canonical_class_code": labels_meta[int(top.indices[0])]["canonical_class_code"], "confidence": float(top.values[0]), "topk": [{"class_index": int(idx), "canonical_class_code": labels_meta[int(idx)]["canonical_class_code"], "confidence": float(score)} for score, idx in zip(top.values, top.indices)]})
    metrics = _metrics(torch.cat(logits_all), torch.cat(labels_all), labels_meta)
    if write_predictions is not None: _write_jsonl(write_predictions, predictions)
    return metrics


def _collate(batch):
    images, labels, rows = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(rows)


def run_classifier(args: argparse.Namespace) -> None:
    rank, world, local = _distributed()
    if world != 1: raise ValueError("classification uses one A800 in v1 to keep balanced sampling exact")
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    root = Path(args.artifact_root); labels_meta = json.loads((root / "label_map.json").read_text(encoding="utf-8"))
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1, 0.05), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    eval_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train = ManifestDataset(root / "manifests" / "train.jsonl", train_transform); dev = ManifestDataset(root / "manifests" / "dev_known.jsonl", eval_transform)
    weights = Counter(row["label"] for row in train.rows); sampler = WeightedRandomSampler([1 / weights[row["label"]] for row in train.rows], num_samples=len(train), replacement=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=_collate)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=_collate)
    model = _classifier_model(len(labels_meta), Path(args.encoder_checkpoint), args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05); scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best = -1.0; history = []; step = 0
    for epoch in range(args.epochs):
        model.train()
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device), label_smoothing=0.1)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); step += 1
            if args.max_steps and step >= args.max_steps: break
        metrics = _evaluate(model, dev_loader, device, labels_meta); metrics.update({"epoch": epoch + 1, "step": step}); history.append(metrics); _write_json(output / "classifier_dev_metrics.json", history)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]; torch.save({"arch": _ARCHITECTURES[args.architecture], "state_dict": model.state_dict(), "label_map": labels_meta, "epoch": epoch + 1, "dev_metrics": metrics}, output / "model_best.pth.tar")
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch + 1 == args.epochs: torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, output / "checkpoint_last.pth.tar")
        if args.max_steps and step >= args.max_steps: break
    _write_json(output / "label_map.json", labels_meta)


def run_evaluate(args: argparse.Namespace) -> None:
    root = Path(args.artifact_root); labels_meta = json.loads((root / "label_map.json").read_text(encoding="utf-8")); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = ManifestDataset(root / "manifests" / f"{args.split}.jsonl", transform); loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, collate_fn=_collate)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False); import timm; model = timm.create_model(payload["arch"], pretrained=False, num_classes=len(labels_meta)); model.load_state_dict(payload["state_dict"]); model.to(device)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); metrics = _evaluate(model, loader, device, labels_meta, output / f"predictions_{args.split}.jsonl"); _write_json(output / f"metrics_{args.split}.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifests"); build.add_argument("--dataset-root", type=Path, required=True); build.add_argument("--artifact-root", type=Path, required=True)
    for name in ("mae", "classifier"):
        child = commands.add_parser(name); child.add_argument("--artifact-root", type=Path, required=True); child.add_argument("--output-dir", type=Path, required=True); child.add_argument("--epochs", type=int, required=True); child.add_argument("--batch-size", type=int, required=True); child.add_argument("--workers", type=int, default=8); child.add_argument("--checkpoint-interval", type=int, default=25); child.add_argument("--max-steps", type=int); child.add_argument("--corpus-limit", type=int)
        child.add_argument("--architecture", choices=tuple(_ARCHITECTURES), required=True)
        if name == "classifier": child.add_argument("--encoder-checkpoint", type=Path, required=True)
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("--artifact-root", type=Path, required=True); evaluate.add_argument("--output-dir", type=Path, required=True); evaluate.add_argument("--checkpoint", type=Path, required=True); evaluate.add_argument("--split", choices=("dev_known", "test_known"), default="test_known")
    args = parser.parse_args()
    random.seed(42); torch.manual_seed(42)
    if args.command == "build-manifests": build_manifests(args.dataset_root, args.artifact_root)
    elif args.command == "mae": run_mae(args)
    elif args.command == "classifier": run_classifier(args)
    else: run_evaluate(args)


if __name__ == "__main__":
    main()
