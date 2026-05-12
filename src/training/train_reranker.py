from __future__ import annotations

import math
import random
from typing import Any

from src.utils.artifact import ensure_dir, is_complete, mark_done, read_json, read_jsonl, stable_hash, write_json
from src.utils.logging import saved, skip


def _split_qids(qids: list[str], seed: int, train_ratio: float = 0.9) -> tuple[list[str], list[str]]:
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []

    split_index = int(len(shuffled) * train_ratio)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    return shuffled[:split_index], shuffled[split_index:]


def train_reranker(config: Any, *, kind: str = "bge") -> None:
    filename = "rerank_train_ready.jsonl" if kind == "bge" else "qwen_train_ready.jsonl"
    model_name = config.rerank_model if kind == "bge" else config.qwen_model
    model_dir = config.dataset_dir / "models" / f"{kind}_reranker"
    marker = model_dir / "train_summary.json"
    params = {
        "epochs": config.reranker_epochs,
        "batch_size": config.reranker_train_batch_size,
        "grad_accum": config.reranker_grad_accum,
        "lr": config.reranker_lr,
        "warmup_ratio": config.reranker_warmup_ratio,
        "max_length": config.reranker_max_length,
        "max_train_examples": config.reranker_max_train_examples,
    }
    if is_complete(marker, expected={"model": model_name, "params": params}) and not config.force:
        skip(model_dir)
        return

    pairs_path = config.dataset_dir / "negatives" / filename
    ready_rows = read_jsonl(pairs_path)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    train_qids = sorted({str(qid) for qid in splits["train"]})
    subtrain_qids, subval_qids = _split_qids(train_qids, config.seed, train_ratio=0.9)
    subtrain_qids_set = set(subtrain_qids)
    subval_qids_set = set(subval_qids)

    train_rows = [row for row in ready_rows if str(row["qid"]) in subtrain_qids_set]
    val_rows = [row for row in ready_rows if str(row["qid"]) in subval_qids_set]
    if not train_rows:
        raise ValueError(f"No reranker training rows found for train split in {pairs_path}")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    examples: list[dict[str, Any]] = []
    for row in train_rows:
        for passage in row["passages"]:
            examples.append({"text_a": row["query"], "text_b": passage["text"], "label": float(passage["label"])})
    random.Random(config.seed).shuffle(examples)
    if config.reranker_max_train_examples > 0:
        examples = examples[: config.reranker_max_train_examples]
    if not examples:
        raise ValueError("No reranker training examples were built.")

    print(f"[train_reranker] first_sample={examples[0]!r}")

    ensure_dir(model_dir)
    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [str(item["text_a"]) for item in batch],
            [str(item["text_b"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=config.reranker_max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32)
        return encoded

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=config.reranker_train_batch_size, collate_fn=collate)
    grad_accum = max(1, int(config.reranker_grad_accum))
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / grad_accum))
    warmup_steps = int(optimizer_steps_per_epoch * config.reranker_epochs * config.reranker_warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.reranker_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, optimizer_steps_per_epoch * config.reranker_epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.reranker_use_amp and device.type == "cuda"))
    losses: list[float] = []
    optimizer_steps = 0
    log_every = 10
    print(f"[train_reranker] training with grad_accum={grad_accum}")
    model.train()
    for epoch in range(config.reranker_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_total = 0.0
        for step, batch in enumerate(train_dataloader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", enabled=bool(config.reranker_use_amp and device.type == "cuda")):
                outputs = model(**batch)
                logits = outputs.logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float())
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            epoch_loss_total += float(loss.detach().cpu())
            if step == 1 or step % log_every == 0 or step == len(train_dataloader):
                avg_loss = epoch_loss_total / step
                print(f"[train_reranker] epoch={epoch + 1}/{config.reranker_epochs} step={step}/{len(train_dataloader)} loss={avg_loss:.4f}")
            if step % grad_accum == 0 or step == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
        print(f"[train_reranker] epoch={epoch + 1}/{config.reranker_epochs} avg_loss={epoch_loss_total / max(len(train_dataloader), 1):.4f}")

    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    write_json(
        marker,
        {
            "status": "complete",
            "base_model": model_name,
            "pairs_path": str(pairs_path),
            "num_ready_rows": len(ready_rows),
            "num_train_rows": len(train_rows),
            "num_val_rows": len(val_rows),
            "num_examples": len(examples),
            "train_qids": len(train_qids),
            "subtrain_qids": len(subtrain_qids),
            "subval_qids": len(subval_qids),
            "optimizer_steps": optimizer_steps,
            "params": params,
        },
    )
    mark_done(marker, config=config, stage=f"train_{kind}_reranker", input_hash=stable_hash({"examples": len(examples), "qids": sorted(train_qids)}), model=model_name, params=params, fmt="json")
    saved(marker)
