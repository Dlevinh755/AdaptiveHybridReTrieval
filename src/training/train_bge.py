from __future__ import annotations

import gc
import math
import random
from typing import Any

from src.utils.artifact import ensure_dir, is_complete, mark_done, read_json, read_jsonl, stable_hash, write_json
from src.utils.logging import saved, skip


def _build_triplets(rows: list[dict[str, Any]], negatives_per_positive: int, seed: int) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    rng = random.Random(seed)

    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        positives = [str(chunk.get("text", "")).strip() for chunk in row.get("pos", []) if str(chunk.get("text", "")).strip()]
        negatives = [str(chunk.get("text", "")).strip() for chunk in row.get("neg", []) if str(chunk.get("text", "")).strip()]
        if not positives or not negatives:
            continue

        for positive in positives:
            sampled_negatives = negatives
            if negatives_per_positive > 0 and len(negatives) > negatives_per_positive:
                sampled_negatives = rng.sample(negatives, negatives_per_positive)
            for negative in sampled_negatives:
                if negative != positive:
                    triplets.append((query, positive, negative))

    rng.shuffle(triplets)
    return triplets


def _collate_triplets(batch: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    return {
        "queries": [row[0] for row in batch],
        "positives": [row[1] for row in batch],
        "negatives": [row[2] for row in batch],
    }


def train_bge(config: Any) -> None:
    model_dir = config.dataset_dir / "models" / "bge_finetuned"
    marker = model_dir / "train_summary.json"
    params = {
        "epochs": config.bge_epochs,
        "batch_size": config.bge_train_batch_size,
        "grad_accum": config.bge_grad_accum,
        "lr": config.bge_lr,
        "warmup_ratio": config.bge_warmup_ratio,
        "max_length": config.bge_max_length,
        "max_train_examples": config.bge_max_train_examples,
        "negatives_per_example": config.bge_negatives_per_example,
        "use_amp": bool(config.bge_use_amp),
        "gradient_checkpointing": bool(config.bge_gradient_checkpointing),
        "auto_batch_reduce": bool(config.bge_auto_batch_reduce),
    }
    if is_complete(marker, expected={"model": config.dense_model, "params": params}) and not config.force:
        skip(model_dir)
        return

    pairs_path = config.dataset_dir / "negatives" / "bge_train_ready.jsonl"
    triplets_path = config.dataset_dir / "negatives" / "bge_triplets.jsonl"
    ready_rows = read_jsonl(pairs_path)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    train_qids = {str(qid) for qid in splits["train"]}
    ready_qids = {str(row.get("qid", "")) for row in ready_rows if str(row.get("qid", "")).strip()}
    leaked_ready_qids = sorted(ready_qids - train_qids)
    if leaked_ready_qids:
        raise ValueError(
            "BGE training ready cache contains qids outside train split. "
            f"Re-run mine_hard_negatives/sample_negatives with --force true. Examples: {leaked_ready_qids[:10]}"
        )
    train_rows = [row for row in ready_rows if str(row["qid"]) in train_qids]
    if not train_rows:
        raise ValueError(f"No BGE training rows found for train split in {pairs_path}")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    if triplets_path.exists():
        all_triplet_rows = read_jsonl(triplets_path)
        triplet_qids = {str(row.get("qid", "")) for row in all_triplet_rows if str(row.get("qid", "")).strip()}
        leaked_triplet_qids = sorted(triplet_qids - train_qids)
        if leaked_triplet_qids:
            raise ValueError(
                "BGE triplet cache contains qids outside train split. "
                f"Re-run sample_negatives with --force true. Examples: {leaked_triplet_qids[:10]}"
            )
        triplet_rows = [row for row in all_triplet_rows if str(row["qid"]) in train_qids]
        triplets = [
            (str(row["query"]), str(row["positive"]), str(row["negative"]))
            for row in triplet_rows
            if str(row.get("query", "")).strip()
            and str(row.get("positive", "")).strip()
            and str(row.get("negative", "")).strip()
        ]
    else:
        triplet_rows = []
        triplets = _build_triplets(train_rows, config.bge_negatives_per_example, config.seed)

    if config.bge_max_train_examples > 0:
        triplets = triplets[: config.bge_max_train_examples]
    if not triplets:
        raise ValueError("No BGE training triplets were built.")

    print(f"[train_bge] first_sample={triplets[0]!r}")

    ensure_dir(model_dir)

    print(f"[train_bge] GPU memory note:")
    print(f"  - Total triplets: {len(triplets)}")
    print(f"  - Batch size: {config.bge_train_batch_size} (reduce if OOM)")
    print(f"  - Grad accumulation: {max(int(config.bge_grad_accum), 1)}")
    print(f"  - Epochs: {config.bge_epochs}")
    print(f"  - Max length: {config.bge_max_length}")
    print(f"  - AMP enabled: {config.bge_use_amp}")
    print(f"  - Gradient checkpointing: {config.bge_gradient_checkpointing}")
    print(f"  - Auto batch reduce: {config.bge_auto_batch_reduce}")
    print(f"  - Use GPU: {config.device == 'cuda'}")
    print(f"  - Custom contrastive loop encodes query/positive/negative in separate forwards to reduce peak memory")
    print(f"  - If CUDA OOM persists: try --bge_train_batch_size 2 --bge_max_length 384")

    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")

    def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_texts(texts: list[str], encoder: Any, tokenizer: Any) -> Any:
        inputs = tokenizer(
            [str(text or "") for text in texts],
            padding=True,
            truncation=True,
            max_length=config.bge_max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = encoder(**inputs)
        embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def contrastive_loss(query_embeddings: Any, positive_embeddings: Any, negative_embeddings: Any) -> Any:
        candidates = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        logits = torch.matmul(query_embeddings, candidates.T) / 0.05
        labels = torch.arange(query_embeddings.size(0), device=device)
        return torch.nn.CrossEntropyLoss()(logits, labels)

    def build_model() -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(config.dense_model)
        encoder = AutoModel.from_pretrained(config.dense_model).to(device)
        if bool(config.bge_gradient_checkpointing) and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
            if getattr(encoder, "config", None) is not None and hasattr(encoder.config, "use_cache"):
                encoder.config.use_cache = False
        return tokenizer, encoder

    def clear_cuda() -> None:
        gc.collect()
        if config.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    batch_size = max(1, int(config.bge_train_batch_size))
    grad_accum = max(1, int(config.bge_grad_accum))
    last_oom: RuntimeError | None = None
    tokenizer = None
    encoder = None
    while batch_size >= 1:
        clear_cuda()
        tokenizer, encoder = build_model()
        train_dataloader = DataLoader(triplets, shuffle=True, batch_size=batch_size, collate_fn=_collate_triplets, num_workers=0)
        optimizer_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / grad_accum))
        warmup_steps = int(optimizer_steps_per_epoch * config.bge_epochs * config.bge_warmup_ratio)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.bge_lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, optimizer_steps_per_epoch * config.bge_epochs),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.bge_use_amp and device.type == "cuda"))
        try:
            print(f"[train_bge] training with batch_size={batch_size}")
            encoder.train()
            for epoch in range(config.bge_epochs):
                total_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for step, batch in enumerate(train_dataloader, start=1):
                    with torch.cuda.amp.autocast(enabled=bool(config.bge_use_amp and device.type == "cuda")):
                        query_embeddings = encode_texts(batch["queries"], encoder, tokenizer)
                        positive_embeddings = encode_texts(batch["positives"], encoder, tokenizer)
                        negative_embeddings = encode_texts(batch["negatives"], encoder, tokenizer)
                        loss = contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings)
                        scaled_loss = loss / grad_accum
                    scaler.scale(scaled_loss).backward()
                    total_loss += float(loss.detach().cpu())
                    if step % grad_accum == 0 or step == len(train_dataloader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    if step == 1 or step % 50 == 0 or step == len(train_dataloader):
                        avg_loss = total_loss / step
                        print(f"[train_bge] epoch={epoch + 1}/{config.bge_epochs} step={step}/{len(train_dataloader)} loss={avg_loss:.4f}")
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message or "cuda oom" in message
            if not is_oom or not config.bge_auto_batch_reduce or batch_size == 1:
                raise
            last_oom = exc
            print(f"[warn] CUDA OOM at batch_size={batch_size}; retrying with batch_size={max(1, batch_size // 2)}")
            del encoder
            del tokenizer
            encoder = None
            tokenizer = None
            clear_cuda()
            batch_size //= 2

    if encoder is None or tokenizer is None:
        raise RuntimeError("BGE training failed before model initialization.") from last_oom

    hf_tmp_dir = model_dir / "_hf_transformer"
    ensure_dir(hf_tmp_dir)
    encoder.save_pretrained(hf_tmp_dir)
    tokenizer.save_pretrained(hf_tmp_dir)
    del encoder
    clear_cuda()

    from sentence_transformers import SentenceTransformer, models

    word_embedding_model = models.Transformer(str(hf_tmp_dir), max_seq_length=config.bge_max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    st_model.save(str(model_dir))
    expected_model_files = ["modules.json", "config_sentence_transformers.json"]
    missing_model_files = [name for name in expected_model_files if not (model_dir / name).exists()]
    if missing_model_files:
        raise RuntimeError(f"BGE model save appears incomplete. Missing files: {missing_model_files}")

    write_json(
        marker,
        {
            "status": "complete",
            "base_model": config.dense_model,
            "checkpoint_path": str(model_dir),
            "pairs_path": str(pairs_path),
            "triplets_path": str(triplets_path) if triplets_path.exists() else "",
            "num_ready_rows": len(ready_rows),
            "num_train_rows": len(train_rows),
            "num_triplet_rows": len(triplet_rows),
            "num_examples": len(triplets),
            "train_qids": len(train_qids),
            "effective_batch_size": batch_size,
            "params": params,
        },
    )
    mark_done(
        marker,
        config=config,
        stage="train_bge",
        input_hash=stable_hash({"examples": len(triplets), "qids": sorted(train_qids)}),
        model=config.dense_model,
        params=params,
        fmt="json",
    )
    saved(marker)
