from __future__ import annotations

import random
from typing import Any

from src.data.loaders import load_questions
from src.utils.artifact import is_complete, mark_done, prepared_dir, read_json, read_jsonl, read_table, stable_hash, write_jsonl
from src.utils.logging import saved, skip


POLICIES = {
    "bge_train_pairs.jsonl": [(20, 40, 4), (40, 60, 4)],
    "rerank_train_pairs.jsonl": [(4, 10, 5), (10, 20, 7), (20, 30, 4)],
    "qwen_train_pairs.jsonl":   [(4, 10, 5), (10, 20, 7), (20, 30, 4)],
}

READY_FILES = {
    "bge_train_pairs.jsonl": "bge_train_ready.jsonl",
    "rerank_train_pairs.jsonl": "rerank_train_ready.jsonl",
    "qwen_train_pairs.jsonl": "qwen_train_ready.jsonl",
}

DERIVED_READY_FILES = {
    "bge_triplets.jsonl",
    "rerank_pairwise.jsonl",
    "qwen_prompt_train.jsonl",
}

QWEN_PROMPT_TEMPLATE = """Bạn là hệ thống đánh giá mức độ liên quan của văn bản pháp luật.
Hãy phân loại đoạn văn bản sau có liên quan đến câu hỏi hay không.
Chỉ trả lời đúng một ký tự: 1 nếu liên quan, 0 nếu không liên quan.

Câu hỏi:
{question}

Văn bản:
{article}

Nhãn:"""


def _sample_ranges(candidates: list[dict[str, Any]], policy: list[tuple[int, int, int]], rng: random.Random) -> list[dict[str, Any]]:
    sampled = []
    target_count = sum(count for _lo, _hi, count in policy)
    for lo, hi, count in policy:
        bucket = [row for row in candidates if lo <= int(row["rank"]) <= hi]
        sampled.extend(rng.sample(bucket, min(count, len(bucket))))

    seen = {str(row["chunk_id"]) for row in sampled}
    if len(sampled) < target_count:
        filler = [row for row in sorted(candidates, key=lambda item: int(item["rank"])) if str(row["chunk_id"]) not in seen]
        sampled.extend(filler[: target_count - len(sampled)])
    return sampled


def _chunk_text_by_id(config: Any) -> dict[str, str]:
    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    return {str(row["chunk_id"]): str(row["text"]) for row in chunks}


def _assert_pair_ground_truth(pair_rows: list[dict[str, Any]], chunk_to_aid: dict[str, str], *, name: str) -> None:
    errors = []
    for row in pair_rows:
        qid = str(row["qid"])
        positive_aid = str(row["positive_aid"])
        for chunk_id in row.get("positive_chunk_ids", []):
            mapped_aid = str(chunk_to_aid.get(str(chunk_id), ""))
            if mapped_aid != positive_aid:
                errors.append(f"qid={qid} positive_aid={positive_aid} chunk_id={chunk_id} maps_to={mapped_aid}")
        for aid, chunk_id in zip(row.get("negative_aids", []), row.get("negative_chunk_ids", [])):
            mapped_aid = str(chunk_to_aid.get(str(chunk_id), ""))
            if mapped_aid != str(aid):
                errors.append(f"qid={qid} negative_aid={aid} chunk_id={chunk_id} maps_to={mapped_aid}")
            if str(aid) == positive_aid:
                errors.append(f"qid={qid} negative_aid equals positive_aid={positive_aid}")
    if errors:
        raise AssertionError(f"{name} has ground-truth/chunk mapping errors: " + "; ".join(errors[:10]))


def _assert_ready_ground_truth(ready_rows: list[dict[str, Any]], chunk_to_aid: dict[str, str], *, name: str) -> None:
    errors = []
    for row in ready_rows:
        positive_aid = str(row["positive_aid"])
        if "pos" in row:
            for passage in row.get("pos", []):
                mapped_aid = str(chunk_to_aid.get(str(passage["chunk_id"]), ""))
                if mapped_aid != positive_aid:
                    errors.append(f"qid={row['qid']} positive_aid={positive_aid} chunk_id={passage['chunk_id']} maps_to={mapped_aid}")
            for passage in row.get("neg", []):
                mapped_aid = str(chunk_to_aid.get(str(passage["chunk_id"]), ""))
                if mapped_aid != str(passage["aid"]) or str(passage["aid"]) == positive_aid:
                    errors.append(f"qid={row['qid']} negative_aid={passage['aid']} chunk_id={passage['chunk_id']} maps_to={mapped_aid} positive_aid={positive_aid}")
        for passage in row.get("passages", []):
            mapped_aid = str(chunk_to_aid.get(str(passage["chunk_id"]), ""))
            expected_label = 1 if str(passage["aid"]) == positive_aid else 0
            if mapped_aid != str(passage["aid"]):
                errors.append(f"qid={row['qid']} aid={passage['aid']} chunk_id={passage['chunk_id']} maps_to={mapped_aid}")
            if int(passage["label"]) != expected_label:
                errors.append(f"qid={row['qid']} aid={passage['aid']} label={passage['label']} expected={expected_label}")
    if errors:
        raise AssertionError(f"{name} has ground-truth label errors: " + "; ".join(errors[:10]))


def _build_ready_rows(pair_rows: list[dict[str, Any]], chunk_text: dict[str, str], kind: str) -> list[dict[str, Any]]:
    ready_rows = []
    for row in pair_rows:
        positive_passages = [
            {"chunk_id": chunk_id, "text": chunk_text[chunk_id]}
            for chunk_id in row.get("positive_chunk_ids", [])
            if chunk_id in chunk_text and chunk_text[chunk_id].strip()
        ]
        negative_passages = [
            {"chunk_id": chunk_id, "aid": aid, "text": chunk_text[chunk_id]}
            for aid, chunk_id in zip(row.get("negative_aids", []), row.get("negative_chunk_ids", []))
            if chunk_id in chunk_text and chunk_text[chunk_id].strip()
        ]
        positive_ids = {item["chunk_id"] for item in positive_passages}
        negative_passages = [item for item in negative_passages if item["chunk_id"] not in positive_ids]
        if not positive_passages or not negative_passages:
            continue

        if kind == "bge":
            ready_rows.append(
                {
                    "qid": row["qid"],
                    "query": row["question"],
                    "positive_aid": row["positive_aid"],
                    "pos": positive_passages,
                    "neg": negative_passages,
                }
            )
        else:
            labeled_passages = [
                {"chunk_id": item["chunk_id"], "aid": row["positive_aid"], "text": item["text"], "label": 1}
                for item in positive_passages
            ]
            labeled_passages.extend(
                {"chunk_id": item["chunk_id"], "aid": item["aid"], "text": item["text"], "label": 0}
                for item in negative_passages
            )
            ready_rows.append(
                {
                    "qid": row["qid"],
                    "query": row["question"],
                    "positive_aid": row["positive_aid"],
                    "passages": labeled_passages,
                }
            )
    return ready_rows


def _build_bge_triplets(ready_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triplets = []
    for row in ready_rows:
        for positive in row.get("pos", []):
            for negative in row.get("neg", []):
                if positive["text"] == negative["text"]:
                    continue
                triplets.append(
                    {
                        "qid": row["qid"],
                        "query": row["query"],
                        "positive": positive["text"],
                        "negative": negative["text"],
                        "positive_chunk_id": positive["chunk_id"],
                        "negative_chunk_id": negative["chunk_id"],
                    }
                )
    return triplets


def _build_pairwise_rows(ready_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairwise = []
    for row in ready_rows:
        for passage in row.get("passages", []):
            pairwise.append(
                {
                    "qid": row["qid"],
                    "query": row["query"],
                    "doc": passage["text"],
                    "chunk_id": passage["chunk_id"],
                    "aid": passage["aid"],
                    "label": int(passage["label"]),
                }
            )
    return pairwise


def _build_qwen_prompt_rows(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "qid": row["qid"],
            "prompt": QWEN_PROMPT_TEMPLATE.format(question=row["query"], article=row["doc"]),
            "label": f" {int(row['label'])}",
            "label_id": int(row["label"]),
            "chunk_id": row["chunk_id"],
            "aid": row["aid"],
        }
        for row in pairwise_rows
    ]


def sample_negatives(config: Any) -> None:
    neg_dir = config.dataset_dir / "negatives"
    done_targets = (
        [neg_dir / name for name in POLICIES]
        + [neg_dir / name for name in READY_FILES.values()]
        + [neg_dir / name for name in DERIVED_READY_FILES]
    )
    expected_params = {"seed": config.seed, "top_k": config.top_k, "positive_chunks_per_aid": config.positive_chunks_per_aid}
    expected = {"params": expected_params}
    if all(is_complete(path, expected=expected) for path in done_targets) and not config.force:
        skip(neg_dir)
        return

    questions = {row["qid"]: row for row in load_questions(config)}
    mined = {row["qid"]: row for row in read_jsonl(neg_dir / "hard_negative_top100_by_qid.jsonl")}
    splits = read_json(prepared_dir(config) / "splits.json")
    train_qids = {str(qid) for qid in splits.get("train", [])}
    missing_mined_qids = sorted(train_qids - {str(qid) for qid in mined})
    if missing_mined_qids:
        preview = ", ".join(missing_mined_qids[:10])
        raise ValueError(
            "Hard negative cache is missing train qids. "
            f"Re-run mine_hard_negatives with --force true. Missing qids: {preview}"
        )

    leaked_qids = sorted(str(qid) for qid in mined if str(qid) not in train_qids)
    if leaked_qids:
        preview = ", ".join(leaked_qids[:10])
        raise ValueError(
            "Hard negative cache contains qids outside the train split. "
            f"Re-run mine_hard_negatives with --force true. Offending qids: {preview}"
        )

    chunk_text = _chunk_text_by_id(config)
    chunk_to_aid = {str(chunk_id): str(aid) for chunk_id, aid in read_json(prepared_dir(config) / "chunk_to_aid.json").items()}
    rng = random.Random(config.seed)

    for filename, policy in POLICIES.items():
        path = neg_dir / filename
        ready_path = neg_dir / READY_FILES[filename]
        rows = []
        for qid, mined_row in mined.items():
            candidates = mined_row.get("hard_negatives", mined_row.get("candidates", []))
            question = questions[qid]
            negatives = _sample_ranges(candidates, policy, rng)
            for positive_aid in question["relevant_laws"]:
                positive_chunks = mined_row.get("positive_chunks_by_aid", {}).get(str(positive_aid), [])
                rows.append(
                    {
                        "qid": qid,
                        "question": question["question"],
                        "positive_aid": positive_aid,
                        "positive_chunk_ids": [row["chunk_id"] for row in positive_chunks],
                        "positive_chunks": positive_chunks,
                        "negative_aids": [row["aid"] for row in negatives],
                        "negative_chunk_ids": [row["chunk_id"] for row in negatives],
                    }
                )
        _assert_pair_ground_truth(rows, chunk_to_aid, name=filename)
        write_jsonl(path, rows)
        kind = "bge" if filename.startswith("bge") else "rerank"
        ready_rows = _build_ready_rows(rows, chunk_text, kind)
        _assert_ready_ground_truth(ready_rows, chunk_to_aid, name=ready_path.name)
        write_jsonl(ready_path, ready_rows)
        mark_done(path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": filename, "rows": len(rows)}), params=expected_params, fmt="jsonl")
        mark_done(ready_path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": ready_path.name, "rows": len(ready_rows)}), params=expected_params, fmt="jsonl")
        if filename == "bge_train_pairs.jsonl":
            triplet_path = neg_dir / "bge_triplets.jsonl"
            triplets = _build_bge_triplets(ready_rows)
            write_jsonl(triplet_path, triplets)
            mark_done(triplet_path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": triplet_path.name, "rows": len(triplets)}), params=expected_params, fmt="jsonl")
            saved(triplet_path)
        elif filename == "rerank_train_pairs.jsonl":
            pairwise_path = neg_dir / "rerank_pairwise.jsonl"
            pairwise_rows = _build_pairwise_rows(ready_rows)
            write_jsonl(pairwise_path, pairwise_rows)
            mark_done(pairwise_path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": pairwise_path.name, "rows": len(pairwise_rows)}), params=expected_params, fmt="jsonl")
            saved(pairwise_path)
        elif filename == "qwen_train_pairs.jsonl":
            prompt_path = neg_dir / "qwen_prompt_train.jsonl"
            pairwise_rows = _build_pairwise_rows(ready_rows)
            prompt_rows = _build_qwen_prompt_rows(pairwise_rows)
            write_jsonl(prompt_path, prompt_rows)
            mark_done(prompt_path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": prompt_path.name, "rows": len(prompt_rows)}), params=expected_params, fmt="jsonl")
            saved(prompt_path)
        saved(path)
        saved(ready_path)
