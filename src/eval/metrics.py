from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


DEFAULT_RANKING_KS = [1, 3, 5, 10, 20, 50]
DEFAULT_NDCG_KS = [3, 5, 10, 20, 50]
DEFAULT_TOPK_TUNE_KS = [1, 2, 3, 4, 5]


def group_ranked(rows: list[dict[str, Any]], score_field: str | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped_by_aid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        qid = row["qid"]
        aid = row["aid"]
        current = grouped_by_aid[qid].get(aid)
        if current is None:
            grouped_by_aid[qid][aid] = row
            continue
        if score_field is None:
            continue
        if float(row.get(score_field, 0.0)) > float(current.get(score_field, 0.0)):
            grouped_by_aid[qid][aid] = row

    grouped = {qid: list(items_by_aid.values()) for qid, items_by_aid in grouped_by_aid.items()}
    if score_field:
        for qid, items in grouped.items():
            grouped[qid] = sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)
    return grouped


def aggregate_by_aid_max(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped = group_ranked(rows, score_field=score_field)
    output: list[dict[str, Any]] = []
    for items in grouped.values():
        output.extend(items)
    return output


def recall_at_k(rows: list[dict[str, Any]], question: dict[str, Any], k: int) -> float:
    grouped = group_ranked(rows)
    hits = {row["aid"] for row in grouped.get(question["qid"], [])[:k]}
    positives = set(question["relevant_laws"])
    return len(hits & positives) / max(len(positives), 1)


def _dcg(labels: list[int]) -> float:
    return sum(label / math.log2(idx + 2) for idx, label in enumerate(labels))


def ranking_metrics(rows: list[dict[str, Any]], questions: list[dict[str, Any]], ks: list[int] | None = None) -> dict[str, float]:
    ks = ks or DEFAULT_RANKING_KS
    score_field = _infer_score_field(rows)
    grouped = group_ranked(rows, score_field=score_field)
    totals: dict[str, float] = defaultdict(float)
    count = max(len(questions), 1)

    for question in questions:
        positives = set(question["relevant_laws"])
        ranked = grouped.get(question["qid"], [])
        aids = [row["aid"] for row in ranked]
        for k in ks:
            top = aids[:k]
            hits = len(set(top) & positives)
            totals[f"hit@{k}"] += 1.0 if hits > 0 else 0.0
            totals[f"recall@{k}"] += hits / max(len(positives), 1)
        for k in DEFAULT_NDCG_KS:
            labels = [1 if aid in positives else 0 for aid in aids[:k]]
            ideal = [1] * min(len(positives), k)
            denom = _dcg(ideal)
            totals[f"ndcg@{k}"] += (_dcg(labels) / denom) if denom > 0 else 0.0

    return {key: value / count for key, value in totals.items()}


def threshold_metrics(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    score_field: str,
    threshold: float,
    fallback_top_k_if_empty: int = 5,
    max_candidates_per_query: int = 5,
) -> dict[str, float]:
    positives_by_qid = {row["qid"]: set(row["relevant_laws"]) for row in questions}
    tp = fp = fn = 0
    predicted_by_qid: dict[str, set[str]] = defaultdict(set)
    grouped = group_ranked(rows, score_field=score_field)

    for qid, items in grouped.items():
        predicted = {
            row["aid"]
            for row in items
            if float(row.get(score_field, 0.0)) >= threshold
        }
        if not predicted and fallback_top_k_if_empty > 0 and items:
            predicted = {row["aid"] for row in items[:fallback_top_k_if_empty]}
        if len(predicted) > max_candidates_per_query and items:
            top_items = sorted(items, key=lambda r: float(r.get(score_field, 0.0)), reverse=True)[:max_candidates_per_query]
            predicted = {row["aid"] for row in top_items}
        predicted_by_qid[qid] = predicted

    for qid, positives in positives_by_qid.items():
        predicted = predicted_by_qid.get(qid, set())
        tp += len(predicted & positives)
        fp += len(predicted - positives)
        fn += len(positives - predicted)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta2 = 4
    f2 = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f2": f2}


def topk_metrics(rows: list[dict[str, Any]], questions: list[dict[str, Any]], *, score_field: str, k: int) -> dict[str, float]:
    positives_by_qid = {row["qid"]: set(row["relevant_laws"]) for row in questions}
    grouped = group_ranked(rows, score_field=score_field)
    tp = fp = fn = 0
    for qid, positives in positives_by_qid.items():
        predicted = {row["aid"] for row in grouped.get(qid, [])[:k]}
        tp += len(predicted & positives)
        fp += len(predicted - positives)
        fn += len(positives - predicted)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta2 = 4
    f2 = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"f2@{k}": f2,
    }


def tune_top_k(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    score_field: str,
    candidate_ks: list[int] | None = None,
) -> dict[str, float]:
    if candidate_ks is None:
        grouped = group_ranked(rows, score_field=score_field)
        max_candidates = max((len(items) for items in grouped.values()), default=0)
        candidate_ks = [k for k in DEFAULT_TOPK_TUNE_KS if k <= max_candidates]
        if max_candidates and max_candidates not in candidate_ks:
            candidate_ks.append(max_candidates)
    candidate_ks = sorted({int(k) for k in candidate_ks if int(k) > 0})
    if not candidate_ks:
        candidate_ks = [1]

    best: dict[str, float] | None = None
    for k in candidate_ks:
        metrics = topk_metrics(rows, questions, score_field=score_field, k=k)
        trial = {"top_k": int(k), **metrics}
        if best is None or (trial["f2"], trial["recall"], trial["precision"], -trial["top_k"]) > (
            best["f2"],
            best["recall"],
            best["precision"],
            -best["top_k"],
        ):
            best = trial
    return best or {"top_k": 1, "precision": 0.0, "recall": 0.0, "f2": 0.0, "precision@1": 0.0, "recall@1": 0.0, "f2@1": 0.0}


def tune_threshold(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    score_field: str,
    candidate_thresholds: list[float] | None = None,
) -> dict[str, float]:
    thresholds = candidate_thresholds
    if thresholds is None:
        scores = sorted({float(row.get(score_field, 0.0)) for row in aggregate_by_aid_max(rows, score_field)})
        thresholds = scores or [0.0]

    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = threshold_metrics(rows, questions, score_field=score_field, threshold=threshold)
        trial = {"threshold": float(threshold), **metrics}
        if best is None or (trial["f2"], trial["recall"], trial["precision"]) > (best["f2"], best["recall"], best["precision"]):
            best = trial
    return best or {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f2": 0.0}


def _infer_score_field(rows: list[dict[str, Any]]) -> str | None:
    for field in ["llm_rerank_score", "rerank_score", "qwen_rerank_score", "hybrid_score", "bge_score_norm", "bm25_score_norm", "bge_score", "bm25_score"]:
        if rows and field in rows[0]:
            return field
    return None


