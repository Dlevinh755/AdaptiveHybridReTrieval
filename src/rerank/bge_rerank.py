from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.data.loaders import load_questions
from src.eval.metrics import aggregate_by_aid_max, ranking_metrics, threshold_metrics, topk_metrics, tune_threshold, tune_top_k
from src.utils.artifact import eval_dir, is_complete, mark_done, prepared_dir, read_json, read_table, retrieval_dir, stable_hash, write_json, write_table
from src.utils.logging import saved, skip


RERANK_BGE_SCHEMA_VERSION = 3
RERANK_BGE_SOURCE = "hybrid_router"


def _limit_candidates(rows: list[dict[str, Any]], top_k: int, score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)
    limited = []
    for items in grouped.values():
        limited.extend(sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)[:top_k])
    return limited


def _add_query_minmax_score(rows: list[dict[str, Any]], raw_field: str, norm_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["qid"]), []).append(dict(row))

    normalized_rows: list[dict[str, Any]] = []
    for items in grouped.values():
        scores = [float(item.get(raw_field, 0.0)) for item in items]
        lo = min(scores) if scores else 0.0
        hi = max(scores) if scores else 0.0
        denom = hi - lo
        for item in items:
            score = float(item.get(raw_field, 0.0))
            item[norm_field] = (score - lo) / denom if abs(denom) > 1e-12 else 0.0
            normalized_rows.append(item)
    return normalized_rows


def _chunk_text_by_id(config: Any) -> dict[str, str]:
    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    return {str(row["chunk_id"]): row["text"] for row in chunks}


def _expand_aid_rows_to_chunks(config: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aid_to_chunks = read_json(prepared_dir(config) / "aid2chunks.json")
    chunk_text = _chunk_text_by_id(config)
    expanded: list[dict[str, Any]] = []
    for row in rows:
        aid = str(row["aid"])
        chunk_ids = [str(chunk_id) for chunk_id in aid_to_chunks.get(aid, [])]
        if row.get("chunk_id") and str(row["chunk_id"]) not in chunk_ids:
            chunk_ids.insert(0, str(row["chunk_id"]))
        for chunk_id in chunk_ids:
            text = chunk_text.get(chunk_id, "")
            if not text:
                continue
            expanded.append(
                {
                    **row,
                    "aid": aid,
                    "chunk_id": chunk_id,
                    "chunk_text": text,
                    "candidate_unit": "chunk",
                }
            )
    return expanded


def _hash_vector(text: str, dim: int = 256) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _score(query: str, text: str) -> float:
    return float(_hash_vector(query) @ _hash_vector(text))


def _cross_encoder_scores(config: Any, pairs: list[tuple[str, str]]) -> list[float] | None:
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(config.rerank_model, device=config.device)
        scores = model.predict(pairs, batch_size=config.batch_size, show_progress_bar=True)
        return [float(score) for score in scores]
    except Exception as exc:
        print(f"[warn] CrossEncoder rerank unavailable, falling back to hash score: {exc}")
        return None


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if str(row["qid"]) in qids]


def _assert_max_candidates(rows: list[dict[str, Any]], *, top_k: int, label: str) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        qid = str(row["qid"])
        counts[qid] = counts.get(qid, 0) + 1
    overflow = {qid: count for qid, count in counts.items() if count > top_k}
    if overflow:
        examples = sorted(overflow.items(), key=lambda item: item[1], reverse=True)[:5]
        raise AssertionError(f"{label} exceeds top_k={top_k}. Examples: {examples}")


def _assert_no_duplicate_chunks(rows: list[dict[str, Any]], *, label: str) -> None:
    seen_qid_chunk: set[tuple[str, str]] = set()
    seen_qid_aid_chunk: set[tuple[str, str, str]] = set()
    duplicate_qid_chunk = 0
    duplicate_qid_aid_chunk = 0
    for row in rows:
        qid = str(row["qid"])
        aid = str(row["aid"])
        chunk_id = str(row.get("chunk_id", ""))
        key_qid_chunk = (qid, chunk_id)
        key_qid_aid_chunk = (qid, aid, chunk_id)
        if key_qid_chunk in seen_qid_chunk:
            duplicate_qid_chunk += 1
        else:
            seen_qid_chunk.add(key_qid_chunk)
        if key_qid_aid_chunk in seen_qid_aid_chunk:
            duplicate_qid_aid_chunk += 1
        else:
            seen_qid_aid_chunk.add(key_qid_aid_chunk)
    if duplicate_qid_chunk or duplicate_qid_aid_chunk:
        raise AssertionError(
            f"{label} has duplicate chunk candidates: "
            f"duplicated(qid, chunk_id)={duplicate_qid_chunk}, "
            f"duplicated(qid, aid, chunk_id)={duplicate_qid_aid_chunk}"
        )


def _assert_valid_rerank_inputs(
    config: Any,
    rows: list[dict[str, Any]],
    questions: dict[str, str],
    *,
    label: str,
) -> None:
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    missing_query = []
    empty_query = []
    empty_chunk_text = []
    chunk_aid_mismatch = []
    for row in rows:
        qid = str(row["qid"])
        aid = str(row["aid"])
        chunk_id = str(row.get("chunk_id", ""))
        if qid not in questions:
            missing_query.append(qid)
        elif not str(questions[qid]).strip():
            empty_query.append(qid)
        if not str(row.get("chunk_text", "")).strip():
            empty_chunk_text.append((qid, aid, chunk_id))
        mapped_aid = str(chunk_to_aid.get(chunk_id, ""))
        if mapped_aid and mapped_aid != aid:
            chunk_aid_mismatch.append((qid, aid, chunk_id, mapped_aid))
    errors = []
    if missing_query:
        errors.append(f"missing_query={missing_query[:5]}")
    if empty_query:
        errors.append(f"empty_query={empty_query[:5]}")
    if empty_chunk_text:
        errors.append(f"empty_chunk_text={empty_chunk_text[:5]}")
    if chunk_aid_mismatch:
        errors.append(f"chunk_aid_mismatch={chunk_aid_mismatch[:5]}")
    if errors:
        raise AssertionError(f"{label} invalid rerank inputs: " + "; ".join(errors))


def _assert_source_subset(source_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], *, top_k: int, split_name: str) -> None:
    source_top = _limit_candidates(source_rows, top_k, "hybrid_score" if source_rows and "hybrid_score" in source_rows[0] else "bge_score")
    source_pairs = {(str(row["qid"]), str(row["aid"])) for row in source_top}
    candidate_pairs = {(str(row["qid"]), str(row["aid"])) for row in candidate_rows}
    extra = sorted(candidate_pairs - source_pairs)[:5]
    if extra:
        raise AssertionError(f"bge_rerank {split_name} candidates are not subset of hybrid_router top{top_k}: {extra}")


def _comparison_with_hybrid_router(config: Any, split_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    baseline_path = eval_dir(config) / f"hybrid_router_{split_name}_metrics.json"
    if not baseline_path.exists():
        return {}
    baseline = read_json(baseline_path)
    baseline_ranking = baseline.get("ranking", {})
    baseline_threshold = baseline.get("threshold", {})
    ranking = payload.get("ranking", {})
    threshold = payload.get("threshold", {})
    return {
        "baseline": "hybrid_router",
        "f2_delta": float(threshold.get("f2", 0.0)) - float(baseline_threshold.get("f2", 0.0)),
        "precision_delta": float(threshold.get("precision", 0.0)) - float(baseline_threshold.get("precision", 0.0)),
        "recall_delta": float(threshold.get("recall", 0.0)) - float(baseline_threshold.get("recall", 0.0)),
        "recall@10_delta": float(ranking.get("recall@10", 0.0)) - float(baseline_ranking.get("recall@10", 0.0)),
        "ndcg@10_delta": float(ranking.get("ndcg@10", 0.0)) - float(baseline_ranking.get("ndcg@10", 0.0)),
    }


def _write_rerank_metrics(
    config: Any,
    split_rows: dict[str, list[dict[str, Any]]],
    split_questions: dict[str, list[dict[str, Any]]],
    *,
    params: dict[str, Any],
    input_hash: str,
) -> None:
    normalized_split_rows = {
        split_name: _add_query_minmax_score(rows, "rerank_score", "rerank_score_norm")
        for split_name, rows in split_rows.items()
    }
    aid_aggregated_rows = {
        split_name: aggregate_by_aid_max(rows, score_field="rerank_score_norm")
        for split_name, rows in normalized_split_rows.items()
    }
    threshold_info = tune_threshold(aid_aggregated_rows["val"], split_questions["val"], score_field="rerank_score_norm")
    threshold = float(threshold_info["threshold"])
    topk_info = tune_top_k(aid_aggregated_rows["val"], split_questions["val"], score_field="rerank_score_norm")
    tuned_top_k = int(topk_info["top_k"])
    metrics_by_split: dict[str, Any] = {}
    for split_name in ["val", "test"]:
        rows = aid_aggregated_rows[split_name]
        questions = split_questions[split_name]
        payload = {
            "split": split_name,
            "score_field": "rerank_score_norm",
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "ranking": ranking_metrics(rows, questions),
            "threshold": {
                "threshold": threshold,
                **threshold_metrics(rows, questions, score_field="rerank_score_norm", threshold=threshold),
            },
            "topk_fixed_3": {
                "top_k": 3,
                **topk_metrics(rows, questions, score_field="rerank_score_norm", k=3),
            },
            "topk_tuned": {
                "top_k": tuned_top_k,
                **topk_metrics(rows, questions, score_field="rerank_score_norm", k=tuned_top_k),
            },
            "topk_selection_split": "val",
            "threshold_selection_split": "val",
            "source": RERANK_BGE_SOURCE,
            "params": params,
            "num_questions": len(questions),
            "num_rows": len(rows),
        }
        payload["comparison"] = _comparison_with_hybrid_router(config, split_name, payload)
        metrics_path = eval_dir(config) / f"bge_rerank_{split_name}_metrics.json"
        write_json(metrics_path, payload)
        mark_done(metrics_path, config=config, stage="rerank_bge_metrics", input_hash=input_hash, model=config.rerank_model, params={**params, "split": split_name}, fmt="json")
        saved(metrics_path)
        metrics_by_split[split_name] = payload

    threshold_path = eval_dir(config) / "bge_rerank_threshold.json"
    write_json(
        threshold_path,
        {
            "selection_split": "val",
            "score_field": "rerank_score_norm",
            "best_threshold": threshold,
            "val": threshold_info,
            "best_top_k": tuned_top_k,
            "topk_val": topk_info,
            "test": metrics_by_split["test"]["threshold"],
            "topk_test": metrics_by_split["test"]["topk_tuned"],
            "source": RERANK_BGE_SOURCE,
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "params": params,
        },
    )
    mark_done(threshold_path, config=config, stage="rerank_bge_threshold", input_hash=input_hash, model=config.rerank_model, params=params, fmt="json")
    saved(threshold_path)


def rerank_bge(config: Any) -> None:
    path = retrieval_dir(config) / "bge_rerank_scores.parquet"
    split_paths = {
        "val": retrieval_dir(config) / "bge_rerank_scores_val.parquet",
        "test": retrieval_dir(config) / "bge_rerank_scores_test.parquet",
    }
    metric_paths = [
        eval_dir(config) / "bge_rerank_val_metrics.json",
        eval_dir(config) / "bge_rerank_test_metrics.json",
        eval_dir(config) / "bge_rerank_threshold.json",
    ]
    expected_params = {
        "schema_version": RERANK_BGE_SCHEMA_VERSION,
        "candidate_top_k": config.candidate_top_k,
        "candidate_unit": "chunk",
        "ranking_unit": "aid",
        "source": RERANK_BGE_SOURCE,
        "splits": ["val", "test"],
    }
    expected_aggregate = {"model": config.rerank_model, "params": {**expected_params, "split": "aggregate"}}
    expected_by_split = {
        split_name: {"model": config.rerank_model, "params": {**expected_params, "split": split_name}}
        for split_name in split_paths
    }
    if (
        is_complete(path, expected=expected_aggregate)
        and all(is_complete(split_path, expected=expected_by_split[split_name]) for split_name, split_path in split_paths.items())
        and all(is_complete(metric_path) for metric_path in metric_paths)
        and not config.force
    ):
        skip(path)
        return

    all_questions = load_questions(config)
    questions = {str(row["qid"]): row["question"] for row in all_questions}
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    eval_qids_by_split = {
        split_name: {str(qid) for qid in splits[split_name]}
        for split_name in ["val", "test"]
    }
    eval_qids = set().union(*eval_qids_by_split.values())

    rows = []
    source_paths = []
    source_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_qids in eval_qids_by_split.items():
        split_source = retrieval_dir(config) / f"hybrid_router_scores_{split_name}.parquet"
        if split_source.exists():
            split_rows = read_table(split_source)
            source_paths.append(str(split_source))
        else:
            aggregate_source = retrieval_dir(config) / "hybrid_router_scores.parquet"
            if not aggregate_source.exists():
                raise FileNotFoundError(
                    f"Missing hybrid router cache for rerank: expected {split_source} "
                    f"or {aggregate_source}. Run train_router before rerank_bge."
                )
            split_rows = [
                row
                for row in read_table(aggregate_source)
                if str(row["qid"]) in split_qids
            ]
            source_paths.append(str(aggregate_source))
        split_rows = [row for row in split_rows if str(row["qid"]) in split_qids]
        if not split_rows:
            raise ValueError(f"No hybrid_router source rows found for split={split_name}.")
        if "hybrid_score" not in split_rows[0]:
            raise ValueError(f"Rerank source for split={split_name} is not hybrid_router-like: missing hybrid_score.")
        source_rows_by_split[split_name] = split_rows
        rows.extend(_limit_candidates(split_rows, config.candidate_top_k, "hybrid_score"))
    rows = _expand_aid_rows_to_chunks(config, rows)
    rows = _limit_candidates(rows, config.candidate_top_k, "hybrid_score" if rows and "hybrid_score" in rows[0] else "bge_score") if rows else []
    _assert_no_duplicate_chunks(rows, label="bge_rerank input rows")
    _assert_valid_rerank_inputs(config, rows, questions, label="bge_rerank input rows")
    for split_name, split_qids in eval_qids_by_split.items():
        split_candidate_rows = [row for row in rows if str(row["qid"]) in split_qids]
        _assert_source_subset(source_rows_by_split[split_name], split_candidate_rows, top_k=config.candidate_top_k, split_name=split_name)
    pairs = [(questions[str(row["qid"])], row.get("chunk_text", "")) for row in rows]
    scores = _cross_encoder_scores(config, pairs)
    if scores is None:
        scores = [_score(query, text) for query, text in pairs]
    ranked = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["rerank_score"] = score
        ranked.append(item)
    ranked = _limit_candidates(ranked, len(ranked), "rerank_score")
    ranked = _limit_candidates(ranked, config.candidate_top_k, "rerank_score")
    _assert_max_candidates(ranked, top_k=config.candidate_top_k, label="bge_rerank ranked rows")
    _assert_no_duplicate_chunks(ranked, label="bge_rerank ranked rows")

    input_hash = stable_hash({"source": sorted(set(source_paths)), "rows": len(ranked), "split_qids": {key: sorted(value) for key, value in eval_qids_by_split.items()}})
    split_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_path in split_paths.items():
        split_qids = eval_qids_by_split[split_name]
        split_rows = [row for row in ranked if str(row["qid"]) in split_qids]
        _assert_max_candidates(split_rows, top_k=config.candidate_top_k, label=f"bge_rerank {split_name} rows")
        _assert_no_duplicate_chunks(split_rows, label=f"bge_rerank {split_name} rows")
        split_rows_by_name[split_name] = split_rows
        split_rows_with_norm = _add_query_minmax_score(split_rows, "rerank_score", "rerank_score_norm")
        split_fmt = write_table(split_path, split_rows_with_norm)
        mark_done(split_path, config=config, stage="rerank_bge", input_hash=input_hash, model=config.rerank_model, params={**expected_params, "split": split_name}, fmt=split_fmt)
        saved(split_path)

    ranked_with_norm = _add_query_minmax_score(ranked, "rerank_score", "rerank_score_norm")
    fmt = write_table(path, ranked_with_norm)
    mark_done(path, config=config, stage="rerank_bge", input_hash=input_hash, model=config.rerank_model, params={**expected_params, "split": "aggregate"}, fmt=fmt)
    split_questions = {
        split_name: _filter_questions(all_questions, qids)
        for split_name, qids in eval_qids_by_split.items()
    }
    _write_rerank_metrics(config, split_rows_by_name, split_questions, params=expected_params, input_hash=input_hash)
    saved(path)
