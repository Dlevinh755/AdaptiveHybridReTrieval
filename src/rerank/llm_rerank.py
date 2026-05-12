from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics, topk_metrics, tune_threshold, tune_top_k
from src.rerank.bge_rerank import _assert_max_candidates, _assert_no_duplicate_chunks, _limit_candidates
from src.training.llm_rerank_backends import get_backend
from src.training.llm_rerank_backends.unsloth_causal_lm import SCORE_FORMULA
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, read_table, retrieval_dir, stable_hash, write_json, write_table
from src.utils.logging import saved, skip


LLM_RERANK_SCHEMA_VERSION = 1
LLM_RERANK_SOURCE = "bge_rerank"


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if str(row["qid"]) in qids]


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


def _comparison_with_bge_rerank(config: Any, split_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    baseline_path = eval_dir(config) / f"bge_rerank_{split_name}_metrics.json"
    if not baseline_path.exists():
        return {}
    baseline = read_json(baseline_path)
    baseline_ranking = baseline.get("ranking", {})
    baseline_threshold = baseline.get("threshold", {})
    baseline_topk = baseline.get("topk_tuned", {})
    ranking = payload.get("ranking", {})
    threshold = payload.get("threshold", {})
    topk = payload.get("topk_tuned", {})
    return {
        "baseline": "bge_rerank",
        "f2_delta": float(threshold.get("f2", 0.0)) - float(baseline_threshold.get("f2", 0.0)),
        "topk_f2_delta": float(topk.get("f2", 0.0)) - float(baseline_topk.get("f2", 0.0)),
        "recall@10_delta": float(ranking.get("recall@10", 0.0)) - float(baseline_ranking.get("recall@10", 0.0)),
        "ndcg@10_delta": float(ranking.get("ndcg@10", 0.0)) - float(baseline_ranking.get("ndcg@10", 0.0)),
    }


def _write_llm_metrics(
    config: Any,
    split_rows: dict[str, list[dict[str, Any]]],
    split_questions: dict[str, list[dict[str, Any]]],
    *,
    params: dict[str, Any],
    input_hash: str,
) -> None:
    normalized_split_rows = {
        split_name: _add_query_minmax_score(rows, "llm_rerank_score", "llm_rerank_score_norm")
        for split_name, rows in split_rows.items()
    }
    threshold_info = tune_threshold(normalized_split_rows["val"], split_questions["val"], score_field="llm_rerank_score_norm")
    threshold = float(threshold_info["threshold"])
    topk_info = tune_top_k(normalized_split_rows["val"], split_questions["val"], score_field="llm_rerank_score_norm")
    tuned_top_k = int(topk_info["top_k"])
    metrics_by_split: dict[str, Any] = {}
    for split_name in ["val", "test"]:
        rows = normalized_split_rows[split_name]
        questions = split_questions[split_name]
        payload = {
            "split": split_name,
            "score_field": "llm_rerank_score_norm",
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "ranking": ranking_metrics(rows, questions),
            "threshold": {
                "threshold": threshold,
                **threshold_metrics(rows, questions, score_field="llm_rerank_score_norm", threshold=threshold),
            },
            "topk_fixed_3": {
                "top_k": 3,
                **topk_metrics(rows, questions, score_field="llm_rerank_score_norm", k=3),
            },
            "topk_tuned": {
                "top_k": tuned_top_k,
                **topk_metrics(rows, questions, score_field="llm_rerank_score_norm", k=tuned_top_k),
            },
            "topk_selection_split": "val",
            "threshold_selection_split": "val",
            "source": LLM_RERANK_SOURCE,
            "params": params,
            "num_questions": len(questions),
            "num_rows": len(rows),
        }
        payload["comparison"] = _comparison_with_bge_rerank(config, split_name, payload)
        metrics_path = eval_dir(config) / f"llm_rerank_{split_name}_metrics.json"
        write_json(metrics_path, payload)
        mark_done(metrics_path, config=config, stage="rerank_llm_metrics", input_hash=input_hash, model=config.llm_rerank_model, params={**params, "split": split_name}, fmt="json")
        saved(metrics_path)
        metrics_by_split[split_name] = payload

    threshold_path = eval_dir(config) / "llm_rerank_threshold.json"
    write_json(
        threshold_path,
        {
            "selection_split": "val",
            "score_field": "llm_rerank_score_norm",
            "best_threshold": threshold,
            "val": threshold_info,
            "best_top_k": tuned_top_k,
            "topk_val": topk_info,
            "test": metrics_by_split["test"]["threshold"],
            "topk_test": metrics_by_split["test"]["topk_tuned"],
            "source": LLM_RERANK_SOURCE,
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "score_formula": SCORE_FORMULA,
            "params": params,
        },
    )
    mark_done(threshold_path, config=config, stage="rerank_llm_threshold", input_hash=input_hash, model=config.llm_rerank_model, params=params, fmt="json")
    saved(threshold_path)


def _assert_source_subset(source_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], *, top_k: int, split_name: str) -> None:
    source_top = _limit_candidates(source_rows, top_k, "rerank_score")
    source_keys = {(str(row["qid"]), str(row.get("chunk_id", ""))) for row in source_top}
    candidate_keys = {(str(row["qid"]), str(row.get("chunk_id", ""))) for row in candidate_rows}
    extra = sorted(candidate_keys - source_keys)[:5]
    if extra:
        raise AssertionError(f"llm_rerank {split_name} candidates are not subset of bge_rerank top{top_k}: {extra}")


def rerank_llm(config: Any) -> None:
    model_dir = config.dataset_dir / "models" / "llm_reranker"
    if not (model_dir / "train_summary.json").exists():
        raise FileNotFoundError(f"Missing LLM reranker model at {model_dir}. Run train_llm_reranker first.")

    path = retrieval_dir(config) / "llm_rerank_scores.parquet"
    split_paths = {
        "val": retrieval_dir(config) / "llm_rerank_scores_val.parquet",
        "test": retrieval_dir(config) / "llm_rerank_scores_test.parquet",
    }
    metric_paths = [
        eval_dir(config) / "llm_rerank_val_metrics.json",
        eval_dir(config) / "llm_rerank_test_metrics.json",
        eval_dir(config) / "llm_rerank_threshold.json",
    ]
    expected_params = {
        "schema_version": LLM_RERANK_SCHEMA_VERSION,
        "backend": config.llm_rerank_backend,
        "candidate_top_k": config.llm_rerank_top_k,
        "batch_size": config.llm_rerank_batch_size,
        "candidate_unit": "chunk",
        "ranking_unit": "aid",
        "source": LLM_RERANK_SOURCE,
        "score_formula": SCORE_FORMULA,
        "splits": ["val", "test"],
    }
    expected_aggregate = {"model": config.llm_rerank_model, "params": {**expected_params, "split": "aggregate"}}
    expected_by_split = {
        split_name: {"model": config.llm_rerank_model, "params": {**expected_params, "split": split_name}}
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
    eval_qids_by_split = {split_name: {str(qid) for qid in splits[split_name]} for split_name in ["val", "test"]}
    backend = get_backend(config.llm_rerank_backend)

    all_ranked: list[dict[str, Any]] = []
    split_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    source_paths: list[str] = []
    for split_name, split_qids in eval_qids_by_split.items():
        source_path = retrieval_dir(config) / f"bge_rerank_scores_{split_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing BGE rerank cache for LLM rerank: {source_path}")
        source_paths.append(str(source_path))
        source_rows = [row for row in read_table(source_path) if str(row["qid"]) in split_qids]
        rows = _limit_candidates(source_rows, int(config.llm_rerank_top_k), "rerank_score")
        _assert_max_candidates(rows, top_k=int(config.llm_rerank_top_k), label=f"llm_rerank {split_name} input rows")
        _assert_no_duplicate_chunks(rows, label=f"llm_rerank {split_name} input rows")
        _assert_source_subset(source_rows, rows, top_k=int(config.llm_rerank_top_k), split_name=split_name)
        pairs = [
            {"question": questions[str(row["qid"])], "chunk_text": str(row.get("chunk_text", ""))}
            for row in rows
        ]
        scores = backend.score(config, model_dir, pairs)
        ranked = []
        for row, score in zip(rows, scores):
            item = dict(row)
            item["llm_rerank_score"] = float(score)
            item["candidate_unit"] = "chunk"
            ranked.append(item)
        ranked = _limit_candidates(ranked, int(config.llm_rerank_top_k), "llm_rerank_score")
        _assert_max_candidates(ranked, top_k=int(config.llm_rerank_top_k), label=f"llm_rerank {split_name} rows")
        _assert_no_duplicate_chunks(ranked, label=f"llm_rerank {split_name} rows")
        bad_scores = [row["llm_rerank_score"] for row in ranked if not (0.0 <= float(row["llm_rerank_score"]) <= 1.0)]
        if bad_scores:
            raise AssertionError(f"llm_rerank {split_name} scores outside [0,1]: {bad_scores[:5]}")
        split_rows_by_name[split_name] = ranked
        all_ranked.extend(ranked)

    input_hash = stable_hash({"source": source_paths, "rows": len(all_ranked), "top_k": config.llm_rerank_top_k})
    for split_name, split_path in split_paths.items():
        split_fmt = write_table(split_path, split_rows_by_name[split_name])
        mark_done(split_path, config=config, stage="rerank_llm", input_hash=input_hash, model=config.llm_rerank_model, params={**expected_params, "split": split_name}, fmt=split_fmt)
        saved(split_path)
    fmt = write_table(path, all_ranked)
    mark_done(path, config=config, stage="rerank_llm", input_hash=input_hash, model=config.llm_rerank_model, params={**expected_params, "split": "aggregate"}, fmt=fmt)

    split_questions = {
        split_name: _filter_questions(all_questions, qids)
        for split_name, qids in eval_qids_by_split.items()
    }
    _write_llm_metrics(config, split_rows_by_name, split_questions, params=expected_params, input_hash=input_hash)
    saved(path)
