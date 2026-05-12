from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, read_table, retrieval_dir, stable_hash, write_json, write_table
from src.utils.logging import saved, skip


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(_clamp01(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value * (1.0 - fraction) + upper_value * fraction


def _observed_scores(items: list[dict[str, Any]], canonical: str, alias: str, raw_field: str) -> list[float]:
    scores: list[float] = []
    for item in items:
        if canonical in item:
            scores.append(_clamp01(item.get(canonical)))
        elif alias in item:
            scores.append(_clamp01(item.get(alias)))
        elif raw_field in item:
            raise ValueError(
                f"Hybrid cache row has raw {raw_field} but missing {canonical}. "
                "Re-run retrieve_cache with --force true to rebuild normalized merged scores."
            )
    return scores


def _normalized_score(item: dict[str, Any], canonical: str, alias: str, raw_field: str, fallback: float) -> float:
    if canonical in item:
        return _clamp01(item.get(canonical))
    if alias in item:
        return _clamp01(item.get(alias))
    if raw_field in item:
        raise ValueError(
            f"Hybrid cache row has raw {raw_field} but missing {canonical}. "
            "Re-run retrieve_cache with --force true to rebuild normalized merged scores."
        )
    return 0.0


def apply_hybrid(rows: list[dict[str, Any]], alpha_by_qid: dict[str, float] | None = None, fixed_alpha: float = 0.5) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(dict(row))

    output = []
    for qid, items in grouped.items():
        alpha = _clamp01(alpha_by_qid.get(qid, fixed_alpha) if alpha_by_qid else fixed_alpha)
        bm25_fallback = _percentile(_observed_scores(items, "bm25_score_norm", "bm25_norm", "bm25_score"), 10.0)
        bge_fallback = _percentile(_observed_scores(items, "bge_score_norm", "bge_norm", "bge_score"), 10.0)
        for item in items:
            bm25_score = _normalized_score(item, "bm25_score_norm", "bm25_norm", "bm25_score", bm25_fallback)
            bge_score = _normalized_score(item, "bge_score_norm", "bge_norm", "bge_score", bge_fallback)
            item["bm25_score_norm"] = bm25_score
            item["bge_score_norm"] = bge_score
            item["bm25_norm"] = bm25_score
            item["bge_norm"] = bge_score
            item["hybrid_alpha"] = alpha
            item["hybrid_score"] = alpha * bge_score + (1.0 - alpha) * bm25_score
        output.extend(sorted(items, key=lambda row: row["hybrid_score"], reverse=True))
    return output


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if row["qid"] in qids]


def tune_hybrid(config: Any) -> None:
    out_path = retrieval_dir(config) / "hybrid_tuned_scores.parquet"
    fixed_path = retrieval_dir(config) / "hybrid_fixed_scores.parquet"
    alpha_path = eval_dir(config) / "hybrid_alpha.json"
    expected = {"params": {"alpha_grid": config.alpha_grid}}
    if is_complete(out_path, expected=expected) and is_complete(fixed_path) and is_complete(alpha_path, expected=expected) and not config.force:
        skip(out_path)
        return

    rows = read_table(retrieval_dir(config) / "merged_scores.parquet")
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    questions = load_questions(config)
    val_questions = _filter_questions(questions, set(splits["val"]))
    alphas = [float(value) for value in config.alpha_grid.split(",")]

    best_alpha = config.hybrid_alpha
    best_recall = -1.0
    for alpha in alphas:
        ranked = apply_hybrid(rows, fixed_alpha=alpha)
        metrics = ranking_metrics(ranked, val_questions, ks=[10])
        score = metrics.get("recall@10", 0.0)
        if score > best_recall:
            best_recall = score
            best_alpha = alpha

    tuned = apply_hybrid(rows, fixed_alpha=best_alpha)
    fixed = apply_hybrid(rows, fixed_alpha=0.5)
    fmt = write_table(out_path, tuned)
    fixed_fmt = write_table(fixed_path, fixed)
    write_json(alpha_path, {"best_alpha": best_alpha, "val_recall@10": best_recall, "grid": alphas})
    input_hash = stable_hash({"rows": len(rows), "alphas": alphas})
    mark_done(out_path, config=config, stage="tune_hybrid", input_hash=input_hash, params={"alpha_grid": config.alpha_grid}, fmt=fmt)
    mark_done(fixed_path, config=config, stage="tune_hybrid", input_hash=input_hash, params={"alpha": 0.5}, fmt=fixed_fmt)
    mark_done(alpha_path, config=config, stage="tune_hybrid", input_hash=input_hash, params={"alpha_grid": config.alpha_grid}, fmt="json")
    saved(out_path)
