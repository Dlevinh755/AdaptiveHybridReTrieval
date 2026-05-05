from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics
from src.retrieval.hybrid import apply_hybrid
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, read_table, retrieval_dir, stable_hash, write_json
from src.utils.logging import saved, skip


EVAL_SOURCES = {
    "bm25": ("bm25_scores.parquet", "bm25_score_norm"),
    "bge": ("bge_scores.parquet", "bge_score_norm"),
    "hybrid_fixed": ("hybrid_fixed_scores.parquet", "hybrid_score"),
    "hybrid_tuned": ("hybrid_tuned_scores.parquet", "hybrid_score"),
    "hybrid_router": ("hybrid_router_scores.parquet", "hybrid_score"),
    "rerank_bge": ("bge_rerank_scores.parquet", "rerank_score"),
    "rerank_qwen": ("qwen_rerank_scores.parquet", "qwen_rerank_score"),
}


def _source_signature(config: Any) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for source, (filename, score_field) in EVAL_SOURCES.items():
        paths = [retrieval_dir(config) / filename]
        stem, suffix = filename.rsplit(".", 1)
        paths.extend(retrieval_dir(config) / f"{stem}_{split_name}.{suffix}" for split_name in ["val", "test"])
        for path in paths:
            if path.exists():
                stat = path.stat()
                signature[str(path.name)] = {
                    "source": source,
                    "score_field": score_field,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
    return signature


def _source_path_for_split(config: Any, filename: str, split_name: str) -> Any:
    stem, suffix = filename.rsplit(".", 1)
    split_path = retrieval_dir(config) / f"{stem}_{split_name}.{suffix}"
    if split_path.exists():
        return split_path
    return retrieval_dir(config) / filename


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if row["qid"] in qids]


def _rank(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)
    output = []
    for items in grouped.values():
        output.extend(sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True))
    return output


def _compare_strategies(summary: dict[str, Any]) -> None:
    """Print comparison between hybrid_tuned (fixed) and hybrid_router (per-query alpha)."""
    print("\n" + "="*70)
    print("[HYBRID ROUTER EVALUATION] Comparing fixed alpha vs per-query alpha")
    print("="*70)
    
    for split in ["val", "test"]:
        tuned_key = f"hybrid_tuned_{split}"
        router_key = f"hybrid_router_{split}"
        
        if tuned_key not in summary or router_key not in summary:
            continue
        
        tuned = summary[tuned_key]
        router = summary[router_key]
        
        print(f"\n[{split.upper()} SET]")
        print(f"{'Metric':<15} {'hybrid_tuned':<15} {'hybrid_router':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for metric in ["hit@10", "recall@10", "ndcg@10"]:
            tuned_val = tuned.get(metric, 0.0)
            router_val = router.get(metric, 0.0)
            if tuned_val > 0:
                improvement = ((router_val - tuned_val) / tuned_val) * 100
                print(f"{metric:<15} {tuned_val:<15.4f} {router_val:<15.4f} {improvement:+.2f}%")
            else:
                print(f"{metric:<15} {tuned_val:<15.4f} {router_val:<15.4f} N/A")
    
    print("="*70 + "\n")


def evaluate(config: Any) -> None:
    path = eval_dir(config) / "summary.json"
    signature = _source_signature(config)
    expected_params = {"threshold": config.threshold, "sources": signature}
    if is_complete(path, expected={"params": expected_params}) and not config.force:
        skip(path)
        return

    questions = load_questions(config)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    summary: dict[str, Any] = {}

    for split_name in ["val", "test"]:
        split_questions = _filter_questions(questions, set(splits[split_name]))
        for source, (filename, score_field) in EVAL_SOURCES.items():
            source_path = _source_path_for_split(config, filename, split_name)
            if not source_path.exists():
                continue
            rows = read_table(source_path)
            if source == "hybrid_tuned":
                alpha = read_json(eval_dir(config) / "hybrid_alpha.json")["best_alpha"]
                rows = apply_hybrid(read_table(retrieval_dir(config) / "merged_scores.parquet"), fixed_alpha=alpha)
            ranked = _rank(rows, score_field)
            key = f"{source}_{split_name}"
            summary[key] = {
                **ranking_metrics(ranked, split_questions),
                **threshold_metrics(ranked, split_questions, score_field=score_field, threshold=config.threshold),
            }

    _compare_strategies(summary)
    write_json(path, summary)
    mark_done(path, config=config, stage="evaluate", input_hash=stable_hash({"summary": sorted(summary.keys()), "sources": signature}), params=expected_params, fmt="json")
    saved(path)
