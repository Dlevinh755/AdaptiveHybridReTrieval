from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics, tune_threshold
from src.indexes.faiss_index import _get_dense_model, dense_index_paths
from src.utils.artifact import eval_dir, prepared_dir, read_json, read_table, retrieval_dir


def _base_config(args: argparse.Namespace) -> Config:
    return Config(
        stage="evaluate",
        dataset_name=args.dataset_name,
        corpus_path=args.corpus_path,
        questions_path=args.questions_path,
        output_dir=args.output_dir,
        force=False,
        seed=42,
        dense_model=args.dense_model,
        rerank_model="",
        qwen_model="",
        use_qwen_rerank=False,
        device=args.device,
        batch_size=args.batch_size,
        bge_train_batch_size=1,
        bge_epochs=1,
        bge_lr=2e-5,
        bge_warmup_ratio=0.1,
        bge_max_length=512,
        bge_max_train_examples=0,
        bge_use_amp=True,
        bge_gradient_checkpointing=True,
        bge_auto_batch_reduce=True,
        bge_negatives_per_example=3,
        reranker_train_batch_size=1,
        reranker_epochs=1,
        reranker_lr=2e-5,
        reranker_warmup_ratio=0.1,
        reranker_max_length=512,
        reranker_max_train_examples=0,
        reranker_use_amp=True,
        bm25_k1=1.2,
        bm25_b=0.9,
        bm25_k1_grid="1.2",
        bm25_b_grid="0.9",
        bm25_tune_metric="recall@10",
        use_tuned_bm25=True,
        hybrid_alpha=0.5,
        alpha_grid="0.5",
        router_model="ridge",
        top_k=args.top_k,
        candidate_top_k=50,
        positive_chunks_per_aid=2,
        threshold=0.5,
        train_ratio=0.7,
        router_train_ratio=0.1,
        val_ratio=0.1,
        test_ratio=0.1,
        max_chunk_tokens=450,
        chunk_overlap_sentences=1,
        corpus_law_id_field="law_id",
        corpus_articles_field="content",
        article_id_field="aid",
        article_text_field="content_Article",
        question_id_field="qid",
        question_text_field="question",
        relevant_ids_field="relevant_laws",
    )


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if str(row["qid"]) in qids]


def _group_top(rows: list[dict[str, Any]], score_field: str, *, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["qid"])].append(row)
    return {
        qid: sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)[:limit]
        for qid, items in grouped.items()
    }


def _label_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row.get("label", 0)) for row in rows]
    scores = [float(row.get("bge_score", 0.0)) for row in rows]
    norm_scores = [float(row.get("bge_score_norm", 0.0)) for row in rows if "bge_score_norm" in row]
    return {
        "rows": len(rows),
        "positive_rows": sum(labels),
        "unique_qids": len({str(row["qid"]) for row in rows}),
        "unique_aids": len({str(row["aid"]) for row in rows}),
        "bge_score_min": min(scores) if scores else None,
        "bge_score_max": max(scores) if scores else None,
        "bge_score_norm_min": min(norm_scores) if norm_scores else None,
        "bge_score_norm_max": max(norm_scores) if norm_scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug BGE retrieval cache, mapping, and metrics.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--questions_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dense_model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()
    config = _base_config(args)

    root = config.dataset_dir
    print("[paths]")
    print("dataset_dir:", root)
    print("prepared exists:", prepared_dir(config).exists())
    print("retrieval_cache exists:", retrieval_dir(config).exists())
    print("eval exists:", eval_dir(config).exists())

    splits = read_json(prepared_dir(config) / "splits.json")
    questions = load_questions(config)
    print("\n[splits]")
    for split_name in ["train", "router", "val", "test"]:
        print(split_name, len(splits.get(split_name, [])))

    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    print("\n[prepared]")
    print("questions:", len(questions))
    print("chunks:", len(chunks))
    print("chunk_to_aid:", len(chunk_to_aid))

    dense_model = _get_dense_model(config)
    dense_paths = dense_index_paths(config)
    print("\n[dense index]")
    print("resolved_model:", dense_model)
    print("index_root:", dense_paths["root"])
    if dense_paths["metadata"].exists():
        print("metadata:", read_json(dense_paths["metadata"]))
    else:
        print("metadata: missing")

    print("\n[bge cache metrics]")
    for split_name in ["val", "test"]:
        qids = {str(qid) for qid in splits[split_name]}
        split_questions = _filter_questions(questions, qids)
        cache_path = retrieval_dir(config) / f"bge_scores_{split_name}.parquet"
        if not cache_path.exists():
            print(split_name, "missing", cache_path)
            continue
        rows = read_table(cache_path)
        print(split_name, _label_stats(rows))
        tuned = tune_threshold(rows, split_questions, score_field="bge_score_norm")
        metrics = {
            **ranking_metrics(rows, split_questions),
            **threshold_metrics(rows, split_questions, score_field="bge_score_norm", threshold=tuned["threshold"]),
        }
        print(split_name, "threshold:", tuned)
        print(split_name, "metrics:", {key: metrics.get(key) for key in ["hit@10", "recall@10", "ndcg@10", "precision", "recall", "f2"]})

        positives_by_qid = {str(row["qid"]): set(map(str, row["relevant_laws"])) for row in split_questions}
        grouped = _group_top(rows, "bge_score_norm", limit=10)
        misses = []
        hits = []
        for qid, top_rows in grouped.items():
            top_aids = {str(row["aid"]) for row in top_rows}
            if top_aids & positives_by_qid.get(qid, set()):
                hits.append(qid)
            else:
                misses.append(qid)
        print(split_name, "top10 hit qids:", len(hits), "miss qids:", len(misses))
        for qid in misses[:3]:
            print("  miss qid:", qid, "positives:", sorted(positives_by_qid.get(qid, set())))
            print("  top aids:", [(row.get("aid"), row.get("bge_score"), row.get("bge_score_norm")) for row in grouped.get(qid, [])[:5]])

    print("\n[summary check]")
    summary_path = eval_dir(config) / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        for key in ["bge_val", "bge_test"]:
            print(key, summary.get(key))
    else:
        print("summary missing:", summary_path)


if __name__ == "__main__":
    main()
