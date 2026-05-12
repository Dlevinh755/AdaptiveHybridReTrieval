from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pathlib import Path as _Path
import sys

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_summary_consistency import (
    DETAIL_METRIC_FILES,
    assert_close,
    check_bge_candidate_pool,
    check_bge_training_split_leakage,
    check_dense_index_checkpoint,
    check_hybrid_formula,
    check_question_text_not_in_corpus,
    check_rows_against_questions,
    check_split_leakage,
    check_training_ground_truth,
    check_val_topk_application,
    check_val_topk_is_best_f2,
    check_val_threshold_is_best_f2,
    check_val_threshold_application,
    describe,
    done_path,
    read_json,
    read_records,
    top_pairs,
)
from src.data.loaders import load_questions
from src.eval.metrics import aggregate_by_aid_max, ranking_metrics, threshold_metrics, topk_metrics


def section(index: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"{index}. {title}")
    print("=" * 80)


def load_questions_by_qid(questions_path: Path) -> dict[str, dict[str, Any]]:
    class _Config:
        question_id_field = "qid"
        question_text_field = "question"
        relevant_ids_field = "relevant_laws"

    _Config.questions_path = questions_path
    return {str(row["qid"]): row for row in load_questions(_Config())}


def check_summary_thresholds(dataset_dir: Path) -> None:
    eval_dir = dataset_dir / "eval"
    summary = read_json(eval_dir / "summary.json")
    check_val_threshold_application(eval_dir, summary)
    for method, pattern in DETAIL_METRIC_FILES.items():
        for split in ["val", "test"]:
            key = f"{method}_{split}"
            detail_path = eval_dir / pattern.format(split=split)
            if key not in summary or not detail_path.exists():
                continue
            detail = read_json(detail_path)
            threshold = detail.get("threshold", {})
            ranking = detail.get("ranking", {})
            for metric in ["precision", "recall", "f2"]:
                assert_close(f"{key}.{metric}", summary[key][metric], threshold[metric])
            for metric in ["hit@10", "recall@10", "ndcg@10"]:
                if metric in summary[key] and metric in ranking:
                    assert_close(f"{key}.{metric}", summary[key][metric], ranking[metric])
            print(f"[ok] {key}: summary matches detail threshold/ranking metrics")


def rerank_paths(dataset_dir: Path) -> dict[str, Path]:
    return {
        "val": dataset_dir / "retrieval_cache" / "bge_rerank_scores_val.parquet",
        "test": dataset_dir / "retrieval_cache" / "bge_rerank_scores_test.parquet",
    }


def check_rerank_num_rows(dataset_dir: Path, *, candidate_top_k: int) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, path in rerank_paths(dataset_dir).items():
        if not path.exists():
            continue
        marker_path = done_path(path)
        if not marker_path.exists():
            raise AssertionError(f"missing marker: {marker_path}")
        marker = read_json(marker_path)
        params = marker.get("params", {})
        expected = {
            "schema_version": 3,
            "candidate_top_k": candidate_top_k,
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "source": "hybrid_router",
            "split": split,
        }
        for key, expected_value in expected.items():
            if params.get(key) != expected_value:
                raise AssertionError(f"{marker_path.name} params[{key!r}]={params.get(key)!r}, expected {expected_value!r}")
        rows = read_records(path)
        counts = Counter(str(row["qid"]) for row in rows)
        overflow = {qid: count for qid, count in counts.items() if count > candidate_top_k}
        if overflow:
            examples = sorted(overflow.items(), key=lambda item: item[1], reverse=True)[:5]
            raise AssertionError(f"{path.name} exceeds candidate_top_k={candidate_top_k}: {examples}")
        print(
            f"[ok] {path.name}: rows={len(rows)}, qids={len(counts)}, "
            f"max_candidates_per_qid={max(counts.values()) if counts else 0}"
        )
        rows_by_split[split] = rows
    return rows_by_split


def check_rerank_duplicates(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in rows_by_split.items():
        qid_chunk = Counter((str(row["qid"]), str(row.get("chunk_id", ""))) for row in rows)
        qid_aid_chunk = Counter((str(row["qid"]), str(row["aid"]), str(row.get("chunk_id", ""))) for row in rows)
        qid_aid = Counter((str(row["qid"]), str(row["aid"])) for row in rows)
        dup_qid_chunk = sum(count - 1 for count in qid_chunk.values() if count > 1)
        dup_qid_aid_chunk = sum(count - 1 for count in qid_aid_chunk.values() if count > 1)
        dup_qid_aid = sum(count - 1 for count in qid_aid.values() if count > 1)
        print(
            f"[duplicates] {split}: duplicated(qid,chunk_id)={dup_qid_chunk}, "
            f"duplicated(qid,aid,chunk_id)={dup_qid_aid_chunk}, duplicated(qid,aid)={dup_qid_aid}"
        )
        if dup_qid_chunk or dup_qid_aid_chunk:
            raise AssertionError(f"bge_rerank_scores_{split} has duplicate chunk-level candidates")


def check_rerank_chunk_to_aid_aggregation(dataset_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]], questions_by_qid: dict[str, dict[str, Any]], *, candidate_top_k: int) -> None:
    chunk_to_aid = read_json(dataset_dir / "prepared" / "chunk_to_aid.json")
    splits = read_json(dataset_dir / "prepared" / "splits.json")
    eval_dir = dataset_dir / "eval"
    retrieval_dir = dataset_dir / "retrieval_cache"
    for split, rows in rows_by_split.items():
        mismatches = []
        for row in rows:
            chunk_id = str(row.get("chunk_id", ""))
            aid = str(row["aid"])
            mapped_aid = str(chunk_to_aid.get(chunk_id, ""))
            if mapped_aid and mapped_aid != aid:
                mismatches.append((str(row["qid"]), aid, chunk_id, mapped_aid))
        if mismatches:
            raise AssertionError(f"bge_rerank_scores_{split} chunk_id->aid mismatches: {mismatches[:5]}")

        source_path = retrieval_dir / f"hybrid_router_scores_{split}.parquet"
        if not source_path.exists():
            source_path = retrieval_dir / "hybrid_router_scores.parquet"
        source_rows = [row for row in read_records(source_path) if str(row["qid"]) in {str(item["qid"]) for item in rows}]
        allowed_pairs = top_pairs(source_rows, score_field="hybrid_score", top_k=candidate_top_k)
        candidate_pairs = {(str(row["qid"]), str(row["aid"])) for row in rows}
        extra = sorted(candidate_pairs - allowed_pairs)[:5]
        if extra:
            raise AssertionError(f"bge_rerank_scores_{split} contains candidates outside hybrid_router top{candidate_top_k}: {extra}")

        detail_path = eval_dir / f"bge_rerank_{split}_metrics.json"
        if detail_path.exists():
            split_qids = {str(qid) for qid in splits[split]}
            questions = [question for qid, question in questions_by_qid.items() if qid in split_qids]
            detail = read_json(detail_path)
            score_field = str(detail.get("score_field", "rerank_score"))
            aid_aggregated_rows = aggregate_by_aid_max(rows, score_field=score_field)
            recomputed_ranking = ranking_metrics(aid_aggregated_rows, questions)
            recomputed_threshold = threshold_metrics(aid_aggregated_rows, questions, score_field=score_field, threshold=detail["threshold"]["threshold"])
            recomputed_topk = topk_metrics(aid_aggregated_rows, questions, score_field=score_field, k=int(detail["topk_tuned"]["top_k"]))
            for metric in ["hit@10", "recall@10", "ndcg@10"]:
                assert_close(f"bge_rerank_{split}.{metric}.aid_max_recomputed", detail["ranking"][metric], recomputed_ranking[metric])
            for metric in ["precision", "recall", "f2"]:
                assert_close(f"bge_rerank_{split}.{metric}.aid_max_recomputed", detail["threshold"][metric], recomputed_threshold[metric])
                assert_close(f"bge_rerank_{split}.topk_{metric}.aid_max_recomputed", detail["topk_tuned"][metric], recomputed_topk[metric])
        print(f"[ok] {split}: chunk->aid mapping valid and metrics recompute with aid_score=max(chunk_scores)")


def check_rerank_sort_direction(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in rows_by_split.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["qid"])].append(row)
        unsorted_qids = []
        for qid, items in grouped.items():
            scores = [float(row.get("rerank_score", 0.0)) for row in items]
            if scores != sorted(scores, reverse=True):
                unsorted_qids.append(qid)
        if unsorted_qids:
            raise AssertionError(f"bge_rerank_scores_{split} is not sorted descending by rerank_score for qids: {unsorted_qids[:10]}")
        positive_scores = [float(row["rerank_score"]) for row in rows if int(row.get("label", 0)) == 1]
        negative_scores = [float(row["rerank_score"]) for row in rows if int(row.get("label", 0)) == 0]
        print(f"[score] {split} positive rerank_score:", describe(positive_scores))
        print(f"[score] {split} negative rerank_score:", describe(negative_scores))
        if positive_scores and negative_scores and sum(positive_scores) / len(positive_scores) < sum(negative_scores) / len(negative_scores):
            print(f"[warn] {split}: positive mean rerank_score is lower than negative mean; check model quality or score direction.")
        print(f"[ok] {split}: rows sorted by rerank_score descending")


def check_router_alpha(dataset_dir: Path) -> None:
    path = dataset_dir / "eval" / "router_metrics.json"
    if not path.exists():
        print("[skip] router_metrics.json missing")
        return
    payload = read_json(path)
    router_train = payload.get("router_train", {})
    if router_train.get("alpha_convention") != "alpha_is_weight_bge":
        raise AssertionError(f"Unexpected router alpha convention: {router_train.get('alpha_convention')}")
    if router_train.get("convention_self_check_passed") is False:
        raise AssertionError("Router alpha convention self-check failed.")
    print("[ok] router alpha convention:", {
        "alpha_convention": router_train.get("alpha_convention"),
        "label_gt_0.5_means": router_train.get("label_gt_0.5_means"),
        "binary_accuracy_alpha_gt_0.5": router_train.get("binary_accuracy_alpha_gt_0.5"),
        "alpha_label_pred_correlation": router_train.get("alpha_label_pred_correlation"),
    })


def check_score_normalization(dataset_dir: Path) -> None:
    retrieval_dir = dataset_dir / "retrieval_cache"
    for split in ["val", "test"]:
        for stem in ["hybrid_fixed_scores", "hybrid_router_scores"]:
            path = retrieval_dir / f"{stem}_{split}.parquet"
            if not path.exists():
                continue
            rows = read_records(path)
            check_hybrid_formula(rows, name=path.name)
            print(f"[ok] {path.name}: hybrid_score = alpha*bge_score_norm + (1-alpha)*bm25_score_norm")


def check_cache_freshness(dataset_dir: Path, *, top_k: int) -> None:
    check_dense_index_checkpoint(dataset_dir)
    check_bge_candidate_pool(dataset_dir, top_k=top_k)
    for split, path in rerank_paths(dataset_dir).items():
        if path.exists():
            marker = read_json(done_path(path))
            params = marker.get("params", {})
            if params.get("source") != "hybrid_router":
                raise AssertionError(f"{path.name} source={params.get('source')!r}, expected hybrid_router")
            print(f"[ok] {path.name}: cache marker source/schema:", params)


def check_label_unit(dataset_dir: Path, questions_by_qid: dict[str, dict[str, Any]], rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    check_training_ground_truth(dataset_dir, questions_by_qid)
    retrieval_dir = dataset_dir / "retrieval_cache"
    for split in ["val", "test"]:
        for path in [retrieval_dir / f"hybrid_router_scores_{split}.parquet", retrieval_dir / f"bge_rerank_scores_{split}.parquet"]:
            if not path.exists():
                continue
            check_rows_against_questions(read_records(path), questions_by_qid, name=path.name)
            print(f"[ok] {path.name}: labels use aid ground truth")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run priority audit checks for suspicious retrieval/rerank metrics.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--questions_path", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--candidate_top_k", type=int, default=50)
    parser.add_argument("--allow_positive_aid_overlap", action="store_true", default=True)
    parser.add_argument("--disallow_positive_aid_overlap", action="store_false", dest="allow_positive_aid_overlap")
    parser.add_argument("--allow_question_text_in_corpus", action="store_true", default=False)
    parser.add_argument("--question_text_leak_min_words", type=int, default=8)
    args = parser.parse_args()

    questions_by_qid = load_questions_by_qid(args.questions_path)

    section(1, "summary.json threshold metrics")
    check_summary_thresholds(args.dataset_dir)
    check_val_topk_application(args.dataset_dir / "eval", read_json(args.dataset_dir / "eval" / "summary.json"))
    check_val_threshold_is_best_f2(args.dataset_dir, questions_by_qid)
    check_val_topk_is_best_f2(args.dataset_dir, questions_by_qid)

    section(2, "rerank num_rows vs candidate_top_k")
    rows_by_split = check_rerank_num_rows(args.dataset_dir, candidate_top_k=args.candidate_top_k)

    section(3, "duplicate qid/chunk_id after rerank")
    check_rerank_duplicates(rows_by_split)

    section(4, "chunk -> aid aggregation in rerank")
    check_rerank_chunk_to_aid_aggregation(args.dataset_dir, rows_by_split, questions_by_qid, candidate_top_k=args.candidate_top_k)

    section(5, "rerank_score sort direction")
    check_rerank_sort_direction(rows_by_split)

    section(6, "hybrid/router alpha convention")
    check_router_alpha(args.dataset_dir)

    section(7, "BM25/BGE score normalization")
    check_score_normalization(args.dataset_dir)

    section(8, "embedding/rerank cache freshness")
    check_cache_freshness(args.dataset_dir, top_k=args.top_k)

    section(9, "label unit aid vs chunk")
    check_label_unit(args.dataset_dir, questions_by_qid, rows_by_split)

    section(10, "split leakage/question duplicate")
    check_split_leakage(args.dataset_dir, questions_by_qid, allow_positive_aid_overlap=args.allow_positive_aid_overlap)
    check_bge_training_split_leakage(args.dataset_dir)
    check_question_text_not_in_corpus(
        args.dataset_dir,
        questions_by_qid,
        min_words=args.question_text_leak_min_words,
        allow_question_text_in_corpus=args.allow_question_text_in_corpus,
    )

    print("\n[done] priority audit passed")


if __name__ == "__main__":
    main()
