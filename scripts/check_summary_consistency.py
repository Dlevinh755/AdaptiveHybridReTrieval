from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pathlib import Path as _Path
import sys

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import ranking_metrics, threshold_metrics, topk_metrics, tune_threshold, tune_top_k
from src.data.loaders import load_questions
from src.utils.artifact import file_hash


DETAIL_METRIC_FILES = {
    "bm25": "bm25_{split}_metrics.json",
    "bge": "bge_{split}_metrics.json",
    "hybrid_fixed": "hybrid_fixed_{split}_metrics.json",
    "hybrid_router": "hybrid_router_{split}_metrics.json",
    "rerank_bge": "bge_rerank_{split}_metrics.json",
    "rerank_llm": "llm_rerank_{split}_metrics.json",
}

SCORE_CACHE_FILES = {
    "bm25": ("bm25_scores_{split}.parquet", "bm25_score_norm"),
    "bge": ("bge_scores_{split}.parquet", "bge_score_norm"),
    "hybrid_fixed": ("hybrid_fixed_scores_{split}.parquet", "hybrid_score"),
    "hybrid_router": ("hybrid_router_scores_{split}.parquet", "hybrid_score"),
    "rerank_bge": ("bge_rerank_scores_{split}.parquet", "rerank_score"),
    "rerank_llm": ("llm_rerank_scores_{split}.parquet", "llm_rerank_score"),
}

CANONICAL_SPLITS = ["train", "router", "val", "test"]
VAL_TUNED_METHODS = set(DETAIL_METRIC_FILES)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def done_path(path: Path) -> Path:
    return path.with_name(path.name + ".done.json")


def assert_close(name: str, left: Any, right: Any, *, tol: float = 1e-12) -> None:
    if abs(float(left) - float(right)) > tol:
        raise AssertionError(f"{name}: summary={left} detail={right}")


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "mean": sum(ordered) / count,
        "min": ordered[0],
        "p25": ordered[int((count - 1) * 0.25)],
        "p50": ordered[int((count - 1) * 0.50)],
        "p75": ordered[int((count - 1) * 0.75)],
        "max": ordered[-1],
    }


def text_len_stats(values: list[str]) -> dict[str, Any]:
    return describe([float(len(str(value).split())) for value in values])


def top_pairs(rows: list[dict[str, Any]], *, score_field: str, top_k: int) -> set[tuple[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["qid"]), []).append(row)
    pairs: set[tuple[str, str]] = set()
    for qid, items in grouped.items():
        for row in sorted(items, key=lambda item: float(item.get(score_field, 0.0)), reverse=True)[:top_k]:
            pairs.add((qid, str(row["aid"])))
    return pairs


def read_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def token_count(text: str) -> int:
    return len([token for token in normalize_question_text(text).split(" ") if token])


def check_split_leakage(dataset_dir: Path, questions_by_qid: dict[str, dict[str, Any]], *, allow_positive_aid_overlap: bool) -> None:
    splits_path = dataset_dir / "prepared" / "splits.json"
    if not splits_path.exists():
        return
    splits = read_json(splits_path)
    qids_by_split = {split_name: {str(qid) for qid in splits.get(split_name, [])} for split_name in CANONICAL_SPLITS}
    assigned = [qid for split_name in CANONICAL_SPLITS for qid in qids_by_split[split_name]]
    if len(assigned) != len(set(assigned)):
        counts = Counter(assigned)
        duplicates = sorted(qid for qid, count in counts.items() if count > 1)[:10]
        raise AssertionError(f"splits.json has qid overlap across splits: {duplicates}")
    known_qids = set(questions_by_qid)
    assigned_set = set(assigned)
    if assigned_set - known_qids:
        raise AssertionError(f"splits.json contains qids missing from questions: {sorted(assigned_set - known_qids)[:10]}")
    if known_qids - assigned_set:
        raise AssertionError(f"splits.json misses question qids: {sorted(known_qids - assigned_set)[:10]}")

    text_by_split = {
        split_name: {
            normalize_question_text(questions_by_qid[qid]["question"])
            for qid in qids
            if qid in questions_by_qid and normalize_question_text(questions_by_qid[qid]["question"])
        }
        for split_name, qids in qids_by_split.items()
    }
    aids_by_split = {
        split_name: {
            str(aid)
            for qid in qids
            for aid in questions_by_qid.get(qid, {}).get("relevant_laws", [])
        }
        for split_name, qids in qids_by_split.items()
    }

    positive_overlap_summary: dict[str, int] = {}
    for left_idx, left in enumerate(CANONICAL_SPLITS):
        for right in CANONICAL_SPLITS[left_idx + 1 :]:
            qid_overlap = qids_by_split[left] & qids_by_split[right]
            if qid_overlap:
                raise AssertionError(f"{left}/{right} qid overlap: {sorted(qid_overlap)[:10]}")
            text_overlap = text_by_split[left] & text_by_split[right]
            if text_overlap:
                raise AssertionError(f"{left}/{right} question text overlap: {sorted(text_overlap)[:5]}")
            positive_overlap = aids_by_split[left] & aids_by_split[right]
            positive_overlap_summary[f"{left}__{right}"] = len(positive_overlap)
            if positive_overlap and not allow_positive_aid_overlap:
                raise AssertionError(f"{left}/{right} positive aid overlap: {sorted(positive_overlap)[:10]}")
    print("[ok] split qids are disjoint/exhaustive and question text has no cross-split overlap")
    print("[split positive aid overlap counts]", positive_overlap_summary)


def split_by_qid(dataset_dir: Path) -> dict[str, str]:
    splits_path = dataset_dir / "prepared" / "splits.json"
    if not splits_path.exists():
        return {}
    splits = read_json(splits_path)
    output = {}
    for split_name in CANONICAL_SPLITS:
        for qid in splits.get(split_name, []):
            output[str(qid)] = split_name
    return output


def check_bge_training_split_leakage(dataset_dir: Path) -> None:
    negatives_dir = dataset_dir / "negatives"
    if not negatives_dir.exists():
        return
    qid_to_split = split_by_qid(dataset_dir)
    if not qid_to_split:
        return
    for filename in ["bge_train_ready.jsonl", "bge_triplets.jsonl"]:
        path = negatives_dir / filename
        if not path.exists():
            continue
        rows = read_jsonl_records(path)
        leaked = sorted({
            str(row.get("qid", ""))
            for row in rows
            if str(row.get("qid", "")) and qid_to_split.get(str(row.get("qid", ""))) != "train"
        })
        if leaked:
            raise AssertionError(f"{filename} contains non-train qids: {leaked[:10]}")
        print(f"[ok] {filename} contains only train qids")


def check_dense_index_checkpoint(dataset_dir: Path) -> None:
    index_root = dataset_dir / "indexes" / "faiss"
    if not index_root.exists():
        return
    metadata_paths = sorted(index_root.glob("*/*/metadata.json"))
    if not metadata_paths:
        return
    finetuned_dir = dataset_dir / "models" / "bge_finetuned"
    finetuned_done = finetuned_dir / "train_summary.json.done.json"
    finetuned_summary = finetuned_dir / "train_summary.json"
    resolved_metadata = []
    for metadata_path in metadata_paths:
        metadata = read_json(metadata_path)
        if int(metadata.get("schema_version", 0)) != 2:
            raise AssertionError(f"{metadata_path} schema_version={metadata.get('schema_version')}, expected 2. Rebuild dense index.")
        resolved = str(metadata.get("dense_model_resolved", ""))
        if resolved == str(finetuned_dir):
            if not finetuned_done.exists():
                raise AssertionError(f"{metadata_path} resolves to fine-tuned BGE but missing {finetuned_done}")
            expected_done_hash = file_hash(finetuned_done)
            expected_summary_hash = file_hash(finetuned_summary) if finetuned_summary.exists() else ""
            if metadata.get("train_summary_done_hash", "") != expected_done_hash:
                raise AssertionError(
                    f"{metadata_path} was built from an older/different fine-tuned checkpoint. "
                    "Re-run train_bge_retriever or build_dense_index with --force true."
                )
            if metadata.get("train_summary_hash", "") != expected_summary_hash:
                raise AssertionError(
                    f"{metadata_path} train_summary_hash mismatch. Re-run build_dense_index with --force true."
                )
        resolved_metadata.append(
            {
                "metadata": str(metadata_path),
                "resolved": resolved,
                "model_key": metadata.get("model_key"),
                "backend": metadata.get("backend"),
                "chunk_count": metadata.get("chunk_count"),
                "embedding_dim": metadata.get("embedding_dim"),
            }
        )
    print("[dense index metadata]", resolved_metadata)
    if finetuned_done.exists() and not any(item["resolved"] == str(finetuned_dir) for item in resolved_metadata):
        raise AssertionError(
            "Fine-tuned BGE checkpoint exists, but no dense index metadata resolves to models/bge_finetuned. "
            "Run train_bge_retriever or build_dense_index after train_bge."
        )
    print("[ok] dense index metadata matches available fine-tuned checkpoint")


def check_bge_candidate_pool(dataset_dir: Path, *, top_k: int) -> None:
    retrieval_dir = dataset_dir / "retrieval_cache"
    eval_dir = dataset_dir / "eval"
    for split in ["router", "val", "test"]:
        bge_path = retrieval_dir / f"bge_scores_{split}.parquet"
        if not bge_path.exists():
            continue
        rows = read_records(bge_path)
        qid_aid = Counter((str(row["qid"]), str(row["aid"])) for row in rows)
        duplicates = [key for key, count in qid_aid.items() if count > 1]
        if duplicates:
            raise AssertionError(f"{bge_path.name} has duplicate (qid, aid) rows after dense aid collapse: {duplicates[:5]}")
        counts = Counter(str(row["qid"]) for row in rows)
        overflow = {qid: count for qid, count in counts.items() if count > top_k}
        if overflow:
            raise AssertionError(f"{bge_path.name} has more than top_k={top_k} aids per query: {sorted(overflow.items())[:5]}")
        merged_path = retrieval_dir / f"merged_scores_{split}.parquet"
        if merged_path.exists():
            merged_pairs = {(str(row["qid"]), str(row["aid"])) for row in read_records(merged_path)}
            bge_pairs = set(qid_aid)
            missing_from_merge = sorted(bge_pairs - merged_pairs)[:5]
            if missing_from_merge:
                raise AssertionError(f"{bge_path.name} has pairs missing from {merged_path.name}: {missing_from_merge}")
        metrics_path = eval_dir / f"bge_{split}_metrics.json"
        if metrics_path.exists():
            payload = read_json(metrics_path)
            pool = payload.get("candidate_pool", {})
            if pool:
                if int(pool.get("aid_rows_after_collapse", -1)) != len(rows):
                    raise AssertionError(f"{metrics_path.name} candidate_pool aid_rows_after_collapse does not match {bge_path.name}")
                if int(pool.get("raw_chunk_rows", 0)) < len(rows):
                    raise AssertionError(f"{metrics_path.name} raw_chunk_rows is smaller than collapsed aid rows")
                print(f"[bge candidate_pool] {split}", pool)
        print(f"[ok] {bge_path.name} is dense-only aid-collapsed top_k cache, not BM25 candidate pool")


def check_question_text_not_in_corpus(
    dataset_dir: Path,
    questions_by_qid: dict[str, dict[str, Any]],
    *,
    min_words: int,
    allow_question_text_in_corpus: bool,
) -> None:
    chunks_path = dataset_dir / "prepared" / "chunks.parquet"
    if not chunks_path.exists():
        return
    qid_to_split = split_by_qid(dataset_dir)
    chunks = read_records(chunks_path)
    normalized_chunks = [
        (str(row.get("chunk_id", "")), str(row.get("parent_aid", row.get("aid", ""))), normalize_question_text(str(row.get("text", ""))))
        for row in chunks
    ]
    hits = []
    for qid, question in questions_by_qid.items():
        query = normalize_question_text(question.get("question", ""))
        if token_count(query) < min_words:
            continue
        for chunk_id, aid, chunk_text in normalized_chunks:
            if query and query in chunk_text:
                hits.append(
                    {
                        "qid": qid,
                        "split": qid_to_split.get(qid, "unknown"),
                        "aid": aid,
                        "chunk_id": chunk_id,
                        "question_preview": query[:120],
                    }
                )
                break
    if hits:
        counts = Counter(row["split"] for row in hits)
        print("[question text in corpus counts]", dict(counts))
        print("[question text in corpus examples]", hits[:5])
        if not allow_question_text_in_corpus:
            raise AssertionError(
                "Corpus chunks contain full question text. This can make BGE look overfit/leaky. "
                f"Examples: {hits[:5]}"
            )
    else:
        print(f"[ok] no full question text with >= {min_words} words found inside corpus chunks")


def check_val_threshold_application(eval_dir: Path, summary: dict[str, Any]) -> None:
    for method, pattern in DETAIL_METRIC_FILES.items():
        val_path = eval_dir / pattern.format(split="val")
        if not val_path.exists():
            continue
        val_detail = read_json(val_path)
        val_threshold = float(val_detail["threshold"]["threshold"])
        for split in ["val", "test"]:
            key = f"{method}_{split}"
            detail_path = eval_dir / pattern.format(split=split)
            if not detail_path.exists() or key not in summary:
                continue
            detail = read_json(detail_path)
            selection_split = detail.get("threshold_selection_split", "val")
            if selection_split != "val":
                raise AssertionError(f"{detail_path.name} threshold_selection_split={selection_split!r}, expected 'val'")
            assert_close(f"{key}.detail_threshold_from_val", detail["threshold"]["threshold"], val_threshold)
            assert_close(f"{key}.summary_threshold_from_val", summary[key]["threshold"], val_threshold)
    print("[ok] val-selected thresholds are reused for val/test in detailed metrics and summary")


def check_val_topk_application(eval_dir: Path, summary: dict[str, Any]) -> None:
    for method, pattern in DETAIL_METRIC_FILES.items():
        val_path = eval_dir / pattern.format(split="val")
        if not val_path.exists():
            continue
        val_detail = read_json(val_path)
        if "topk_tuned" not in val_detail:
            continue
        val_top_k = int(val_detail["topk_tuned"]["top_k"])
        for split in ["val", "test"]:
            key = f"{method}_{split}"
            detail_path = eval_dir / pattern.format(split=split)
            if not detail_path.exists() or key not in summary:
                continue
            detail = read_json(detail_path)
            selection_split = detail.get("topk_selection_split", "val")
            if selection_split != "val":
                raise AssertionError(f"{detail_path.name} topk_selection_split={selection_split!r}, expected 'val'")
            assert_close(f"{key}.detail_topk_from_val", detail["topk_tuned"]["top_k"], val_top_k)
            if "best_top_k" in summary[key]:
                assert_close(f"{key}.summary_topk_from_val", summary[key]["best_top_k"], val_top_k)
    print("[ok] val-selected top-k values are reused for val/test in detailed metrics and summary")


def check_val_threshold_is_best_f2(dataset_dir: Path, questions_by_qid: dict[str, dict[str, Any]]) -> None:
    splits_path = dataset_dir / "prepared" / "splits.json"
    if not splits_path.exists() or not questions_by_qid:
        return
    split_qids = {str(qid) for qid in read_json(splits_path)["val"]}
    val_questions = [question for qid, question in questions_by_qid.items() if qid in split_qids]
    retrieval_dir = dataset_dir / "retrieval_cache"
    eval_dir = dataset_dir / "eval"

    for method, (cache_pattern, score_field) in SCORE_CACHE_FILES.items():
        cache_path = retrieval_dir / cache_pattern.format(split="val")
        detail_path = eval_dir / DETAIL_METRIC_FILES[method].format(split="val")
        if not cache_path.exists() or not detail_path.exists():
            continue
        rows = read_records(cache_path)
        detail = read_json(detail_path)
        detail_score_field = str(detail.get("score_field", score_field))
        best = tune_threshold(rows, val_questions, score_field=detail_score_field)
        threshold = detail.get("threshold", {})
        for metric in ["threshold", "precision", "recall", "f2"]:
            assert_close(f"{method}_val.best_f2_{metric}", threshold[metric], best[metric])
        print(f"[ok] {method}_val threshold maximizes F2 on val:", best)


def check_val_topk_is_best_f2(dataset_dir: Path, questions_by_qid: dict[str, dict[str, Any]]) -> None:
    splits_path = dataset_dir / "prepared" / "splits.json"
    if not splits_path.exists() or not questions_by_qid:
        return
    split_qids = {str(qid) for qid in read_json(splits_path)["val"]}
    val_questions = [question for qid, question in questions_by_qid.items() if qid in split_qids]
    retrieval_dir = dataset_dir / "retrieval_cache"
    eval_dir = dataset_dir / "eval"

    for method, (cache_pattern, score_field) in SCORE_CACHE_FILES.items():
        cache_path = retrieval_dir / cache_pattern.format(split="val")
        detail_path = eval_dir / DETAIL_METRIC_FILES[method].format(split="val")
        if not cache_path.exists() or not detail_path.exists():
            continue
        rows = read_records(cache_path)
        detail = read_json(detail_path)
        detail_score_field = str(detail.get("score_field", score_field))
        best = tune_top_k(rows, val_questions, score_field=detail_score_field)
        topk = detail.get("topk_tuned", {})
        for metric in ["top_k", "precision", "recall", "f2"]:
            assert_close(f"{method}_val.best_topk_{metric}", topk[metric], best[metric])
        fixed_3 = detail.get("topk_fixed_3", {})
        recomputed_3 = {"top_k": 3, **topk_metrics(rows, val_questions, score_field=detail_score_field, k=3)}
        for metric in ["top_k", "precision", "recall", "f2", "precision@3", "recall@3", "f2@3"]:
            assert_close(f"{method}_val.topk3_{metric}", fixed_3[metric], recomputed_3[metric])
        print(f"[ok] {method}_val top-k maximizes F2 on val:", best)


def check_rows_against_questions(rows: list[dict[str, Any]], questions_by_qid: dict[str, dict[str, Any]], *, name: str) -> None:
    errors = []
    for idx, row in enumerate(rows):
        if "qid" not in row or "aid" not in row or "label" not in row:
            continue
        qid = str(row["qid"])
        question = questions_by_qid.get(qid)
        if question is None:
            errors.append(f"row={idx} qid={qid} missing from questions")
            continue
        positives = {str(aid) for aid in question.get("relevant_laws", [])}
        expected = 1 if str(row["aid"]) in positives else 0
        actual = int(row.get("label", 0))
        if actual != expected:
            errors.append(f"row={idx} qid={qid} aid={row['aid']} label={actual} expected={expected}")
    if errors:
        raise AssertionError(f"{name} labels do not match questions.relevant_laws: " + "; ".join(errors[:10]))


def check_hybrid_formula(rows: list[dict[str, Any]], *, name: str, tol: float = 1e-9) -> None:
    required = {"hybrid_score", "hybrid_alpha", "bm25_score_norm", "bge_score_norm"}
    if rows:
        missing = sorted(required - set(rows[0].keys()))
        if missing:
            raise AssertionError(f"{name} missing normalized hybrid columns: {missing}")
    for idx, row in enumerate(rows):
        alpha = float(row.get("hybrid_alpha", 0.5))
        bm25 = float(row.get("bm25_score_norm", 0.0))
        bge = float(row.get("bge_score_norm", 0.0))
        if not (0.0 <= bm25 <= 1.0 and 0.0 <= bge <= 1.0):
            raise AssertionError(f"{name} row {idx} has norm score outside [0,1]: bm25={bm25}, bge={bge}")
        expected = alpha * bge + (1.0 - alpha) * bm25
        actual = float(row.get("hybrid_score", 0.0))
        if abs(actual - expected) > tol:
            raise AssertionError(f"{name} row {idx} hybrid_score={actual}, expected={expected}")


def check_aid_labels(rows: list[dict[str, Any]], *, name: str) -> None:
    for idx, row in enumerate(rows):
        if "label" not in row or "relevant_laws" not in row:
            continue
        expected = 1 if str(row["aid"]) in {str(aid) for aid in row.get("relevant_laws", [])} else 0
        actual = int(row.get("label", 0))
        if actual != expected:
            raise AssertionError(f"{name} row {idx} label={actual}, expected {expected} from aid/relevant_laws")


def check_training_ground_truth(dataset_dir: Path, questions_by_qid: dict[str, dict[str, Any]]) -> None:
    negatives_dir = dataset_dir / "negatives"
    if not negatives_dir.exists():
        return
    chunk_map_path = dataset_dir / "prepared" / "chunk_to_aid.json"
    chunk_to_aid = read_json(chunk_map_path) if chunk_map_path.exists() else {}
    for filename in ["rerank_pairwise.jsonl", "qwen_prompt_train.jsonl"]:
        path = negatives_dir / filename
        if not path.exists():
            continue
        rows = read_jsonl_records(path)
        check_rows_against_questions(rows, questions_by_qid, name=filename)
        mismatches = []
        for row in rows:
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id:
                continue
            mapped_aid = str(chunk_to_aid.get(chunk_id, ""))
            if mapped_aid and mapped_aid != str(row.get("aid", "")):
                mismatches.append((str(row.get("qid", "")), str(row.get("aid", "")), chunk_id, mapped_aid))
        if mismatches:
            raise AssertionError(f"{filename} chunk_id->aid mismatches: {mismatches[:5]}")
        print(f"[ok] {filename} labels match questions.relevant_laws and chunk_id maps to aid")
    for filename in ["bge_train_pairs.jsonl", "rerank_train_pairs.jsonl", "qwen_train_pairs.jsonl"]:
        path = negatives_dir / filename
        if not path.exists():
            continue
        rows = read_jsonl_records(path)
        errors = []
        for row in rows:
            qid = str(row["qid"])
            positives = {str(aid) for aid in questions_by_qid.get(qid, {}).get("relevant_laws", [])}
            positive_aid = str(row.get("positive_aid", ""))
            if positive_aid not in positives:
                errors.append(f"qid={qid} positive_aid={positive_aid} not in relevant_laws")
            for chunk_id in row.get("positive_chunk_ids", []):
                mapped_aid = str(chunk_to_aid.get(str(chunk_id), ""))
                if mapped_aid != positive_aid:
                    errors.append(f"qid={qid} positive_aid={positive_aid} chunk_id={chunk_id} maps_to={mapped_aid}")
            for aid, chunk_id in zip(row.get("negative_aids", []), row.get("negative_chunk_ids", [])):
                mapped_aid = str(chunk_to_aid.get(str(chunk_id), ""))
                if str(aid) in positives:
                    errors.append(f"qid={qid} negative_aid={aid} is in relevant_laws")
                if mapped_aid != str(aid):
                    errors.append(f"qid={qid} negative_aid={aid} chunk_id={chunk_id} maps_to={mapped_aid}")
        if errors:
            raise AssertionError(f"{filename} ground-truth joins are invalid: " + "; ".join(errors[:10]))
        print(f"[ok] {filename} positive/negative aids match ground truth")


def check_qwen_metadata(path: Path, *, candidate_top_k: int) -> None:
    marker_path = done_path(path)
    if not marker_path.exists():
        raise AssertionError(f"missing marker: {marker_path}")
    marker = read_json(marker_path)
    params = marker.get("params", {})
    expected_params = {
        "schema_version": 2,
        "candidate_top_k": candidate_top_k,
        "candidate_unit": "chunk",
        "ranking_unit": "aid",
        "source": "bge_rerank",
    }
    for key, expected_value in expected_params.items():
        if params.get(key) != expected_value:
            raise AssertionError(f"{marker_path.name} params[{key!r}]={params.get(key)!r}, expected {expected_value!r}")
    rows = read_records(path)
    required_columns = {"qid", "aid", "chunk_id", "chunk_text", "qwen_rerank_score", "label"}
    if rows:
        missing_columns = sorted(required_columns - set(rows[0].keys()))
        if missing_columns:
            raise AssertionError(f"{path.name} missing columns: {missing_columns}")
    check_aid_labels(rows, name=path.name)
    print(f"[ok] {path.name} metadata/input rows use candidate_unit=chunk and ranking_unit=aid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check eval/summary.json matches per-method detailed metrics.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--candidate_top_k", type=int, default=50)
    parser.add_argument("--llm_rerank_top_k", type=int, default=20)
    parser.add_argument("--rerank_model")
    parser.add_argument("--corpus_path", type=Path)
    parser.add_argument("--questions_path", type=Path)
    parser.add_argument("--allow_positive_aid_overlap", action="store_true", default=True)
    parser.add_argument("--disallow_positive_aid_overlap", action="store_false", dest="allow_positive_aid_overlap")
    parser.add_argument("--allow_question_text_in_corpus", action="store_true", default=False)
    parser.add_argument("--question_text_leak_min_words", type=int, default=8)
    args = parser.parse_args()

    eval_dir = args.dataset_dir / "eval"
    summary_path = eval_dir / "summary.json"
    summary = read_json(summary_path)
    checked = 0
    questions_by_qid: dict[str, dict[str, Any]] = {}
    if args.questions_path:
        class _BaseConfig:
            questions_path = args.questions_path
            question_id_field = "qid"
            question_text_field = "question"
            relevant_ids_field = "relevant_laws"

        questions_by_qid = {str(row["qid"]): row for row in load_questions(_BaseConfig())}
        check_split_leakage(args.dataset_dir, questions_by_qid, allow_positive_aid_overlap=args.allow_positive_aid_overlap)
        check_bge_training_split_leakage(args.dataset_dir)
        check_dense_index_checkpoint(args.dataset_dir)
        check_bge_candidate_pool(args.dataset_dir, top_k=args.top_k)
        check_question_text_not_in_corpus(
            args.dataset_dir,
            questions_by_qid,
            min_words=args.question_text_leak_min_words,
            allow_question_text_in_corpus=args.allow_question_text_in_corpus,
        )
        check_training_ground_truth(args.dataset_dir, questions_by_qid)
    check_val_threshold_application(eval_dir, summary)
    check_val_topk_application(eval_dir, summary)
    check_val_threshold_is_best_f2(args.dataset_dir, questions_by_qid)
    check_val_topk_is_best_f2(args.dataset_dir, questions_by_qid)

    router_metrics_path = eval_dir / "router_metrics.json"
    if router_metrics_path.exists():
        router_metrics = read_json(router_metrics_path)
        router_train = router_metrics.get("router_train", {})
        if router_train.get("alpha_convention") and router_train.get("alpha_convention") != "alpha_is_weight_bge":
            raise AssertionError(f"Unexpected router alpha convention: {router_train.get('alpha_convention')}")
        if router_train.get("convention_self_check_passed") is False:
            raise AssertionError("Router alpha convention self-check failed.")
        print(
            "[router]",
            {
                "alpha_convention": router_train.get("alpha_convention"),
                "label_gt_0.5_means": router_train.get("label_gt_0.5_means"),
                "alpha_label_gt_0.5_rate": router_train.get("alpha_label_gt_0.5_rate"),
                "alpha_pred_gt_0.5_rate": router_train.get("alpha_pred_gt_0.5_rate"),
                "binary_accuracy_alpha_gt_0.5": router_train.get("binary_accuracy_alpha_gt_0.5"),
                "alpha_label_pred_correlation": router_train.get("alpha_label_pred_correlation"),
            },
        )

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
            checked += 1
            print(f"[ok] {key} matches {detail_path.name}")

    retrieval_dir = args.dataset_dir / "retrieval_cache"
    for split in ["val", "test"]:
        for stem in ["hybrid_fixed_scores", "hybrid_router_scores"]:
            hybrid_path = retrieval_dir / f"{stem}_{split}.parquet"
            if hybrid_path.exists():
                hybrid_rows = read_records(hybrid_path)
                check_hybrid_formula(hybrid_rows, name=hybrid_path.name)
                check_aid_labels(hybrid_rows, name=hybrid_path.name)
                if questions_by_qid:
                    check_rows_against_questions(hybrid_rows, questions_by_qid, name=hybrid_path.name)
                print(f"[ok] {hybrid_path.name} uses normalized BM25/BGE scores in hybrid formula")

    qwen_path = retrieval_dir / "qwen_rerank_scores.parquet"
    if qwen_path.exists():
        check_qwen_metadata(qwen_path, candidate_top_k=args.candidate_top_k)

    for split in ["val", "test"]:
        path = retrieval_dir / f"bge_rerank_scores_{split}.parquet"
        if not path.exists():
            continue
        marker_path = done_path(path)
        if not marker_path.exists():
            raise AssertionError(f"missing marker: {marker_path}")
        marker = read_json(marker_path)
        params = marker.get("params", {})
        expected_params = {
            "schema_version": 3,
            "candidate_top_k": args.candidate_top_k,
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "source": "hybrid_router",
            "split": split,
        }
        for key, expected_value in expected_params.items():
            if params.get(key) != expected_value:
                raise AssertionError(f"{marker_path.name} params[{key!r}]={params.get(key)!r}, expected {expected_value!r}")
        if args.rerank_model and marker.get("model") != args.rerank_model:
            raise AssertionError(f"{marker_path.name} model={marker.get('model')!r}, expected {args.rerank_model!r}")
        print(f"[ok] {marker_path.name} metadata matches schema/source/top_k/unit/split")
        rows = read_records(path)
        required_columns = {"qid", "aid", "chunk_id", "chunk_text", "rerank_score", "label"}
        if rows:
            missing_columns = sorted(required_columns - set(rows[0].keys()))
            if missing_columns:
                raise AssertionError(f"{path.name} missing columns: {missing_columns}")
        check_aid_labels(rows, name=path.name)
        if questions_by_qid:
            check_rows_against_questions(rows, questions_by_qid, name=path.name)
        source_path = retrieval_dir / f"hybrid_router_scores_{split}.parquet"
        if not source_path.exists():
            source_path = retrieval_dir / "hybrid_router_scores.parquet"
        if not source_path.exists():
            raise AssertionError(f"missing rerank source cache: hybrid_router_scores_{split}.parquet or hybrid_router_scores.parquet")
        source_rows = read_records(source_path)
        source_rows = [row for row in source_rows if str(row["qid"]) in {str(item["qid"]) for item in rows}]
        if source_rows and "hybrid_score" not in source_rows[0]:
            raise AssertionError(f"{source_path.name} is not a hybrid_router source: missing hybrid_score")
        allowed_pairs = top_pairs(source_rows, score_field="hybrid_score", top_k=args.candidate_top_k)
        candidate_pairs = {(str(row["qid"]), str(row["aid"])) for row in rows}
        extra_pairs = sorted(candidate_pairs - allowed_pairs)[:5]
        if extra_pairs:
            raise AssertionError(f"bge_rerank_scores_{split} contains candidates outside hybrid_router top{args.candidate_top_k}: {extra_pairs}")
        print(f"[ok] bge_rerank_scores_{split} candidates come from hybrid_router top{args.candidate_top_k}")
        counts = Counter(str(row["qid"]) for row in rows)
        overflow = {qid: count for qid, count in counts.items() if count > args.candidate_top_k}
        if overflow:
            examples = sorted(overflow.items(), key=lambda item: item[1], reverse=True)[:5]
            raise AssertionError(f"bge_rerank_scores_{split} exceeds candidate_top_k={args.candidate_top_k}: {examples}")
        if counts:
            print(f"[ok] bge_rerank_scores_{split} max_candidates_per_qid={max(counts.values())}")
        qid_chunk = Counter((str(row["qid"]), str(row.get("chunk_id", ""))) for row in rows)
        qid_aid_chunk = Counter((str(row["qid"]), str(row["aid"]), str(row.get("chunk_id", ""))) for row in rows)
        qid_aid = Counter((str(row["qid"]), str(row["aid"])) for row in rows)
        dup_qid_chunk = sum(count - 1 for count in qid_chunk.values() if count > 1)
        dup_qid_aid_chunk = sum(count - 1 for count in qid_aid_chunk.values() if count > 1)
        dup_qid_aid = sum(count - 1 for count in qid_aid.values() if count > 1)
        print(
            f"[duplicates] bge_rerank_scores_{split}: "
            f"duplicated(qid, chunk_id)={dup_qid_chunk}, "
            f"duplicated(qid, aid, chunk_id)={dup_qid_aid_chunk}, "
            f"duplicated(qid, aid)={dup_qid_aid}"
        )
        if dup_qid_chunk or dup_qid_aid_chunk:
            raise AssertionError(f"bge_rerank_scores_{split} has duplicate chunk-level candidates")
        if args.questions_path:
            class _Config:
                questions_path = args.questions_path
                question_id_field = "qid"
                question_text_field = "question"
                relevant_ids_field = "relevant_laws"

            question_text_by_qid = {str(row["qid"]): row["question"] for row in load_questions(_Config())}
            missing_query = [str(row["qid"]) for row in rows if str(row["qid"]) not in question_text_by_qid]
            empty_query = [str(row["qid"]) for row in rows if str(row["qid"]) in question_text_by_qid and not str(question_text_by_qid[str(row["qid"])]).strip()]
            if missing_query or empty_query:
                raise AssertionError(f"bge_rerank_scores_{split} query issues: missing={missing_query[:5]} empty={empty_query[:5]}")
            print(f"[text] bge_rerank_scores_{split} query token lengths:", text_len_stats([question_text_by_qid[str(row["qid"])] for row in rows]))
        empty_chunk_text = [(str(row["qid"]), str(row["aid"]), str(row.get("chunk_id", ""))) for row in rows if not str(row.get("chunk_text", "")).strip()]
        if empty_chunk_text:
            raise AssertionError(f"bge_rerank_scores_{split} has empty chunk_text examples: {empty_chunk_text[:5]}")
        print(f"[text] bge_rerank_scores_{split} chunk_text token lengths:", text_len_stats([str(row.get("chunk_text", "")) for row in rows]))
        chunk_map_path = args.dataset_dir / "prepared" / "chunk_to_aid.json"
        if chunk_map_path.exists():
            chunk_to_aid = read_json(chunk_map_path)
            mismatches = []
            for row in rows:
                chunk_id = str(row.get("chunk_id", ""))
                aid = str(row["aid"])
                mapped_aid = str(chunk_to_aid.get(chunk_id, ""))
                if mapped_aid and mapped_aid != aid:
                    mismatches.append((str(row["qid"]), aid, chunk_id, mapped_aid))
            if mismatches:
                raise AssertionError(f"bge_rerank_scores_{split} chunk_id->aid mismatches: {mismatches[:5]}")
            print(f"[ok] bge_rerank_scores_{split} chunk_id maps to aid")
        positive_scores = [float(row["rerank_score"]) for row in rows if int(row.get("label", 0)) == 1]
        negative_scores = [float(row["rerank_score"]) for row in rows if int(row.get("label", 0)) == 0]
        pos_desc = describe(positive_scores)
        neg_desc = describe(negative_scores)
        print(f"[score] bge_rerank_scores_{split} positive rerank_score:", pos_desc)
        print(f"[score] bge_rerank_scores_{split} negative rerank_score:", neg_desc)
        if positive_scores and negative_scores and float(pos_desc["mean"]) < float(neg_desc["mean"]):
            print(f"[warn] bge_rerank_scores_{split}: positive mean rerank_score is lower than negative mean. Check model quality or score direction.")
        detail_path = eval_dir / f"bge_rerank_{split}_metrics.json"
        if args.questions_path and detail_path.exists():
            splits_path = args.dataset_dir / "prepared" / "splits.json"
            split_qids = set(map(str, read_json(splits_path)[split]))
            questions = [row for row in load_questions(_Config()) if str(row["qid"]) in split_qids]
            detail = read_json(detail_path)
            recomputed_ranking = ranking_metrics(rows, questions)
            score_field = str(detail.get("score_field", "rerank_score"))
            recomputed_threshold = threshold_metrics(rows, questions, score_field=score_field, threshold=detail["threshold"]["threshold"])
            recomputed_topk = topk_metrics(rows, questions, score_field=score_field, k=int(detail["topk_tuned"]["top_k"]))
            for metric in ["hit@10", "recall@10", "ndcg@10"]:
                assert_close(f"bge_rerank_{split}.{metric}.recomputed", detail["ranking"][metric], recomputed_ranking[metric])
            for metric in ["precision", "recall", "f2"]:
                assert_close(f"bge_rerank_{split}.{metric}.recomputed", detail["threshold"][metric], recomputed_threshold[metric])
                assert_close(f"bge_rerank_{split}.topk_{metric}.recomputed", detail["topk_tuned"][metric], recomputed_topk[metric])
            print(f"[ok] bge_rerank_scores_{split} recomputes with aid_score=max(chunk_scores)")

    for split in ["val", "test"]:
        path = retrieval_dir / f"llm_rerank_scores_{split}.parquet"
        if not path.exists():
            continue
        marker_path = done_path(path)
        if not marker_path.exists():
            raise AssertionError(f"missing marker: {marker_path}")
        marker = read_json(marker_path)
        params = marker.get("params", {})
        expected_params = {
            "schema_version": 1,
            "candidate_top_k": args.llm_rerank_top_k,
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "source": "bge_rerank",
            "split": split,
        }
        for key, expected_value in expected_params.items():
            if params.get(key) != expected_value:
                raise AssertionError(f"{marker_path.name} params[{key!r}]={params.get(key)!r}, expected {expected_value!r}")
        rows = read_records(path)
        counts = Counter(str(row["qid"]) for row in rows)
        overflow = {qid: count for qid, count in counts.items() if count > args.llm_rerank_top_k}
        if overflow:
            raise AssertionError(f"llm_rerank_scores_{split} exceeds top_k={args.llm_rerank_top_k}: {sorted(overflow.items())[:5]}")
        qid_chunk = Counter((str(row["qid"]), str(row.get("chunk_id", ""))) for row in rows)
        dup_qid_chunk = sum(count - 1 for count in qid_chunk.values() if count > 1)
        if dup_qid_chunk:
            raise AssertionError(f"llm_rerank_scores_{split} has duplicate chunk-level candidates: {dup_qid_chunk}")
        bad_scores = [float(row.get("llm_rerank_score", -1.0)) for row in rows if not (0.0 <= float(row.get("llm_rerank_score", -1.0)) <= 1.0)]
        if bad_scores:
            raise AssertionError(f"llm_rerank_scores_{split} has scores outside [0,1]: {bad_scores[:5]}")
        source_path = retrieval_dir / f"bge_rerank_scores_{split}.parquet"
        if not source_path.exists():
            raise AssertionError(f"missing LLM rerank source cache: {source_path}")
        source_rows = read_records(source_path)
        allowed = {
            (str(row["qid"]), str(row.get("chunk_id", "")))
            for row in sorted(source_rows, key=lambda item: (str(item["qid"]), -float(item.get("rerank_score", 0.0))))
        }
        source_top = {}
        for row in source_rows:
            source_top.setdefault(str(row["qid"]), []).append(row)
        allowed = {
            (qid, str(row.get("chunk_id", "")))
            for qid, items in source_top.items()
            for row in sorted(items, key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)[: args.llm_rerank_top_k]
        }
        extra = sorted({(str(row["qid"]), str(row.get("chunk_id", ""))) for row in rows} - allowed)[:5]
        if extra:
            raise AssertionError(f"llm_rerank_scores_{split} contains candidates outside bge_rerank top{args.llm_rerank_top_k}: {extra}")
        print(f"[ok] llm_rerank_scores_{split} uses bge_rerank top{args.llm_rerank_top_k} chunk candidates with scores in [0,1]")

    print(f"[done] checked={checked}")


if __name__ == "__main__":
    main()
