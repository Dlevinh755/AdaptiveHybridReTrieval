from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics, topk_metrics, tune_threshold, tune_top_k
from src.retrieval.hybrid import apply_hybrid
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, read_table, retrieval_dir, stable_hash, write_json, write_jsonl, write_pickle, write_table
from src.utils.logging import saved, skip


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ROUTER_SCHEMA_VERSION = 3
ROUTER_LABEL_TEMPERATURE = 0.2


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _build_tfidf(texts: list[str]) -> dict[str, Any]:
    doc_tokens = [_tokenize(text) for text in texts]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    vocab = {term: idx for idx, (term, _count) in enumerate(sorted(doc_freq.items()))}
    total_docs = max(len(texts), 1)
    idf = {
        term: math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }
    return {"vocab": vocab, "idf": idf}


def _transform_tfidf(texts: list[str], vectorizer: dict[str, Any]) -> np.ndarray:
    vocab: dict[str, int] = vectorizer["vocab"]
    idf: dict[str, float] = vectorizer["idf"]
    features = np.zeros((len(texts), len(vocab) + 1), dtype=np.float64)
    features[:, 0] = 1.0
    for row_idx, text in enumerate(texts):
        counts = Counter(_tokenize(text))
        total = max(sum(counts.values()), 1)
        for term, count in counts.items():
            col_idx = vocab.get(term)
            if col_idx is None:
                continue
            features[row_idx, col_idx + 1] = (count / total) * float(idf.get(term, 0.0))
    return features


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _group_rows(rows: list[dict[str, Any]], score_field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["qid"])].append(row)
    return {
        qid: sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)
        for qid, items in grouped.items()
    }


def _recall_at_k_for_qid(rows: list[dict[str, Any]], question: dict[str, Any], k: int = 10) -> float:
    positives = {str(aid) for aid in question["relevant_laws"]}
    hits = {str(row["aid"]) for row in rows[:k]}
    return len(hits & positives) / max(len(positives), 1)


def _ndcg_at_k_for_qid(rows: list[dict[str, Any]], question: dict[str, Any], k: int = 10) -> float:
    positives = {str(aid) for aid in question["relevant_laws"]}
    labels = [1 if str(row["aid"]) in positives else 0 for row in rows[:k]]
    dcg = sum(label / math.log2(idx + 2) for idx, label in enumerate(labels))
    ideal = [1] * min(len(positives), k)
    idcg = sum(label / math.log2(idx + 2) for idx, label in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _make_alpha_labels(router_rows: list[dict[str, Any]], router_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bm25_grouped = _group_rows(router_rows, "bm25_score")
    bge_grouped = _group_rows(router_rows, "bge_score")
    labels = []
    for question in router_questions:
        qid = str(question["qid"])
        bm25_ranked = bm25_grouped.get(qid, [])
        bge_ranked = bge_grouped.get(qid, [])
        r10_bm25 = _recall_at_k_for_qid(bm25_ranked, question, k=10)
        r10_bge = _recall_at_k_for_qid(bge_ranked, question, k=10)
        ndcg10_bm25 = _ndcg_at_k_for_qid(bm25_ranked, question, k=10)
        ndcg10_bge = _ndcg_at_k_for_qid(bge_ranked, question, k=10)
        r20_bm25 = _recall_at_k_for_qid(bm25_ranked, question, k=20)
        r20_bge = _recall_at_k_for_qid(bge_ranked, question, k=20)
        r50_bm25 = _recall_at_k_for_qid(bm25_ranked, question, k=50)
        r50_bge = _recall_at_k_for_qid(bge_ranked, question, k=50)
        ndcg20_bm25 = _ndcg_at_k_for_qid(bm25_ranked, question, k=20)
        ndcg20_bge = _ndcg_at_k_for_qid(bge_ranked, question, k=20)
        dense_perf = 0.2 * r50_bge + 0.6 * r20_bge + 0.2 * ndcg20_bge
        sparse_perf = 0.2 * r50_bm25 + 0.6 * r20_bm25 + 0.2 * ndcg20_bm25
        alpha_soft = _sigmoid((dense_perf - sparse_perf) / ROUTER_LABEL_TEMPERATURE)
        labels.append(
            {
                "qid": qid,
                "question": question["question"],
                "r10_bm25": r10_bm25,
                "r10_bge": r10_bge,
                "ndcg10_bm25": ndcg10_bm25,
                "ndcg10_bge": ndcg10_bge,
                "r20_bm25": r20_bm25,
                "r20_bge": r20_bge,
                "r50_bm25": r50_bm25,
                "r50_bge": r50_bge,
                "ndcg20_bm25": ndcg20_bm25,
                "ndcg20_bge": ndcg20_bge,
                "dense_perf": dense_perf,
                "sparse_perf": sparse_perf,
                "alpha_soft": alpha_soft,
                "alpha_convention": "alpha_is_weight_bge",
                "preferred_by_label": "bge" if alpha_soft > 0.5 else "bm25" if alpha_soft < 0.5 else "tie",
            }
        )
    return labels


def _fit_ridge(features: np.ndarray, targets: np.ndarray, reg: float = 1.0) -> np.ndarray:
    penalty = reg * np.eye(features.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(features.T @ features + penalty) @ features.T @ targets


def _predict_alpha(questions: list[dict[str, Any]], vectorizer: dict[str, Any], weights: np.ndarray) -> dict[str, float]:
    features = _transform_tfidf([row["question"] for row in questions], vectorizer)
    predictions = features @ weights
    return {
        str(row["qid"]): max(0.0, min(1.0, float(alpha)))
        for row, alpha in zip(questions, predictions)
    }


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if str(row["qid"]) in qids]


def _write_hybrid_split(
    config: Any,
    split_name: str,
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    alpha_by_qid: dict[str, float] | None,
    fixed_alpha: float,
    output_name: str,
    params: dict[str, Any],
    input_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    ranked = apply_hybrid(rows, alpha_by_qid=alpha_by_qid, fixed_alpha=fixed_alpha)
    score_path = retrieval_dir(config) / f"{output_name}_scores_{split_name}.parquet"
    fmt = write_table(score_path, ranked)
    threshold = tune_threshold(ranked, questions, score_field="hybrid_score") if split_name == "val" else None
    topk = tune_top_k(ranked, questions, score_field="hybrid_score") if split_name == "val" else None
    mark_done(score_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params={**params, "split": split_name, "output": output_name}, fmt=fmt)
    return ranked, threshold or {}, topk or {}


def _write_hybrid_metrics(
    config: Any,
    split_name: str,
    output_name: str,
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    threshold: float,
    tuned_top_k: int,
    params: dict[str, Any],
    input_hash: str,
) -> dict[str, Any]:
    payload = {
        "split": split_name,
        "score_field": "hybrid_score",
        "ranking": ranking_metrics(rows, questions),
        "threshold": {
            "threshold": threshold,
            **threshold_metrics(rows, questions, score_field="hybrid_score", threshold=threshold),
        },
        "topk_fixed_3": {
            "top_k": 3,
            **topk_metrics(rows, questions, score_field="hybrid_score", k=3),
        },
        "topk_tuned": {
            "top_k": tuned_top_k,
            **topk_metrics(rows, questions, score_field="hybrid_score", k=tuned_top_k),
        },
        "topk_selection_split": "val",
        "threshold_selection_split": "val",
        "params": params,
        "num_questions": len(questions),
        "num_rows": len(rows),
    }
    metrics_path = eval_dir(config) / f"{output_name}_{split_name}_metrics.json"
    write_json(metrics_path, payload)
    mark_done(metrics_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params={**params, "split": split_name, "output": output_name}, fmt="json")
    return payload


def _binary_rate(values: list[float], threshold: float = 0.5) -> float:
    return float(np.mean([value > threshold for value in values])) if values else 0.0


def _alpha_diagnostics(labels: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    y = np.asarray([float(row["alpha_soft"]) for row in labels], dtype=np.float64)
    clipped = np.clip(predictions, 0.0, 1.0)
    dense_minus_sparse = np.asarray([float(row["dense_perf"]) - float(row["sparse_perf"]) for row in labels], dtype=np.float64)
    label_sign = y > 0.5
    pred_sign = clipped > 0.5
    convention_ok = bool(np.all((dense_minus_sparse > 0) == label_sign) and np.all((dense_minus_sparse < 0) == (y < 0.5)))
    correlation = 0.0
    if len(y) > 1 and float(np.std(clipped)) > 0 and float(np.std(y)) > 0:
        correlation = float(np.corrcoef(clipped, y)[0, 1])
    return {
        "alpha_convention": "alpha_is_weight_bge",
        "hybrid_formula": "hybrid_score = alpha * bge_score_norm + (1 - alpha) * bm25_score_norm",
        "label_formula": "sigmoid((dense_perf - sparse_perf) / temperature)",
        "label_gt_0.5_means": "BGE preferred",
        "label_lt_0.5_means": "BM25 preferred",
        "convention_self_check_passed": convention_ok,
        "alpha_label_mean": float(np.mean(y)) if len(y) else 0.0,
        "alpha_pred_mean": float(np.mean(clipped)) if len(clipped) else 0.0,
        "alpha_label_gt_0.5_rate": _binary_rate([float(value) for value in y]),
        "alpha_pred_gt_0.5_rate": _binary_rate([float(value) for value in clipped]),
        "binary_accuracy_alpha_gt_0.5": float(np.mean(pred_sign == label_sign)) if len(y) else 0.0,
        "alpha_label_pred_correlation": correlation,
        "num_router_labels": int(len(y)),
    }


def train_router(config: Any) -> None:
    model_path = config.dataset_dir / "models" / "router_alpha_regressor.joblib"
    labels_path = eval_dir(config) / "router_alpha_labels.jsonl"
    metrics_path = eval_dir(config) / "router_metrics.json"
    config_path = eval_dir(config) / "router_config.json"
    expected_params = {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "top_k": config.top_k,
        "label_temperature": ROUTER_LABEL_TEMPERATURE,
        "model": "tfidf_ridge",
    }
    expected = {"model": config.router_model, "params": expected_params}
    expected_outputs = [
        model_path,
        labels_path,
        metrics_path,
        config_path,
        retrieval_dir(config) / "hybrid_fixed_scores_val.parquet",
        retrieval_dir(config) / "hybrid_fixed_scores_test.parquet",
        retrieval_dir(config) / "hybrid_router_scores_val.parquet",
        retrieval_dir(config) / "hybrid_router_scores_test.parquet",
    ]
    if all(is_complete(path, expected=expected) for path in expected_outputs[:1]) and all(is_complete(path) for path in expected_outputs[1:]) and not config.force:
        skip(model_path)
        return

    questions = load_questions(config)
    questions_by_qid = {str(row["qid"]): row for row in questions}
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    split_questions = {
        "router": _filter_questions(questions, set(map(str, splits.get("router", splits.get("router_train", []))))),
        "val": _filter_questions(questions, set(map(str, splits["val"]))),
        "test": _filter_questions(questions, set(map(str, splits["test"]))),
    }
    if not split_questions["router"]:
        raise ValueError("Router split is empty. Cannot train router alpha regressor.")

    split_rows = {
        split_name: read_table(retrieval_dir(config) / f"merged_scores_{split_name}.parquet")
        for split_name in ["router", "val", "test"]
    }
    router_labels = _make_alpha_labels(split_rows["router"], split_questions["router"])
    if not router_labels:
        raise ValueError("No router alpha labels were created.")

    vectorizer = _build_tfidf([row["question"] for row in router_labels])
    x = _transform_tfidf([row["question"] for row in router_labels], vectorizer)
    y = np.asarray([float(row["alpha_soft"]) for row in router_labels], dtype=np.float64)
    weights = _fit_ridge(x, y, reg=1.0)
    train_predictions = x @ weights
    clipped_train_predictions = np.clip(train_predictions, 0.0, 1.0)
    mse = float(np.mean((clipped_train_predictions - y) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(clipped_train_predictions - y)))
    alpha_diagnostics = _alpha_diagnostics(router_labels, train_predictions)

    alpha_by_split = {
        split_name: _predict_alpha(rows_questions, vectorizer, weights)
        for split_name, rows_questions in split_questions.items()
    }
    for row in router_labels:
        row["alpha_pred"] = alpha_by_split["router"].get(str(row["qid"]), 0.5)

    write_pickle(
        model_path,
        {
            "model_type": "tfidf_ridge",
            "schema_version": ROUTER_SCHEMA_VERSION,
            "vocab": vectorizer["vocab"],
            "idf": vectorizer["idf"],
            "weights": weights.tolist(),
        },
    )
    write_jsonl(labels_path, router_labels)
    write_json(
        config_path,
        {
            "model_type": "tfidf_ridge",
            "label_formula": "sigmoid(((0.6*r50_bge + 0.2*r20_bge + 0.2*ndcg20_bge) - (0.6*r50_bm25 + 0.2*r20_bm25 + 0.2*ndcg20_bm25)) / temperature)",
            "alpha_convention": "alpha_is_weight_bge",
            "hybrid_formula": "hybrid_score = alpha * bge_score_norm + (1 - alpha) * bm25_score_norm",
            "label_temperature": ROUTER_LABEL_TEMPERATURE,
            "train_split": "router",
            "fixed_alpha": 0.5,
            "vocab_size": len(vectorizer["vocab"]),
        },
    )

    input_hash = stable_hash({"router_qids": [row["qid"] for row in router_labels], "params": expected_params})
    fixed_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    router_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    fixed_val_threshold: dict[str, Any] = {}
    router_val_threshold: dict[str, Any] = {}
    fixed_val_topk: dict[str, Any] = {}
    router_val_topk: dict[str, Any] = {}

    for split_name in ["router", "val", "test"]:
        fixed_rows, fixed_threshold, fixed_topk = _write_hybrid_split(
            config,
            split_name,
            split_rows[split_name],
            split_questions[split_name],
            alpha_by_qid=None,
            fixed_alpha=0.5,
            output_name="hybrid_fixed",
            params=expected_params,
            input_hash=input_hash,
        )
        router_rows, router_threshold, router_topk = _write_hybrid_split(
            config,
            split_name,
            split_rows[split_name],
            split_questions[split_name],
            alpha_by_qid=alpha_by_split[split_name],
            fixed_alpha=0.5,
            output_name="hybrid_router",
            params=expected_params,
            input_hash=input_hash,
        )
        fixed_rows_by_split[split_name] = fixed_rows
        router_rows_by_split[split_name] = router_rows
        if split_name == "val":
            fixed_val_threshold = fixed_threshold
            router_val_threshold = router_threshold
            fixed_val_topk = fixed_topk
            router_val_topk = router_topk

    fixed_threshold_value = float(fixed_val_threshold.get("threshold", 0.0))
    router_threshold_value = float(router_val_threshold.get("threshold", 0.0))
    fixed_topk_value = int(fixed_val_topk.get("top_k", 3))
    router_topk_value = int(router_val_topk.get("top_k", 3))
    metrics_payload: dict[str, Any] = {
        "router_train": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            **alpha_diagnostics,
        },
        "fixed_alpha": 0.5,
        "router_model": "tfidf_ridge",
        "label_temperature": ROUTER_LABEL_TEMPERATURE,
        "alpha_convention": "alpha_is_weight_bge",
        "hybrid_formula": "hybrid_score = alpha * bge_score_norm + (1 - alpha) * bm25_score_norm",
        "splits": {},
    }
    for split_name in ["router", "val", "test"]:
        metrics_payload["splits"][f"hybrid_fixed_{split_name}"] = _write_hybrid_metrics(
            config,
            split_name,
            "hybrid_fixed",
            fixed_rows_by_split[split_name],
            split_questions[split_name],
            threshold=fixed_threshold_value,
            tuned_top_k=fixed_topk_value,
            params=expected_params,
            input_hash=input_hash,
        )
        metrics_payload["splits"][f"hybrid_router_{split_name}"] = _write_hybrid_metrics(
            config,
            split_name,
            "hybrid_router",
            router_rows_by_split[split_name],
            split_questions[split_name],
            threshold=router_threshold_value,
            tuned_top_k=router_topk_value,
            params=expected_params,
            input_hash=input_hash,
        )

    fixed_all = fixed_rows_by_split["router"] + fixed_rows_by_split["val"] + fixed_rows_by_split["test"]
    router_all = router_rows_by_split["router"] + router_rows_by_split["val"] + router_rows_by_split["test"]
    fixed_fmt = write_table(retrieval_dir(config) / "hybrid_fixed_scores.parquet", fixed_all)
    router_fmt = write_table(retrieval_dir(config) / "hybrid_router_scores.parquet", router_all)
    mark_done(retrieval_dir(config) / "hybrid_fixed_scores.parquet", config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=expected_params, fmt=fixed_fmt)
    mark_done(retrieval_dir(config) / "hybrid_router_scores.parquet", config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=expected_params, fmt=router_fmt)

    write_json(metrics_path, metrics_payload)
    mark_done(model_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=expected_params, fmt="pickle")
    mark_done(labels_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=expected_params, fmt="jsonl")
    mark_done(config_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=expected_params, fmt="json")
    mark_done(metrics_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=expected_params, fmt="json")
    saved(model_path)
