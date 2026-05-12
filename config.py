from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


STAGES = [
    "prepare_data",
    "split",
    "prepare_training_data",
    "tune_bm25",
    "tune_and_build_bm25",
    "build_bm25",
    "build_dense_index",
    "mine_hard_negatives",
    "sample_negatives",
    "train_bge",
    "train_bge_retriever",
    "train_reranker",
    "retrieve_cache",
    "tune_hybrid",
    "train_router",
    "rerank_bge",
    "rerank_qwen",
    "train_llm_reranker",
    "rerank_llm",
    "evaluate",
    "all",
]


@dataclass(frozen=True)
class Config:
    stage: str
    dataset_name: str
    corpus_path: Path
    questions_path: Path
    output_dir: Path
    force: bool
    seed: int

    dense_model: str
    rerank_model: str
    qwen_model: str
    use_qwen_rerank: bool
    llm_rerank_model: str
    llm_rerank_backend: str
    use_llm_rerank: bool
    llm_rerank_top_k: int
    llm_rerank_train_batch_size: int
    llm_rerank_batch_size: int
    llm_rerank_grad_accum: int
    llm_rerank_epochs: int
    llm_rerank_lr: float
    llm_rerank_max_length: int
    llm_rerank_max_train_examples: int
    llm_rerank_lora_r: int
    llm_rerank_lora_alpha: int
    llm_rerank_load_in_4bit: bool
    device: str
    batch_size: int
    bge_train_batch_size: int
    bge_epochs: int
    bge_lr: float
    bge_warmup_ratio: float
    bge_max_length: int
    bge_max_train_examples: int
    bge_use_amp: bool
    bge_gradient_checkpointing: bool
    bge_auto_batch_reduce: bool
    bge_negatives_per_example: int
    bge_grad_accum: int
    reranker_train_batch_size: int
    reranker_epochs: int
    reranker_lr: float
    reranker_warmup_ratio: float
    reranker_max_length: int
    reranker_max_train_examples: int
    reranker_use_amp: bool
    reranker_grad_accum: int

    bm25_k1: float
    bm25_b: float
    bm25_k1_grid: str
    bm25_b_grid: str
    bm25_tune_metric: str
    use_tuned_bm25: bool
    hybrid_alpha: float
    alpha_grid: str
    router_model: str
    top_k: int
    candidate_top_k: int
    positive_chunks_per_aid: int
    threshold: float

    train_ratio: float
    router_train_ratio: float
    val_ratio: float
    test_ratio: float
    max_chunk_tokens: int
    chunk_overlap_sentences: int
    max_chunks: int
    max_chunks_per_article: int

    corpus_law_id_field: str
    corpus_articles_field: str
    article_id_field: str
    article_text_field: str
    question_id_field: str
    question_text_field: str
    relevant_ids_field: str

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / self.dataset_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular retrieval pipeline")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--questions_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--force", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dense_model", default="BAAI/bge-m3")
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--use_qwen_rerank", type=str2bool, default=False)
    parser.add_argument("--llm_rerank_model", default="unsloth/Qwen3.5-4B-Base")
    parser.add_argument("--llm_rerank_backend", default="unsloth_causal_lm", choices=["unsloth_causal_lm"])
    parser.add_argument("--use_llm_rerank", type=str2bool, default=False)
    parser.add_argument("--llm_rerank_top_k", type=int, default=20)
    parser.add_argument("--llm_rerank_train_batch_size", type=int, default=1)
    parser.add_argument("--llm_rerank_batch_size", type=int, default=16)
    parser.add_argument("--llm_rerank_grad_accum", type=int, default=2)
    parser.add_argument("--llm_rerank_epochs", type=int, default=1)
    parser.add_argument("--llm_rerank_lr", type=float, default=2e-5)
    parser.add_argument("--llm_rerank_max_length", type=int, default=630)
    parser.add_argument("--llm_rerank_max_train_examples", type=int, default=12000)
    parser.add_argument("--llm_rerank_lora_r", type=int, default=16)
    parser.add_argument("--llm_rerank_lora_alpha", type=int, default=16)
    parser.add_argument("--llm_rerank_load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bge_train_batch_size", type=int, default=16)
    parser.add_argument("--bge_epochs", type=int, default=2)
    parser.add_argument("--bge_lr", type=float, default=2e-5)
    parser.add_argument("--bge_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--bge_max_length", type=int, default=522)
    parser.add_argument("--bge_max_train_examples", type=int, default=0)
    parser.add_argument("--bge_use_amp", type=str2bool, default=True)
    parser.add_argument("--bge_gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--bge_auto_batch_reduce", type=str2bool, default=True)
    parser.add_argument("--bge_negatives_per_example", type=int, default=3)
    parser.add_argument("--bge_grad_accum", type=int, default=1)
    parser.add_argument("--reranker_train_batch_size", type=int, default=4)
    parser.add_argument("--reranker_epochs", type=int, default=1)
    parser.add_argument("--reranker_lr", type=float, default=2e-5)
    parser.add_argument("--reranker_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--reranker_max_length", type=int, default=582)
    parser.add_argument("--reranker_max_train_examples", type=int, default=0)
    parser.add_argument("--reranker_use_amp", type=str2bool, default=True)
    parser.add_argument("--reranker_grad_accum", type=int, default=2)

    parser.add_argument("--bm25_k1", type=float, default=1.2)
    parser.add_argument("--bm25_b", type=float, default=0.9)
    parser.add_argument("--bm25_k1_grid", default="1.2")
    parser.add_argument("--bm25_b_grid", default="0.9")
    parser.add_argument("--bm25_tune_metric", default="recall@20")
    parser.add_argument("--use_tuned_bm25", type=str2bool, default=True)
    parser.add_argument("--hybrid_alpha", type=float, default=0.5)
    parser.add_argument("--alpha_grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--router_model", default="ridge", choices=["ridge"])
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--candidate_top_k", type=int, default=50)
    parser.add_argument("--positive_chunks_per_aid", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.75)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--router_train_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--max_chunk_tokens", type=int, default=512)
    parser.add_argument("--chunk_overlap_sentences", type=int, default=1)
    parser.add_argument("--max_chunks", type=int, default=0, help="If >0, fail prepare_data when total chunks exceeds this value.")
    parser.add_argument("--max_chunks_per_article", type=int, default=0, help="If >0, keep only this many chunks per article.")

    parser.add_argument("--corpus_law_id_field", default="law_id")
    parser.add_argument("--corpus_articles_field", default="content")
    parser.add_argument("--article_id_field", default="aid")
    parser.add_argument("--article_text_field", default="content_Article")
    parser.add_argument("--question_id_field", default="qid")
    parser.add_argument("--question_text_field", default="question")
    parser.add_argument("--relevant_ids_field", default="relevant_laws")
    return parser


def parse_args() -> Config:
    args = build_parser().parse_args()
    ratios = args.train_ratio + args.router_train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError("--train_ratio + --router_train_ratio + --val_ratio + --test_ratio must equal 1.0")
    return Config(**vars(args))
