# Modular Legal Retrieval Pipeline

Default workflow: run grouped stages first, then train/evaluate. Individual stages are still available for debugging or rebuilding a specific artifact.

Minimal smoke test for the full pipeline:

```bash
python3 run.py \
  --stage all \
  --dataset_name mini \
  --corpus_path examples/mini_corpus.json \
  --questions_path examples/mini_questions.json \
  --output_dir outputs \
  --top_k 5
```

Useful stages:

```text
prepare_training_data
-> train_bge_retriever -> train_reranker
-> tune_bm25 -> build_bm25 -> retrieve_cache -> tune_hybrid
-> train_router -> rerank_bge -> train_llm_reranker(optional) -> rerank_llm(optional) -> rerank_qwen(optional) -> evaluate
```

`prepare_training_data` is the default way to run the data-preparation block. It replaces running these stages one by one:

```text
prepare_data -> split -> build_dense_index -> mine_hard_negatives -> sample_negatives
```

Prepare all training-data artifacts in one command:

```bash
python3 run.py \
  --stage prepare_training_data \
  --dataset_name legalraw_full \
  --corpus_path raw_data/legalraw/full/legal_corpus.json \
  --questions_path raw_data/legalraw/full/train.json \
  --output_dir outputs \
  --dense_model BAAI/bge-m3 \
  --device cuda \
  --batch_size 32 \
  --top_k 100
```

The pipeline writes `.done.json` markers next to completed artifacts. Re-running a stage skips completed work unless `--force true` is set.

Default qid split is:

```text
train=70%, router=10%, val=10%, test=10%
```

`router` is used only for the router soft-label regression. `router_train` is kept as a backward-compatible alias in `splits.json`. `val` remains clean for alpha tuning/evaluation, and `test` is never used for training or model selection.

Large tables use Parquet when `pandas` and a parquet engine are installed. Without them, the same paths are written as JSONL fallback and the `.done.json` metadata records `format: jsonl_fallback`.

## Kaggle LegalRaw Data

Download the dataset into `raw_data/legalraw/full`:

```bash
python3 -m pip install kagglehub
python3 scripts/download_legalraw.py
```

Create a path-compatible sample:

```bash
python3 scripts/make_legalraw_sample.py \
  --input_dir raw_data/legalraw/full \
  --output_dir raw_data/legalraw/sample_100q \
  --num_questions 100 \
  --num_extra_articles 300 \
  --seed 42
```

The sample script defaults to article-level sampling: it keeps only the relevant `aid` articles for sampled qids, then adds random extra non-positive `aid` articles. Avoid `--num_extra_laws` for smoke tests because one law can contain many articles and inflate chunk count.

Prepare training data on the sample by only changing paths:

```bash
python3 run.py \
  --stage prepare_training_data \
  --dataset_name legalraw_sample_100q \
  --corpus_path raw_data/legalraw/sample_100q/legal_corpus.json \
  --questions_path raw_data/legalraw/sample_100q/train.json \
  --output_dir outputs \
  --dense_model BAAI/bge-m3 \
  --device cuda \
  --batch_size 32 \
  --top_k 100
```

Prepare training data on full data with the same code and file structure:

```bash
python3 run.py \
  --stage prepare_training_data \
  --dataset_name legalraw_full \
  --corpus_path raw_data/legalraw/full/legal_corpus.json \
  --questions_path raw_data/legalraw/full/train.json \
  --output_dir outputs \
  --dense_model BAAI/bge-m3 \
  --device cuda \
  --batch_size 32 \
  --top_k 100
```

After `prepare_training_data` finishes, continue with model training stages such as `train_bge`, `train_reranker`, and the downstream retrieval/evaluation stages.

`train_bge_retriever` runs `train_bge -> build_dense_index` together. The first dense index from `prepare_training_data` is for mining hard negatives with the base model; the second dense index from `train_bge_retriever` uses `outputs/<dataset>/models/bge_finetuned` for retrieval/evaluation caches.

BGE retrieval training uses a custom Hugging Face `AutoModel` contrastive loop instead of `SentenceTransformer.fit(TripletLoss)`. Query, positive, and negative texts are encoded in separate forwards to reduce peak GPU memory. The saved output is still converted to a SentenceTransformer-compatible directory for dense indexing. The training stage supports memory guards:

```text
--bge_train_batch_size 4
--bge_max_length 384
--bge_use_amp true
--bge_gradient_checkpointing true
--bge_auto_batch_reduce true
```

If CUDA OOM still happens, lower `--bge_train_batch_size` to `2` or `1`, then reduce `--bge_negatives_per_example`.

BM25 tuning writes split-specific caches and metrics:

```text
outputs/<dataset>/retrieval_cache/bm25_scores_router.parquet
outputs/<dataset>/retrieval_cache/bm25_scores_val.parquet
outputs/<dataset>/retrieval_cache/bm25_scores_test.parquet
outputs/<dataset>/eval/bm25_router_metrics.json
outputs/<dataset>/eval/bm25_val_metrics.json
outputs/<dataset>/eval/bm25_test_metrics.json
outputs/<dataset>/eval/bm25_threshold.json
```

BM25 ranking metrics include Hit/Recall/NDCG at `3`, `5`, `10`, and `20`; threshold is tuned on `val` and applied to `router`/`test`.

BGE retrieval caches are also split-specific after `retrieve_cache`:

```text
outputs/<dataset>/retrieval_cache/bge_scores_router.parquet
outputs/<dataset>/retrieval_cache/bge_scores_val.parquet
outputs/<dataset>/retrieval_cache/bge_scores_test.parquet
outputs/<dataset>/eval/bge_router_metrics.json
outputs/<dataset>/eval/bge_val_metrics.json
outputs/<dataset>/eval/bge_test_metrics.json
outputs/<dataset>/eval/bge_threshold.json
```

BGE threshold is tuned on `val` and applied to `router`/`test`.

Router alpha training reads the split-specific BM25/BGE caches from `retrieve_cache`. It trains on the `router` split, then writes both fixed-alpha and predicted-alpha hybrid caches:

```text
outputs/<dataset>/eval/router_alpha_labels.jsonl
outputs/<dataset>/eval/router_config.json
outputs/<dataset>/eval/router_metrics.json
outputs/<dataset>/models/router_alpha_regressor.joblib
outputs/<dataset>/retrieval_cache/hybrid_fixed_scores_router.parquet
outputs/<dataset>/retrieval_cache/hybrid_fixed_scores_val.parquet
outputs/<dataset>/retrieval_cache/hybrid_fixed_scores_test.parquet
outputs/<dataset>/retrieval_cache/hybrid_router_scores_router.parquet
outputs/<dataset>/retrieval_cache/hybrid_router_scores_val.parquet
outputs/<dataset>/retrieval_cache/hybrid_router_scores_test.parquet
```

`hybrid_fixed` always uses `alpha=0.5`. `hybrid_router` uses the TF-IDF + Ridge router prediction per query.

BGE reranking takes the top-50 `hybrid_router` aid candidates, expands those aids back to chunks, scores each chunk, then writes split-specific evaluation caches and a backward-compatible aggregate cache:

```text
outputs/<dataset>/retrieval_cache/bge_rerank_scores_val.parquet
outputs/<dataset>/retrieval_cache/bge_rerank_scores_test.parquet
outputs/<dataset>/retrieval_cache/bge_rerank_scores.parquet
outputs/<dataset>/eval/bge_rerank_val_metrics.json
outputs/<dataset>/eval/bge_rerank_test_metrics.json
outputs/<dataset>/eval/bge_rerank_threshold.json
```

The BGE rerank threshold is tuned on `val` with `rerank_score`, then applied to `test` for precision/recall/F2. The rerank metrics include deltas against `hybrid_router` when the baseline metrics are available. `evaluate` prefers the split-specific cache when it exists, then falls back to the aggregate cache for older runs.

Optional LLM reranking uses BGE rerank top-20 chunk candidates by default. The first backend is `unsloth_causal_lm`, which fine-tunes `unsloth/Qwen3.5-4B-Base` with a small LoRA causal-LM loop and does not use TRL. It scores each candidate with:

```text
llm_rerank_score = sigmoid(logit("1") - logit("0"))
```

Example:

```bash
python3 run.py \
  --stage train_llm_reranker \
  --dataset_name legalraw_full \
  --corpus_path raw_data/legalraw/full/legal_corpus.json \
  --questions_path raw_data/legalraw/full/train.json \
  --output_dir outputs \
  --use_llm_rerank true \
  --llm_rerank_model unsloth/Qwen3.5-4B-Base \
  --llm_rerank_backend unsloth_causal_lm \
  --llm_rerank_train_batch_size 1 \
  --llm_rerank_grad_accum 8

python3 run.py \
  --stage rerank_llm \
  --dataset_name legalraw_full \
  --corpus_path raw_data/legalraw/full/legal_corpus.json \
  --questions_path raw_data/legalraw/full/train.json \
  --output_dir outputs \
  --use_llm_rerank true \
  --llm_rerank_top_k 20 \
  --llm_rerank_batch_size 4
```

LLM rerank writes:

```text
outputs/<dataset>/models/llm_reranker/train_summary.json
outputs/<dataset>/retrieval_cache/llm_rerank_scores_val.parquet
outputs/<dataset>/retrieval_cache/llm_rerank_scores_test.parquet
outputs/<dataset>/retrieval_cache/llm_rerank_scores.parquet
outputs/<dataset>/eval/llm_rerank_val_metrics.json
outputs/<dataset>/eval/llm_rerank_test_metrics.json
outputs/<dataset>/eval/llm_rerank_threshold.json
```

`evaluate` builds `eval/summary.json` from the detailed per-method metrics files when they exist, so threshold metrics in the summary match files like `hybrid_router_val_metrics.json` and `bge_rerank_test_metrics.json`.
