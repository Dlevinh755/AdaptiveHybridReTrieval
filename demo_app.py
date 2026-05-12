from __future__ import annotations

import argparse
import html
import json
import math
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from src.eval.metrics import aggregate_by_aid_max
from src.indexes.bm25_index import bm25_index_path
from src.indexes.faiss_index import dense_index_paths
from src.rerank.bge_rerank import _add_query_minmax_score, _expand_aid_rows_to_chunks, _score
from src.retrieval.bm25 import search_bm25
from src.retrieval.cache_scores import _merge_scores
from src.retrieval.dense import add_dense_labels_and_norm, search_dense_with_stats
from src.retrieval.hybrid import apply_hybrid
from src.retrieval.tune_bm25 import _add_labels_and_norm as add_bm25_labels_and_norm
from src.training.train_router import _transform_tfidf
from src.utils.artifact import prepared_dir, read_json, read_pickle, read_table


DEMO_QID = "demo_query"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple web demo for legal retrieval.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--questions_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--dense_model",
        default="",
        help="Optional dense model override. Defaults to the model recorded in the existing dense index metadata when available, otherwise BAAI/bge-m3.",
    )
    parser.add_argument(
        "--rerank_model",
        default="",
        help="Optional reranker model override. Defaults to the trained checkpoint when available, otherwise BAAI/bge-reranker-v2-m3.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bm25_k1", type=float, default=1.2)
    parser.add_argument("--bm25_b", type=float, default=0.9)
    parser.add_argument("--use_tuned_bm25", type=str2bool, default=True)
    parser.add_argument("--router_model", default="ridge")
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--final_top_k", type=int, default=5)
    return parser


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def make_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        corpus_path=args.corpus_path,
        questions_path=args.questions_path,
        output_dir=args.output_dir,
        dataset_dir=args.output_dir / args.dataset_name,
        force=False,
        dense_model=args.dense_model or "BAAI/bge-m3",
        rerank_model=args.rerank_model or "BAAI/bge-reranker-v2-m3",
        device=args.device,
        batch_size=args.batch_size,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        use_tuned_bm25=args.use_tuned_bm25,
        router_model=args.router_model,
        top_k=args.retrieval_top_k,
        candidate_top_k=args.retrieval_top_k,
        corpus_law_id_field="law_id",
        corpus_articles_field="content",
        article_id_field="aid",
        article_text_field="content_Article",
        question_id_field="qid",
        question_text_field="question",
        relevant_ids_field="relevant_laws",
    )


class DemoPipeline:
    def __init__(self, config: SimpleNamespace, *, retrieval_top_k: int, final_top_k: int):
        self.config = config
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self._lock = threading.Lock()
        self._cross_encoder = None
        self._router = None
        self._articles_by_aid: dict[str, dict[str, Any]] | None = None
        self.config.dense_model = self._resolve_dense_model()
        self.rerank_model = self._resolve_rerank_model()
        self._validate_artifacts()

    def _resolve_dense_model(self) -> str:
        if self.config.dense_model != "BAAI/bge-m3":
            return str(self.config.dense_model)

        metadata_paths = sorted((self.config.dataset_dir / "indexes" / "faiss").glob("*/*/metadata.json"))
        if not metadata_paths:
            return str(self.config.dense_model)

        latest_path = max(metadata_paths, key=lambda path: path.stat().st_mtime)
        try:
            metadata = read_json(latest_path)
        except Exception:
            return str(self.config.dense_model)
        return str(metadata.get("dense_model_requested") or self.config.dense_model)

    def _resolve_rerank_model(self) -> str:
        if self.config.rerank_model != "BAAI/bge-reranker-v2-m3":
            return str(self.config.rerank_model)
        trained_model_dir = self.config.dataset_dir / "models" / "bge_reranker"
        marker = trained_model_dir / "train_summary.json.done.json"
        if trained_model_dir.exists() and marker.exists():
            return str(trained_model_dir)
        return str(self.config.rerank_model)

    def _validate_artifacts(self) -> None:
        missing = []
        if not bm25_index_path(self.config).exists():
            missing.append(str(bm25_index_path(self.config)))
        dense_paths = dense_index_paths(self.config)
        for key in ["embeddings", "chunk_ids", "metadata"]:
            if not dense_paths[key].exists():
                missing.append(str(dense_paths[key]))
        if not (self.config.dataset_dir / "models" / "router_alpha_regressor.joblib").exists():
            missing.append(str(self.config.dataset_dir / "models" / "router_alpha_regressor.joblib"))
        if not (prepared_dir(self.config) / "articles.parquet").exists():
            missing.append(str(prepared_dir(self.config) / "articles.parquet"))
        if missing:
            joined = "\n- ".join(missing)
            raise FileNotFoundError(
                "Missing demo artifacts. Run pipeline stages through train_router first:\n"
                "prepare_training_data -> train_bge_retriever -> tune_bm25 -> build_bm25 "
                "-> retrieve_cache -> train_router\n\nMissing:\n- "
                f"{joined}"
            )

    def _load_articles(self) -> dict[str, dict[str, Any]]:
        if self._articles_by_aid is None:
            self._articles_by_aid = {str(row["aid"]): row for row in read_table(prepared_dir(self.config) / "articles.parquet")}
        return self._articles_by_aid

    def _load_router(self) -> dict[str, Any]:
        if self._router is None:
            self._router = read_pickle(self.config.dataset_dir / "models" / "router_alpha_regressor.joblib")
        return self._router

    def _predict_alpha(self, question: str) -> float:
        router = self._load_router()
        vectorizer = {"vocab": router["vocab"], "idf": router["idf"]}
        weights = np.asarray(router["weights"], dtype=np.float64)
        prediction = float((_transform_tfidf([question], vectorizer) @ weights)[0])
        return max(0.0, min(1.0, prediction))

    def _rerank_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            with self._lock:
                if self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder

                    self._cross_encoder = CrossEncoder(self.rerank_model, device=self.config.device)
                scores = self._cross_encoder.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
            return [float(score) for score in scores]
        except Exception as exc:
            print(f"[warn] CrossEncoder unavailable, using hash fallback: {exc}")
            return [_score(query, text) for query, text in pairs]

    def search(self, question_text: str, *, final_top_k: int | None = None) -> dict[str, Any]:
        question_text = question_text.strip()
        if not question_text:
            raise ValueError("Question is empty.")
        requested_top_k = self._resolve_final_top_k(final_top_k)

        question = {"qid": DEMO_QID, "question": question_text, "relevant_laws": []}
        questions = [question]

        bm25_rows = add_bm25_labels_and_norm(search_bm25(self.config, questions, self.retrieval_top_k), questions)
        dense_raw_rows, dense_stats = search_dense_with_stats(self.config, questions, self.retrieval_top_k)
        dense_rows = add_dense_labels_and_norm(dense_raw_rows, questions)
        merged_rows = _merge_scores(bm25_rows, dense_rows)

        alpha = self._predict_alpha(question_text)
        hybrid_rows = apply_hybrid(merged_rows, alpha_by_qid={DEMO_QID: alpha})
        hybrid_top20 = hybrid_rows[: self.retrieval_top_k]

        chunk_rows = _expand_aid_rows_to_chunks(self.config, hybrid_top20)
        pairs = [(question_text, str(row.get("chunk_text", ""))) for row in chunk_rows]
        scores = self._rerank_scores(pairs)
        reranked_chunks = []
        for row, score in zip(chunk_rows, scores):
            item = dict(row)
            item["rerank_score"] = score
            reranked_chunks.append(item)
        reranked_chunks = _add_query_minmax_score(reranked_chunks, "rerank_score", "rerank_score_norm")
        aid_rows = aggregate_by_aid_max(reranked_chunks, score_field="rerank_score_norm")
        aid_rows = sorted(aid_rows, key=lambda row: float(row.get("rerank_score_norm", 0.0)), reverse=True)

        articles = self._load_articles()
        results = []
        for rank, row in enumerate(aid_rows[:requested_top_k], start=1):
            aid = str(row["aid"])
            article = articles.get(aid, {})
            chunk_text = str(row.get("chunk_text", ""))
            article_text = str(article.get("text", ""))
            results.append(
                {
                    "rank": rank,
                    "aid": aid,
                    "law_id": article.get("law_id", ""),
                    "score": safe_float(row.get("rerank_score_norm", row.get("rerank_score", 0.0))),
                    "rerank_score_raw": safe_float(row.get("rerank_score", 0.0)),
                    "hybrid_score": safe_float(row.get("hybrid_score", 0.0)),
                    "hybrid_alpha": safe_float(row.get("hybrid_alpha", alpha)),
                    "bm25_score_norm": safe_float(row.get("bm25_score_norm", 0.0)),
                    "bge_score_norm": safe_float(row.get("bge_score_norm", 0.0)),
                    "chunk_id": row.get("chunk_id", ""),
                    "excerpt": make_excerpt(chunk_text or article_text),
                    "chunk_text": chunk_text,
                    "article_text": article_text,
                }
            )

        return {
            "query": question_text,
            "pipeline": {
                "hybrid_router_top_k": self.retrieval_top_k,
                "bge_rerank_input_unit": "chunks expanded from top aid candidates",
                "final_top_k": requested_top_k,
                "max_final_top_k": self.retrieval_top_k,
                "alpha": alpha,
                "dense_model": self.config.dense_model,
                "rerank_model": self.rerank_model,
                "dense_pool": dense_stats,
            },
            "results": results,
        }

    def _resolve_final_top_k(self, value: int | None) -> int:
        if value is None:
            return self.final_top_k
        return max(1, min(int(value), self.retrieval_top_k))


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def make_excerpt(text: str, limit: int = 700) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def page_html(error: str = "") -> bytes:
    error_html = f"<div class='error'>{html.escape(error)}</div>" if error else ""
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Legal Retrieval Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5b6776;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      width: min(1100px, calc(100vw - 32px));
      margin: 28px auto;
    }}
    h1 {{ font-size: 26px; margin: 0 0 14px; }}
    form {{
      display: grid;
      grid-template-columns: 1fr 150px auto;
      gap: 12px;
      align-items: stretch;
      margin-bottom: 14px;
    }}
    textarea {{
      min-height: 104px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      font-size: 15px;
      line-height: 1.45;
      background: var(--panel);
      color: var(--ink);
    }}
    label {{
      display: grid;
      gap: 6px;
      align-content: start;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    input[type="number"] {{
      height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      font-size: 15px;
      background: var(--panel);
      color: var(--ink);
    }}
    button {{
      border: 0;
      border-radius: 8px;
      padding: 0 22px;
      font-size: 15px;
      font-weight: 700;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
      min-width: 118px;
    }}
    button:hover {{ background: var(--accent-dark); }}
    .status, .error {{
      min-height: 24px;
      color: var(--muted);
      margin: 8px 0 16px;
      font-size: 14px;
    }}
    .error {{ color: var(--warn); white-space: pre-wrap; }}
    .result {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 12px 0;
    }}
    .result header {{
      display: flex;
      gap: 10px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .rank {{
      color: var(--accent-dark);
      font-weight: 800;
    }}
    .aid {{
      font-weight: 800;
      font-size: 18px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .scores {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 10px 0;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfe;
    }}
    details {{
      border-top: 1px solid var(--line);
      margin-top: 12px;
      padding-top: 10px;
    }}
    summary {{ cursor: pointer; color: var(--accent-dark); font-weight: 700; }}
    p {{ line-height: 1.55; margin: 8px 0 0; white-space: pre-wrap; }}
    @media (max-width: 720px) {{
      form {{ grid-template-columns: 1fr; }}
      button {{ height: 44px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Legal Retrieval Demo</h1>
    <form id="search-form">
      <textarea id="question" name="question" placeholder="Nhập câu hỏi pháp luật..."></textarea>
      <label>
        Số aid
        <input id="top-k" name="top_k" type="number" min="1" max="20" value="5">
      </label>
      <button type="submit">Tìm kiếm</button>
    </form>
    <div id="status" class="status"></div>
    {error_html}
    <section id="results"></section>
  </main>
  <script>
    const form = document.getElementById('search-form');
    const statusEl = document.getElementById('status');
    const resultsEl = document.getElementById('results');
    const fmt = (n) => Number(n || 0).toFixed(4);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const question = document.getElementById('question').value.trim();
      const topK = Math.max(1, Math.min(20, Number(document.getElementById('top-k').value || 5)));
      if (!question) {{
        statusEl.textContent = 'Vui lòng nhập câu hỏi.';
        return;
      }}
      statusEl.textContent = `Đang chạy hybrid router top 20 và BGE rerank, lấy top ${{topK}} aid...`;
      resultsEl.innerHTML = '';
      try {{
        const response = await fetch('/api/search', {{
          method: 'POST',
          headers: {{ 'content-type': 'application/json' }},
          body: JSON.stringify({{ question, top_k: topK }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Search failed');
        statusEl.textContent = `Dense: ${{payload.pipeline.dense_model}} · Reranker: ${{payload.pipeline.rerank_model}} · Router alpha: ${{fmt(payload.pipeline.alpha)}} · Trả về ${{payload.results.length}} aid`;
        resultsEl.innerHTML = payload.results.map(row => `
          <article class="result">
            <header>
              <span class="rank">#${{row.rank}}</span>
              <span class="aid">${{esc(row.aid)}}</span>
              <span class="meta">law_id: ${{esc(row.law_id || 'N/A')}} · chunk: ${{esc(row.chunk_id || 'N/A')}}</span>
            </header>
            <div class="scores">
              <span class="chip">rerank_norm ${{fmt(row.score)}}</span>
              <span class="chip">rerank_raw ${{fmt(row.rerank_score_raw)}}</span>
              <span class="chip">hybrid ${{fmt(row.hybrid_score)}}</span>
              <span class="chip">bm25 ${{fmt(row.bm25_score_norm)}}</span>
              <span class="chip">bge ${{fmt(row.bge_score_norm)}}</span>
            </div>
            <p>${{esc(row.excerpt || row.chunk_text || row.article_text)}}</p>
            <details>
              <summary>Xem chi tiết</summary>
              <p>${{esc(row.article_text || row.chunk_text)}}</p>
            </details>
          </article>
        `).join('');
      }} catch (error) {{
        statusEl.textContent = '';
        resultsEl.innerHTML = `<div class="error">${{esc(error.message)}}</div>`;
      }}
    }});
  </script>
</body>
</html>""".encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    pipeline: DemoPipeline | None = None
    init_error = ""

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        self._send(200, page_html(self.init_error), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/search":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            if self.headers.get("content-type", "").startswith("application/json"):
                payload = json.loads(body or "{}")
            else:
                payload = {key: value[0] for key, value in parse_qs(body).items()}
            if self.pipeline is None:
                raise RuntimeError(self.init_error or "Pipeline is not initialized.")
            result = self.pipeline.search(str(payload.get("question", "")), final_top_k=parse_int(payload.get("top_k")))
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    args = build_parser().parse_args()
    config = make_config(args)
    try:
        DemoHandler.pipeline = DemoPipeline(config, retrieval_top_k=args.retrieval_top_k, final_top_k=args.final_top_k)
    except Exception as exc:
        DemoHandler.init_error = str(exc)
        print(f"[warn] Demo starts with initialization error:\n{exc}")

    with socketserver.ThreadingTCPServer((args.host, args.port), DemoHandler) as server:
        server.daemon_threads = True
        print(f"Demo UI: http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
