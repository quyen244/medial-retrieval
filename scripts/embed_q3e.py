"""Doc index thứ hai: text dài hơn (mặc định 8.000 ký tự) + Qwen3-Embedding-4B, collection `medical_docs_q3e4b`.

Hai bước, tách nhau để bước 1 chạy được trên máy yếu:

  texts  đọc data/raw_documents, dựng text bằng đúng doc_text() của build_doc_texts.py (title | description |
         heading | thân bài, đã bóc markdown và tẩy rác khuôn mẫu), chỉ nới ngân sách. KHÔNG cần GPU.
  embed  encode bằng Qwen3-Embedding-4B, cắt vector còn --dim chiều (model học kiểu Matryoshka nên cắt đầu
         vector vẫn dùng được), nạp vào Qdrant. Cần GPU.

Vì sao 8.000 ký tự (đo 2026-10-27 trên 3.000 doc lấy từ top 50 của pool top300, gần với gold):
  trọn bài trong 1.500 ký tự: 16% (doc vi: 8%) | 3.000: 42% (vi 33%) | 8.000: 95% (vi 94%).
  en/zh vốn ngắn (trung vị 1.664 / 744), nên ngân sách lớn chủ yếu thêm thân bài cho doc tiếng Việt.

Câu hỏi khi tìm phải kèm instruction (doc thì không), xem format_query(). Đổi instruction là đổi vector câu hỏi:
ghi rõ instruction dùng cho mỗi pool.

Payload chỉ có doc_id, language, domain, title, char_count. Text đưa vào reranker vẫn lấy từ payload `text` của
`medical_docs` theo doc_id, vì điểm rerank cũ chấm trên text đó.

    python -m scripts.embed_q3e texts --limit 10000 --shuffle --out data/doc_texts_8k_sample.jsonl
    python -m scripts.embed_q3e embed --input data/doc_texts_8k_sample.jsonl --limit 10000 --collection medical_docs_q3e4b_smoke --recreate
    python -m scripts.embed_q3e texts --out data/doc_texts_8k.jsonl
    python -m scripts.embed_q3e embed --input data/doc_texts_8k.jsonl --recreate
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "Qwen/Qwen3-Embedding-4B"
COLLECTION = "medical_docs_q3e4b"
QUERY_INSTRUCTION = ("Given a Vietnamese medical question, retrieve medical documents in Vietnamese, English or "
                     "Chinese that are relevant to the question")


def format_query(query, instruction=QUERY_INSTRUCTION):
    """Định dạng câu hỏi theo model card Qwen3-Embedding. Doc encode nguyên văn, không instruction."""
    return f"Instruct: {instruction}\nQuery: {query}"


def truncate_normalize(vecs, dim):
    """Giữ dim chiều đầu rồi chuẩn hoá L2 lại (cắt làm vector hết độ dài 1)."""
    import numpy as np
    v = np.asarray(vecs, dtype=np.float32)[:, :dim]
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


# ------------------------------------------------------------------ bước 1: text
def build_texts(args):
    from scripts.build_doc_texts import INDEX, RAW, clean, doc_text
    idx = json.load(open(INDEX))
    ids = list(idx)
    if args.shuffle:
        import random
        random.Random(0).shuffle(ids)        # index xếp theo domain: không trộn thì N id đầu toàn một nguồn
    if args.limit:
        ids = ids[: args.limit]
    print(f"[texts] {len(ids):,} doc, ngân sách {args.budget} ký tự -> {args.out}", flush=True)
    lens, full, langs, skipped = [], 0, Counter(), Counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for n, did in enumerate(ids, 1):
            try:
                with open(os.path.join(RAW, idx[did], did + ".json"), encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped["đọc lỗi"] += 1
                continue
            if doc.get("quality", {}).get("is_empty"):
                skipped["rỗng"] += 1
                continue
            text = doc_text(doc, args.budget, args.head_budget, args.max_heads)
            if len(text) < 40:
                skipped["quá ngắn"] += 1
                continue
            src = doc.get("source", {})
            lang = src.get("language_category") or "?"
            langs[lang] += 1
            lens.append(len(text))
            full += len(text) < args.budget
            fh.write(json.dumps({"doc_id": int(did), "language": lang, "domain": src.get("domain"),
                                 "title": clean(src.get("title") or "")[:300],
                                 "char_count": doc.get("content", {}).get("char_count", 0),
                                 "text": text}, ensure_ascii=False) + "\n")
            if n % 100_000 == 0:
                print(f"  {n:,}/{len(ids):,}", flush=True)
    lens.sort()
    print(f"[texts] ghi {len(lens):,} doc | median {lens[len(lens) // 2]} p90 {lens[int(.9 * len(lens))]} ký tự | "
          f"trọn bài (chưa chạm ngân sách) {full / max(len(lens), 1):.0%}")
    print(f"[texts] ngôn ngữ {dict(langs.most_common())}" + (f" | bỏ qua {dict(skipped)}" if skipped else ""))


# ------------------------------------------------------------------ bước 2: embed
def ensure_collection(client, coll, dim, recreate, int8):
    from qdrant_client import models
    if recreate and client.collection_exists(coll):
        client.delete_collection(coll)
    if not client.collection_exists(coll):
        # int8: vector gốc nằm trên đĩa, bản int8 trong RAM (khoảng 1/4 dung lượng), tìm xong chấm lại bằng gốc.
        client.create_collection(
            collection_name=coll,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=int8)},
            quantization_config=(models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, always_ram=True)) if int8 else None))
    for field, schema in (("language", "keyword"), ("domain", "keyword"), ("doc_id", "integer")):
        try:
            client.create_payload_index(collection_name=coll, field_name=field, field_schema=schema)
        except Exception:
            pass                             # đã có


def load_encoder(args):
    import torch
    from sentence_transformers import SentenceTransformer
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    mk = {"torch_dtype": dt}
    if args.flash_attn:
        mk["attn_implementation"] = "flash_attention_2"
    # padding trái: Qwen3-Embedding lấy vector ở token CUỐI, model card dặn padding_side="left".
    model = SentenceTransformer(args.model, device=args.device, model_kwargs=mk,
                                tokenizer_kwargs={"padding_side": "left"})
    model.max_seq_length = args.max_tokens
    return model


def rows_of(path, resume_from, limit):
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < resume_from:
                continue
            if limit and i - resume_from >= limit:
                return
            yield json.loads(line)


def count_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def embed(args, model=None, client=None):
    from qdrant_client import QdrantClient, models

    from scripts.ingest_doc_index import point_id
    client = client or QdrantClient(url=args.qdrant_url, timeout=600)
    ensure_collection(client, args.collection, args.dim, args.recreate, not args.full_ram)
    t0 = time.time()
    model = model or load_encoder(args)
    print(f"[embed] {args.model} | {args.dtype} | max {args.max_tokens} token | {args.dim} chiều | nạp {time.time() - t0:.0f}s",
          flush=True)
    total = count_lines(args.input) - args.resume_from
    if args.limit:
        total = min(total, args.limit)
    # Tắt build HNSW trong lúc nạp hàng loạt, bật lại ở cuối: build một lần thay vì liên tục.
    client.update_collection(collection_name=args.collection,
                             optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    done, toks, cut, t_enc, start, buf = 0, 0, 0, 0.0, time.time(), []

    def flush(rows):
        nonlocal done, toks, cut, t_enc
        texts = [r["text"] for r in rows]
        lens = [len(x) for x in model.tokenizer(texts, add_special_tokens=True)["input_ids"]]
        toks += sum(min(n, args.max_tokens) for n in lens)
        cut += sum(n > args.max_tokens for n in lens)
        t = time.time()
        vecs = truncate_normalize(model.encode(texts, batch_size=args.batch_size, normalize_embeddings=False,
                                               convert_to_numpy=True, show_progress_bar=False), args.dim)
        t_enc += time.time() - t
        client.upsert(collection_name=args.collection, wait=False, points=[
            models.PointStruct(id=point_id(r["doc_id"]), vector={"dense": v.tolist()},
                               payload={"doc_id": int(r["doc_id"]), "language": r.get("language"),
                                        "domain": r.get("domain"), "title": r.get("title"),
                                        "char_count": r.get("char_count", 0)})
            for r, v in zip(rows, vecs)])
        done += len(rows)
        el = time.time() - start
        rate = done / max(el, 1e-9)
        print(f"[embed] {done:,}/{total:,} doc (dòng {args.resume_from + done:,}) | {rate:.1f} doc/s | "
              f"{toks / max(t_enc, 1e-9):,.0f} token/s khi encode | TB {toks / done:.0f} token/doc | "
              f"{cut} doc bị cắt ở {args.max_tokens} token | còn ~{(total - done) / max(rate, 1e-9) / 3600:.1f} giờ",
              flush=True)

    for r in rows_of(args.input, args.resume_from, args.limit):
        buf.append(r)
        if len(buf) >= args.chunk:          # ST tự xếp theo độ dài bên trong một lần encode, nên chunk lớn đỡ độn
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    client.update_collection(collection_name=args.collection,
                             optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))
    el = time.time() - start
    print(f"[embed] xong {done:,} doc trong {el / 60:.1f} phút ({done / max(el, 1e-9):.1f} doc/s, encode chiếm "
          f"{t_enc / max(el, 1e-9):.0%}). Ước tính 1.003.391 doc: {1_003_391 / max(done / max(el, 1e-9), 1e-9) / 3600:.1f} giờ.")
    print(f"[embed] Qdrant đang build HNSW nền: curl {args.qdrant_url}/collections/{args.collection}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("texts", help="dựng text dài từ data/raw_documents (không cần GPU)")
    p.add_argument("--out", default="data/doc_texts_8k.jsonl")
    p.add_argument("--budget", type=int, default=8000, help="tổng ký tự mỗi doc")
    p.add_argument("--head-budget", type=int, default=1000, help="ký tự cho chuỗi heading (bản 1.500 dùng 300)")
    p.add_argument("--max-heads", type=int, default=30, help="số heading tối đa (bản 1.500 dùng 8)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.set_defaults(fn=build_texts)

    p = sub.add_parser("embed", help="encode Qwen3-Embedding-4B và nạp Qdrant (cần GPU)")
    p.add_argument("--input", default="data/doc_texts_8k.jsonl")
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--dim", type=int, default=2048, help="cắt vector còn N chiều (model đầy đủ 2.560)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="bf16 cho GPU Ampere trở lên; T4/Turing dùng fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--flash-attn", action="store_true", help="cần gói flash-attn và GPU Ampere trở lên")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk", type=int, default=2048, help="số doc mỗi lần encode + upsert")
    p.add_argument("--limit", type=int, default=0, help="chỉ N doc, để đo tốc độ")
    p.add_argument("--resume-from", type=int, default=0, help="bỏ qua N dòng đầu của --input (point id cố định nên chạy lại là ghi đè)")
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--full-ram", action="store_true", help="giữ vector float32 trong RAM, không lượng tử hoá int8")
    p.set_defaults(fn=embed)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
