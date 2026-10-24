"""Encode data/doc_texts.jsonl và nạp vào collection Qdrant `medical_docs`.

Đây là bước DUY NHẤT của Phase 1.4 cần GPU. Chạy trên server.

Thiết kế:
  - Một forward pass BGE-M3 cho ra CẢ dense lẫn sparse (encode_with_sparse),
    nên doc index có hybrid ngay, không phải chạy lại lần hai.
  - point id suy định từ doc_id, nên chạy lại là ghi đè chứ không nhân bản.
  - Payload nhẹ: doc_id, language, domain, title, url, char_count. KHÔNG lưu
    `text` - văn bản nộp bài lấy từ data/raw_documents (Phase 0 đã chứng minh
    nộp cửa sổ markdown gốc thắng nộp chunk).

Vì sao collection riêng thay vì dùng lại `medical_documents`:
  - 1,0M điểm thay vì 3,93M, nên tìm với limit lớn rẻ hơn hẳn. Đây cũng là
    cách sửa tường latency 478s/câu đã đụng ở 1.1 trên máy local.
  - Biểu diễn khác hẳn (title + description + heading + lead), không trộn lẫn
    với vector chunk để còn so sánh được hai bên.

Dung lượng: 1,0M x 1024-d fp32 ~ 4 GB, fp16 ~ 2 GB, cộng sparse.

    python -m scripts.ingest_doc_index --qdrant-url http://localhost:6333
"""
import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient, models

COLL = "medical_docs"
NS = uuid.UUID("7c4f2a10-93bd-4e55-8f21-6a0d9b3c1e47")


def point_id(doc_id) -> str:
    return str(uuid.uuid5(NS, f"doc:{doc_id}"))


def ensure_collection(client, recreate: bool):
    if recreate and client.collection_exists(COLL):
        client.delete_collection(COLL)
    if not client.collection_exists(COLL):
        client.create_collection(
            collection_name=COLL,
            vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    for field, schema in (("language", "keyword"), ("domain", "keyword"), ("doc_id", "integer")):
        try:
            client.create_payload_index(collection_name=COLL, field_name=field, field_schema=schema)
        except Exception:
            pass  # đã có


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/doc_texts.jsonl")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--batch", type=int, default=256, help="số doc mỗi lần upsert")
    ap.add_argument("--encode-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recreate", action="store_true", help="xoá collection cũ trước khi nạp")
    ap.add_argument("--resume-from", type=int, default=0, help="bỏ qua N dòng đầu")
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=600)
    ensure_collection(client, args.recreate)

    from src.rag.embedding import BgeM3Embedder
    t0 = time.time()
    embedder = BgeM3Embedder()
    print(f"[docidx] nạp encoder {time.time() - t0:.0f}s")
    if not hasattr(embedder, "encode_with_sparse"):
        print("encoder không hỗ trợ encode_with_sparse, dừng")
        return 1

    # Tắt indexing trong lúc nạp hàng loạt, bật lại ở cuối -> một lần build
    # HNSW thay vì rebuild liên tục.
    client.update_collection(collection_name=COLL,
                             optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    n, t_enc, t_up, start = 0, 0.0, 0.0, time.time()
    batch = []

    def flush(rows):
        nonlocal t_enc, t_up
        if not rows:
            return
        t = time.time()
        dense, lexical = embedder.encode_with_sparse([r["text"] for r in rows],
                                                     batch_size=args.encode_batch)
        t_enc += time.time() - t
        points = []
        for r, dv, lw in zip(rows, dense, lexical):
            points.append(models.PointStruct(
                id=point_id(r["doc_id"]),
                vector={"dense": dv.tolist() if hasattr(dv, "tolist") else list(dv),
                        "sparse": embedder.lexical_weights_to_sparse_vector(lw)},
                payload={"doc_id": int(r["doc_id"]), "language": r.get("language"),
                         "domain": r.get("domain"), "title": r.get("title"),
                         "url": r.get("url"), "char_count": r.get("char_count", 0)},
            ))
        t = time.time()
        client.upsert(collection_name=COLL, points=points, wait=False)
        t_up += time.time() - t

    with open(args.input, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < args.resume_from:
                continue
            batch.append(json.loads(line))
            if len(batch) >= args.batch:
                flush(batch)
                n += len(batch)
                batch = []
                if n % (args.batch * 20) == 0:
                    el = time.time() - start
                    print(f"[docidx] {n:,} doc | encode {t_enc:.0f}s upsert {t_up:.0f}s | "
                          f"{n/el:.0f} doc/s", flush=True)
            if args.limit and n >= args.limit:
                break
    flush(batch)
    n += len(batch)

    client.update_collection(collection_name=COLL,
                             optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))
    el = time.time() - start
    print(f"\n[docidx] xong {n:,} doc trong {el/60:.1f} phút ({n/el:.0f} doc/s)")
    print(f"[docidx] encode {t_enc/60:.1f} phút | upsert {t_up/60:.1f} phút")
    print(f"[docidx] đang build HNSW nền, theo dõi bằng "
          f"curl {args.qdrant_url}/collections/{COLL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
