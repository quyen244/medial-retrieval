"""Bước 4 của pipeline sinh nhãn: lấy ứng viên để chấm utility.

Tương ứng mục "Bước 4 - Lấy ứng viên" trong docs/buoc-4-5-cham-utility.md.
Bước 5 (chấm hai tầng) nằm ở judge.py.

Hai quyết định thiết kế quan trọng, cả hai đều nằm trong doc:

  1. Pool chia theo NGÔN NGỮ (3 lần search riêng, mỗi lần một hạn ngạch), không
     lấy top-50 thô trên toàn corpus. Top-k thô bị corpus lệch (en phủ 99,8%)
     áp đảo, ứng viên vi/zh gần như không bao giờ lọt vào pool - trong khi mục
     đích của tập nhãn là dạy reranker so khớp xuyên ngôn ngữ cho cả ba.

  2. Chunk nguồn luôn có mặt trong pool nhưng TRỘN ẨN DANH. Nếu mặc định chunk
     nguồn = positive thì reranker chỉ học đếm độ trùng từ vựng, vì query được
     viết ra từ chính chunk đó. Field `is_src` ghi ra file chỉ để audit; nó
     KHÔNG BAO GIỜ được đưa vào prompt ở bước 5.
"""
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import models


# ------------------------------------------------- hạn ngạch ngôn ngữ

def lang_targets(cfg) -> Dict[str, int]:
    """Chia `candidates_per_query` theo `candidate_lang_quota`, tổng đúng bằng
    candidates_per_query.

    Làm tròn xuống trước rồi phát phần dư cho ngôn ngữ có phần thập phân lớn
    nhất: nếu chỉ round() từng cái thì tổng có thể lệch khỏi 50 (ví dụ quota
    1/3-1/3-1/3), và mọi chỗ sau đó giả định pool đúng 50 sẽ sai âm thầm.
    """
    quota = cfg.candidate_lang_quota
    n = cfg.candidates_per_query
    raw = {lang: n * ratio for lang, ratio in quota.items()}
    out = {lang: int(v) for lang, v in raw.items()}
    du = n - sum(out.values())
    # Thứ tự ổn định: phần thập phân giảm dần, hoà thì theo tên ngôn ngữ.
    for lang in sorted(raw, key=lambda l: (-(raw[l] - int(raw[l])), l))[:du]:
        out[lang] += 1
    return out


# ------------------------------------------------- search theo ngôn ngữ

def _search_lang(vectorstore, query: str, qvec, top_k: int, lang: str, mode: str = "hybrid"):
    """Một lần search giới hạn trong một ngôn ngữ.

    Lọc bằng payload field `language`. Đây ĐÚNG là `source.language_category`
    chứ không phải `source.language` thô: ChunkerDispatcher.detect_language()
    đọc `language_category` TRƯỚC (`language_category or language`, xem
    src/rag/chunker/dispatcher.py) rồi gán vào Chunk.language, và
    chunk_to_payload() đưa thẳng field đó vào payload. Corpus không có field
    `language_category` riêng trong Qdrant - nó đã được nướng vào `language`
    lúc ingest. `language` thô của source (hay thiếu, hay sai, "en-US"...)
    không bao giờ tới được payload.

    Hybrid nếu encoder hỗ trợ sparse, không thì dense - cùng luật với
    steps.search(). Ở đây cần recall thô để lấy ứng viên, không phải precision
    cuối cùng, nên hybrid đáng dùng dù pipeline submission đang để dense làm
    mặc định.
    """
    if mode == "hybrid" and hasattr(vectorstore.encoder, "encode_with_sparse"):
        return vectorstore.hybrid_search(query, top_k=top_k, language=lang)
    if qvec is None:
        qvec = vectorstore.encoder.encode(query, normalize_embeddings=True).tolist()
    return vectorstore.search(qvec, top_k=top_k, language=lang)


def _as_candidate(payload: Dict[str, Any], fallback_lang: str) -> Optional[Dict[str, Any]]:
    pl = payload or {}
    cid = pl.get("chunk_id")
    content = pl.get("content") or pl.get("text") or ""
    if not cid or not content:
        return None
    return {
        "chunk_id": cid,
        "doc_id": pl.get("doc_id"),
        "language": pl.get("language") or fallback_lang,
        "content": content,
        "is_src": False,
    }


def _fetch_src_chunk(vectorstore, chunk_id: str) -> Optional[Dict[str, Any]]:
    """Lấy chunk nguồn trực tiếp từ Qdrant khi không có trong source_chunks.jsonl.

    Đây là đường dự phòng: thường thì entrypoint đã nạp sẵn source_chunks.jsonl
    của bước 1 nên không cần gọi tới đây. Dùng scroll theo payload chunk_id,
    không phải vector search - ta biết chính xác cần point nào.
    """
    try:
        points, _ = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None
    if not points:
        return None
    return _as_candidate(points[0].payload, "")


# ------------------------------------------------- bước 4

def fetch_candidates(vectorstore, records: List[Dict[str, Any]], cfg,
                     src_lookup: Optional[Dict[str, Dict[str, Any]]] = None
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Mỗi query -> `candidates_per_query` ứng viên, chia theo hạn ngạch ngôn ngữ.

    `records` là các dòng của gen_queries_kept.jsonl (có `qid`, `query`,
    `src_chunk_id`, `src_language`). `src_lookup` map chunk_id -> dict có
    `content` (nạp từ source_chunks.jsonl), dùng để chèn chunk nguồn mà không
    phải truy Qdrant thêm lần nữa.

    Trả về (rows, stats). Mỗi row: {"qid", "query", "src_chunk_id",
    "src_language", "candidates": [...]}.
    """
    src_lookup = src_lookup or {}
    targets = lang_targets(cfg)
    # Encode trước theo lô 32 thay vì từng câu trong mỗi luồng.
    mode = getattr(cfg, "candidate_search", "hybrid")
    texts = [r["query"] for r in records]
    dense_map = {}
    if mode == "hybrid" and hasattr(vectorstore, "prime_hybrid"):
        vectorstore.prime_hybrid(texts, batch_size=32)
    else:
        dense_map = {t: v.tolist() for t, v in zip(
            texts, vectorstore.encoder.encode(texts, normalize_embeddings=True, batch_size=32))}
    stats: Dict[str, Any] = defaultdict(int)

    def one(rec: Dict[str, Any]):
        q = rec["query"]
        # Encode MỘT lần rồi dùng cho cả ba lần search. Đường dense tự encode
        # mỗi lần search thì cùng một query chạy qua BGE-M3 ba lượt.
        qvec = dense_map.get(q)

        groups: Dict[str, List[Dict[str, Any]]] = {}
        local_stats: Dict[str, int] = defaultdict(int)
        for lang, k in targets.items():
            if k <= 0:
                groups[lang] = []
                continue
            try:
                hits = _search_lang(vectorstore, q, qvec, k, lang, mode)
            except Exception:
                hits = []
                local_stats[f"search_fail_{lang}"] += 1
            seen = set()
            got: List[Dict[str, Any]] = []
            for pt in hits:
                c = _as_candidate(getattr(pt, "payload", None), lang)
                if c is None or c["chunk_id"] in seen:
                    continue
                seen.add(c["chunk_id"])
                got.append(c)
            if len(got) < k:
                # Corpus ngôn ngữ đó không đủ phủ. KHÔNG bù bằng ngôn ngữ khác:
                # doc chốt hạn ngạch theo ngôn ngữ, bù sang en thì chính cái
                # lệch mà bước 4 tồn tại để chống lại sẽ quay lại.
                local_stats[f"thieu_{lang}"] += k - len(got)
            groups[lang] = got

        # --- chunk nguồn phải có mặt, và chỉ trong nhóm ngôn ngữ CỦA NÓ ---
        src_id = rec.get("src_chunk_id")
        src_lang = rec.get("src_language")
        if src_id and src_lang in groups:
            grp = groups[src_lang]
            pos = next((i for i, c in enumerate(grp) if c["chunk_id"] == src_id), None)
            if pos is not None:
                grp[pos]["is_src"] = True
            else:
                src_c = src_lookup.get(src_id) or _fetch_src_chunk(vectorstore, src_id)
                if src_c is None:
                    local_stats["src_khong_tra_duoc"] += 1
                else:
                    local_stats["src_missing_from_pool"] += 1
                    local_stats[f"src_missing_{src_lang}"] += 1
                    inserted = {
                        "chunk_id": src_id,
                        "doc_id": src_c.get("doc_id"),
                        "language": src_lang,
                        "content": src_c.get("content") or "",
                        "is_src": True,
                    }
                    # Đẩy cái điểm thấp nhất CÙNG NHÓM ra (hits đã xếp theo
                    # điểm giảm dần nên cái cuối là thấp nhất). Nhóm chưa đầy
                    # hạn ngạch thì chèn thêm, không phải đẩy ai ra.
                    if len(grp) >= targets[src_lang]:
                        grp.pop()
                    grp.append(inserted)

        pool = [c for lang in targets for c in groups.get(lang, [])]
        # Xáo để chunk nguồn không nằm ở vị trí đoán được. Seed dẫn xuất từ
        # cfg.seed + qid: xác định được giữa hai lần chạy (random.Random nhận
        # chuỗi thì seed qua sha512, không phụ thuộc PYTHONHASHSEED) nhưng mỗi
        # query một thứ tự khác.
        random.Random(f"{cfg.seed}:{rec['qid']}").shuffle(pool)

        return {
            "qid": rec["qid"],
            "query": q,
            "src_chunk_id": src_id,
            "src_language": src_lang,
            "candidates": pool,
        }, local_stats

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=getattr(cfg, "search_concurrency", 8)) as ex:
        # ex.map giữ nguyên thứ tự input -> output xác định được.
        for row, local_stats in ex.map(one, records):
            rows.append(row)
            for k, v in local_stats.items():
                stats[k] += v
            stats["b4_n_query"] += 1
            stats["b4_n_candidate"] += len(row["candidates"])
            if len(row["candidates"]) != cfg.candidates_per_query:
                stats["b4_pool_khong_du"] += 1

    out = {f"b4_{k}" if not k.startswith("b4_") else k: v for k, v in stats.items()}
    out["b4_pool_trung_binh"] = round(
        out.get("b4_n_candidate", 0) / max(out.get("b4_n_query", 0), 1), 2
    )
    return rows, dict(out)
