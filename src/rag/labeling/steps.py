"""Ba bước đầu của pipeline sinh nhãn: chọn chunk, sinh query, lọc round-trip.

Tương ứng bước 1-2-3 trong docs/pipeline-sinh-nhan-reranker.md.
Bước 4-5 (lấy ứng viên, chấm utility hai tầng) nằm ở module khác.
"""
import csv
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

csv.field_size_limit(10**9)

_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹ]+")


# ---------------------------------------------------------------- tiện ích

def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def ngrams(text: str, nmax: int = 6) -> set:
    w = [t.lower() for t in _WORD.findall(text)]
    return {" ".join(w[i:i + k]) for k in range(1, nmax + 1) for i in range(len(w) - k + 1)}


def search(vectorstore, query: str, top_k: int, language: Optional[str] = None):
    """Hybrid nếu encoder hỗ trợ sparse, không thì dense.

    Hybrid đáng dùng ở đây dù pipeline submission đang để dense làm mặc định:
    bước này cần recall thô để lấy ứng viên, không phải precision cuối cùng.
    """
    if hasattr(vectorstore.encoder, "encode_with_sparse"):
        return vectorstore.hybrid_search(query, top_k=top_k, language=language)
    return vectorstore.search(
        vectorstore.encoder.encode(query, normalize_embeddings=True).tolist(),
        top_k=top_k,
        language=language,
    )


# ------------------------------------------------- bảng thuật ngữ ICD

class IcdTerms:
    """Tra thuật ngữ bệnh trong chunk, trả về tên tiếng Việt tương ứng.

    Dùng ở bước 2 làm ràng buộc từ ngữ: chunk tiếng Anh nói "acute vaginitis",
    ta đưa "viêm âm đạo cấp tính" vào prompt để model viết query bằng giọng Việt
    thay vì dịch máy móc. Đồng thời ép query dùng từ KHÁC chunk nguồn, nên chunk
    gốc không còn thắng chỉ nhờ trùng từ vựng.
    """

    def __init__(self, path: str):
        self.en2vi: Dict[str, str] = {}
        self.vi_terms: set = set()
        p = Path(path)
        if not p.exists():
            return
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                en = (r.get("en_term") or "").strip().lower()
                vi = (r.get("vi_name") or "").strip()
                if not (en and vi):
                    continue
                # ICD-10-CM hay gộp hai bệnh vào một mã ("Neonatal conjunctivitis
                # and dacryocystitis"). Tài liệu thật thường chỉ nhắc một vế, nên
                # index cả cụm đầy đủ lẫn từng vế.
                for form in [en] + [s.strip() for s in en.split(" and ")]:
                    # Term một từ quá chung ("allergy", "fever") nên khớp bừa.
                    if len(form.split()) >= 2:
                        self.en2vi.setdefault(form, vi)
                for a in filter(None, (r.get("vi_aliases") or "").split("|")):
                    if len(a.split()) >= 2:
                        self.vi_terms.add(a.strip().lower())

    def lookup(self, text: str, language: str, limit: int = 6) -> List[str]:
        g = ngrams(text)
        if language == "en":
            hits = [self.en2vi[t] for t in (g & set(self.en2vi))]
        elif language == "vi":
            hits = [t for t in (g & self.vi_terms)]
        else:
            hits = []  # bảng không có cột tiếng Trung, xem muc 7 cua doc
        # Term dài hơn thì đặc hiệu hơn, ưu tiên trước.
        return sorted(set(hits), key=len, reverse=True)[:limit]


# ------------------------------------------------- bước 1: chọn chunk

def select_source_chunks(vectorstore, real_queries: List[str], cfg) -> List[Dict[str, Any]]:
    """Chọn chunk nguồn bằng cách truy xuất quanh từng query thật.

    KHÔNG dùng k-means. Ban đầu spec ghi "cluster thành 30 cụm rồi lấy chunk
    quanh mỗi tâm cụm", nhưng cách đó kém hơn vì hai lý do:

      1. k là tham số bịa ra, không có cách nào chọn đúng.
      2. Cụm to và cụm nhỏ được cấp số chunk bằng nhau, làm méo phân phối chủ đề
         so với phân phối thật của query.

    Truy xuất quanh từng query thì phân phối chủ đề của chunk nguồn tự khớp với
    phân phối của query: chủ đề nào nhiều query hỏi thì tự nhiên nhiều chunk hơn.
    Không có k, không có tâm cụm, và rẻ hơn.

    Hạn ngạch ngôn ngữ vẫn áp riêng, vì nó phục vụ mục đích khác (ép tập train
    có cặp vi->en) chứ không phải khớp phân phối chủ đề.

    Cách gom là ROUND-ROBIN theo tầng độ sâu, không phải duyệt hết query này rồi
    mới sang query kia:

        tầng 1: mỗi query góp chunk gần nhất của nó
        tầng 2: mỗi query góp chunk gần thứ hai
        ...  dừng ngay khi đủ hạn ngạch

    Nếu duyệt tuần tự (query 1 góp đủ 8, rồi query 2 góp đủ 8...) thì khi hạn
    ngạch đầy ở query thứ 500, bảy trăm query còn lại không góp được gì cả, và
    chủ đề của chúng biến mất khỏi tập train. Round-robin đảm bảo mọi query đều
    góp phần, và vì mỗi query góp đều nhau nên chủ đề nào đông query thì tự có
    nhiều chunk hơn - đúng cái ta muốn.
    """
    rng = random.Random(cfg.seed)
    per_lang_target = {
        lang: int(cfg.n_source_chunks * ratio) for lang, ratio in cfg.lang_quota.items()
    }
    queries = list(real_queries)
    rng.shuffle(queries)

    # Encode 1.200 query MỘT lần rồi dùng lại cho cả ba ngôn ngữ. Nếu để mỗi
    # lần search tự encode thì cùng một query bị chạy qua BGE-M3 ba lượt, tức
    # gấp ba chi phí nặng nhất của bước này.
    print(f"  encode {len(queries)} query...", flush=True)
    qvecs = [vectorstore.encoder.encode(q, normalize_embeddings=True).tolist() for q in queries]

    def collect(lang: str, target: int) -> Dict[str, Dict[str, Any]]:
        if target <= 0:
            return {}
        # Search một lần cho mỗi query, lấy sẵn cả chiều sâu, rồi mới bóc từng
        # tầng. Bóc tầng mà search lại từ đầu thì tốn gấp `chunks_per_real_query`
        # lần cho cùng một kết quả.
        hits_by_query = [
            list(vectorstore.search(v, top_k=cfg.chunks_per_real_query, language=lang))
            for v in qvecs
        ]
        pool: Dict[str, Dict[str, Any]] = {}
        for depth in range(cfg.chunks_per_real_query):
            if len(pool) >= target:
                break
            for q, hits in zip(queries, hits_by_query):
                if len(pool) >= target:
                    break
                if depth >= len(hits):
                    continue
                pl = hits[depth].payload or {}
                cid = pl.get("chunk_id")
                content = pl.get("content") or ""
                if not cid or cid in pool or len(content) < 200:
                    continue
                pool[cid] = {
                    "chunk_id": cid,
                    "doc_id": pl.get("doc_id"),
                    "language": pl.get("language", lang),
                    "content": content,
                    "title": pl.get("title"),
                    "url": pl.get("url"),
                    "seed_query": q,
                    "seed_depth": depth + 1,
                    "retrieval_score": float(getattr(hits[depth], "score", 0.0)),
                }
        return pool

    # Gom các ngôn ngữ phủ kém trước, để biết hụt bao nhiêu. Corpus zh chỉ phủ
    # 4,1% nên hạn ngạch zh gần như chắc chắn hụt; nếu để tổng số chunk hụt theo
    # thì ngân sách tính toán lệch khỏi bảng trong doc. Tiếng Anh phủ 99,8% nên
    # gom sau cùng với phần bù đã biết.
    out: List[Dict[str, Any]] = []
    shortfall = 0
    for lang, target in per_lang_target.items():
        if lang == "en":
            continue
        got = list(collect(lang, target).values())
        if len(got) < target:
            shortfall += target - len(got)
            print(f"  ! {lang}: chỉ gom được {len(got)}/{target} chunk "
                  f"(corpus {lang} không đủ phủ, xem docs mục 7)")
        out.extend(got)

    en_target = per_lang_target.get("en", 0) + shortfall
    if en_target:
        got_en = list(collect("en", en_target).values())
        if shortfall:
            print(f"  bù {min(shortfall, max(len(got_en) - per_lang_target['en'], 0))} "
                  f"chunk thiếu bằng tiếng Anh")
        out.extend(got_en)

    rng.shuffle(out)
    return out


# ------------------------------------------------- bước 2: sinh query

class FewShotPool:
    """Query thật làm mẫu văn phong, tách theo độ dài.

    Phải tách: `get_few_shot()` cũ chỉ lấy top-2 theo similarity nên chế độ
    narrative gần như không bao giờ có mẫu đúng dạng, và model sẽ chỉ sinh ra
    câu ngắn sạch trong khi 33,8% query thật là đoạn kể bệnh dài.
    """

    def __init__(self, real_queries: List[str], cfg):
        self.cfg = cfg
        self.bands: Dict[str, List[str]] = {"short": [], "narrative": []}
        for q in real_queries:
            band = "narrative" if word_count(q) >= cfg.narrative_word_threshold else "short"
            self.bands[band].append(q)
        self.rng = random.Random(cfg.seed)

    def pick(self, vectorstore, chunk_text: str, mode: str) -> List[str]:
        """Lấy mẫu gần chunk về chủ đề NHƯNG đúng dải độ dài của chế độ."""
        band = self.bands[mode]
        picked: List[str] = []
        try:
            for pt in search(vectorstore, chunk_text[:2000], top_k=self.cfg.few_shot_pool):
                t = (pt.payload or {}).get("text") or (pt.payload or {}).get("content")
                if not t:
                    continue
                got = "narrative" if word_count(t) >= self.cfg.narrative_word_threshold else "short"
                if got == mode:
                    picked.append(t)
                if len(picked) >= self.cfg.few_shot_k:
                    break
        except Exception:
            pass  # thiếu collection query_documents thì rơi xuống lấy ngẫu nhiên
        while len(picked) < self.cfg.few_shot_k and band:
            picked.append(self.rng.choice(band))
        return picked[:self.cfg.few_shot_k]


def _parse_json(text: str) -> Optional[dict]:
    """Bóc JSON từ output model. Model hay bọc thêm ```json hoặc lời dẫn."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _chat(cfg, prompt: str) -> Optional[str]:
    body = {
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.llm_temperature,
        "max_tokens": cfg.llm_max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(cfg.llm_max_retries):
        try:
            r = requests.post(f"{cfg.llm_url}/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == cfg.llm_max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def generate_queries(vectorstore, chunks: List[Dict[str, Any]], prompts: Dict[str, Any],
                     icd: IcdTerms, few_shots: FewShotPool, cfg) -> Tuple[List[Dict], Dict]:
    """Sinh query tiếng Việt từ chunk. Trả về (bản ghi, thống kê).

    Chế độ được gán trước theo tỷ lệ cố định rồi mới xáo, chứ không bốc xúc xắc
    từng chunk: 5.000 lần bốc ngẫu nhiên sẽ lệch khỏi 66/34 vài phần trăm, mà
    khớp đúng phân phối độ dài chính là điểm của bước này.
    """
    rng = random.Random(cfg.seed)
    n = len(chunks)
    modes = (["short"] * round(n * cfg.mode_ratio["short"]) +
             ["narrative"] * (n - round(n * cfg.mode_ratio["short"])))
    rng.shuffle(modes)

    stats = defaultdict(int)

    def one(idx_chunk):
        i, ch = idx_chunk
        mode = modes[i]
        terms = icd.lookup(ch["content"], ch["language"])
        shots = few_shots.pick(vectorstore, ch["content"], mode)
        prompt = prompts[f"gen_query_{mode}"].format(
            chunk=ch["content"],
            chunk_language=ch["language"],
            few_shots="\n".join(f"- {s}" for s in shots),
            vi_terms=", ".join(terms),
        )
        raw = _chat(cfg, prompt)
        return i, ch, mode, terms, raw

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as ex:
        for i, ch, mode, terms, raw in ex.map(one, enumerate(chunks)):
            if raw is None:
                stats["llm_fail"] += 1
                continue
            obj = _parse_json(raw)
            if not obj:
                stats["json_fail"] += 1
                continue
            for slot in range(1, cfg.queries_per_chunk + 1):
                q = (obj.get(f"query_{slot}") or "").strip()
                if not q:
                    stats["slot_rong"] += 1
                    continue
                records.append({
                    "qid": f"g{i:06d}_{slot}",
                    "query": q,
                    "mode": mode,
                    "n_words": word_count(q),
                    "src_chunk_id": ch["chunk_id"],
                    "src_doc_id": ch["doc_id"],
                    "src_language": ch["language"],
                    "vi_terms": terms,
                })
                stats[f"sinh_{mode}"] += 1
    return records, dict(stats)


# ------------------------------------------- bước 3: lọc round-trip

def roundtrip_filter(vectorstore, records: List[Dict[str, Any]], cfg) -> Tuple[List[Dict], Dict]:
    """Giữ query nếu truy xuất lại bằng chính nó thì chunk nguồn nằm trong top-k.

    Thay cho bước "LLM judge chọn câu hỏi". Khách quan hơn vì nó đo đúng thứ ta
    quan tâm (query có định vị được tài liệu không) thay vì hỏi model xem câu hỏi
    có hay không, và rẻ hơn hàng trăm lần vì không gọi LLM.

    `src_rank` được giữ lại dù query bị loại: nó là tín hiệu độ khó truy xuất
    miễn phí, không cần model nào sinh ra.
    """
    def one(rec):
        try:
            hits = search(vectorstore, rec["query"], top_k=cfg.roundtrip_top_k)
        except Exception:
            return rec, None
        for rank, pt in enumerate(hits, 1):
            if (pt.payload or {}).get("chunk_id") == rec["src_chunk_id"]:
                return rec, rank
        return rec, None

    kept: List[Dict[str, Any]] = []
    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as ex:
        for rec, rank in ex.map(one, records):
            rec["src_rank"] = rank
            rec["kept"] = rank is not None
            stats["tong"] += 1
            if rank is None:
                stats[f"loai_{rec['mode']}"] += 1
                continue
            stats[f"giu_{rec['mode']}"] += 1
            kept.append(rec)

    stats["keep_rate"] = round(len(kept) / max(stats["tong"], 1), 4)
    return kept, dict(stats)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
