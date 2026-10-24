"""Dịch câu hỏi GỐC sang en và zh, offline, cache ra jsonl.

Nhánh dịch của Phase 1 (docs/audit-dich-query-va-retrieval.md mục 3). Mỗi câu
MỘT lời gọi LLM trả cả hai ngôn ngữ (prompts/query_translation.txt). Bản dịch
dùng làm biến thể truy xuất RIÊNG, không ghép vào chuỗi câu gốc - ghép là lỗi
đã trả giá ở vòng 1 (B, B' đều thua A).

Đầu vào của prompt:
  - medical_entities: thực thể TIẾNG VIỆT augment_v2 đã trích, chia nhóm
    diseases / symptoms / medications. KHÔNG kèm bản en/zh của augment_v2:
    smoke 20 cho thấy bản dịch đó sai là model chép theo ("dắt" -> dyspareunia,
    "u nhú" -> leiomyoma).
  - few_shots: các cặp vi | en | zh của bảng ICD (data/vocab/trilingual_terms.tsv)
    GẦN NHẤT với câu hỏi, retrieve bằng BGE-M3 dense. Mỗi nguồn (câu hỏi và từng
    thực thể) lấy vài cặp đầu, gộp theo điểm, giữ TOP_SHOTS cặp.

Index ICD encode một lần, cache ra ICD_EMB:

    python -m scripts.translate_queries --build-icd-index
    python -m scripts.translate_queries --sample 20 --out data/translate/smoke20.jsonl --show
    python -m scripts.translate_queries                      # cả 1.200 câu, chạy tiếp được
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.augment.config import AugmentConfig
from src.rag.augment.steps import json_object, make_chat
from src.rag.labeling import prompts as prompt_store

csv.field_size_limit(10 ** 9)

QUERIES = "data/queries/real_queries.jsonl"
AUGMENTED = "data/augment_v2_full/augmented.jsonl"
GLOSSARY = "data/vocab/trilingual_terms.tsv"
ICD_EMB = "data/vocab/icd_terms_bge_m3.npy"
PROMPT = "query_translation"

# Số cặp mỗi nguồn (câu hỏi, từng thực thể) và tổng số cặp đưa vào prompt.
PER_SOURCE = 3
TOP_SHOTS = 8

_CJK = re.compile(r"[一-鿿]")
_LATIN_WORD = re.compile(r"[A-Za-z]{4,}")
# Dấu tiếng Việt lọt vào bản en/zh = model chép nguyên cụm thay vì dịch.
_VI_MARK = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)


def entities_of(rec):
    """Thực thể tiếng Việt theo nhóm, đúng chữ người hỏi (đã bỏ trùng ở augment)."""
    return {key: [t["vi"] for t in rec.get(key, []) if t.get("vi")]
            for key in ("diseases", "symptoms", "medications")}


class IcdShots:
    """Retrieve cặp ICD gần câu hỏi trong không gian BGE-M3.

    Mỗi dòng encode dạng "vi | en | zh" (bỏ phần rỗng): cột vi chỉ có ở 7.845 /
    41.204 dòng, nên phần lớn dòng khớp với câu tiếng Việt nhờ dense đa ngữ qua
    en/zh.
    """

    def __init__(self, path=GLOSSARY, emb_path=ICD_EMB):
        seen, self.rows = set(), []
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                row = tuple((r.get(k) or "").strip() for k in ("vi", "en", "zh"))
                if row[1] and row not in seen:
                    seen.add(row)
                    self.rows.append(row)
        self.emb_path = emb_path
        self.emb = None
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            if len(emb) != len(self.rows):
                raise SystemExit(f"{emb_path} có {len(emb)} dòng, bảng có {len(self.rows)}. "
                                 f"Chạy lại --build-icd-index.")
            self.emb = emb.astype(np.float32)

    @staticmethod
    def text(row):
        return " | ".join(p for p in row if p)

    def build(self, embedder, batch_size=64):
        vecs = embedder.encode([self.text(r) for r in self.rows], batch_size=batch_size)
        np.save(self.emb_path, np.asarray(vecs, dtype=np.float16))
        self.emb = np.asarray(vecs, dtype=np.float32)

    def retrieve(self, source_vecs):
        """source_vecs: vector của câu hỏi và từng thực thể. -> [(vi, en, zh, điểm)]."""
        best = {}
        for v in source_vecs:
            sims = self.emb @ np.asarray(v, dtype=np.float32)
            for i in np.argsort(-sims)[:PER_SOURCE]:
                best[int(i)] = max(best.get(int(i), -1.0), float(sims[i]))
        top = sorted(best.items(), key=lambda t: -t[1])[:TOP_SHOTS]
        return [(*self.rows[i], round(s, 3)) for i, s in top]


def retrieve_shots(icd, queries, augmented, embedder):
    """Encode mọi câu hỏi + thực thể trong một lượt, -> id -> danh sách cặp."""
    texts, spans = [], {}
    for q in queries:
        ents = entities_of(augmented.get(int(q["id"]), {}))
        srcs = [q["query"]] + [e for group in ents.values() for e in group]
        spans[q["id"]] = (len(texts), len(texts) + len(srcs))
        texts += srcs
    vecs = embedder.encode(texts, batch_size=64)
    return {qid: icd.retrieve(vecs[a:b]) for qid, (a, b) in spans.items()}


def few_shots_text(shots):
    if not shots:
        return "(none)"
    lines = ["ICD terminology pairs closest to this query (vi | en | zh; '-' = not available):"]
    lines += [f"- {vi or '-'} | {en} | {zh or '-'}" for vi, en, zh, _ in shots]
    return "\n".join(lines)


def build_prompt(template, query, rec, shots, no_entities=False):
    """-> (prompt, thực thể đưa vào). Dùng chung cho chạy thật và
    scripts/trace_translation.py, để trace đúng thứ đã gửi LLM."""
    ents = {} if no_entities else entities_of(rec)
    prompt = prompt_store.fill(template, query=query,
                               medical_entities=json.dumps(ents, ensure_ascii=False),
                               few_shots=few_shots_text(shots))
    return prompt, ents


def check(en, zh, query=""):
    """Cờ lỗi để soát, không tự sửa."""
    flags = []
    if not en:
        flags.append("en_empty")
    if not zh:
        flags.append("zh_empty")
    if zh and not _CJK.search(zh):
        flags.append("zh_no_cjk")
    if en and _CJK.search(en):
        flags.append("en_has_cjk")
    if _VI_MARK.search(en or "") or _VI_MARK.search(zh or ""):
        flags.append("vi_leak")
    # Chữ Latin trong bản zh mà câu gốc không có: model bỏ dở giữa chừng
    # ("芒果和 pineapple"). Tên thuốc người hỏi viết sẵn thì không tính.
    q = query.lower()
    if any(w.lower() not in q for w in _LATIN_WORD.findall(zh or "")):
        flags.append("zh_latin")
    return flags


def load_embedder():
    from src.rag.embedding import BgeM3Embedder
    return BgeM3Embedder()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/translate/query_translation.jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="lấy N câu ngẫu nhiên, nửa ngắn nửa dài, để soát bằng mắt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="câu dài nhất 283 từ; en + zh của nó cỡ 700 token")
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--show", action="store_true", help="in từng câu kèm bản dịch")
    ap.add_argument("--no-entities", action="store_true",
                    help="đưa medical_entities rỗng (few_shots vẫn retrieve như cũ)")
    ap.add_argument("--build-icd-index", action="store_true",
                    help=f"encode bảng ICD bằng BGE-M3 ra {ICD_EMB} rồi thoát")
    args = ap.parse_args()

    icd = IcdShots()
    if args.build_icd_index:
        t0 = time.time()
        icd.build(load_embedder())
        print(f"[translate] encode {len(icd.rows):,} cặp ICD trong {time.time() - t0:.0f}s -> {ICD_EMB}")
        return
    if icd.emb is None:
        raise SystemExit(f"Chưa có {ICD_EMB}. Chạy --build-icd-index trước.")

    cfg = AugmentConfig()
    if args.llm_url:
        cfg.llm_url = args.llm_url
    chat = make_chat(cfg)
    template = prompt_store.load(PROMPT)
    augmented = {int(json.loads(l)["qid"]): json.loads(l) for l in open(AUGMENTED, encoding="utf-8")}
    queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")]

    if args.sample:
        rnd = random.Random(args.seed)
        long_ = [q for q in queries if augmented.get(int(q["id"]), {}).get("is_long")]
        short = [q for q in queries if not augmented.get(int(q["id"]), {}).get("is_long")]
        half = args.sample // 2
        queries = rnd.sample(short, args.sample - half) + rnd.sample(long_, half)

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    todo = [q for q in queries if q["id"] not in done]
    print(f"[translate] {len(queries)} câu, đã có {len(done)}, còn {len(todo)} | "
          f"ICD {len(icd.rows):,} cặp | {cfg.llm_model} @ {cfg.llm_url}")

    # Retrieve few-shot trước, rồi nhả encoder: vLLM cần VRAM cho cả lượt gọi LLM.
    t0 = time.time()
    embedder = load_embedder()
    shots = retrieve_shots(icd, todo, augmented, embedder)
    del embedder
    print(f"[translate] retrieve few-shot ICD cho {len(todo)} câu trong {time.time() - t0:.0f}s")

    def one(q):
        rec = augmented.get(int(q["id"]), {})
        prompt, ents = build_prompt(template, q["query"], rec, shots[q["id"]], args.no_entities)
        t = time.time()
        raw = chat(prompt, args.max_tokens)
        obj = json_object(raw) or {}
        en = (obj.get("query_en") or "").strip()
        zh = (obj.get("query_zh") or "").strip()
        flags = check(en, zh, q["query"]) if obj else ["parse_fail"]
        return {"id": q["id"], "query": q["query"], "query_en": en, "query_zh": zh,
                "is_long": bool(rec.get("is_long")), "entities": ents,
                "few_shots": shots[q["id"]], "flags": flags,
                "latency_s": round(time.time() - t, 2), "raw": None if obj else raw}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n, n_flag = time.time(), 0, 0
    with open(args.out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(one, todo):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            n_flag += bool(r["flags"])
            if args.show:
                n_ent = sum(len(v) for v in r["entities"].values())
                print(f"\n--- id={r['id']} {'DÀI' if r['is_long'] else 'ngắn'} | {n_ent} thực thể, "
                      f"{len(r['few_shots'])} cặp ICD | {r['latency_s']}s {r['flags'] or ''}")
                print(f"VI: {r['query']}")
                print(f"EN: {r['query_en']}")
                print(f"ZH: {r['query_zh']}")
            elif n % 100 == 0:
                print(f"  {n}/{len(todo)} | {n / (time.time() - t0):.1f} câu/s | gắn cờ {n_flag}", flush=True)

    print(f"\n[translate] xong {n} câu trong {time.time() - t0:.0f}s | gắn cờ {n_flag} -> {args.out}")


if __name__ == "__main__":
    main()
