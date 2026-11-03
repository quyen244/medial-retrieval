"""Nhãn train reranker cấp DOCUMENT trên 1.200 câu thật đã sửa chính tả (pseudo-label).

Khác pipeline cũ (steps.py, candidates.py, judge.py) ở bốn chỗ:
  - câu hỏi là chính 1.200 câu thật (`corrected_query`), không sinh câu từ chunk;
  - ứng viên là doc của pool đã rerank sẵn (bge len 2048 trên text 8k), text là data/doc_texts_8k.jsonl cắt
    còn phần reranker nhìn thấy (run_doc_labels.DOC_TOKENS), không phải chunk;
  - không có tầng 1 riêng: điểm bge-reranker-v2-m3 (len 1024, câu sửa) trong pool là tầng 1;
  - thang 3 mức 2/1/0, few-shot lấy từ nhãn tay theo độ giống câu hỏi, kèm entity làm gợi ý.

Chọn 60 doc mỗi câu: 20 hạng đầu + 40 hạng rải đều trên 21-300. F2 chấm trên TẬP doc nộp
(m_doc = 160), nên reranker chỉ tạo khác biệt ở ranh giới hạng 160; nhãn của pipeline cũ
(top 10 + 5 quanh trung vị) bỏ trống đúng vùng đó.
"""
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Nhãn trang gán tay (scripts/build_labeling_page.py) -> thang 3 mức. Nút "Chưa chắc" được
# dùng cho mức 1: đúng đối tượng nhưng không đáp ứng nhu cầu (quyết định 2026-11-03).
HUMAN_SCALE = {"relevant": 2, "unsure": 1, "irrelevant": 0}
SCORES = (0, 1, 2)
# Chấm hỏng sau mọi lần retry. Không dùng 0: 0 là negative thật, lẫn lỗi hạ tầng vào là nhiễm nhãn.
JUDGE_FAIL = -1
NONE_TEXT = "(không có)"
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def fill(template: str, **values: str) -> str:
    """Thay {{ten_bien}} như prompts.fill, nhưng không kéo theo mlflow: module này chạy cả trên Kaggle."""
    return _VAR_RE.sub(lambda m: str(values.get(m.group(1), m.group(0))), template)


# ------------------------------------------------------------------ đọc / ghi
def read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def append_jsonl(fh, row: Dict[str, Any]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def read_done(path) -> set:
    """id đã ghi xong; dòng cuối ghi dở (bị ngắt) thì bỏ qua."""
    done = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                done.add(int(json.loads(line)["id"]))
            except (ValueError, KeyError):
                continue
    return done


# ------------------------------------------------------------------ chọn ứng viên
def pick_ranks(n_pool: int = 300, top: int = 20, spread: int = 40) -> List[int]:
    """Hạng (1-based) cần chấm: `top` hạng đầu + `spread` hạng rải đều trên phần còn lại,
    luôn gồm hạng cuối. Cố định, không ngẫu nhiên: hai lần chạy chấm đúng cùng tập cặp."""
    top = min(top, n_pool)
    rest = list(range(top + 1, n_pool + 1))
    if spread >= len(rest):
        return list(range(1, top + 1)) + rest
    step = (len(rest) - 1) / max(spread - 1, 1)
    picked = sorted({rest[round(i * step)] for i in range(spread)})
    return list(range(1, top + 1)) + picked


def select_candidates(pool_rows: Iterable[Dict[str, Any]], ranks: Sequence[int]) -> List[Dict[str, Any]]:
    """Pool (đã xếp theo rerank_score) -> mỗi câu chỉ giữ các hạng `ranks`."""
    out = []
    for r in pool_rows:
        cands = r["candidates"]
        scores = [c["rerank_score"] for c in cands]
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            raise ValueError(f"câu {r['id']}: pool chưa xếp theo rerank_score")
        keep = [{"doc_id": str(cands[k - 1]["doc_id"]), "rank": k,
                 "rerank_score": round(float(cands[k - 1]["rerank_score"]), 6),
                 "language": cands[k - 1].get("language") or "vi"}
                for k in ranks if k <= len(cands)]
        out.append({"id": int(r["id"]), "query": r["query"], "rr_model": r.get("rr_model"),
                    "rr_len": r.get("rr_len"), "candidates": keep})
    return out


# ------------------------------------------------------------------ nhãn tay
def load_human_labels(paths: Iterable[str]) -> Dict[int, Dict[str, Any]]:
    """labels_query_<id>_<fp>.json -> {qid: {"query", "pool", "labels": {doc_id: (score, note, rank)}}}.

    Một câu có nhiều file xuất (xuất lại sau khi sửa nhãn) thì lấy file `exported_at` mới nhất.
    """
    latest: Dict[int, Dict[str, Any]] = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        qid = int(d["query_id"])
        if qid in latest and latest[qid]["exported_at"] >= d.get("exported_at", ""):
            continue
        labels = {}
        for lab in d["labels"]:
            if lab.get("label") not in HUMAN_SCALE:
                continue
            labels[str(lab["doc_id"])] = (HUMAN_SCALE[lab["label"]], (lab.get("note") or "").strip(),
                                          lab.get("rank_2048") or lab.get("rank_1024"))
        latest[qid] = {"query": d["query"], "pool": d.get("pool"), "exported_at": d.get("exported_at", ""),
                       "file": str(p), "labels": labels}
    return latest


# ------------------------------------------------------------------ biến của prompt
def cosine_neighbours(vecs: Dict[int, Any], qid: int, pool_ids: Sequence[int], k: int) -> List[int]:
    """k câu trong `pool_ids` giống `qid` nhất (cosine; vector đã chuẩn hoá), không gồm chính nó."""
    import numpy as np
    q = np.asarray(vecs[qid], dtype=np.float32)
    scored = sorted(((float(np.dot(q, np.asarray(vecs[o], dtype=np.float32))), o)
                     for o in pool_ids if o != qid), key=lambda x: (-x[0], x[1]))
    return [o for _, o in scored[:k]]


def pick_examples(labels: Dict[str, Tuple[int, str, Any]], per_label: int = 1) -> List[Tuple[str, int, str]]:
    """Mỗi mức 2/1/0 lấy `per_label` doc có hạng tốt nhất: doc 0 hạng cao là hard negative,
    doc 2 hạng cao là positive rõ nhất. -> [(doc_id, score, note)] theo thứ tự 2, 1, 0."""
    out = []
    for s in (2, 1, 0):
        docs = sorted(((r if r is not None else 10**9, d, note) for d, (sc, note, r) in labels.items() if sc == s))
        out += [(d, s, note) for _, d, note in docs[:per_label]]
    return out


def select_few_shots(ranked: Sequence[int], human: Dict[int, Dict[str, Any]], texts: Dict[str, str],
                     shots: int = 2, per_label: int = 1) -> List[Tuple[int, List[Tuple[str, int, str]]]]:
    """Chọn ví dụ few-shot, LUÔN đủ `per_label` doc cho mỗi mức 0/1/2 (khi tập gán tay có mức đó).

    `shots` câu giống nhất góp đủ mọi mức chúng có. Mức nào còn thiếu (vd câu 2 gán tay không có
    mức 1) thì lấy thêm đúng mức đó từ câu giống tiếp theo có nó. -> [(qid, [(doc_id, điểm, ghi chú)])]
    theo thứ tự giống giảm dần, trong mỗi câu theo mức 2, 1, 0.
    """
    chosen: Dict[int, List[Tuple[str, int, str]]] = {}
    count: Counter = Counter()
    for i, qid in enumerate(ranked):
        missing = {s: per_label - count[s] for s in SCORES if count[s] < per_label}
        if i >= shots and not missing:
            break
        ex = [e for e in pick_examples(human[qid]["labels"], per_label) if texts.get(e[0])]
        if i >= shots:      # ngoài `shots` câu đầu: chỉ lấy mức còn thiếu, đủ số thì thôi
            take = []
            for e in ex:
                if missing.get(e[1], 0) > 0:
                    take.append(e)
                    missing[e[1]] -= 1
            ex = take
        if ex:
            chosen[qid] = ex
            count.update(s for _, s, _ in ex)
    return [(q, chosen[q]) for q in ranked if q in chosen]


def render_few_shots(selection: Sequence[Tuple[int, List[Tuple[str, int, str]]]], human: Dict[int, Dict[str, Any]],
                     texts: Dict[str, str], max_chars: int = 500) -> str:
    """Khối few-shot dạng văn bản. Doc cắt `max_chars` ký tự: đủ thấy đối tượng và ý chính,
    không để vài ví dụ chiếm phần lớn ngữ cảnh."""
    blocks = []
    for n, (qid, ex) in enumerate(selection, 1):
        lines = [f"### Ví dụ {n}", f"Câu hỏi: {human[qid]['query']}"]
        for d, s, note in ex:
            body = " ".join(texts[d].split())
            body = body[:max_chars] + ("..." if len(body) > max_chars else "")
            lines.append(f"- Tài liệu: {body}")
            lines.append(f"  Điểm: {s}" + (f" ({note})" if note else ""))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else NONE_TEXT


def make_vars_of(human: Dict[int, Dict[str, Any]], texts: Dict[str, str],
                 vecs: Dict[int, Any], shots: int = 2, per_label: int = 1, shot_chars: int = 500):
    """-> hàm row -> {"few_shots": ...}, biến duy nhất ngoài {{query}} và {{documents}}. Few-shot lấy từ các câu
    gán tay giống nhất, không bao giờ là chính câu đang chấm, luôn đủ cả 3 mức (select_few_shots).

    Không đưa entity / aspects vào prompt (quyết định 2026-11-04): smoke agree cho thấy danh sách 9-12 aspect
    kéo LLM chấm dễ dãi (doc chạm bất kỳ khía cạnh nào cũng thành 2), và entity tự trích có lúc sót hay sai."""
    human_ids = sorted(human)

    def vars_of(row):
        ranked = cosine_neighbours(vecs, row["id"], human_ids, len(human_ids))
        sel = select_few_shots(ranked, human, texts, shots, per_label)
        return {"few_shots": render_few_shots(sel, human, texts, shot_chars)}
    return vars_of


def documents_var(batch: Sequence[Tuple[int, str]]) -> str:
    """[(id cục bộ, text)] -> mảng JSON. Không đưa doc_id thật, hạng, điểm hay ngôn ngữ."""
    return json.dumps([{"doc_id": lid, "text": t} for lid, t in batch], ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ gọi LLM + đọc kết quả
def parse_scores(text: Optional[str], expected_ids: Sequence[int]) -> Optional[Dict[int, int]]:
    """Mảng JSON [{"doc_id": int, "score": 0|1|2, ...}] -> {id: score}. Trường thêm
    (same_subject, answers...) được bỏ qua. Thiếu/thừa id hay điểm ngoài thang -> None (retry)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    got: Dict[int, int] = {}
    for item in arr if isinstance(arr, list) else []:
        try:
            did, sc = int(item["doc_id"]), int(item["score"])
        except (KeyError, TypeError, ValueError):
            return None
        if sc not in SCORES or did in got:
            return None
        got[did] = sc
    return got if set(got) == set(expected_ids) else None


def prompt_sha(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


def judge_query(row: Dict[str, Any], doc_ids: Sequence[str], texts: Dict[str, str], template: str,
                fill: Callable[..., str], vars_common: Dict[str, str], chat: Callable[[str], Optional[str]],
                batch_size: int, retries: int, seed: int) -> Tuple[Dict[str, int], Counter]:
    """Chấm `doc_ids` của một câu theo lô `batch_size`. -> ({doc_id: điểm}, thống kê).

    Thứ tự doc xáo theo seed + id câu trước khi chia lô: chia theo hạng thì lô đầu toàn doc mạnh,
    model dễ chấm theo mặt bằng của lô thay vì từng doc. Few-shot và entity giống nhau ở mọi lô
    của câu, nên đặt chúng TRƯỚC {{documents}} trong prompt để vLLM dùng lại prefix cache.
    """
    st = Counter()
    order = list(doc_ids)
    random.Random(seed * 100_003 + int(row["id"])).shuffle(order)
    out: Dict[str, int] = {}
    for b in range(0, len(order), batch_size):
        part = order[b: b + batch_size]
        local = list(range(1, len(part) + 1))
        prompt = fill(template, query=row["query"], documents=documents_var(list(zip(local, (texts[d] for d in part)))),
                      **vars_common)
        got = None
        for attempt in range(retries):
            got = parse_scores(chat(prompt, int(row["id"])), local)
            st["llm_call"] += 1
            if got is not None:
                break
            st["llm_retry"] += 1
        if got is None:
            st["batch_fail"] += 1
            out.update({d: JUDGE_FAIL for d in part})
        else:
            out.update({d: got[i] for i, d in zip(local, part)})
    for s in out.values():
        st[f"score_{s}"] += 1
    return out, st


def make_chat(model: str, urls: Sequence[str], temperature: float = 0.0, max_tokens: int = 320,
              seed: int = 42) -> Callable[..., Optional[str]]:
    """Client vLLM (OpenAI-compatible): nhiệt độ 0 + seed cố định để chấm lặp lại được.

    Nhiều server (mỗi GPU một bản Qwen): câu `key` luôn đi server `key % n`, nên các lô của cùng
    một câu gặp lại prefix (instruction + entity + few-shot) đã cache trên đúng server đó.
    """
    import requests
    urls = list(urls)

    def chat(prompt: str, key: int = 0) -> Optional[str]:
        url = urls[key % len(urls)]
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(3):
            try:
                r = requests.post(f"{url}/chat/completions", json=body, timeout=300)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(2 ** attempt)
        return None
    return chat


def run_judge(rows: List[Dict[str, Any]], doc_ids_of: Callable[[Dict[str, Any]], List[str]],
              vars_of: Callable[[Dict[str, Any]], Dict[str, str]], texts: Dict[str, str], template: str,
              fill: Callable[..., str], chat: Callable[..., Optional[str]], out_path, meta: Dict[str, Any],
              batch_size: int = 5, retries: int = 3, seed: int = 42, concurrency: int = 24,
              deadline: Optional[float] = None) -> Counter:
    """Chấm mọi câu chưa có trong `out_path`, ghi tới đâu flush tới đó (chạy lại cùng lệnh là làm tiếp).

    Song song theo câu: `concurrency` câu cùng lúc, mỗi câu tuần tự các lô của nó, tức
    `concurrency` request bay cùng lúc và các lô cùng câu đi liền nhau (trúng prefix cache).
    `deadline` (time.time()): quá hạn thì không mở câu mới, câu đang chấm vẫn chấm xong và ghi.
    """
    done = read_done(out_path)
    todo = [r for r in rows if r["id"] not in done]
    total = Counter()
    if not todo:
        return total
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def one(row):
        if deadline and time.time() > deadline:
            return None
        ids = doc_ids_of(row)
        scores, st = judge_query(row, ids, texts, template, fill, vars_of(row), chat, batch_size, retries, seed)
        return row, ids, scores, st

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, res in enumerate(ex.map(one, todo), 1):
            if res is None:
                total["skipped_deadline"] += 1
                continue
            row, ids, scores, st = res
            by = {c["doc_id"]: c for c in row.get("candidates", [])}
            append_jsonl(fh, {"id": row["id"], "query": row["query"], **meta,
                              "judged": [{**by.get(d, {"doc_id": d}), "llm_score": scores[d]} for d in ids]})
            total += st
            if i == 1 or i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[judge] {i}/{len(todo)} câu | {total['llm_call'] / max(el, 1e-9):.1f} lời gọi/s "
                      f"| còn ~{(len(todo) - i) * el / i / 60:.0f} phút | hỏng {total['batch_fail']} lô", flush=True)
    return total


# ------------------------------------------------------------------ đo độ khớp với nhãn tay
def agree_rows(human: Dict[int, Dict[str, Any]], candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Các cặp đã gán tay, dạng hàng của run_judge. Hạng lấy từ candidates nếu doc nằm trong 60 hạng
    đã chọn, không thì từ hạng trang gán nhãn ghi lại (rank_2048, thiếu thì rank_1024)."""
    rank_of = {}
    for r in candidates:
        if int(r["id"]) in human:
            rank_of.update({(int(r["id"]), c["doc_id"]): c for c in r["candidates"]})
    return [{"id": qid, "query": h["query"],
             "candidates": [rank_of.get((qid, d), {"doc_id": d, "rank": h["labels"][d][2]}) for d in h["labels"]]}
            for qid, h in sorted(human.items())]


def agreement(rows: Iterable[Dict[str, Any]], human: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """So llm_score với nhãn tay trên cùng cặp. Báo ma trận nhầm lẫn 3x3, P/R của mức 2 và của
    mức >=1, tách theo dải hạng (vùng ranh giới 160 là vùng quan trọng nhất)."""
    conf = Counter()
    bands = defaultdict(Counter)
    for r in rows:
        labels = human.get(int(r["id"]), {}).get("labels", {})
        for c in r["judged"]:
            h = labels.get(str(c["doc_id"]))
            if h is None or c["llm_score"] == JUDGE_FAIL:
                continue
            conf[(h[0], c["llm_score"])] += 1
            rank = c.get("rank") or h[2] or 0
            band = "1-20" if rank <= 20 else "21-100" if rank <= 100 else "101-300"
            bands[band][(h[0], c["llm_score"])] += 1

    def pr(cm, thr):
        tp = sum(v for (h, l), v in cm.items() if h >= thr and l >= thr)
        pred = sum(v for (h, l), v in cm.items() if l >= thr)
        gold = sum(v for (h, l), v in cm.items() if h >= thr)
        return {"p": round(tp / pred, 3) if pred else None, "r": round(tp / gold, 3) if gold else None,
                "n_human": gold, "n_llm": pred}

    n = sum(conf.values())
    return {"n_pairs": n, "exact": round(sum(v for (h, l), v in conf.items() if h == l) / n, 3) if n else None,
            "confusion_human_x_llm": {f"{h}->{l}": conf[(h, l)] for h in SCORES for l in SCORES},
            "level2": pr(conf, 2), "level_ge1": pr(conf, 1),
            "by_band": {b: {"n": sum(cm.values()), "level2": pr(cm, 2), "level_ge1": pr(cm, 1)}
                        for b, cm in sorted(bands.items())}}


# ------------------------------------------------------------------ dựng tập train
def build_pairs(judged: Iterable[Dict[str, Any]], human: Dict[int, Dict[str, Any]], texts: Dict[str, str],
                partial_label: Optional[float], val_frac: float, seed: int) -> Tuple[Dict[str, List], Counter]:
    """-> ({"train": [...], "val": [...]}, thống kê); mỗi cặp {"query", "passage", "label"}.

    Nhãn tay đè nhãn LLM trên cùng cặp, và cặp gán tay nằm ngoài 60 doc đã chấm cũng được thêm
    vào (mẫu 30 hạng của trang gán nhãn không trùng hết hạng chọn ở đây). Mức 2 -> 1.0,
    mức 0 -> 0.0, mức 1 -> `partial_label`
    (None = bỏ cặp). Mức 1 không làm negative: gold ~51 doc mỗi câu là một cụm cùng chủ đề,
    ép doc đúng đối tượng xuống là đánh đổi recall. Câu không có cặp mức 2 nào bị bỏ.
    Tách val theo CÂU, seed cố định.
    """
    st = Counter()
    rows = list(judged)
    qids = sorted(int(r["id"]) for r in rows)
    random.Random(seed).shuffle(qids)
    val_q = set(qids[: int(round(len(qids) * val_frac))])
    out = {"train": [], "val": []}
    for r in rows:
        qid = int(r["id"])
        labels = human.get(qid, {}).get("labels", {})
        items, has_pos = [], False
        pairs = [(str(c["doc_id"]), c["llm_score"]) for c in r["judged"]]
        seen = {d for d, _ in pairs}
        pairs += [(d, None) for d in labels if d not in seen]
        for d, llm in pairs:
            s = labels[d][0] if d in labels else llm
            st["from_human" if d in labels else "from_llm"] += 1
            if s == JUDGE_FAIL:
                st["drop_fail"] += 1
                continue
            label = {2: 1.0, 0: 0.0}.get(s, partial_label)
            if label is None:
                st["drop_level1"] += 1
                continue
            if not texts.get(d):
                st["drop_no_text"] += 1
                continue
            has_pos |= s == 2
            items.append({"query": r["query"], "passage": texts[d], "label": float(label)})
            st[f"level_{s}"] += 1
        if not has_pos:
            st["query_no_level2"] += 1
            continue
        split = "val" if qid in val_q else "train"
        out[split] += items
        st[f"query_{split}"] += 1
    rng = random.Random(seed)
    for items in out.values():
        rng.shuffle(items)
    return out, st
