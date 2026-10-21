"""Bước 5 của pipeline sinh nhãn: chấm utility hai tầng.

Tương ứng mục "Bước 5 - Chấm utility hai tầng" trong
docs/buoc-4-5-cham-utility.md. Bước 4 (lấy ứng viên) nằm ở candidates.py.

Tầng 1 (5a): cross-encoder BAAI/bge-reranker-v2-m3 checkpoint GỐC (chưa train)
chấm toàn bộ pool -> `t1_score`. Rẻ, chạy GPU local, không cần vLLM.

Tầng 2 (5b): LLM chỉ chấm `tier2_per_query` = 15/50 ứng viên, chọn theo rank
của t1_score (10 cao nhất + 5 quanh trung vị - vùng CE lưỡng lự). 35 ứng viên
còn lại gán thẳng `t2_score = 0`, không gọi LLM. LLM 8B là phần đắt nhất của
toàn pipeline nên tầng 1 tồn tại chỉ để nó không phải đọc những cái rõ ràng
không liên quan.

15 ứng viên KHÔNG gộp vào một lời gọi: tách thành các lô `tier2_batch_size = 3`
-> đúng 5 lời gọi LLM mỗi query. Quyết định đã chốt trong doc, không gộp lại.

LƯU Ý về prompt: `prompts/judge_tier2.txt` khai báo biến `{{query}}` +
`{{chunks}}` và output là MỘT MẢNG JSON `[{"chunk_id": int, "score": int}]`.
Doc thiết kế viết dự kiến `{{documents}}` và `{"scores": [x,x,x]}` - prompt
mới hơn và prompt là nguồn sự thật, nên code ở đây bám theo prompt.
`chunk_id` đưa vào prompt là SỐ NGUYÊN CỤC BỘ (1..15) của riêng query đó, không
phải chunk_id thật: vừa khớp yêu cầu "giữ nguyên kiểu số nguyên" của prompt,
vừa không để lộ xuất xứ gì cho model.
"""
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

TIER1_MODEL = "BAAI/bge-reranker-v2-m3"

# Điểm -> trọng số mẫu lúc train. 1 điểm KHÔNG phải negative (corpus 1,22 triệu
# tài liệu có hàng chục bản ghi đúng cho một câu hỏi, ép 1 thành negative là tự
# tạo negative giả) nên weight = None = bỏ khỏi tập train.
SCORE_WEIGHT: Dict[int, Optional[float]] = {3: 1.0, 2: 0.6, 1: None, 0: 0.0}

# Điểm gán khi LLM trả JSON hỏng hết số lần retry. KHÔNG dùng 0: 0 là nhãn
# negative thật, lẫn lỗi hạ tầng vào đó thì tập nhãn nhiễm âm thầm. -1 là cờ
# "chưa chấm được", và weight để None nên mẫu đó không vào tập train.
JUDGE_FAIL_SCORE = -1


# ------------------------------------------------- tầng 1: cross-encoder

class CrossEncoderScorer:
    """Bọc sentence_transformers.CrossEncoder cho tầng 1.

    Import nằm trong __init__ chứ không ở đầu module: smoke test của bước 4-5
    thay scorer bằng đồ giả, không có lý gì phải nạp torch + 2GB weights chỉ để
    chạy test logic chọn rank.
    """

    def __init__(self, model_name: str = TIER1_MODEL, device: Optional[str] = None,
                 max_length: int = 1024, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []
        scores = self.model.predict([(query, d) for d in docs], batch_size=self.batch_size)
        return [float(s) for s in scores]


def tier1_score(scorer, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gán `t1_score` cho toàn bộ ứng viên của mọi query. Sửa `rows` tại chỗ.

    Không chạy song song: scorer là một model GPU, nhiều thread cùng gọi
    predict() chỉ tranh nhau một thiết bị. Việc gộp batch đã nằm trong
    CrossEncoder.predict().
    """
    stats: Dict[str, Any] = defaultdict(int)
    all_scores: List[float] = []
    for row in rows:
        cands = row["candidates"]
        scores = scorer.score(row["query"], [c["content"] for c in cands])
        for c, s in zip(cands, scores):
            c["t1_score"] = float(s)
        all_scores.extend(c["t1_score"] for c in cands)
        stats["b5a_n_cap"] += len(cands)

    if all_scores:
        s = sorted(all_scores)
        for name, q in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            stats[f"b5a_t1_{name}"] = round(s[min(int(q * len(s)), len(s) - 1)], 4)
    return dict(stats)


# ------------------------------------------------- 5b: chọn ai được LLM chấm

def select_tier2(cands: List[Dict[str, Any]], cfg) -> List[int]:
    """Chỉ số (trong `cands`) của các ứng viên sẽ gọi LLM.

    `tier2_top_n` cao nhất theo t1_score, cộng `tier2_median_n` nằm quanh trung
    vị của cả pool. Hoà điểm thì phá theo chunk_id để hai lần chạy ra cùng một
    tập - sorted() ổn định nhưng thứ tự vào đã xáo theo query, nên chỉ dựa vào
    tính ổn định của sort là chưa đủ.
    """
    order = sorted(range(len(cands)),
                   key=lambda i: (-cands[i]["t1_score"], str(cands[i]["chunk_id"])))
    n = len(order)
    top = order[:cfg.tier2_top_n]

    # Cửa sổ median_n ôm giữa thứ hạng, đẩy xuống dưới vùng top để không trùng.
    start = max(n // 2 - cfg.tier2_median_n // 2, len(top))
    start = min(start, max(n - cfg.tier2_median_n, len(top)))
    mid = order[start:start + cfg.tier2_median_n]

    picked: List[int] = []
    for i in top + mid:
        if i not in picked:
            picked.append(i)
    return picked[:cfg.tier2_per_query]


# ------------------------------------------------- 5b: gọi LLM

def _chat_judge(cfg, prompt: str) -> Optional[str]:
    """Một lời gọi chấm điểm. Khác steps._chat ở nhiệt độ và max_tokens.

    `judge_temperature = 0.0` và `seed` cố định: tầng 2 cần chấm nhất quán, hai
    lần chạy phải ra cùng một nhãn cho cùng cặp (query, chunk). Nhiệt độ 0.7
    của bước sinh query sẽ phá tính lặp lại của toàn bộ tập nhãn.
    """
    body = {
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.judge_temperature,
        "max_tokens": cfg.judge_max_tokens,
        "seed": cfg.seed,
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


def parse_scores(text: Optional[str], expected_ids: List[int]) -> Optional[Dict[int, int]]:
    """Bóc mảng JSON [{"chunk_id": int, "score": int}] từ output model.

    Model hay bọc thêm ```json hoặc lời dẫn nên lấy theo mảng ngoài cùng.
    Trả None nếu thiếu/thừa chunk_id hoặc điểm ngoài 0-3: thà retry còn hơn
    nhận một nửa kết quả rồi im lặng gán bừa phần còn lại.
    """
    if not text:
        return None
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None

    got: Dict[int, int] = {}
    for item in arr:
        if not isinstance(item, dict):
            return None
        try:
            cid = int(item["chunk_id"])
            sc = int(item["score"])
        except (KeyError, TypeError, ValueError):
            return None
        if sc not in SCORE_WEIGHT or cid in got:
            return None
        got[cid] = sc

    if set(got) != set(expected_ids):
        return None
    return got


def _batches(picked: List[int], size: int) -> List[List[int]]:
    return [picked[i:i + size] for i in range(0, len(picked), size)]


def judge_tier2(rows: List[Dict[str, Any]], prompt, cfg,
                chat: Optional[Callable[[Any, str], Optional[str]]] = None
                ) -> Dict[str, Any]:
    """Chấm tầng 2 cho mọi query. Sửa `rows` tại chỗ, trả về thống kê.

    `prompt` là object có .format(**vars) - PromptVersion của MLflow registry
    (xem prompts.py) hoặc bất cứ thứ gì cùng giao diện. `chat` cho phép test
    thay bằng client giả.

    Mỗi query sinh ra `tier2_per_query / tier2_batch_size` lời gọi. Tất cả lời
    gọi của tất cả query được làm phẳng vào một ThreadPoolExecutor duy nhất:
    nếu chạy song song theo query và tuần tự trong query thì chỉ có
    llm_concurrency/5 request thực sự bay cùng lúc.
    """
    chat = chat or _chat_judge
    stats: Dict[str, Any] = defaultdict(int)

    # Tất cả ứng viên mặc định t2_score = 0: 35 cái không gọi LLM là suy ra 0
    # theo doc, 15 cái được chấm sẽ bị ghi đè ngay bên dưới.
    jobs: List[Tuple[int, List[int], str]] = []  # (row_idx, local_ids, prompt_text)
    for ri, row in enumerate(rows):
        cands = row["candidates"]
        for c in cands:
            c["t2_score"] = 0
            c["t2_judged"] = False
        picked = select_tier2(cands, cfg)
        # local_id 1..15 theo đúng thứ tự rank đã chọn. Đây là thứ duy nhất
        # model thấy - không chunk_id thật, không is_src, không ngôn ngữ.
        row["_tier2"] = {lid: ci for lid, ci in enumerate(picked, 1)}
        for batch in _batches(list(range(1, len(picked) + 1)), cfg.tier2_batch_size):
            payload = [
                {"chunk_id": lid, "text": cands[row["_tier2"][lid]]["content"]}
                for lid in batch
            ]
            text = prompt.format(
                query=row["query"],
                chunks=json.dumps(payload, ensure_ascii=False, indent=2),
            )
            jobs.append((ri, batch, text))

    def one(job):
        ri, batch, text = job
        for attempt in range(cfg.llm_max_retries):
            raw = chat(cfg, text)
            got = parse_scores(raw, batch)
            if got is not None:
                return ri, batch, got, attempt
        return ri, batch, None, cfg.llm_max_retries

    with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as ex:
        for ri, batch, got, retries in ex.map(one, jobs):
            # n_llm_call đếm số LÔ (= số query x 5 theo doc); retry đếm riêng,
            # gộp vào đây thì chỉ số của doc không còn khớp công thức nữa.
            stats["b5b_n_llm_call"] += 1
            stats["b5b_n_llm_retry"] += retries
            cands = rows[ri]["candidates"]
            if got is None:
                # Hết retry. Không im lặng gán 0 - xem JUDGE_FAIL_SCORE.
                stats["b5b_json_fail"] += 1
                print(f"  ! [{rows[ri]['qid']}] lô {batch} parse hỏng sau "
                      f"{cfg.llm_max_retries} lần, gán t2_score={JUDGE_FAIL_SCORE}")
                for lid in batch:
                    c = cands[rows[ri]["_tier2"][lid]]
                    c["t2_score"] = JUDGE_FAIL_SCORE
                    c["t2_judged"] = True
                continue
            for lid, sc in got.items():
                c = cands[rows[ri]["_tier2"][lid]]
                c["t2_score"] = sc
                c["t2_judged"] = True
                stats[f"b5b_diem_{sc}"] += 1

    for row in rows:
        row.pop("_tier2", None)
        for c in row["candidates"]:
            c["weight"] = SCORE_WEIGHT.get(c["t2_score"])
    return dict(stats)


# ------------------------------------------------- tổng kết + loại query

def finalize(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gắn cờ `kept` cho từng query và tính các chỉ số cuối của doc.

    Query không có ứng viên nào đạt >=2 bị loại: không có bằng chứng nào để học
    từ nó, giữ lại chỉ làm loãng tập train.
    """
    stats: Dict[str, Any] = defaultdict(int)
    pos_by_lang: Dict[str, List[int]] = defaultdict(list)
    src_total = src_win = 0

    for row in rows:
        cands = row["candidates"]
        row["kept"] = any(c["t2_score"] >= 2 for c in cands)
        stats["n_query"] += 1
        if not row["kept"]:
            stats["n_query_bi_loai"] += 1
        for c in cands:
            if c["t2_judged"] and c["t2_score"] >= 0:
                pos_by_lang[c["language"]].append(1 if c["t2_score"] >= 2 else 0)
            if c["is_src"]:
                src_total += 1
                src_win += 1 if c["t2_score"] >= 2 else 0

    out: Dict[str, Any] = dict(stats)
    out["ty_le_query_bi_loai"] = round(
        stats["n_query_bi_loai"] / max(stats["n_query"], 1), 4)
    out["ty_le_src_thang"] = round(src_win / src_total, 4) if src_total else 0.0
    for lang, vals in pos_by_lang.items():
        # Tỷ lệ ứng viên đạt >=2 trong số ĐÃ CHẤM của ngôn ngữ đó. Nếu zh gần 0
        # qua hàng nghìn mẫu thì reranker có thể đang học đường tắt "thấy tiếng
        # Trung -> chấm thấp"; xem lại candidate_lang_quota trước khi train.
        out[f"t2_positive_rate_{lang}"] = round(sum(vals) / len(vals), 4)
        out[f"t2_n_judged_{lang}"] = len(vals)
    return [r for r in rows if r["kept"]], out


def judged_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dạng ghi ra judged.jsonl theo schema trong doc.

    Không ghi lại `content`: nó đã nằm trong candidates.jsonl và join được bằng
    (qid, chunk_id). Nhân đôi 50 đoạn văn bản cho mỗi query chỉ làm file phình
    lên mà không thêm thông tin nào.
    """
    return [
        {
            "qid": r["qid"],
            "query": r["query"],
            "kept": r["kept"],
            "candidates": [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c.get("doc_id"),
                    "language": c["language"],
                    "t1_score": round(c["t1_score"], 6),
                    "t2_score": c["t2_score"],
                    "t2_judged": c["t2_judged"],
                    "weight": c["weight"],
                    "is_src": c["is_src"],
                }
                for c in r["candidates"]
            ],
        }
        for r in rows
    ]
