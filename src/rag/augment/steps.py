"""Các bước sinh dữ liệu augment cho một câu hỏi.

Thứ tự: trích xuất thực thể -> tra bảng chuẩn + dịch -> rẽ nhánh theo độ dài
-> sinh mục tiêu truy xuất -> ghép.

Mọi lời gọi LLM đi qua tham số `chat` chứ không gọi thẳng, để chạy thử được
toàn bộ luồng bằng client giả, không cần vLLM.
"""
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from src.rag.augment.compose import compose
from src.rag.augment.lookup import IcdLookup, RxNormLookup, normalize
from src.rag.augment.models import AugmentedQuery, Target, Term, Unit

_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹ]+")

ChatFn = Callable[[str, int], Optional[str]]


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


# ----------------------------------------------------------------- gọi LLM

def make_chat(cfg) -> ChatFn:
    """Client thật. Nhiệt độ 0 và seed cố định: trích xuất và dịch phải ra
    cùng kết quả giữa hai lần chạy, nếu không thì mọi so sánh thí nghiệm sau
    đó đều lẫn nhiễu của chính bước này."""

    def chat(prompt: str, max_tokens: int) -> Optional[str]:
        body = {
            "model": cfg.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.temperature,
            "max_tokens": max_tokens,
            "seed": cfg.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(cfg.llm_max_retries):
            try:
                r = requests.post(
                    f"{cfg.llm_url}/chat/completions", json=body, timeout=cfg.llm_timeout
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt == cfg.llm_max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None

    return chat


def _json_blob(text: Optional[str], opener: str, closer: str) -> Optional[Any]:
    """Bóc khối JSON ngoài cùng. Model hay bọc thêm ```json hoặc lời dẫn."""
    if not text:
        return None
    start, end = text.find(opener), text.rfind(closer)
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def json_object(text: Optional[str]) -> Optional[Dict]:
    out = _json_blob(text, "{", "}")
    return out if isinstance(out, dict) else None


def json_array(text: Optional[str]) -> Optional[List]:
    out = _json_blob(text, "[", "]")
    return out if isinstance(out, list) else None


def _clean_list(values: Any) -> List[str]:
    """Danh sách chuỗi đã bỏ rỗng và bỏ trùng, giữ nguyên thứ tự."""
    out: List[str] = []
    if not isinstance(values, list):
        return out
    for v in values:
        if not isinstance(v, str):
            continue
        v = v.strip()
        if v and normalize(v) not in {normalize(x) for x in out}:
            out.append(v)
    return out


# --------------------------------------------------- bước 1: trích xuất

def extract_entities(chat: ChatFn, prompts, cfg, query: str
                     ) -> Tuple[List[str], List[str], List[str], bool]:
    """-> (bệnh, thuốc, triệu chứng, hỏng?)."""
    raw = chat(prompts["medical_entity_extraction"].format(input_text=query),
               cfg.max_tokens_extract)
    obj = json_object(raw)
    if obj is None:
        return [], [], [], True
    return (_clean_list(obj.get("disease")),
            _clean_list(obj.get("medication")),
            _clean_list(obj.get("symptom")),
            False)


# ------------------------------------------- bước 2: tra bảng và dịch

def resolve_diseases(chat: ChatFn, prompts, cfg, names: List[str],
                     icd: IcdLookup) -> Tuple[List[Term], int]:
    """Mỗi bệnh một lời gọi dịch (prompt nhận `{{disease_name}}`, không nhận
    danh sách). Bệnh nào tra được trong ICD thì trường `en` lấy của ICD và bỏ
    bản dịch của model - bảng chuẩn thắng. Nhưng bảng ICD KHÔNG có cột tiếng
    Trung, nên zh vẫn luôn phải lấy từ model.
    """
    terms: List[Term] = []
    n_fail = 0
    for name in names:
        icd_en = icd.find_en(name)
        raw = chat(prompts["disease_translation"].format(disease_name=name),
                   cfg.max_tokens_translate)
        obj = json_object(raw) or {}
        if not obj:
            n_fail += 1
        llm_en = (obj.get("en") or "").strip()
        zh = (obj.get("zh") or "").strip()

        if icd_en and cfg.icd_overrides_llm_en:
            en, source = icd_en, "icd"
        else:
            en, source = llm_en, "llm"
        terms.append(Term(vi=name, en=en, zh=zh, source_en=source))
    return terms, n_fail


def resolve_medications(names: List[str], rx: RxNormLookup) -> List[Term]:
    """Thuốc không dịch: "omeprazole" viết y hệt trong tài liệu cả ba thứ
    tiếng. Tra RxNorm chỉ để lấy dạng viết chuẩn; tra không ra thì giữ nguyên
    chữ người hỏi viết."""
    terms: List[Term] = []
    for name in names:
        canonical = rx.find_canonical(name)
        terms.append(Term(
            vi=name,
            en=canonical or name,
            zh="",
            source_en="rxnorm" if canonical else "original",
        ))
    return terms


def resolve_symptoms(chat: ChatFn, prompts, cfg, names: List[str]
                     ) -> Tuple[List[Term], bool]:
    """Cả danh sách triệu chứng đi trong MỘT lời gọi (prompt nhận
    `{{symptoms}}` là một mảng), khác với bệnh.

    Model phải trả về đúng số phần tử theo đúng thứ tự. Lệch thì bỏ toàn bộ
    phần dịch thay vì ghép bừa theo vị trí - ghép lệch sẽ gán tên tiếng Anh
    của triệu chứng này cho triệu chứng khác, và lỗi đó không ai thấy.
    """
    if not names:
        return [], False
    raw = chat(prompts["symptom_translation"].format(symptoms=json.dumps(names, ensure_ascii=False)),
               cfg.max_tokens_translate)
    arr = json_array(raw)
    if not isinstance(arr, list) or len(arr) != len(names):
        return [Term(vi=n, source_en="original", en=n) for n in names], True

    terms: List[Term] = []
    for name, item in zip(names, arr):
        item = item if isinstance(item, dict) else {}
        terms.append(Term(
            vi=name,
            en=(item.get("en") or "").strip(),
            zh=(item.get("zh") or "").strip(),
            source_en="llm",
        ))
    return terms, False


# ------------------------------------- bước 3: rẽ nhánh và sinh mục tiêu

def decompose(chat: ChatFn, prompts, cfg, query: str) -> Tuple[List[str], bool]:
    raw = chat(prompts["query_decomposition"].format(query=query), cfg.max_tokens_decompose)
    obj = json_object(raw)
    subs = _clean_list((obj or {}).get("sub_queries"))
    return subs, not subs


def retrieval_targets(chat: ChatFn, prompts, cfg, text: str) -> Tuple[List[Target], bool]:
    raw = chat(prompts["retrieval_target"].format(query=text), cfg.max_tokens_target)
    obj = json_object(raw)
    items = (obj or {}).get("retrieval_targets")
    if not isinstance(items, list) or not items:
        return [], True
    targets = [
        Target(
            aspect=(it.get("aspect") or "").strip(),
            vi=(it.get("vi") or "").strip(),
            en=(it.get("en") or "").strip(),
            zh=(it.get("zh") or "").strip(),
        )
        for it in items if isinstance(it, dict)
    ]
    targets = [t for t in targets if t.forms()]
    return targets, not targets


def coverage(entities: List[str], sub_queries: List[str]) -> Dict[str, Any]:
    """Kiểm tra phân rã có đánh rơi thực thể nào không.

    Hai chiều, vì hai chiều hỏng theo hai kiểu khác nhau:
    - `missing`: thực thể có trong câu gốc nhưng không sub-query nào nhắc tới,
      tức phân rã làm rụng mất một phần câu hỏi.
    - Không kiểm chiều ngược (sub-query bịa thực thể mới) ở đây được, vì
      sub-query là câu tự do chứ không phải danh sách thực thể.

    Chỉ ghi số, không tự sửa: rụng thực thể vẫn còn cứu được, bởi khối thuật
    ngữ dùng chung được ghép vào MỌI unit, nên thực thể đó vẫn có mặt trong
    chuỗi truy xuất.
    """
    joined = normalize(" ".join(sub_queries))
    missing = [e for e in entities if normalize(e) not in joined]
    return {
        "n_entity": len(entities),
        "n_entity_missing": len(missing),
        "entity_missing": missing,
    }


# --------------------------------------------------- dựng một query

def build(chat: ChatFn, prompts, cfg, qid: int, query: str,
          icd: IcdLookup, rx: RxNormLookup) -> AugmentedQuery:
    aq = AugmentedQuery(qid=qid, original=query, n_words=word_count(query))
    aq.is_long = aq.n_words > cfg.narrative_word_threshold

    diseases, medications, symptoms, failed = extract_entities(chat, prompts, cfg, query)
    if failed:
        aq.failures.append("extract")

    aq.diseases, n_dis_fail = resolve_diseases(chat, prompts, cfg, diseases, icd)
    if n_dis_fail:
        aq.failures.append("disease_translation")
    aq.medications = resolve_medications(medications, rx)
    aq.symptoms, sym_failed = resolve_symptoms(chat, prompts, cfg, symptoms)
    if sym_failed:
        aq.failures.append("symptom_translation")

    if aq.is_long:
        subs, failed = decompose(chat, prompts, cfg, query)
        if failed:
            # Phân rã hỏng thì tụt về cách xử lý của nhánh ngắn, chứ không bỏ
            # query: một câu truy vấn augment vẫn hơn không có gì.
            aq.failures.append("decompose")
            subs = [query]
        aq.stats.update(coverage(diseases + medications + symptoms, subs))
        for sub in subs:
            targets, t_failed = retrieval_targets(chat, prompts, cfg, sub)
            if t_failed:
                aq.failures.append("retrieval_target")
            aq.units.append(Unit(kind="sub", text=sub, targets=targets))
        if cfg.include_raw_unit:
            aq.units.append(Unit(kind="raw", text=query))
    else:
        targets, t_failed = retrieval_targets(chat, prompts, cfg, query)
        if t_failed:
            aq.failures.append("retrieval_target")
        aq.units.append(Unit(kind="main", text=query, targets=targets))

    aq.stats["n_disease"] = len(aq.diseases)
    aq.stats["n_medication"] = len(aq.medications)
    aq.stats["n_symptom"] = len(aq.symptoms)
    aq.stats["n_disease_from_icd"] = sum(1 for t in aq.diseases if t.source_en == "icd")
    aq.stats["n_medication_from_rxnorm"] = sum(1 for t in aq.medications if t.source_en == "rxnorm")
    return compose(aq)
