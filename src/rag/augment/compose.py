"""Ghép các mảnh đã sinh thành một chuỗi truy vấn duy nhất.

Không dùng nhãn đoạn kiểu `[DISEASES]`: nhãn đó chỉ để con người nhìn cho dễ
lúc bàn thiết kế. Chuỗi thật ghép thẳng, vì nó còn đi qua encoder dense chứ
không riêng nhánh từ vựng, mà một cái biểu mẫu đầy dấu ngoặc vuông thì không
giống bất kỳ đoạn tài liệu nào trong corpus.

Thứ tự đoạn giữ đúng như thiết kế: câu hỏi trước, rồi thuật ngữ, rồi mục tiêu
truy xuất. Câu hỏi đứng đầu vì nó là thứ duy nhất do người thật viết; phần
sau là chữ máy sinh ra để mở rộng vùng phủ từ vựng.
"""
from typing import List

from src.rag.augment.lookup import normalize
from src.rag.augment.models import AugmentedQuery, Term, Unit

# Ngăn giữa các biến thể cùng một ý (vi; en; zh), giống ví dụ thiết kế.
INNER_SEP = "; "
# Ngăn giữa các đoạn khác loại.
BLOCK_SEP = "\n"


def _term_line(terms: List[Term]) -> str:
    """Mỗi thực thể một dòng, ba ngôn ngữ ngăn bằng dấu chấm phẩy."""
    lines = [INNER_SEP.join(t.forms()) for t in terms]
    return BLOCK_SEP.join(l for l in lines if l)


def scope_terms(aq: AugmentedQuery, unit: Unit):
    """Thực thể thuộc về một sub-unit: những cái mà chính sub-query nhắc tới.

    Thực thể mà KHÔNG sub-query nào nhắc (phân rã làm rụng) thì ở lại mọi
    sub-unit, đúng tinh thần cũ: đã trích được thì không để mất khỏi chuỗi
    truy xuất. Chỉ áp cho unit `sub`; unit `main` (nhánh ngắn) giữ tất cả.
    """
    if unit.kind != "sub":
        return aq.diseases, aq.medications, aq.symptoms
    subs = " ".join(normalize(u.text) for u in aq.units if u.kind == "sub")
    here = normalize(unit.text)

    def pick(terms: List[Term]) -> List[Term]:
        out = []
        for t in terms:
            n = normalize(t.vi)
            if n in here or n not in subs:
                out.append(t)
        return out

    return pick(aq.diseases), pick(aq.medications), pick(aq.symptoms)


def compose_unit(aq: AugmentedQuery, unit: Unit, scoped: bool = False) -> str:
    """Chuỗi truy xuất của một unit.

    Unit `raw` cố tình KHÔNG augment: nó là lưới an toàn giữ nguyên chữ người
    hỏi, phòng khi trích xuất hoặc phân rã đi chệch ý.
    """
    if unit.kind == "raw":
        return unit.text

    parts: List[str] = [unit.text.strip()]

    groups = scope_terms(aq, unit) if scoped else (aq.diseases, aq.medications, aq.symptoms)
    for group in groups:
        line = _term_line(group)
        if line:
            parts.append(line)

    target_lines = [INNER_SEP.join(t.forms()) for t in unit.targets]
    target_lines = [l for l in target_lines if l]
    if target_lines:
        parts.append(BLOCK_SEP.join(target_lines))

    return BLOCK_SEP.join(p for p in parts if p)


def compose(aq: AugmentedQuery, scoped: bool = False) -> AugmentedQuery:
    """Điền `augmented` cho mọi unit, tại chỗ."""
    for unit in aq.units:
        unit.augmented = compose_unit(aq, unit, scoped)
    lengths = [len(u.retrieval_text()) for u in aq.units]
    aq.stats["len_max"] = max(lengths) if lengths else 0
    aq.stats["n_unit"] = len(aq.units)
    return aq
