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

from src.rag.augment.models import AugmentedQuery, Term, Unit

# Ngăn giữa các biến thể cùng một ý (vi; en; zh), giống ví dụ thiết kế.
INNER_SEP = "; "
# Ngăn giữa các đoạn khác loại.
BLOCK_SEP = "\n"


def _term_line(terms: List[Term]) -> str:
    """Mỗi thực thể một dòng, ba ngôn ngữ ngăn bằng dấu chấm phẩy."""
    lines = [INNER_SEP.join(t.forms()) for t in terms]
    return BLOCK_SEP.join(l for l in lines if l)


def compose_unit(aq: AugmentedQuery, unit: Unit) -> str:
    """Chuỗi truy xuất của một unit.

    Unit `raw` cố tình KHÔNG augment: nó là lưới an toàn giữ nguyên chữ người
    hỏi, phòng khi trích xuất hoặc phân rã đi chệch ý.
    """
    if unit.kind == "raw":
        return unit.text

    parts: List[str] = [unit.text.strip()]

    for group in (aq.diseases, aq.medications, aq.symptoms):
        line = _term_line(group)
        if line:
            parts.append(line)

    target_lines = [INNER_SEP.join(t.forms()) for t in unit.targets]
    target_lines = [l for l in target_lines if l]
    if target_lines:
        parts.append(BLOCK_SEP.join(target_lines))

    return BLOCK_SEP.join(p for p in parts if p)


def compose(aq: AugmentedQuery) -> AugmentedQuery:
    """Điền `augmented` cho mọi unit, tại chỗ."""
    for unit in aq.units:
        unit.augmented = compose_unit(aq, unit)
    lengths = [len(u.retrieval_text()) for u in aq.units]
    aq.stats["len_max"] = max(lengths) if lengths else 0
    aq.stats["n_unit"] = len(aq.units)
    return aq
