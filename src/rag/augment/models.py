"""Cấu trúc dữ liệu của pha augment.

Một `AugmentedQuery` là toàn bộ những gì sinh ra cho một câu hỏi, đủ để
retrieval chạy lại mà không cần gọi LLM lần nào nữa.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Term:
    """Một thực thể y tế kèm bản tương đương ba ngôn ngữ."""
    vi: str
    en: str = ""
    zh: str = ""
    # Nguồn của trường `en`: "icd" (bảng chuẩn), "rxnorm", "llm" (model dịch),
    # "original" (giữ nguyên chữ người hỏi). Ghi lại để soát xem bảng chuẩn
    # phủ được bao nhiêu phần, và để truy ngược khi một thuật ngữ trông sai.
    source_en: str = "llm"

    def forms(self) -> List[str]:
        """Các dạng chữ đem đi ghép vào câu truy vấn, bỏ rỗng và trùng."""
        out: List[str] = []
        for s in (self.vi, self.en, self.zh):
            s = (s or "").strip()
            if s and s not in out:
                out.append(s)
        return out


@dataclass
class Target:
    """Một khía cạnh cần truy xuất, viết sẵn bằng ba thứ tiếng."""
    aspect: str = ""
    vi: str = ""
    en: str = ""
    zh: str = ""

    def forms(self) -> List[str]:
        out: List[str] = []
        for s in (self.vi, self.en, self.zh):
            s = (s or "").strip()
            if s and s not in out:
                out.append(s)
        return out


@dataclass
class Unit:
    """Một lượt truy xuất.

    Nhánh ngắn có đúng một unit (`kind="main"`). Nhánh dài có ba unit `"sub"`
    cộng một unit `"raw"` mang câu gốc chưa augment.
    """
    kind: str
    text: str
    targets: List[Target] = field(default_factory=list)
    augmented: str = ""

    def retrieval_text(self) -> str:
        """Chuỗi thật sự đem đi truy xuất."""
        return self.augmented or self.text


@dataclass
class AugmentedQuery:
    qid: int
    original: str
    n_words: int = 0
    is_long: bool = False
    diseases: List[Term] = field(default_factory=list)
    medications: List[Term] = field(default_factory=list)
    symptoms: List[Term] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)
    # Cờ hỏng của từng bước, để biết query nào chỉ augment được một nửa thay
    # vì coi mọi dòng trong file là như nhau.
    failures: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
