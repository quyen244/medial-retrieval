"""Tiền xử lý câu hỏi bằng bảng luật: bung viết tắt chat, viết tắt y khoa nghĩa
rõ, cách nói tránh bộ phận sinh dục, khẩu ngữ -> từ y khoa.

Tất định, không gọi LLM, chạy TRƯỚC scripts/correct_queries.py. Bảng ở
query_normalize.tsv (cùng thư mục, có version theo git); cách viết luật ghi ở
đầu file đó.

Lý do mỗi luật cần điều kiện ngữ cảnh, đo trên 1.200 câu thật:
  - "cu nhóc của em" là đứa trẻ, không phải bộ phận sinh dục
  - "Cozaar XQ 5/50mg" là tên thuốc, "chụp XQ ngực" mới là X-quang
  - "đái tháo đường" là tên bệnh, "đái rắt" mới là khẩu ngữ của "tiểu rắt"
  - "Đến tháng 9/2019" là mốc thời gian, "như kiểu đến tháng" mới là hành kinh

    from src.rag.augment.query_normalize import QueryNormalizer
    text, changes = QueryNormalizer().apply("e bị đau bụng ko rõ nguyên nhân")
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

TABLE = Path(__file__).with_name("query_normalize.tsv")
WINDOW = 40


@dataclass
class Rule:
    pattern: str
    replacement: str
    kind: str
    require: Optional[re.Pattern]
    forbid: Optional[re.Pattern]
    regex: re.Pattern


class QueryNormalizer:
    def __init__(self, path: Path = TABLE):
        self.rules: List[Rule] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cols = (line.split("\t") + [""] * 6)[:6]
            pattern, repl, kind, require, forbid, _note = cols
            # Mẫu viết thường nhiều chữ còn bắt cả dạng viết hoa chữ đầu ở đầu câu
            # ("Ko", "Trc", "Đái rắt", "Mẹ bầu"). Một chữ cái thì KHÔNG: "K",
            # "E", "J" viết hoa thường là mã (K+, vitamin E).
            body = re.escape(pattern) if not any(c in pattern for c in "\\[(?") else pattern
            if len(pattern) >= 2 and pattern[0].islower():
                body = f"(?:{body}|{re.escape(pattern[0].upper() + pattern[1:])})"
            self.rules.append(Rule(
                pattern=pattern, replacement=repl, kind=kind,
                require=re.compile(require) if require else None,
                forbid=re.compile(forbid) if forbid else None,
                regex=re.compile(rf"(?<!\w){body}(?!\w)"),
            ))

    def abbrev_map(self) -> dict:
        """Viết tắt chat dạng chữ trơn -> chữ đầy đủ, cho luật kiểm chỗ sửa của
        LLM (scripts/correct_queries.py) dùng CHUNG một nguồn với bảng này."""
        return {r.pattern.lower(): r.replacement for r in self.rules
                if r.kind == "abbrev_chat" and r.pattern.isalpha()}

    def apply(self, text: str) -> Tuple[str, List[dict]]:
        """-> (câu đã chuẩn hoá, [chỗ thay: quy tắc, chữ cũ, chữ mới, vị trí])."""
        text = unicodedata.normalize("NFC", text or "")
        changes: List[dict] = []
        for rule in self.rules:
            out, last = [], 0
            for m in rule.regex.finditer(text):
                left = text[max(0, m.start() - WINDOW):m.start()]
                right = text[m.end():m.end() + WINDOW]
                # Điều kiện trong bảng viết theo dạng của mẫu ("⟨đái⟩"), nên đưa
                # chỗ khớp viết hoa đầu câu ("Đái") về đúng dạng đó trước khi so.
                token = m.group(0)
                if rule.pattern[:1].islower():
                    token = token[:1].lower() + token[1:]
                window = f"{left}⟨{token}⟩{right}"
                if rule.require and not rule.require.search(window):
                    continue
                if rule.forbid and rule.forbid.search(window):
                    continue
                new = rule.replacement
                # Chỉ viết hoa chữ thay khi mẫu viết thường mà chỗ khớp viết hoa
                # (đầu câu). Mẫu vốn viết hoa (THA, NST) thì giữ nguyên chữ thay.
                if rule.pattern[:1].islower() and m.group(0)[:1].isupper() and new[:1].islower():
                    new = new[:1].upper() + new[1:]
                out.append(text[last:m.start()])
                out.append(new)
                last = m.end()
                changes.append({"rule": rule.pattern, "kind": rule.kind, "old": m.group(0),
                                "new": new, "context": f"{left[-25:]}⟨{m.group(0)}⟩{right[:25]}"})
            if out:
                out.append(text[last:])
                text = "".join(out)
        return text, changes
