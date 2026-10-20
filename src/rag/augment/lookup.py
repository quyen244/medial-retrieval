"""Tra thuật ngữ đã trích xuất trong bảng chuẩn: ICD cho bệnh, RxNorm cho thuốc.

Khác hẳn cách tra của pha labeling: ở đó ta quét cả một đoạn văn bản nên phải
sinh n-gram rồi dò. Ở đây LLM đã khoanh sẵn từng thực thể ("viêm dạ dày"), nên
việc còn lại chỉ là đối chiếu MỘT cụm với bảng. Không cần quét, không cần cắt
tiền tố - tức cũng không dính lỗi khớp bừa mà cách quét n-gram gặp phải.

Cách tra ICD còn đang để ngỏ (user sẽ chốt sau). `IcdLookup` ở đây cố tình
giản dị nhất có thể - khớp đúng sau chuẩn hoá - và nằm sau một giao diện hẹp
(`find_en`) để thay bằng cách tra khác mà không đụng tới phần còn lại.
"""
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

csv.field_size_limit(10**9)

_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹ]+")


def normalize(text: str) -> str:
    """Chỉ giữ ký tự từ, thường hoá, nối một khoảng trắng.

    Alias nguồn mang dấu phẩy và định tố ("Loét tá tràng, cấp tính"), còn
    người hỏi viết "loét tá tràng". Không chuẩn hoá thì hai chuỗi không bao
    giờ bằng nhau, và lỗi đó im lặng - không ai báo gì cả.
    """
    return " ".join(t.lower() for t in _WORD.findall(text))


class IcdLookup:
    """vi -> en theo bảng ICD song ngữ.

    Một cụm tiếng Việt có thể xuất hiện ở nhiều mã ICD với tên tiếng Anh khác
    nhau. Những cụm như vậy bị LOẠI hẳn thay vì chọn bừa một cái: thuật ngữ
    sai được bơm vào câu truy vấn sẽ kéo về cả một rổ tài liệu sai bệnh, hại
    hơn là không bơm gì.
    """

    def __init__(self, path: str):
        self.vi2en: Dict[str, str] = {}
        self.n_ambiguous: int = 0

        p = Path(path)
        if not p.exists():
            return

        candidates: Dict[str, set] = {}
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                en = (r.get("en_term") or "").strip()
                if not en:
                    continue
                names = [(r.get("vi_name") or "").strip()]
                names += (r.get("vi_aliases") or "").split("|")
                for name in filter(None, (n.strip() for n in names)):
                    key = normalize(name)
                    if len(key.split()) >= 2:
                        candidates.setdefault(key, set()).add(en)

        for key, ens in candidates.items():
            # So không phân biệt hoa thường: bảng nguồn có cả "pneumonia" lẫn
            # "Pneumonia" cho cùng một bệnh. So chuỗi thô sẽ coi đó là hai
            # thuật ngữ khác nhau rồi loại nhầm cả cụm - đã gặp thật với
            # "viêm phổi".
            distinct = {e.lower(): e for e in ens}
            if len(distinct) == 1:
                # min() cho kết quả cố định giữa các lần chạy, và vì chữ hoa
                # đứng trước chữ thường nên nó chọn dạng viết hoa của ICD.
                self.vi2en[key] = min(ens)
            else:
                self.n_ambiguous += 1

    def find_en(self, vi_term: str) -> Optional[str]:
        return self.vi2en.get(normalize(vi_term))


class RxNormLookup:
    """Tên thuốc -> tên chuẩn RxNorm.

    Thuốc không dịch: "omeprazole" viết y hệt trong tài liệu cả ba thứ tiếng.
    Việc của bảng này chỉ là xác nhận đó đúng là thuốc và trả về dạng viết
    chuẩn; tra không ra thì giữ nguyên chữ người hỏi viết.
    """

    def __init__(self, path: str):
        self.name2canonical: Dict[str, str] = {}

        p = Path(path)
        if not p.exists():
            return

        with p.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                name = (r.get("name") or "").strip()
                if not name:
                    continue
                forms = [name] + [a.strip() for a in (r.get("aliases") or "").split("|")]
                for form in filter(None, forms):
                    key = normalize(form)
                    if key:
                        # Gặp lại thì giữ bản ghi đầu: file đã sắp theo RXCUI,
                        # ghi đè sẽ làm kết quả đổi theo thứ tự đọc file.
                        self.name2canonical.setdefault(key, name)

    def find_canonical(self, term: str) -> Optional[str]:
        return self.name2canonical.get(normalize(term))


def load_tables(cfg) -> Tuple[IcdLookup, RxNormLookup]:
    return IcdLookup(cfg.icd_table), RxNormLookup(cfg.rxnorm_table)
