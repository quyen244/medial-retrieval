"""Nạp prompt từ file .txt và đồng bộ vào MLflow Prompt Registry.

Luồng: **file .txt là nguồn sự thật, MLflow là sổ ghi version.**

Lý do không làm ngược lại: nếu registry là nguồn sự thật thì mỗi lần sửa một
chữ trong prompt phải chạy một lệnh đăng ký, rất dễ quên và dẫn tới chuyện code
chạy một prompt khác với cái đang mở trên màn hình. Để file làm nguồn thì sửa là
ăn ngay, còn registry tự bắt kịp.

`sync()` so nội dung file với version mới nhất trong registry. Giống thì dùng lại
version cũ (không đẻ version rác mỗi lần chạy), khác thì đăng ký version mới.
URI trả về được log vào run, nên sau này truy ngược được run nào dùng prompt nào.

Biến dùng cú pháp MLflow: {{ten_bien}}, không phải f-string.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import mlflow

PROMPT_DIR = Path("prompts")

# Biến bắt buộc của từng prompt. Thiếu thì từ chối chạy ngay, kèm thông báo rõ.
# Thà hỏng ở giây thứ nhất còn hơn chạy 5 tiếng rồi mới phát hiện prompt sai.
REQUIRED_VARS: Dict[str, List[str]] = {
    "gen_query_short": ["chunk", "chunk_language", "few_shots", "vi_terms"],
    "gen_query_narrative": ["chunk", "chunk_language", "few_shots", "vi_terms"],
    # Bước 5b. Tên biến lấy đúng từ prompts/judge_tier2.txt: {{chunks}} (một
    # mảng JSON các {"chunk_id": int, "text": str}), không phải {{documents}}
    # như doc thiết kế dự kiến ban đầu.
    "judge_tier2": ["query", "chunks"],
    # Pha augment câu truy vấn (src/rag/augment). Lưu ý hai prompt dịch khác
    # nhau về độ hạt: disease_translation nhận MỘT tên bệnh một lần, còn
    # symptom_translation nhận cả mảng triệu chứng trong một lần.
    "medical_entity_extraction": ["input_text"],
    "disease_translation": ["disease_name", "query"],
    "symptom_translation": ["symptoms", "query"],
    "query_decomposition": ["query"],
    "retrieval_target": ["query", "glossary"],
    # Nhánh dịch câu gốc sang en/zh (scripts/translate_queries.py).
    "query_translation": ["query", "medical_entities", "few_shots"],
    # Sửa chính tả câu gốc, chạy TRƯỚC trích thực thể và dịch (scripts/correct_queries.py).
    "query_spelling_correction": ["query", "few_shots"],
    # HyDE cho biến thể truy xuất hyde (scripts/gen_hyde.py).
    "hyde_doc": ["query", "retrieval_targets", "diseases", "symptoms", "medications"],
}

# Prompt của riêng pha augment, để gọi sync_all() cho đúng nhóm thay vì đồng
# bộ cả những prompt không liên quan tới run đang chạy.
AUGMENT_PROMPTS = [
    "medical_entity_extraction",
    "disease_translation",
    "symptom_translation",
    "query_decomposition",
    "retrieval_target",
]

_STUB = "CHUA VIET PROMPT"
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptError(RuntimeError):
    pass


def _read(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise PromptError(f"Không thấy {path}. Xem prompts/README.md.")
    text = path.read_text(encoding="utf-8").strip()
    if not text or _STUB in text:
        raise PromptError(
            f"{path} chưa được viết. Xem prompts/README.md để biết các biến bắt buộc."
        )
    return text


def _validate(name: str, text: str) -> None:
    found = set(_VAR_RE.findall(text))
    # {chunk} một ngoặc sẽ KHÔNG được MLflow thay thế: prompt đi tới model
    # nguyên si và không ai báo gì cả. Đây là lỗi dễ mắc nhất nên kiểm trước,
    # để thông báo chỉ thẳng vào nguyên nhân thay vì chỉ nói "thiếu biến".
    single = set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", text))
    required = set(REQUIRED_VARS.get(name, []))
    one_brace = sorted((single & required) - found)
    if one_brace:
        raise PromptError(
            f"{name}.txt viết {', '.join('{%s}' % v for v in one_brace)} bằng MỘT ngoặc. "
            f"MLflow cần HAI: {', '.join('{{%s}}' % v for v in one_brace)}"
        )

    missing = [v for v in REQUIRED_VARS.get(name, []) if v not in found]
    if missing:
        raise PromptError(
            f"{name}.txt thiếu biến bắt buộc: {', '.join('{{%s}}' % m for m in missing)}\n"
            f"  Đang có: {', '.join(sorted(found)) or '(không có biến nào)'}\n"
            f"  Xem prompts/README.md."
        )


def load(name: str) -> str:
    """Nạp và kiểm biến như sync(), nhưng KHÔNG đụng MLflow. Cho các script
    offline không mở run (scripts/translate_queries.py). Điền biến bằng
    fill()."""
    text = _read(name)
    _validate(name, text)
    return text


def fill(template: str, **values: str) -> str:
    """Thay {{ten_bien}} đúng như MLflow làm, không cần PromptVersion."""
    return _VAR_RE.sub(lambda m: str(values.get(m.group(1), m.group(0))), template)


def _latest_template(name: str) -> Optional[str]:
    try:
        return mlflow.genai.load_prompt(f"prompts:/{name}@production").template
    except Exception:
        return None


def sync(name: str, commit_message: str = "cập nhật từ file .txt"):
    """Nạp prompt từ file, kiểm biến, đăng ký vào registry nếu nội dung đổi.

    Trả về PromptVersion. Dùng `.format(**vars)` để điền biến, `.uri` để log.
    """
    text = _read(name)
    _validate(name, text)

    if _latest_template(name) == text:
        return mlflow.genai.load_prompt(f"prompts:/{name}@production")

    version = mlflow.genai.register_prompt(
        name=name, template=text, commit_message=commit_message
    )
    mlflow.genai.set_prompt_alias(name=name, alias="production", version=version.version)
    return version


def sync_all(names: Optional[List[str]] = None) -> Dict[str, object]:
    return {n: sync(n) for n in (names or list(REQUIRED_VARS))}
