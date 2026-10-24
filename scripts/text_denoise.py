"""Tẩy rác khuôn mẫu khỏi văn bản tài liệu, trả về text thuần.

Dùng cho VĂN BẢN ĐEM ENCODE (scripts/build_doc_texts.py), không dùng cho
văn bản nộp bài. Bài nộp bị ràng buộc "trích nguyên văn" và chấm bằng LCS;
văn bản encode thì không, nên ở đây tẩy mạnh tay được.

Rác nhắm tới, quan sát trên mẫu thật:
  - dấu thời gian: `2024-06-06 17:13`, `2019-07-30 04:20`, `06/06/2024`
  - khung hỏi đáp Trung: `发病时间：`, `补充说明：`, `精选回答(1)`, `我要咨询`,
    `举报`, `向医生提问`
  - dòng bài liên quan tiếng Việt: `Xem thêm: <tiêu đề bài khác>` - đây là rác
    LẠC CHỦ ĐỀ, kéo vector sang hướng khác chứ không chỉ vô dụng
  - ký tự trang trí còn sót sau khi bóc
"""
import re

# --- dấu thời gian ---
_DT = [
    re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?(?:\s+\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)?"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}(?:\s+\d{1,2}[:：]\d{2})?"),
    re.compile(r"\b\d{1,2}[:：]\d{2}(?:[:：]\d{2})?\b"),
    re.compile(r"\b\d{1,2}\s*(?:giờ|phút)\s*\d{0,2}\b", re.I),
]

# --- nhãn khuôn mẫu, bỏ nguyên cụm ---
_LABELS = [
    "我要咨询", "医生回答", "向医生提问", "举报", "有帮助", "展开全部", "收起",
    "免费提问", "相关问题", "健康百科", "补充说明：", "病情描述：",
    "找医生", "立即咨询", "免费咨询", "预约挂号",
    "医生建议：", "追问", "回复", "查看更多", "点击查看",
    "Chia sẻ", "Lưu bài viết", "Bài viết liên quan", "Tin liên quan",
]
_LABEL_RE = re.compile("|".join(re.escape(x) for x in _LABELS))
_PICKED = re.compile(r"精选回答\s*[（(]\s*\d+\s*[)）]")
# nhãn kèm giá trị ngắn: "发病时间：不清楚", "发病时间：半年"
_FIELD = re.compile(r"发病时间\s*[：:]\s*\S{0,8}?(?=\s|$)")

# --- dòng bài liên quan: bỏ tới hết dòng ---
_SEEALSO = re.compile(
    r"(?:Xem thêm|Đọc thêm|Tham khảo thêm|Nguồn tham khảo|Theo dõi)\s*[:：].*",
    re.I)

# Lý lịch bác sĩ trả lời: "擅长：尿道炎，膀胱炎，肾炎..." - danh sách chuyên khoa
# của NGƯỜI trả lời, không liên quan chủ đề bài. Trên ask.39.net nó dài hơn cả
# câu trả lời và kéo vector sang hẳn chuyên khoa khác.
_BIO = re.compile(r"擅长\s*[：:].*")

# --- dọn sau khi bóc ---
_EMPTY_BRACKET = re.compile(r"[（(\[]\s*[)）\]]")
_PUNCT_RUN = re.compile(r"([，。、；：,.;:!?！？])\s*(?:\1\s*)+")
_ORPHAN = re.compile(r"(?:^|(?<=\s))[|·•∙‣▪►–—\-]{1,}(?=\s|$)", re.M)
_WS = re.compile(r"[ \t]{2,}")
_NL = re.compile(r"\n{3,}")


def denoise(text: str) -> str:
    if not text:
        return text
    t = _SEEALSO.sub(" ", text)
    t = _BIO.sub(" ", t)
    t = _PICKED.sub(" ", t)
    t = _FIELD.sub(" ", t)
    t = _LABEL_RE.sub(" ", t)
    for r in _DT:
        t = r.sub(" ", t)
    t = _EMPTY_BRACKET.sub(" ", t)
    t = _PUNCT_RUN.sub(r"\1 ", t)
    t = _ORPHAN.sub(" ", t)
    t = _WS.sub(" ", t)
    t = _NL.sub("\n\n", t)
    return t.strip()
