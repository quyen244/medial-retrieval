"""Cấu hình cho pha augment câu truy vấn.

Cùng vai trò với `LabelingConfig`: một object khai báo duy nhất, và chính nó
được log vào mlflow để dựng lại được một run bất kỳ.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class AugmentConfig:
    # --- rẽ nhánh dài/ngắn ---
    # Dùng lại đúng ngưỡng của LabelingConfig, vốn đo trên 1.200 query thật
    # (66,3% dưới 30 từ). Không đặt ngưỡng mới để hai pha không nói hai chuyện
    # khác nhau về thế nào là "query dài".
    narrative_word_threshold: int = 30

    # Nhánh dài truy xuất bằng 3 sub-query cộng câu gốc = 4 lượt. Câu gốc đi
    # vào dạng THÔ, không augment: nó là lưới an toàn cho trường hợp phân rã
    # đi chệch ý người hỏi, nên phải giữ đúng thứ người ta viết.
    include_raw_unit: bool = True

    # --- tra bảng ---
    icd_table: str = "data/vocab/icd_bilingual.tsv"
    rxnorm_table: str = "data/knowledge/rxnorm_concepts_final.csv"

    # Thuật ngữ tiếng Anh lấy từ ICD thì bỏ bản dịch của LLM (bảng chuẩn thắng
    # model). Nhưng bảng ICD không có cột tiếng Trung, nên zh luôn lấy từ LLM.
    icd_overrides_llm_en: bool = True

    # --- LLM ---
    llm_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B-AWQ"
    llm_max_retries: int = 3
    llm_timeout: int = 300
    # Số query xử lý song song. Mỗi query tự chạy các bước của nó tuần tự vì
    # bước sau cần kết quả bước trước.
    concurrency: int = 16

    # Trích xuất, dịch, phân rã đều là việc cần ĐÚNG và lặp lại được, không
    # cần sáng tạo - khác hẳn bước sinh query của pha labeling (0.7).
    temperature: float = 0.0
    seed: int = 42
    max_tokens_extract: int = 512
    max_tokens_translate: int = 512
    max_tokens_decompose: int = 640
    max_tokens_target: int = 768

    # --- vào/ra ---
    queries_file: str = "data/queries/test_queries.jsonl"
    out_dir: str = "data/augment"

    # --- mlflow ---
    experiment: str = "query_augmentation"
    run_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in d.items()}
