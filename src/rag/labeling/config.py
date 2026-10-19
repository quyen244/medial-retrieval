"""Cấu hình cho pipeline sinh nhãn.

Cùng vai trò với `PipelineConfig` của retrieval pipeline: một object khai báo duy
nhất, và chính nó là thứ được log vào mlflow để dựng lại được một run bất kỳ.

Mọi con số mặc định ở đây đều có lý do trong docs/pipeline-sinh-nhan-reranker.md.
Đổi chúng thì sửa cả doc, đừng để hai chỗ lệch nhau.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class LabelingConfig:
    # --- quy mô ---
    n_source_chunks: int = 5000
    queries_per_chunk: int = 2

    # Tỷ lệ ngôn ngữ của chunk nguồn. KHÔNG chia đều ba ngôn ngữ: corpus tiếng
    # Anh phủ 99,8% còn tiếng Việt 41%, tiếng Trung 4,1%, nên đa số bằng chứng
    # thật nằm ở tiếng Anh. Đây cũng là cách duy nhất để tập train có cặp vi->en,
    # vốn là đường truy xuất chính của cuộc thi.
    lang_quota: Dict[str, float] = field(
        default_factory=lambda: {"en": 0.60, "vi": 0.30, "zh": 0.10}
    )

    # Tỷ lệ hai chế độ query, bám theo phân phối đo được trên 1.200 query thật:
    # 66,3% dưới 30 từ, 33,8% trên 30 từ.
    mode_ratio: Dict[str, float] = field(
        default_factory=lambda: {"short": 0.66, "narrative": 0.34}
    )
    narrative_word_threshold: int = 30

    # --- chọn chunk nguồn ---
    # Số chunk lấy cho mỗi query thật. Xem steps.select_source_chunks: ta truy
    # xuất quanh từng query thật thay vì cluster, nên không có tham số k nào.
    chunks_per_real_query: int = 8

    # --- few-shot ---
    few_shot_k: int = 2
    few_shot_pool: int = 20  # lấy rộng rồi mới lọc theo độ dài của chế độ

    # --- lọc round-trip ---
    # Giữ query nếu chunk nguồn nằm trong top-k khi truy xuất lại bằng chính
    # query đó. Khách quan hơn và rẻ hơn hàng trăm lần so với LLM judge.
    roundtrip_top_k: int = 50
    # Dưới ngưỡng này thì prompt sinh query đang có vấn đề, dừng lại sửa.
    roundtrip_min_keep_rate: float = 0.40

    # --- LLM ---
    llm_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B-AWQ"
    llm_temperature: float = 0.7  # sinh query cần đa dạng, không cần xác định
    llm_max_tokens: int = 640
    llm_concurrency: int = 16
    llm_max_retries: int = 3

    # --- hạ tầng ---
    qdrant_url: str = "http://localhost:6333"
    corpus_collection: str = "medical_documents"
    query_collection: str = "query_documents"
    icd_table: str = "data/vocab/icd_bilingual.tsv"
    out_dir: str = "data/labeling"
    seed: int = 42

    # --- mlflow ---
    experiment: str = "labeling_query_generation"
    run_name: str = ""  # rỗng -> tự sinh theo timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Phẳng hoá cho mlflow.log_params. Dict lồng được stringify vì mlflow
        chỉ nhận giá trị vô hướng."""
        d = asdict(self)
        return {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in d.items()}

    @property
    def n_pilot(self) -> int:
        """Quy mô pilot mặc định, theo lộ trình trong doc."""
        return 300
