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

    # Tỷ lệ ngôn ngữ của chunk nguồn (câu hỏi train được sinh từ tài liệu tiếng
    # gì). vi 50 / en 40 / zh 10: corpus đã ingest nghiêng hẳn về tiếng Việt
    # (khoảng 80% chunk), nhưng vẫn giữ en đủ lớn để tập train có cặp vi->en.
    lang_quota: Dict[str, float] = field(
        default_factory=lambda: {"en": 0.40, "vi": 0.50, "zh": 0.10}
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
    # Tìm lại chunk nguồn trong ĐÚNG ngôn ngữ của nó, không phải cả corpus.
    # Corpus ~80% tiếng Việt: tìm trên cả corpus thì câu hỏi sinh từ chunk en/zh
    # phải chen với rất nhiều chunk vi cùng chủ đề và bị loại oan - tức mất đúng
    # các cặp vi->en mà lang_quota nhắm tới. Lúc nộp bài mỗi nhánh ngôn ngữ cũng
    # tìm riêng, nên điều kiện này khớp với cái reranker thật sự gặp.
    roundtrip_same_language: bool = True
    # Dưới ngưỡng này thì prompt sinh query đang có vấn đề, dừng lại sửa.
    roundtrip_min_keep_rate: float = 0.40

    # --- bước 4: lấy ứng viên để chấm utility ---
    # Pool chia theo ngôn ngữ, không lấy top-k thô trên toàn corpus: top-k thô
    # bị corpus lệch áp đảo, ứng viên ngôn ngữ yếu gần như không bao giờ xuất
    # hiện. Tách field riêng khỏi lang_quota của bước 1: bước 1 chọn ngôn ngữ
    # NGUỒN câu hỏi, bước 4 chọn ngôn ngữ ỨNG VIÊN chấm - hai việc khác nhau.
    # Khớp pool 50 ứng viên lúc inference (PipelineConfig.branch_keep: vi 26 /
    # en 16 / zh 8), để reranker được train trên pool có phân bố ngôn ngữ giống
    # pool nó chấm khi nộp bài. Với 50 ứng viên, tỷ lệ này ra đúng 26 / 16 / 8.
    candidates_per_query: int = 50
    # Cách tìm ứng viên ở bước 4: "hybrid" (dense + sparse) hoặc "dense". Nên
    # khớp với retriever lúc nộp bài (thí nghiệm dense hay hybrid): train trên
    # pool hybrid rồi nộp bằng dense là reranker gặp pool khác pool nó học.
    candidate_search: str = "hybrid"
    candidate_lang_quota: Dict[str, float] = field(
        default_factory=lambda: {"vi": 50 / 95, "en": 30 / 95, "zh": 15 / 95}
    )

    # --- bước 5b: tầng 2 chỉ chấm một phần cố định của pool ---
    # Chọn theo t1_score của tầng 1: top_n cao nhất (nghi positive) cộng
    # median_n quanh trung vị (vùng CE lưỡng lự). Phần còn lại
    # (candidates_per_query - tier2_per_query) không gọi LLM, gán thẳng
    # t2_score=0.
    tier2_top_n: int = 10
    tier2_median_n: int = 5

    @property
    def tier2_per_query(self) -> int:
        return self.tier2_top_n + self.tier2_median_n

    # 15 ứng viên của 1 query không gộp vào 1 lời gọi - tách thành từng lô
    # nhỏ, mỗi lô 1 lời gọi LLM riêng. 15 / 3 = 5 lời gọi mỗi query.
    tier2_batch_size: int = 3

    # --- LLM ---
    llm_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B-AWQ"
    llm_temperature: float = 0.7  # sinh query cần đa dạng, không cần xác định
    llm_max_tokens: int = 640
    # Số request đồng thời tới vLLM (cũng là số luồng search Qdrant song song).
    llm_concurrency: int = 24
    llm_max_retries: int = 3

    # Nhiệt độ chấm điểm PHẢI thấp, khác hẳn llm_temperature ở trên (0.7, dùng
    # cho sinh query cần đa dạng). Vai trò tầng 2 là chấm nhất quán, không
    # phải sáng tạo - nhiệt độ cao làm hai lần chạy ra hai nhãn khác nhau cho
    # cùng một cặp (query, chunk), phá khả năng lặp lại của toàn bộ nhãn.
    judge_temperature: float = 0.0
    # Mỗi chunk trả thêm same_subject + answers trước score (ép model đi qua hai
    # cổng trước khi chấm) nên cần nhiều token hơn chỉ trả điểm.
    judge_max_tokens: int = 320

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
