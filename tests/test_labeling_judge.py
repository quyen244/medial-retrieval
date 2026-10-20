"""Smoke test bước 4-5 của pipeline sinh nhãn, không cần GPU/Qdrant/vLLM.

Thay ba thứ bằng đồ giả: vectorstore (trả payload dựng sẵn theo ngôn ngữ),
scorer tầng 1 (điểm xác định được theo chunk_id), và client LLM (trả đúng mảng
JSON mà prompts/judge_tier2.txt khai báo). Mục đích là kiểm các bất biến của
doc, không phải chất lượng nhãn:

  - pool đúng candidates_per_query, đúng hạn ngạch ngôn ngữ
  - chunk nguồn LUÔN có trong pool, trong đúng nhóm ngôn ngữ của nó
  - đúng tier2_per_query ứng viên được LLM chấm, phần còn lại t2_score = 0
  - đúng tier2_per_query / tier2_batch_size lời gọi LLM mỗi query
  - chạy hai lần ra output giống nhau từng byte
"""
import json
import unittest
from typing import Any, Dict, List

from src.rag.labeling.candidates import fetch_candidates, lang_targets
from src.rag.labeling.config import LabelingConfig
from src.rag.labeling.judge import (
    JUDGE_FAIL_SCORE,
    finalize,
    judge_tier2,
    judged_rows,
    parse_scores,
    select_tier2,
    tier1_score,
)


# --------------------------------------------------------------- đồ giả

class FakePoint:
    def __init__(self, payload: Dict[str, Any], score: float):
        self.payload = payload
        self.score = score


class FakeEncoder:
    """Không có encode_with_sparse -> candidates.py đi đường dense."""

    def encode(self, text, normalize_embeddings=True):
        class _V:
            def tolist(self_inner):
                return [0.1, 0.2, 0.3]
        return _V()


class FakeVectorStore:
    """Mỗi ngôn ngữ có 60 chunk sẵn, search trả top_k đầu tiên của ngôn ngữ đó.

    Chunk nguồn `src_zh_absent` KHÔNG nằm trong corpus giả, để ép nhánh "chèn
    chunk nguồn vào nhóm ngôn ngữ của nó" phải chạy.
    """

    def __init__(self):
        self.encoder = FakeEncoder()
        self.collection_name = "fake"
        self.client = None
        self.calls: List[str] = []
        self.corpus = {
            lang: [
                {
                    "chunk_id": f"{lang}_c{i:03d}",
                    "doc_id": f"{lang}_d{i:03d}",
                    "language": lang,
                    "content": f"noi dung {lang} so {i} " * 8,
                }
                for i in range(60)
            ]
            for lang in ("en", "vi", "zh")
        }

    def search(self, query_vector, top_k=5, language=None, query_filter=None):
        assert language is not None, "bước 4 phải search tách theo từng ngôn ngữ"
        self.calls.append(language)
        pool = self.corpus[language][:top_k]
        return [FakePoint(dict(p), 1.0 - i * 0.01) for i, p in enumerate(pool)]


class FakeScorer:
    """Điểm tầng 1 xác định được, không ngẫu nhiên: băm ổn định từ chunk_id."""

    def score(self, query: str, docs: List[str]) -> List[float]:
        return [(hash_str(d) % 1000) / 1000.0 for d in docs]


def hash_str(s: str) -> int:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 1_000_003
    return h


class FakePrompt:
    """Giao diện .format() giống PromptVersion của MLflow."""

    def __init__(self, text: str):
        self.text = text

    def format(self, **kw):
        out = self.text
        for k, v in kw.items():
            out = out.replace("{{%s}}" % k, str(v))
        return out


class FakeChat:
    """Trả đúng mảng JSON mà judge_tier2.txt khai báo, kèm rác bọc ngoài."""

    def __init__(self, fail_first_n: int = 0):
        self.prompts: List[str] = []
        self.fail_first_n = fail_first_n

    def __call__(self, cfg, prompt_text: str):
        self.prompts.append(prompt_text)
        if len(self.prompts) <= self.fail_first_n:
            return "xin loi toi khong the tra loi"
        chunks = json.loads(prompt_text.split("CHUNKS=")[1].split("\nEND")[0])
        arr = [{"chunk_id": c["chunk_id"], "score": c["chunk_id"] % 4} for c in chunks]
        return "```json\n" + json.dumps(arr) + "\n```"


PROMPT_TEXT = 'QUERY={{query}}\nCHUNKS={{chunks}}\nEND'


def make_records():
    return [
        {"qid": "g000001_1", "query": "cau hoi 1", "src_chunk_id": "en_c007",
         "src_language": "en"},
        {"qid": "g000002_1", "query": "cau hoi 2", "src_chunk_id": "src_zh_absent",
         "src_language": "zh"},
        {"qid": "g000003_1", "query": "cau hoi 3", "src_chunk_id": "vi_c000",
         "src_language": "vi"},
    ]


SRC_LOOKUP = {
    "src_zh_absent": {"chunk_id": "src_zh_absent", "doc_id": "zh_dXXX",
                      "language": "zh", "content": "chunk nguon zh khong trong top-k " * 6}
}


def run_pipeline(cfg, chat=None):
    store = FakeVectorStore()
    rows, b4 = fetch_candidates(store, make_records(), cfg, SRC_LOOKUP)
    b5a = tier1_score(FakeScorer(), rows)
    chat = chat or FakeChat()
    b5b = judge_tier2(rows, FakePrompt(PROMPT_TEXT), cfg, chat=chat)
    kept, final = finalize(rows)
    return rows, kept, {"b4": b4, "b5a": b5a, "b5b": b5b, "final": final}, store, chat


# --------------------------------------------------------------- bước 4

class TestBuoc4(unittest.TestCase):
    def setUp(self):
        self.cfg = LabelingConfig()
        self.rows, self.b4 = fetch_candidates(
            FakeVectorStore(), make_records(), self.cfg, SRC_LOOKUP)

    def test_han_ngach_dung_so(self):
        self.assertEqual(lang_targets(self.cfg), {"en": 25, "vi": 15, "zh": 10})
        self.assertEqual(sum(lang_targets(self.cfg).values()), self.cfg.candidates_per_query)

    def test_pool_dung_50_va_dung_quota(self):
        targets = lang_targets(self.cfg)
        for row in self.rows:
            self.assertEqual(len(row["candidates"]), self.cfg.candidates_per_query, row["qid"])
            by_lang = {}
            for c in row["candidates"]:
                by_lang[c["language"]] = by_lang.get(c["language"], 0) + 1
            self.assertEqual(by_lang, targets, row["qid"])

    def test_chunk_nguon_luon_trong_pool_dung_nhom_ngon_ngu(self):
        for row in self.rows:
            srcs = [c for c in row["candidates"] if c["is_src"]]
            self.assertEqual(len(srcs), 1, row["qid"])
            self.assertEqual(srcs[0]["chunk_id"], row["src_chunk_id"])
            self.assertEqual(srcs[0]["language"], row["src_language"])

    def test_dem_lan_chen_thu_cong(self):
        # Chỉ g000002_1 có chunk nguồn ngoài top-k; hai query kia tự có sẵn.
        self.assertEqual(self.b4["b4_src_missing_from_pool"], 1)
        self.assertEqual(self.b4["b4_src_missing_zh"], 1)
        self.assertNotIn("b4_src_missing_en", self.b4)

    def test_pool_khong_lo_is_src_qua_vi_tri(self):
        # Chunk nguồn không được nằm ở vị trí cố định nào giữa các query.
        pos = [next(i for i, c in enumerate(r["candidates"]) if c["is_src"]) for r in self.rows]
        self.assertGreater(len(set(pos)), 1)

    def test_search_tach_theo_ngon_ngu(self):
        store = FakeVectorStore()
        fetch_candidates(store, make_records()[:1], self.cfg, SRC_LOOKUP)
        self.assertEqual(sorted(store.calls), ["en", "vi", "zh"])


# --------------------------------------------------------------- bước 5

class TestBuoc5(unittest.TestCase):
    def setUp(self):
        self.cfg = LabelingConfig()
        self.rows, self.kept, self.stats, _, self.chat = run_pipeline(self.cfg)

    def test_moi_ung_vien_co_t1_score(self):
        for row in self.rows:
            for c in row["candidates"]:
                self.assertIsInstance(c["t1_score"], float)

    def test_dung_15_duoc_llm_cham(self):
        for row in self.rows:
            judged = [c for c in row["candidates"] if c["t2_judged"]]
            self.assertEqual(len(judged), self.cfg.tier2_per_query, row["qid"])
            khong_cham = [c for c in row["candidates"] if not c["t2_judged"]]
            self.assertEqual(len(khong_cham),
                             self.cfg.candidates_per_query - self.cfg.tier2_per_query)
            self.assertTrue(all(c["t2_score"] == 0 for c in khong_cham))

    def test_15_duoc_chon_dung_la_top10_cong_5_quanh_trung_vi(self):
        for row in self.rows:
            cands = row["candidates"]
            order = sorted(range(len(cands)),
                           key=lambda i: (-cands[i]["t1_score"], str(cands[i]["chunk_id"])))
            picked = set(select_tier2(cands, self.cfg))
            self.assertEqual(picked & set(order[:10]), set(order[:10]))
            # 5 cái còn lại nằm quanh trung vị (hạng 23-27 của 50), không phải
            # hạng 11-15.
            self.assertEqual(picked - set(order[:10]), set(order[23:28]))

    def test_dung_5_loi_goi_llm_moi_query(self):
        moi_query = self.cfg.tier2_per_query // self.cfg.tier2_batch_size
        self.assertEqual(moi_query, 5)
        self.assertEqual(len(self.chat.prompts), moi_query * len(self.rows))
        self.assertEqual(self.stats["b5b"]["b5b_n_llm_call"], moi_query * len(self.rows))
        self.assertEqual(self.stats["b5b"].get("b5b_json_fail", 0), 0)

    def test_moi_loi_goi_dung_3_tai_lieu_va_khong_lo_is_src(self):
        for p in self.chat.prompts:
            chunks = json.loads(p.split("CHUNKS=")[1].split("\nEND")[0])
            self.assertEqual(len(chunks), self.cfg.tier2_batch_size)
            self.assertEqual(sorted(chunks[0].keys()), ["chunk_id", "text"])
            for c in chunks:
                self.assertIsInstance(c["chunk_id"], int)
            self.assertNotIn("is_src", p)
            # chunk_id thật không được lọt vào prompt.
            self.assertNotIn("en_c", p)
            self.assertNotIn("src_zh_absent", p)
            # query đứng trước documents.
            self.assertLess(p.index("QUERY="), p.index("CHUNKS="))

    def test_weight_theo_bang_diem(self):
        bang = {3: 1.0, 2: 0.6, 1: None, 0: 0.0}
        for row in self.rows:
            for c in row["candidates"]:
                self.assertEqual(c["weight"], bang[c["t2_score"]])

    def test_chi_so_cuoi(self):
        f = self.stats["final"]
        self.assertEqual(f["n_query"], 3)
        self.assertIn("ty_le_src_thang", f)
        self.assertIn("ty_le_query_bi_loai", f)
        for lang in ("en", "vi", "zh"):
            self.assertIn(f"t2_positive_rate_{lang}", f)

    def test_query_khong_co_ung_vien_dat_2_bi_loai(self):
        for row in self.rows:
            co = any(c["t2_score"] >= 2 for c in row["candidates"])
            self.assertEqual(row["kept"], co)
        self.assertEqual(len(self.kept), sum(1 for r in self.rows if r["kept"]))


class TestRetryVaFallback(unittest.TestCase):
    def test_retry_roi_thanh_cong(self):
        cfg = LabelingConfig()
        # Lời gọi đầu trả rác -> phải retry trong cùng lô, không được bỏ qua.
        chat = FakeChat(fail_first_n=1)
        rows, _, stats, _, chat = run_pipeline(cfg, chat=chat)
        self.assertEqual(stats["b5b"].get("b5b_json_fail", 0), 0)
        self.assertEqual(stats["b5b"]["b5b_n_llm_retry"], 1)
        self.assertEqual(len(chat.prompts), 5 * len(rows) + 1)

    def test_het_retry_thi_gan_diem_fallback_ro_rang(self):
        cfg = LabelingConfig()
        chat = FakeChat(fail_first_n=10 ** 6)  # luôn hỏng
        rows, kept, stats, _, _ = run_pipeline(cfg, chat=chat)
        self.assertEqual(stats["b5b"]["b5b_json_fail"], 5 * len(rows))
        for row in rows:
            judged = [c for c in row["candidates"] if c["t2_judged"]]
            self.assertEqual(len(judged), cfg.tier2_per_query)
            # KHÔNG được âm thầm thành 0 (0 là nhãn negative thật).
            self.assertTrue(all(c["t2_score"] == JUDGE_FAIL_SCORE for c in judged))
            self.assertTrue(all(c["weight"] is None for c in judged))
        self.assertEqual(len(kept), 0)

    def test_parse_tu_choi_khi_thieu_hoac_thua_chunk(self):
        self.assertIsNone(parse_scores('[{"chunk_id":1,"score":3}]', [1, 2, 3]))
        self.assertIsNone(parse_scores('[{"chunk_id":1,"score":9}]', [1]))
        self.assertIsNone(parse_scores("khong co json", [1]))
        self.assertIsNone(parse_scores('[{"chunk_id":1,"score":3},{"chunk_id":1,"score":0}]', [1]))
        self.assertEqual(parse_scores('[{"chunk_id":1,"score":2}]', [1]), {1: 2})


class TestXacDinhDuoc(unittest.TestCase):
    def test_chay_hai_lan_ra_output_giong_nhau(self):
        cfg = LabelingConfig()
        a, _, _, _, chat_a = run_pipeline(cfg)
        b, _, _, _, chat_b = run_pipeline(cfg)
        self.assertEqual(
            json.dumps(judged_rows(a), ensure_ascii=False, sort_keys=True),
            json.dumps(judged_rows(b), ensure_ascii=False, sort_keys=True),
        )
        # Cả các prompt gửi đi cũng phải giống nhau từng ký tự.
        self.assertEqual(chat_a.prompts, chat_b.prompts)


if __name__ == "__main__":
    unittest.main()
