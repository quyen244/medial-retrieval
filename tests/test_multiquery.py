"""Nhánh nhiều truy vấn: hợp tập rồi rerank một lần bằng câu gốc.

Pipeline, Qdrant và re-ranker đều là đồ giả. Cái cần khoá lại ở đây là luật
điều phối, không phải chất lượng truy xuất.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.rag.pipeline.models import Candidate, QueryItem
from src.rag.pipeline.multiquery import (
    MultiQueryConfig,
    MultiQueryPipeline,
    load_augmented,
    union_candidates,
    unit_texts,
)

ORIGINAL = "Em bị viêm dạ dày, uống thuốc xong vẫn buồn nôn, có nguy hiểm không?"


def cand(chunk_id, score=1.0):
    return Candidate(chunk_id=str(chunk_id), doc_id=f"d{chunk_id}",
                     content=f"noi dung {chunk_id}", language="vi",
                     fusion_score=score)


class FakeReranker:
    def __init__(self):
        self.seen_queries = []
        self.seen_sizes = []

    def rerank(self, query, candidates, top_k):
        self.seen_queries.append(query.text)
        self.seen_sizes.append(len(candidates))
        # Sắp cố định theo chunk_id để kết quả lặp lại được.
        ordered = sorted(candidates, key=lambda c: int(c.chunk_id))
        return ordered[:top_k]

    def get_params(self):
        return {"type": "fake"}


class FakeConfig:
    rerank_top_k = 5
    fusion_top_k = 10


class FakePipeline:
    """Đủ bề mặt mà MultiQueryPipeline dùng tới."""

    def __init__(self, per_text=None):
        self.reranker = FakeReranker()
        self.config = FakeConfig()
        self.per_text = per_text or {}
        self.fuse_calls = []
        self.run_calls = []

    def retrieve_and_fuse(self, query, trace, parallel_retrieval=True, span_prefix=""):
        self.fuse_calls.append(query.text)
        ids = self.per_text.get(query.text, [1, 2, 3])
        fused = [cand(i) for i in ids]
        return {"vi": list(fused)}, fused

    def run(self, query, parallel_retrieval=True):
        self.run_calls.append(query.text)
        from src.rag.pipeline.pipeline import PipelineResult
        from src.rag.pipeline.tracer import Trace
        return PipelineResult(query=query, branch_results={}, fused=[],
                              reranked=[cand(99)], trace=Trace(query.id, query.text))

    def component_params(self):
        return {"config": {}}


def record(qid, units, is_long=False):
    return {"qid": qid, "original": ORIGINAL, "is_long": is_long, "units": units}


class TestUnitTexts(unittest.TestCase):
    def test_dung_augmented_neu_co(self):
        r = record(1, [{"kind": "main", "text": "goc", "augmented": "da ghep"}])
        self.assertEqual(unit_texts(r), ["da ghep"])

    def test_tut_ve_text_khi_augmented_rong(self):
        # Unit raw cố tình không augment.
        r = record(1, [{"kind": "raw", "text": ORIGINAL, "augmented": ""}])
        self.assertEqual(unit_texts(r), [ORIGINAL])

    def test_bo_unit_rong(self):
        r = record(1, [{"kind": "main", "text": "", "augmented": ""},
                       {"kind": "sub", "text": "co chu", "augmented": ""}])
        self.assertEqual(unit_texts(r), ["co chu"])


class TestUnion(unittest.TestCase):
    def test_khu_trung_theo_chunk_id(self):
        union, hits = union_candidates([[cand(1), cand(2)], [cand(2), cand(3)]])
        self.assertEqual(sorted(int(c.chunk_id) for c in union), [1, 2, 3])
        self.assertEqual(hits["2"], 2)

    def test_giu_ban_gap_dau_tien(self):
        a, b = cand(1, score=0.9), cand(1, score=0.1)
        a.content = "ban dau"
        b.content = "ban sau"
        union, _ = union_candidates([[a], [b]])
        self.assertEqual(union[0].content, "ban dau")


class TestDieuPhoi(unittest.TestCase):
    def setUp(self):
        self.query = QueryItem(id=7, text=ORIGINAL)

    def mq(self, units, per_text=None, cfg=None, is_long=False):
        pipe = FakePipeline(per_text)
        aug = {7: record(7, units, is_long)}
        return MultiQueryPipeline(pipe, aug, cfg or MultiQueryConfig()), pipe

    def test_nhanh_ngan_mot_luot(self):
        mq, pipe = self.mq([{"kind": "main", "text": ORIGINAL, "augmented": "A"}])
        out = mq.run(self.query)
        self.assertEqual(pipe.fuse_calls, ["A"])
        self.assertEqual(out.n_unit, 1)

    def test_nhanh_dai_bon_luot(self):
        units = [{"kind": "sub", "text": f"s{i}", "augmented": f"S{i}"} for i in range(3)]
        units.append({"kind": "raw", "text": ORIGINAL, "augmented": ""})
        mq, pipe = self.mq(units, is_long=True)
        out = mq.run(self.query)
        self.assertEqual(pipe.fuse_calls, ["S0", "S1", "S2", ORIGINAL])
        self.assertEqual(out.n_unit, 4)

    def test_mac_dinh_rerank_bang_cau_goc(self):
        mq, pipe = self.mq([{"kind": "main", "text": ORIGINAL, "augmented": "CHUOI AUGMENT"}])
        mq.run(self.query)
        self.assertEqual(pipe.reranker.seen_queries, [ORIGINAL])

    def test_bat_co_thi_rerank_bang_chuoi_augment(self):
        cfg = MultiQueryConfig(rerank_with_augmented=True)
        mq, pipe = self.mq([{"kind": "main", "text": ORIGINAL, "augmented": "CHUOI AUGMENT"}], cfg=cfg)
        mq.run(self.query)
        self.assertEqual(pipe.reranker.seen_queries, ["CHUOI AUGMENT"])

    def test_bat_co_thi_lay_chuoi_cua_unit_dau(self):
        # Nhánh dài: lấy sub-query thứ nhất, không ghép cả bốn unit lại.
        units = [{"kind": "sub", "text": f"s{i}", "augmented": f"S{i}"} for i in range(3)]
        units.append({"kind": "raw", "text": ORIGINAL, "augmented": ""})
        cfg = MultiQueryConfig(rerank_with_augmented=True)
        mq, pipe = self.mq(units, cfg=cfg, is_long=True)
        mq.run(self.query)
        self.assertEqual(pipe.reranker.seen_queries, ["S0"])

    def test_truy_xuat_luon_dung_chuoi_augment(self):
        # Dù rerank dùng câu gì, phần TRUY XUẤT luôn đưa chuỗi augment vào.
        for flag in (False, True):
            cfg = MultiQueryConfig(rerank_with_augmented=flag)
            mq, pipe = self.mq([{"kind": "main", "text": ORIGINAL, "augmented": "CHUOI AUGMENT"}], cfg=cfg)
            mq.run(self.query)
            self.assertEqual(pipe.fuse_calls, ["CHUOI AUGMENT"])

    def test_rerank_dung_mot_lan_tren_ca_tap(self):
        units = [{"kind": "sub", "text": f"s{i}", "augmented": f"S{i}"} for i in range(3)]
        per_text = {"S0": [1, 2], "S1": [2, 3], "S2": [4]}
        mq, pipe = self.mq(units, per_text, is_long=True)
        out = mq.run(self.query)
        # Một lời gọi rerank duy nhất, trên tập hợp đã khử trùng: {1,2,3,4}.
        self.assertEqual(len(pipe.reranker.seen_sizes), 1)
        self.assertEqual(pipe.reranker.seen_sizes[0], 4)
        self.assertEqual(out.n_union, 4)
        self.assertEqual(out.n_overlap, 1)  # chunk 2 do hai lượt tìm ra

    def test_cat_bot_truoc_khi_rerank(self):
        units = [{"kind": "sub", "text": f"s{i}", "augmented": f"S{i}"} for i in range(2)]
        per_text = {"S0": [1, 2, 3], "S1": [4, 5, 6]}
        cfg = MultiQueryConfig(max_rerank_candidates=4)
        mq, pipe = self.mq(units, per_text, cfg, is_long=True)
        mq.run(self.query)
        self.assertEqual(pipe.reranker.seen_sizes[0], 4)

    def test_khong_cat_khi_khong_dat_nguong(self):
        units = [{"kind": "sub", "text": "s", "augmented": "S0"}]
        mq, pipe = self.mq(units, {"S0": list(range(20))})
        mq.run(self.query)
        self.assertEqual(pipe.reranker.seen_sizes[0], 20)

    def test_thieu_qid_thi_tut_ve_baseline(self):
        pipe = FakePipeline()
        mq = MultiQueryPipeline(pipe, {}, MultiQueryConfig())
        out = mq.run(self.query)
        self.assertTrue(out.used_fallback)
        self.assertEqual(pipe.run_calls, [ORIGINAL])
        self.assertEqual(pipe.fuse_calls, [])

    def test_tat_fallback_thi_bao_loi(self):
        pipe = FakePipeline()
        mq = MultiQueryPipeline(pipe, {}, MultiQueryConfig(fallback_to_baseline=False))
        with self.assertRaises(KeyError):
            mq.run(self.query)

    def test_chay_hai_lan_ra_cung_ket_qua(self):
        units = [{"kind": "sub", "text": f"s{i}", "augmented": f"S{i}"} for i in range(3)]
        per_text = {"S0": [3, 1], "S1": [2, 1], "S2": [5]}
        a = self.mq(units, per_text, is_long=True)[0].run(self.query)
        b = self.mq(units, per_text, is_long=True)[0].run(self.query)
        self.assertEqual([c.chunk_id for c in a.result.reranked],
                         [c.chunk_id for c in b.result.reranked])


class TestNapFile(unittest.TestCase):
    def test_doc_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "augmented.jsonl"
            p.write_text(json.dumps(record(3, [{"kind": "main", "text": "x", "augmented": "X"}]),
                                    ensure_ascii=False) + "\n", encoding="utf-8")
            out = load_augmented(str(p))
            self.assertEqual(list(out), [3])

    def test_thieu_file_bao_loi_ro(self):
        with self.assertRaises(FileNotFoundError) as e:
            load_augmented("khong/ton/tai.jsonl")
        self.assertIn("run_augment", str(e.exception))


if __name__ == "__main__":
    unittest.main()
