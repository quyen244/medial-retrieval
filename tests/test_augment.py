"""Chạy toàn bộ pha augment bằng LLM giả và bảng tra giả.

Không cần vLLM, không cần Qdrant, không cần GPU. Mục tiêu là khoá lại các bất
biến của thiết kế: rẽ nhánh theo độ dài, unit raw không bị augment, bảng chuẩn
thắng model, và chạy hai lần ra cùng kết quả.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.rag.augment.compose import compose_unit
from src.rag.augment.config import AugmentConfig
from src.rag.augment.lookup import IcdLookup, RxNormLookup, normalize
from src.rag.augment.models import AugmentedQuery, Term, Unit
from src.rag.augment.steps import build, coverage, json_object, word_count

SHORT_Q = "Em bị viêm dạ dày, uống omeprazole xong vẫn buồn nôn, có nguy hiểm không?"
LONG_Q = (
    "Em bị viêm xung huyết một phần ba dưới thực quản, viêm dạ dày trợt xung huyết "
    "mạn tính, sẹo loét hành tá tràng. Cho em hỏi các bệnh này của em có nặng và có "
    "thể khỏi được không. Em vừa uống xong đợt thuốc được một hai ngày thì giờ em "
    "lại thấy triệu chứng bị nghẹn ở cổ và hơi buồn nôn, không biết có phải bệnh "
    "nặng lên hay do sao."
)

ICD_TSV = (
    "code\tvi_name\tvi_aliases\ten_name_raw\ten_term\tjoin\n"
    "K29.7\tViêm dạ dày\tViêm dạ dày|Viêm dạ dày, không xác định\tGastritis\tGastritis\texact\n"
    "K26.9\tLoét tá tràng\tLoét tá tràng\tDuodenal ulcer\tDuodenal ulcer\texact\n"
    # Hai mã cùng mang cụm "ung thư" -> cụm đó phải bị loại vì mơ hồ.
    "C22.0\tUng thư gan\tUng thư gan\tLiver cancer\tLiver cancer\texact\n"
    "C46.0\tUng thư da\tUng thư da\tSkin cancer\tSkin cancer\texact\n"
)
RXNORM_CSV = (
    "RXCUI,name,tty,tty_meaning,semantic_type,sources,aliases\n"
    "7646,Omeprazole,IN,Ingredient,Organic Chemical,RXNORM,omeprazol|Omeprazole sodium\n"
)


class FakeTemplate:
    """Thay cho mlflow PromptVersion: chỉ cần .format(**vars)."""

    def __init__(self, name):
        self.name = name
        self.uri = f"fake:/{name}"

    def format(self, **kwargs):
        return json.dumps({"prompt": self.name, "vars": kwargs}, ensure_ascii=False)


PROMPTS = {
    n: FakeTemplate(n)
    for n in ("medical_entity_extraction", "disease_translation",
              "symptom_translation", "query_decomposition", "retrieval_target")
}


class FakeChat:
    """LLM giả, trả lời theo tên prompt. Đếm luôn số lời gọi từng loại."""

    def __init__(self, broken=()):
        self.broken = set(broken)
        self.calls = []

    def __call__(self, prompt, max_tokens):
        req = json.loads(prompt)
        name, v = req["prompt"], req["vars"]
        self.calls.append(name)
        if name in self.broken:
            return "model nói linh tinh, không có JSON"

        if name == "medical_entity_extraction":
            return json.dumps({
                "disease": ["viêm dạ dày", "ung thư"],
                "medication": ["omeprazole", "thuốc nam"],
                "symptom": ["buồn nôn", "nghẹn ở cổ"],
            }, ensure_ascii=False)
        if name == "disease_translation":
            return json.dumps({"vi": v["disease_name"], "en": "LLM_EN", "zh": "中文病"},
                              ensure_ascii=False)
        if name == "symptom_translation":
            names = json.loads(v["symptoms"])
            return json.dumps([{"vi": n, "en": f"en_{i}", "zh": f"zh_{i}"}
                               for i, n in enumerate(names)], ensure_ascii=False)
        if name == "query_decomposition":
            return json.dumps({"sub_queries": ["sub một", "sub hai", "sub ba"]},
                              ensure_ascii=False)
        if name == "retrieval_target":
            return json.dumps({"retrieval_targets": [
                {"aspect": "a", "vi": "muc tieu vi", "en": "target en", "zh": "目标中文"},
            ]}, ensure_ascii=False)
        raise AssertionError(name)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "icd.tsv").write_text(ICD_TSV, encoding="utf-8")
        (d / "rx.csv").write_text(RXNORM_CSV, encoding="utf-8")
        cls.icd = IcdLookup(str(d / "icd.tsv"))
        cls.rx = RxNormLookup(str(d / "rx.csv"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_build(self, query, broken=(), **cfg_kwargs):
        cfg = AugmentConfig(**cfg_kwargs)
        chat = FakeChat(broken=broken)
        aq = build(chat, PROMPTS, cfg, 1, query, self.icd, self.rx)
        return aq, chat


class TestLookup(Base):
    def test_icd_khop_sau_chuan_hoa(self):
        # Alias nguồn có dấu phẩy, câu hỏi thì không.
        self.assertEqual(self.icd.find_en("Viêm dạ dày, không xác định"), "Gastritis")
        self.assertEqual(self.icd.find_en("viêm dạ dày"), "Gastritis")

    def test_icd_loai_cum_mo_ho(self):
        # "ung thư" không có trong bảng (chỉ "ung thư gan"/"ung thư da"), và
        # kể cả nếu có thì cũng phải bị loại. Đây là lỗi đã gặp thật.
        self.assertIsNone(self.icd.find_en("ung thư"))

    def test_icd_bo_cum_mot_tu(self):
        self.assertIsNone(self.icd.find_en("Gastritis"))

    def test_rxnorm_tra_duoc_ca_alias(self):
        self.assertEqual(self.rx.find_canonical("omeprazole"), "Omeprazole")
        self.assertEqual(self.rx.find_canonical("Omeprazol"), "Omeprazole")
        self.assertIsNone(self.rx.find_canonical("thuốc nam"))

    def test_normalize_bo_dau_cau(self):
        self.assertEqual(normalize("Loét tá tràng, cấp tính"), "loét tá tràng cấp tính")


class TestNhanhNgan(Base):
    def test_mot_unit(self):
        aq, _ = self.run_build(SHORT_Q)
        self.assertFalse(aq.is_long)
        self.assertEqual([u.kind for u in aq.units], ["main"])

    def test_bang_chuan_thang_llm(self):
        aq, _ = self.run_build(SHORT_Q)
        gastritis = next(t for t in aq.diseases if t.vi == "viêm dạ dày")
        self.assertEqual(gastritis.en, "Gastritis")
        self.assertEqual(gastritis.source_en, "icd")
        # Bảng ICD không có cột tiếng Trung -> zh vẫn phải lấy từ model.
        self.assertEqual(gastritis.zh, "中文病")

    def test_tra_khong_ra_thi_dung_ban_dich_llm(self):
        aq, _ = self.run_build(SHORT_Q)
        ungthu = next(t for t in aq.diseases if t.vi == "ung thư")
        self.assertEqual(ungthu.en, "LLM_EN")
        self.assertEqual(ungthu.source_en, "llm")

    def test_thuoc_giu_nguyen_khong_dich(self):
        aq, _ = self.run_build(SHORT_Q)
        ome = next(t for t in aq.medications if t.vi == "omeprazole")
        self.assertEqual(ome.en, "Omeprazole")
        self.assertEqual(ome.source_en, "rxnorm")
        self.assertEqual(ome.zh, "")
        khac = next(t for t in aq.medications if t.vi == "thuốc nam")
        self.assertEqual(khac.en, "thuốc nam")
        self.assertEqual(khac.source_en, "original")

    def test_khong_goi_llm_dich_thuoc(self):
        _, chat = self.run_build(SHORT_Q)
        # 2 bệnh -> 2 lời gọi dịch bệnh; thuốc không có lời gọi nào.
        self.assertEqual(chat.calls.count("disease_translation"), 2)
        self.assertEqual(chat.calls.count("symptom_translation"), 1)

    def test_chuoi_augment_co_du_ba_ngon_ngu(self):
        aq, _ = self.run_build(SHORT_Q)
        text = aq.units[0].retrieval_text()
        for expect in (SHORT_Q, "Gastritis", "中文病", "Omeprazole",
                       "target en", "目标中文"):
            self.assertIn(expect, text)

    def test_cau_goc_dung_dau_chuoi(self):
        aq, _ = self.run_build(SHORT_Q)
        self.assertTrue(aq.units[0].retrieval_text().startswith(SHORT_Q))

    def test_khong_con_nhan_doan(self):
        aq, _ = self.run_build(SHORT_Q)
        text = aq.units[0].retrieval_text()
        for marker in ("[ORIGINAL]", "[DISEASES]", "[SYMPTOMS]", "[RETRIEVAL_TARGETS]"):
            self.assertNotIn(marker, text)


class TestNhanhDai(Base):
    def test_ba_sub_cong_mot_raw(self):
        aq, _ = self.run_build(LONG_Q)
        self.assertTrue(aq.is_long)
        self.assertEqual([u.kind for u in aq.units], ["sub", "sub", "sub", "raw"])

    def test_raw_khong_bi_augment(self):
        aq, _ = self.run_build(LONG_Q)
        raw = aq.units[-1]
        self.assertEqual(raw.retrieval_text(), LONG_Q)
        self.assertNotIn("Gastritis", raw.retrieval_text())

    def test_sub_dung_chung_khoi_thuat_ngu(self):
        aq, _ = self.run_build(LONG_Q)
        for u in aq.units[:3]:
            self.assertIn("Gastritis", u.retrieval_text())
            self.assertTrue(u.retrieval_text().startswith(u.text))

    def test_mot_lan_goi_target_cho_moi_sub(self):
        _, chat = self.run_build(LONG_Q)
        self.assertEqual(chat.calls.count("query_decomposition"), 1)
        self.assertEqual(chat.calls.count("retrieval_target"), 3)

    def test_tat_raw_thi_con_ba_unit(self):
        aq, _ = self.run_build(LONG_Q, include_raw_unit=False)
        self.assertEqual([u.kind for u in aq.units], ["sub", "sub", "sub"])

    def test_dem_thuc_the_bi_rung(self):
        aq, _ = self.run_build(LONG_Q)
        # Sub-query giả ("sub một"...) không nhắc thực thể nào.
        self.assertEqual(aq.stats["n_entity"], 6)
        self.assertEqual(aq.stats["n_entity_missing"], 6)

    def test_thuc_the_rung_van_con_trong_chuoi(self):
        # Rụng khỏi sub-query không có nghĩa là mất: khối thuật ngữ dùng chung
        # được ghép vào mọi unit, nên thuật ngữ vẫn đi vào truy xuất.
        aq, _ = self.run_build(LONG_Q)
        self.assertIn("viêm dạ dày", aq.units[0].retrieval_text())


class TestHong(Base):
    def test_trich_xuat_hong_van_ra_ket_qua(self):
        aq, _ = self.run_build(SHORT_Q, broken=["medical_entity_extraction"])
        self.assertIn("extract", aq.failures)
        self.assertEqual(aq.diseases, [])
        # Vẫn còn unit để truy xuất, không bỏ query.
        self.assertEqual(len(aq.units), 1)
        self.assertIn(SHORT_Q, aq.units[0].retrieval_text())

    def test_phan_ra_hong_thi_tut_ve_mot_sub(self):
        aq, _ = self.run_build(LONG_Q, broken=["query_decomposition"])
        self.assertIn("decompose", aq.failures)
        self.assertEqual([u.kind for u in aq.units], ["sub", "raw"])
        self.assertEqual(aq.units[0].text, LONG_Q)

    def test_dich_trieu_chung_lech_so_luong_thi_bo_ca_cum(self):
        class LechChat(FakeChat):
            def __call__(self, prompt, max_tokens):
                req = json.loads(prompt)
                if req["prompt"] == "symptom_translation":
                    self.calls.append("symptom_translation")
                    return json.dumps([{"vi": "x", "en": "y", "zh": "z"}])  # thiếu 1
                return super().__call__(prompt, max_tokens)

        cfg = AugmentConfig()
        aq = build(LechChat(), PROMPTS, cfg, 1, SHORT_Q, self.icd, self.rx)
        self.assertIn("symptom_translation", aq.failures)
        # Không ghép lệch: giữ nguyên chữ gốc thay vì gán nhầm tên tiếng Anh.
        for t in aq.symptoms:
            self.assertEqual(t.en, t.vi)
            self.assertEqual(t.zh, "")

    def test_target_hong_van_con_cau_hoi(self):
        aq, _ = self.run_build(SHORT_Q, broken=["retrieval_target"])
        self.assertIn("retrieval_target", aq.failures)
        self.assertIn(SHORT_Q, aq.units[0].retrieval_text())


class TestXacDinh(Base):
    def test_chay_hai_lan_ra_cung_ket_qua(self):
        a, _ = self.run_build(LONG_Q)
        b, _ = self.run_build(LONG_Q)
        self.assertEqual(json.dumps(a.to_dict(), ensure_ascii=False, sort_keys=True),
                         json.dumps(b.to_dict(), ensure_ascii=False, sort_keys=True))

    def test_nguong_dai_ngan_theo_so_tu(self):
        cfg = AugmentConfig()
        self.assertLessEqual(word_count(SHORT_Q), cfg.narrative_word_threshold)
        self.assertGreater(word_count(LONG_Q), cfg.narrative_word_threshold)


class TestTienIch(unittest.TestCase):
    def test_boc_json_co_rac_bao_quanh(self):
        self.assertEqual(json_object('nói thêm ```json\n{"a": 1}\n``` hết'), {"a": 1})
        self.assertIsNone(json_object("không có json"))

    def test_coverage_khop_sau_chuan_hoa(self):
        out = coverage(["viêm dạ dày", "buồn nôn"], ["Viêm dạ dày có nặng không?"])
        self.assertEqual(out["entity_missing"], ["buồn nôn"])

    def test_term_bo_truong_rong_va_trung(self):
        self.assertEqual(Term(vi="a", en="a", zh="").forms(), ["a"])

    def test_unit_raw_khong_ghep_gi(self):
        aq = AugmentedQuery(qid=1, original="x", diseases=[Term(vi="b", en="c")])
        u = Unit(kind="raw", text="x")
        self.assertEqual(compose_unit(aq, u), "x")


if __name__ == "__main__":
    unittest.main()
