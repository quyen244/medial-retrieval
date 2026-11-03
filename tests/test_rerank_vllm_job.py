"""Test chấm rerank bằng vLLM trên Kaggle (kaggle/rerank_vllm_job.py) và gộp pool (scripts/kaggle_rerank_vllm.py).

Không cần GPU/vLLM: hai server HTTP giả trả /score theo một hàm của (câu, doc). Tokenizer bge-reranker-v2-m3 thật
(cache HF tại chỗ) để kiểm cắt doc. Các bất biến:
  - câu k luôn đi server k % 2, mỗi request tối đa per_request doc
  - check.jsonl lệch thang -> dừng trước khi chấm
  - doc dài bị cắt để q + d + 4 <= max_length; doc ngắn giữ nguyên
  - chạy lại thì bỏ qua câu đã có; merge ra đủ câu, xếp theo điểm mới, điểm cũ ở _prev
"""
import http.server
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

from scripts import kaggle_rerank_vllm as krv   # noqa: E402
from scripts.kaggle_jobs import write_job        # noqa: E402


def fake_score(query, doc):
    return round((len(query) * 7 + len(doc) * 13) % 97 / 97, 6)


def load_job_module():
    spec = importlib.util.spec_from_file_location("rerank_vllm_job", "kaggle/rerank_vllm_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeServers:
    def __init__(self, n=2, offset=0.0):
        self.hits = []
        hits = self.hits

        def handler(idx):
            class H(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    hits.append((idx, body["text_1"], len(body["text_2"]), self.path))
                    data = [{"index": i, "score": fake_score(body["text_1"], d) + offset}
                            for i, d in enumerate(body["text_2"])][::-1]      # thứ tự ngược: client phải xếp lại
                    out = json.dumps({"data": data}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)

                def log_message(self, *a):
                    pass
            return H
        self.servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler(i)) for i in range(n)]
        for s in self.servers:
            threading.Thread(target=s.serve_forever, daemon=True).start()
        self.urls = ",".join(f"http://127.0.0.1:{s.server_address[1]}" for s in self.servers)

    def close(self):
        for s in self.servers:
            s.shutdown()


class TestRerankVllmJob(unittest.TestCase):
    def setUp(self):
        self.job = load_job_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.long = "bệnh gan " * 2000                       # chắc chắn vượt 2048 token
        texts = {f"{q}{i}": (self.long if (q, i) == (11, 0) else f"doc {q}{i} " * (i + 1)) for q in (10, 11) for i in range(5)}
        texts["90"] = "đã chấm"
        queries = {10: "câu mười", 11: "câu mười một", 9: "câu chín"}
        pairs = {10: [f"10{i}" for i in range(5)], 11: [f"11{i}" for i in range(5)]}
        self.check_ok = [{"id": 9, "doc_id": "90", "expected": fake_score("câu chín", "đã chấm")}]
        self.args = (queries, pairs, texts)

    def tearDown(self):
        self.tmp.cleanup()

    def make_job(self, check):
        queries, pairs, texts = self.args
        meta = {"model": "BAAI/bge-reranker-v2-m3", "kind": "vllm-score", "max_length": 2048}
        return write_job("t_test", meta, queries, pairs, texts, check, dataset_dir=str(self.dir / "ds"))

    def test_scores_routing_trim_resume(self):
        zpath = self.make_job(self.check_ok)
        fs = FakeServers()
        try:
            out = self.dir / "out"
            self.job.main(["--job", zpath, "--out", str(out), "--urls", fs.urls, "--per-request", "2"])
            rows = {json.loads(l)["id"]: json.loads(l)["scores"] for l in open(out / "scores_t_test.jsonl", encoding="utf-8")}
            _, pairs, texts = self.args
            self.assertEqual(sorted(rows), [10, 11])
            for d in pairs[10]:
                self.assertEqual(rows[10][d], fake_score("câu mười", texts[d]))
            # doc dài đã bị cắt trước khi gửi -> điểm khác điểm của text nguyên
            self.assertNotEqual(rows[11]["110"], fake_score("câu mười một", self.long))
            routed = {q: {i for i, t, n, p in fs.hits if t == q} for q in ("câu mười", "câu mười một")}
            self.assertEqual(routed, {"câu mười": {0}, "câu mười một": {1}})
            self.assertTrue(all(n <= 2 and p == "/score" for _, _, n, p in fs.hits))
            n_hits = len(fs.hits)
            self.job.main(["--job", zpath, "--out", str(out), "--urls", fs.urls])
            self.assertEqual(len(fs.hits), n_hits)                    # đã đủ câu: không gọi server nữa
        finally:
            fs.close()

    def test_trimmer_budget(self):
        trim = self.job.Trimmer("BAAI/bge-reranker-v2-m3", 2048)
        q = "câu mười một"
        cut = trim(q, self.long)
        n = len(trim.tok(q, cut)["input_ids"])
        self.assertLessEqual(n, 2048 + 2)                             # decode rồi tokenize lại có thể lệch vài token
        self.assertGreater(n, 2000)
        self.assertEqual(trim(q, "ngắn"), "ngắn")
        self.assertEqual(trim.n_cut, 1)

    def test_check_scale_stops(self):
        zpath = self.make_job(self.check_ok)
        fs = FakeServers(offset=0.05)
        try:
            with self.assertRaises(SystemExit):
                self.job.main(["--job", zpath, "--out", str(self.dir / "o2"), "--urls", fs.urls])
            self.assertFalse((self.dir / "o2" / "scores_t_test.jsonl").exists()
                             and (self.dir / "o2" / "scores_t_test.jsonl").stat().st_size)
            self.assertEqual({t for _, t, _, _ in fs.hits}, {"câu chín"})   # chỉ chấm cặp check
        finally:
            fs.close()


class TestMerge(unittest.TestCase):
    def test_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cand = lambda q: [{"doc_id": f"{q}{i}", "rerank_score": 0.1 * i} for i in range(3)]
            base = [{"id": q, "query": f"q{q}", "rr_model": "retrieval", "rr_len": 0, "candidates": cand(q)} for q in (1, 2, 3)]
            done = [{**base[0], "rr_model": krv.BGE, "rr_len": 2048, "rr_text": "q3e"}]
            for name, rows in (("base.jsonl", base), ("done.jsonl", done)):
                (tmp / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (tmp / "sc").mkdir()
            (tmp / "sc" / "scores_x.jsonl").write_text(
                json.dumps({"id": 2, "scores": {"20": 0.9, "21": 0.1, "22": 0.5}}) + "\n" + '{"id": 3, "sco', encoding="utf-8")
            common = ["--pool", str(tmp / "base.jsonl"), "--done", str(tmp / "done.jsonl"),
                      "--scores", str(tmp / "sc"), "--out", str(tmp / "full.jsonl")]
            with self.assertRaises(SystemExit):                       # câu 3 chưa có điểm
                krv.main(["merge"] + common)
            krv.main(["merge"] + common + ["--allow-partial"])
            rows = [json.loads(l) for l in open(tmp / "full.jsonl", encoding="utf-8")]
        self.assertEqual([r["id"] for r in rows], [1, 2])
        self.assertEqual(rows[0], done[0])
        self.assertEqual([c["doc_id"] for c in rows[1]["candidates"]], ["20", "22", "21"])
        self.assertEqual(rows[1]["candidates"][0]["rerank_score_prev"], 0.0)
        self.assertEqual((rows[1]["rr_len"], rows[1]["rr_text"]), (2048, "q3e"))


if __name__ == "__main__":
    unittest.main()
