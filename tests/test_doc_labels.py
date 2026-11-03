"""Test pipeline nhãn cấp doc 3 mức (src/rag/labeling/doc_labels.py), không cần GPU/Qdrant/vLLM.

LLM giả đọc mảng {{documents}} trong prompt và chấm theo từ khoá trong text, nên kiểm được
đường đi id cục bộ -> doc_id thật mà không phụ thuộc chất lượng nhãn. Các bất biến:
  - đúng 60 hạng mỗi câu (20 đầu + 40 rải, gồm hạng 300), không trùng
  - few-shot không bao giờ lấy chính câu đang chấm
  - lời gọi hỏng hết retry ra -1, không ra 0
  - chạy lại cùng file ra là làm tiếp, không chấm lại câu đã xong
  - nhãn tay đè nhãn LLM; mức 1 theo partial_label; câu không có mức 2 bị bỏ
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rag.labeling import doc_labels as dl
fill = dl.fill

TEMPLATE = "Q: {{query}}\nF: {{few_shots}}\nDOCS: {{documents}}"


def fake_chat(prompt, calls=None, broken=False):
    if calls is not None:
        calls.append(prompt)
    if broken:
        return "không phải JSON"
    docs = json.loads(prompt.split("DOCS: ", 1)[1])
    out = [{"doc_id": d["doc_id"], "same_subject": True,
            "score": 2 if "POS" in d["text"] else 1 if "HALF" in d["text"] else 0} for d in docs]
    return "```json\n" + json.dumps(out) + "\n```"


def pool_row(qid, n=300):
    return {"id": qid, "query": f"câu {qid}", "rr_model": "bge", "rr_len": 1024,
            "candidates": [{"doc_id": str(qid * 1000 + k), "rerank_score": 1 - k / 1000, "language": "vi"}
                           for k in range(1, n + 1)]}


class TestSelect(unittest.TestCase):
    def test_pick_ranks(self):
        r = dl.pick_ranks(300, 20, 40)
        self.assertEqual(len(r), 60)
        self.assertEqual(len(set(r)), 60)
        self.assertEqual(r[:20], list(range(1, 21)))
        self.assertEqual((r[20], r[-1]), (21, 300))
        self.assertEqual(r, sorted(r))

    def test_select_keeps_ranks_and_rejects_unsorted(self):
        rows = dl.select_candidates([pool_row(1)], dl.pick_ranks())
        self.assertEqual([c["rank"] for c in rows[0]["candidates"]], dl.pick_ranks())
        self.assertEqual(rows[0]["candidates"][0]["doc_id"], "1001")
        bad = pool_row(2)
        bad["candidates"][0]["rerank_score"] = -1
        with self.assertRaises(ValueError):
            dl.select_candidates([bad], dl.pick_ranks())


class TestPromptVars(unittest.TestCase):
    def test_few_shot_excludes_self(self):
        vecs = {1: np.array([1.0, 0]), 2: np.array([0.9, 0.436]), 3: np.array([0, 1.0])}
        self.assertEqual(dl.cosine_neighbours(vecs, 1, [1, 2, 3], 2), [2, 3])
        self.assertNotIn(2, dl.cosine_neighbours(vecs, 2, [1, 2, 3], 5))

    def test_render_few_shots(self):
        human = {7: {"query": "hỏi 7", "labels": {"a": (0, "", 3), "b": (2, "đúng bệnh", 9), "c": (2, "", 1),
                                                   "d": (1, "", 5)}}}
        texts = {"a": "A" * 600, "b": "B", "c": "C", "d": "D"}
        s = dl.render_few_shots(dl.select_few_shots([7], human, texts), human, texts, max_chars=500)
        self.assertIn("Câu hỏi: hỏi 7", s)
        # mỗi mức một doc, hạng tốt nhất: 2 -> c (hạng 1), 1 -> d, 0 -> a
        self.assertEqual(re.findall(r"Tài liệu: (\w)", s), ["C", "D", "A"])
        self.assertIn("A" * 500 + "...", s)
        self.assertEqual(dl.render_few_shots([], human, {}), dl.NONE_TEXT)

    def test_few_shots_always_cover_three_levels(self):
        # 2 câu giống nhất (1, 2) đều thiếu mức 1 -> lấy đúng 1 doc mức 1 từ câu 3, không lấy gì từ câu 4
        human = {1: {"query": "q1", "labels": {"a": (2, "", 1), "b": (0, "", 2)}},
                 2: {"query": "q2", "labels": {"c": (2, "", 1)}},
                 3: {"query": "q3", "labels": {"d": (2, "", 1), "e": (1, "", 4), "f": (0, "", 5)}},
                 4: {"query": "q4", "labels": {"g": (1, "", 1)}}}
        texts = {k: k for k in "abcdefg"}
        sel = dl.select_few_shots([1, 2, 3, 4], human, texts, shots=2)
        self.assertEqual(sel, [(1, [("a", 2, ""), ("b", 0, "")]), (2, [("c", 2, "")]), (3, [("e", 1, "")])])
        # đã đủ 3 mức trong 2 câu đầu -> không lấy thêm
        self.assertEqual([q for q, _ in dl.select_few_shots([3, 4, 1], human, texts, shots=2)], [3, 4])
        # per_label=2: bù tới đủ 2 doc mỗi mức nếu có
        cnt = {}
        for _, ex in dl.select_few_shots([1, 2, 3, 4], human, texts, shots=2, per_label=2):
            for _, sc, _ in ex:
                cnt[sc] = cnt.get(sc, 0) + 1
        self.assertEqual(cnt, {2: 2, 0: 2, 1: 2})          # mức 2 đã đủ từ câu 1, 2 nên d không được lấy


class TestParse(unittest.TestCase):
    def test_parse(self):
        ok = '<think>x</think>[{"doc_id": 1, "score": 2, "answers": true}, {"doc_id": 2, "score": 0}]'
        self.assertEqual(dl.parse_scores(ok, [1, 2]), {1: 2, 2: 0})
        self.assertIsNone(dl.parse_scores('[{"doc_id": 1, "score": 3}]', [1]))      # ngoài thang
        self.assertIsNone(dl.parse_scores('[{"doc_id": 1, "score": 1}]', [1, 2]))   # thiếu id
        self.assertIsNone(dl.parse_scores('[{"doc_id": 1, "score": 1}, {"doc_id": 1, "score": 1}]', [1]))
        self.assertIsNone(dl.parse_scores(None, [1]))


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.texts = {f"d{i}": ("POS " if i % 3 == 0 else "HALF " if i % 3 == 1 else "") + f"text {i}"
                      for i in range(12)}
        self.rows = [{"id": q, "query": f"câu {q}", "candidates": [{"doc_id": f"d{i}", "rank": i + 1} for i in range(12)]}
                     for q in (1, 2)]
        self.vars = lambda r: {"few_shots": "-"}
        self.ids = lambda r: [c["doc_id"] for c in r["candidates"]]

    def test_scores_map_back_and_resume(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "j.jsonl"
            st = dl.run_judge(self.rows[:1], self.ids, self.vars, self.texts, TEMPLATE, fill,
                              lambda p, k=0: fake_chat(p, calls), out, {"prompt_sha": "x"}, batch_size=5, concurrency=2)
            self.assertEqual(st["llm_call"], 3)                         # 12 doc / lô 5 = 3 lời gọi
            row = next(dl.read_jsonl(out))
            got = {c["doc_id"]: c["llm_score"] for c in row["judged"]}
            self.assertEqual(got, {f"d{i}": (2 if i % 3 == 0 else 1 if i % 3 == 1 else 0) for i in range(12)})
            self.assertEqual(row["judged"][0]["rank"], 1)
            dl.run_judge(self.rows, self.ids, self.vars, self.texts, TEMPLATE, fill,
                         lambda p, k=0: fake_chat(p, calls), out, {}, batch_size=5, concurrency=2)
            self.assertEqual(len(calls), 6)                              # câu 1 không bị chấm lại
            self.assertEqual([r["id"] for r in dl.read_jsonl(out)], [1, 2])
            self.assertTrue(all("{{" not in p for p in calls))

    def test_batch_order_is_shuffled_but_deterministic(self):
        orders = []
        for _ in range(2):
            calls = []
            dl.judge_query(self.rows[0], self.ids(self.rows[0]), self.texts, TEMPLATE, fill, self.vars(None),
                           lambda p, k=0: fake_chat(p, calls), 5, 3, 42)
            orders.append([re.findall(r"text (\d+)", p) for p in calls])
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(sum(orders[0], []), [str(i) for i in range(12)])

    def test_failure_is_minus_one(self):
        scores, st = dl.judge_query(self.rows[0], self.ids(self.rows[0]), self.texts, TEMPLATE, fill, self.vars(None),
                                    lambda p, k=0: fake_chat(p, broken=True), 5, 3, 42)
        self.assertEqual(set(scores.values()), {dl.JUDGE_FAIL})
        self.assertEqual((st["batch_fail"], st["llm_call"]), (3, 9))


class TestHumanAgreeBuild(unittest.TestCase):
    def test_load_human_latest_and_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stamp, lab in (("2026-11-01", "irrelevant"), ("2026-11-02", "relevant")):
                Path(tmp, f"labels_query_5_{stamp}.json").write_text(json.dumps(
                    {"query_id": 5, "query": "q", "pool": "p", "exported_at": stamp,
                     "labels": [{"doc_id": "9", "label": lab, "rank_1024": 3},
                                {"doc_id": "8", "label": "unsure", "rank_1024": 4},
                                {"doc_id": "7", "label": "", "rank_1024": 5}]}), encoding="utf-8")
            h = dl.load_human_labels(sorted(Path(tmp).glob("*.json")))
        self.assertEqual(h[5]["labels"], {"9": (2, "", 3), "8": (1, "", 4)})

    def test_agreement(self):
        human = {1: {"labels": {"a": (2, "", 1), "b": (0, "", 2), "c": (1, "", 150)}}}
        rows = [{"id": 1, "judged": [{"doc_id": "a", "rank": 1, "llm_score": 2}, {"doc_id": "b", "rank": 2, "llm_score": 2},
                                     {"doc_id": "c", "rank": 150, "llm_score": dl.JUDGE_FAIL}]}]
        rep = dl.agreement(rows, human)
        self.assertEqual(rep["n_pairs"], 2)
        self.assertEqual(rep["level2"], {"p": 0.5, "r": 1.0, "n_human": 1, "n_llm": 2})
        self.assertEqual(rep["confusion_human_x_llm"]["0->2"], 1)

    def test_build(self):
        judged = [{"id": 1, "query": "q1", "judged": [{"doc_id": "a", "llm_score": 0}, {"doc_id": "b", "llm_score": 1},
                                                     {"doc_id": "c", "llm_score": dl.JUDGE_FAIL}]},
                  {"id": 2, "query": "q2", "judged": [{"doc_id": "d", "llm_score": 1}]}]
        human = {1: {"labels": {"a": (2, "", 1), "z": (0, "", 9)}}}
        texts = {k: k.upper() for k in "abcdz"}
        split, st = dl.build_pairs(judged, human, texts, 0.4, 0.0, 42)
        got = sorted((p["passage"], p["label"]) for p in split["train"])
        self.assertEqual(got, [("A", 1.0), ("B", 0.4), ("Z", 0.0)])   # tay đè LLM; z chỉ có nhãn tay
        self.assertEqual((st["drop_fail"], st["query_no_level2"]), (1, 1))
        split, st = dl.build_pairs(judged, human, texts, None, 0.0, 42)
        self.assertEqual(st["drop_level1"], 2)



class TestKaggleJobEndToEnd(unittest.TestCase):
    """kaggle/doc_label_job.py chạy thật trên zip job, gọi HTTP tới 2 server OpenAI giả (không GPU/vLLM).
    Kiểm: tìm + giải nén job, agree + báo cáo, judge, và câu k luôn đi server k % 2."""

    def setUp(self):
        import http.server
        import importlib.util
        import sys
        import threading
        sys.modules["doc_labels"] = dl          # trên Kaggle doc_labels.py nằm cạnh doc_label_job.py
        spec = importlib.util.spec_from_file_location("doc_label_job", "kaggle/doc_label_job.py")
        self.job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.job)
        self.hits = []
        hits = self.hits

        def handler(idx):
            class H(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    prompt = body["messages"][0]["content"]
                    hits.append((idx, prompt.split("\n", 1)[0]))
                    out = json.dumps({"choices": [{"message": {"content": fake_chat(prompt)}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)

                def log_message(self, *a):
                    pass
            return H
        self.servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler(i)) for i in range(2)]
        for s in self.servers:
            threading.Thread(target=s.serve_forever, daemon=True).start()
        self.urls = ",".join(f"http://127.0.0.1:{s.server_address[1]}/v1" for s in self.servers)

    def tearDown(self):
        for s in self.servers:
            s.shutdown()

    def test_run(self):
        import zipfile
        texts = {f"{q}{i}": ("POS " if i < 2 else "HALF " if i == 2 else "") + f"doc {q}{i}" for q in (1, 2) for i in range(6)}
        cands = [{"id": q, "query": f"câu {q}", "candidates": [{"doc_id": f"{q}{i}", "rank": i + 1} for i in range(6)]}
                 for q in (1, 2)]
        human = {"2": {"query": "câu 2", "labels": {"20": [2, "", 1], "25": [0, "", 6], "22": [1, "", 3]}},
                 "1": {"query": "câu 1", "labels": {"10": [2, "", 1], "14": [0, "", 5]}}}
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "doc_labels_test.zip"
            buf = Path(tmp) / "v.npz"
            np.savez(buf, ids=np.array([1, 2]), vecs=np.eye(2, dtype=np.float32))
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("meta.json", json.dumps({"name": "doc_labels_test", "prompt_sha": dl.prompt_sha(TEMPLATE)}))
                z.writestr("prompt.txt", TEMPLATE)
                z.writestr("candidates.jsonl", "".join(json.dumps(r) + "\n" for r in cands))
                z.writestr("texts.jsonl", "".join(json.dumps({"doc_id": d, "text": t}) + "\n" for d, t in texts.items()))
                z.writestr("human.json", json.dumps(human))
                z.writestr("queries.jsonl", "")
                z.write(buf, "query_vecs.npz")
            out = Path(tmp) / "out"
            self.job.main(["run", "--job", str(zpath), "--out", str(out), "--steps", "agree,judge",
                           "--urls", self.urls, "--concurrency", "2", "--batch-size", "4"])
            sha = dl.prompt_sha(TEMPLATE)
            judged = list(dl.read_jsonl(out / f"judged_{sha}.jsonl"))
            rep = json.load(open(out / f"agree_{sha}_report.json", encoding="utf-8"))
        self.assertEqual(sorted(r["id"] for r in judged), [1, 2])
        for r in judged:
            got = {c["doc_id"]: c["llm_score"] for c in r["judged"]}
            self.assertEqual(got, {f"{r['id']}{i}": 2 if i < 2 else 1 if i == 2 else 0 for i in range(6)})
        self.assertEqual((rep["n_pairs"], rep["exact"]), (5, 1.0))
        # câu 1 -> server 1, câu 2 -> server 0; mọi lô của một câu cùng một server (prefix cache)
        self.assertEqual({q: {i for i, h in self.hits if h == f"Q: câu {q}"} for q in (1, 2)}, {1: {1}, 2: {0}})


if __name__ == "__main__":
    unittest.main()
