import json

import numpy as np

import scripts.embed_q3e as eq


def test_truncate_normalize_keeps_prefix_and_unit_length():
    v = eq.truncate_normalize([[3.0, 4.0, 100.0], [0.0, 2.0, -7.0]], 2)
    assert v.shape == (2, 2)
    assert np.allclose(v[0], [0.6, 0.8]) and np.allclose(v[1], [0.0, 1.0])


def test_format_query_follows_model_card():
    assert eq.format_query("đau đầu", "X") == "Instruct: X\nQuery: đau đầu"


class FakeTok:
    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [list(range(len(t))) for t in texts]}


class FakeModel:
    tokenizer = FakeTok()

    def encode(self, texts, **kw):
        return np.array([[len(t), 1.0, 2.0, 3.0] for t in texts], dtype=np.float32)


class FakeClient:
    def __init__(self):
        self.points, self.colls, self.opt = [], {}, []

    def collection_exists(self, c):
        return c in self.colls

    def delete_collection(self, c):
        self.colls.pop(c)

    def create_collection(self, collection_name, vectors_config, quantization_config=None):
        self.colls[collection_name] = (vectors_config, quantization_config)

    def create_payload_index(self, **kw):
        pass

    def update_collection(self, collection_name, optimizer_config):
        self.opt.append(optimizer_config.indexing_threshold)

    def upsert(self, collection_name, points, wait):
        self.points += points


def test_embed_upserts_truncated_vectors_with_stable_ids_and_resumes(tmp_path):
    from scripts.ingest_doc_index import point_id
    src = tmp_path / "t.jsonl"
    with open(src, "w", encoding="utf-8") as fh:
        for d in (11, 12, 13, 14, 15):
            fh.write(json.dumps({"doc_id": d, "language": "vi", "text": "x" * d}) + "\n")
    c = FakeClient()
    args = eq.main.__globals__["argparse"].Namespace(
        input=str(src), collection="c", qdrant_url="u", model="m", dim=2, max_tokens=12, dtype="fp16",
        device="cpu", flash_attn=False, batch_size=2, chunk=2, limit=3, resume_from=1, recreate=True, full_ram=False)
    assert eq.embed(args, model=FakeModel(), client=c) == 3
    assert [p.payload["doc_id"] for p in c.points] == [12, 13, 14]           # bỏ 1 dòng đầu, lấy 3
    assert c.points[0].id == point_id(12)
    assert all(len(p.vector["dense"]) == 2 for p in c.points)
    assert np.isclose(np.linalg.norm(c.points[0].vector["dense"]), 1.0)
    vc, qc = c.colls["c"]
    assert vc["dense"].size == 2 and vc["dense"].on_disk and qc is not None
    assert c.opt == [0, 20000]                                                 # tắt index lúc nạp, bật lại cuối


class P:
    def __init__(self, d, s):
        self.payload, self.score = {"doc_id": d}, s


class SearchClient:
    def __init__(self, n=5):
        self.n, self.reqs = n, []

    def count(self, coll, exact):
        return type("C", (), {"count": self.n})()

    def get_collection(self, coll):
        vec = type("V", (), {"size": 2})()
        return type("G", (), {"config": type("X", (), {"params": type("Y", (), {"vectors": {"dense": vec}})()})()})()

    def query_batch_points(self, collection_name, requests, timeout):
        self.reqs += requests
        return [type("R", (), {"points": [P(7, 0.91), P(3, 0.5)]})() for _ in requests]


def test_search_writes_corrected_candidates_with_instruction_queries(tmp_path, monkeypatch):
    import scripts.kaggle_jobs as kj
    c = tmp_path / "c.jsonl"
    c.write_text(json.dumps({"id": 4, "query": "dau", "corrected_query": "đau"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(kj, "CORRECTED", str(c))
    seen = []

    class M(FakeModel):
        def encode(self, texts, **kw):
            seen.extend(texts)
            return super().encode(texts)

    out = tmp_path / "cands.jsonl"
    ins = tmp_path / "q3e_instruct_p7.txt"
    ins.write_text("X\n  y\n", encoding="utf-8")
    args = eq.main.__globals__["argparse"].Namespace(
        collection="c", qdrant_url="u", query_source="corrected", instruction_file=str(ins), top_k=2, hnsw_ef=8,
        min_points=5, tag=None, out=str(out), limit=0, batch_size=4)
    client = SearchClient()
    eq.search(args, model=M(), client=client)
    assert seen == ["Instruct: X y\nQuery: đau"]
    assert len(client.reqs[0].query) == 2 and client.reqs[0].limit == 2
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row == {"id": 4, "query": "đau", "retriever": "q3e4b_p7", "query_source": "corrected",
                   "doc_ids": ["7", "3"], "scores": [0.91, 0.5]}
    args.min_points = 6
    import pytest
    with pytest.raises(SystemExit):
        eq.search(args, model=M(), client=client)


def test_repo_instruction_files_load_and_tag():
    for n in ("p1", "p2"):
        path = f"prompts/q3e_instruct_{n}.txt"
        assert eq.instruction_tag(path) == n and "\n" not in eq.load_instruction(path)
    assert eq.load_instruction(eq.INSTRUCTION_FILE).startswith("Given a Vietnamese medical question, retrieve medically")
