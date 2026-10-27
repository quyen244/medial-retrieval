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
