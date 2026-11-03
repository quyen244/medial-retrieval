import json
import os
import zipfile

import numpy as np
import pytest

from scripts import kaggle_embed as ke

ej = ke.embed_job_module()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def job(tmp_path, monkeypatch):
    import scripts.kaggle_jobs as kj
    c = tmp_path / "c.jsonl"
    write_jsonl(c, [{"id": 1, "query": "dau", "corrected_query": "đau"}, {"id": 2, "query": "sốt", "corrected_query": ""}])
    monkeypatch.setattr(kj, "CORRECTED", str(c))
    docs = tmp_path / "docs.jsonl"
    write_jsonl(docs, [{"doc_id": 100 + i, "language": "vi" if i % 2 else "en", "title": "t", "text": "x" * (i + 1)}
                       for i in range(7)])
    zpath, n, nq = ke.write_job("j", str(docs), {"p1": "A", "p2": "B"}, dataset_dir=str(tmp_path / "ds"), shard_size=3)
    return tmp_path, zpath


def test_write_job_has_corrected_queries_and_instructions(job):
    _, zpath = job
    with zipfile.ZipFile(zpath) as z:
        assert set(z.namelist()) == set(ej.FILES)
        meta = json.loads(z.read("meta.json"))
        qs = [json.loads(x) for x in z.read("queries.jsonl").decode().splitlines()]
        assert json.loads(z.read("instructions.json")) == {"p1": "A", "p2": "B"}
    assert meta["n_docs"] == 7 and meta["query_source"] == "corrected"
    assert qs == [{"id": 1, "query": "đau"}, {"id": 2, "query": "sốt"}]


def test_shards_split_over_parts_and_workers_exactly_once():
    got = [s for p in range(3) for w in range(2) for s in ej.my_shards(1000, 70, f"{p}/3", f"{w}/2")]
    assert sorted(got) == list(range(15))


class FakeEnc:
    """vector = [độ dài text, 1, 0]; doc 103 ra NaN để kiểm search loại nó."""
    dim, batch_tokens, auto_eod = 3, 100, True

    def ids(self, texts):
        return [[0] * len(t) for t in texts], sum(len(t) > 5 for t in texts)

    def encode_ids(self, ids):
        v = np.array([[len(x), 1.0, 0.0] for x in ids], dtype=np.float16)
        v[[len(x) == 4 for x in ids]] = np.nan
        return v


def run_embed(zpath, out, worker="0/1", **kw):
    kw.setdefault("done_dirs", [])
    args = ej.main.__globals__["argparse"].Namespace(
        job=zpath, model="m", out=str(out), part="0/1", worker=worker, device="cpu", dtype="fp16", max_tokens=8,
        batch_tokens=100, shard_size=3, limit_docs=0, max_hours=1.0, **kw)
    ej.embed(args, enc=FakeEnc())


def test_embed_writes_shards_queries_stats_and_resumes(job):
    tmp, zpath = job
    out = tmp / "out"
    run_embed(zpath, out, "0/2")
    run_embed(zpath, out, "1/2")
    assert sorted(os.listdir(out)) == sorted(
        [f"docs_{s:05d}{e}" for s in range(3) for e in (".npy", ".ids.npy")] + [f"stats_{s:05d}.json" for s in range(3)]
        + ["queries_p1.npy", "queries_p2.npy", "queries.ids.npy", "queries.jsonl", "instructions.json"])
    assert np.load(out / "docs_00001.ids.npy").tolist() == [103, 104, 105]
    st = json.load(open(out / "stats_00001.json"))
    assert st["n"] == 3 and st["nan"] == 1 and st["truncated"] == 1 and st["by_language"]["vi"]["n"] == 2
    assert json.load(open(out / "instructions.json"))["_query_source"] == "corrected"
    mtime = os.path.getmtime(out / "docs_00000.npy")
    run_embed(zpath, out, "0/1")                                       # đủ hết: không ghi lại
    assert os.path.getmtime(out / "docs_00000.npy") == mtime


def test_search_is_exact_cosine_over_all_shards_and_skips_nan(tmp_path):
    rng = np.random.default_rng(0)
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    docs = rng.normal(size=(9, 4)).astype(np.float16)
    docs[5] = np.nan
    ids = np.arange(9) + 500
    np.save(a / "docs_00000.npy", docs[:4]); np.save(a / "docs_00000.ids.npy", ids[:4])
    np.save(b / "docs_00001.npy", docs[4:]); np.save(b / "docs_00001.ids.npy", ids[4:])   # shard ở phiên khác
    q = rng.normal(size=(2, 4)).astype(np.float16)
    np.save(a / "queries_p1.npy", q); np.save(a / "queries.ids.npy", np.array([7, 8]))
    write_jsonl(a / "queries.jsonl", [{"id": 7, "query": "q7"}, {"id": 8, "query": "q8"}])
    json.dump({"p1": "A", "_query_source": "corrected"}, open(a / "instructions.json", "w"))
    out = tmp_path / "c.jsonl"
    ej.main(["search", "--vecs", str(a), str(b), "--instruction", "p1", "--top-k", "5", "--retriever", "r", "--dim", "3",
             "--out", str(out)])
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    dn, qn = ej.normalize(docs, 3), ej.normalize(q, 3)
    for r, qv in zip(rows, qn):
        s = dn @ qv
        s[5] = -np.inf
        want = [str(ids[i]) for i in np.argsort(-s)[:5]]
        assert r["doc_ids"] == want and r["retriever"] == "r" and r["query_source"] == "corrected"
        assert np.allclose(r["scores"], np.sort(s)[::-1][:5], atol=1e-5)
    assert rows[0]["query"] == "q7"


def test_encoder_last_token_with_left_padding_keeps_order_and_backs_off_on_oom():
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    class Tok:
        def pad(self, enc, padding, return_tensors):
            L = max(map(len, enc["input_ids"]))
            ids = [[0] * (L - len(x)) + x for x in enc["input_ids"]]
            mask = [[0] * (L - len(x)) + [1] * len(x) for x in enc["input_ids"]]
            return Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    class Model:
        calls = 0

        def __call__(self, input_ids, attention_mask):
            Model.calls += 1
            if input_ids.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("fake")
            # vị trí cuối (đệm trái) mang số token thật của câu
            h = attention_mask.sum(1, keepdim=True).float().unsqueeze(-1).expand(-1, input_ids.shape[1], 2)
            return type("O", (), {"last_hidden_state": h})()

    enc = ej.Encoder.__new__(ej.Encoder)
    enc.torch, enc.tok, enc.model, enc.device, enc.dim, enc.batch_tokens = torch, Tok(), Model(), "cpu", 2, 40
    ids = [[1] * n for n in (3, 9, 1, 5, 2)]
    v = enc.encode_ids(ids)
    assert v[:, 0].tolist() == [3, 9, 1, 5, 2] and enc.batch_tokens < 40


def test_kernel_embeds_job_source_and_models(job, monkeypatch):
    tmp, _ = job
    monkeypatch.setattr(ke, "ROOT", str(tmp / "k"))
    kdir, kid = ke.write_kernel("emb_smoke2k", "me", "me/ds", ["q3e06b", "q3e4b"], extra=["--shard-size", "250"])
    code = open(os.path.join(kdir, "run.py"), encoding="utf-8").read()
    compile(code, "run.py", "exec")
    assert repr(open(ke.EMBED_JOB, encoding="utf-8").read()) in code
    assert "('q3e06b', 'Qwen/Qwen3-Embedding-0.6B', 32768)" in code and '"--shard-size", "250"' in code
    meta = json.load(open(os.path.join(kdir, "kernel-metadata.json")))
    assert kid == "me/vibiomir-emb-smoke2k" and meta["enable_gpu"] and meta["dataset_sources"] == ["me/ds"]


def test_summarize_extrapolates_from_per_gpu_rate():
    stats = [{"n": 100, "encode_s": 9.0, "tokenize_s": 1.0, "tokens": 50000, "truncated": 2, "nan": 0, "device": f"cuda:{i}",
              "chars": 150000} for i in range(2)]
    s = ke.summarize(stats, n_gpu=2)
    assert s["doc_per_s_per_gpu"] == 10.0 and s["tokens_per_doc"] == 500 and s["gpus_used"] == 2
    assert s["hours_full_corpus_2gpu"] == round(ke.N_CORPUS / 20 / 3600, 1)


def test_find_job_when_kaggle_unzips_into_dataset_root(job, tmp_path):
    _, zpath = job
    root = tmp_path / "input"
    d = root / "vibiomir-embed-jobs"
    d.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(d)                            # meta.json nằm thẳng ở gốc dataset, không còn tên emb
    assert ej.find_job("j", str(root)) == str(d)
    with pytest.raises(SystemExit, match="đang có"):
        ej.find_job("khac", str(root))


def test_build_parts_makes_one_zip_and_one_kernel_per_session(job, tmp_path, monkeypatch):
    docs = tmp_path / "docs.jsonl"
    monkeypatch.setattr(ke, "ROOT", str(tmp_path / "k"))
    monkeypatch.setattr(ke, "DATASET_DIR", str(tmp_path / "k" / "ds"))
    ke.main(["build", "--owner", "me", "--docs", str(docs), "--name", "emb_full", "--models", "q3e06b", "--parts", "2"])
    assert sorted(os.listdir(tmp_path / "k" / "ds")) == ["dataset-metadata.json", "emb_full.zip"]
    for i in range(2):
        code = open(tmp_path / "k" / f"kernel_emb_full_p{i}of2" / "run.py", encoding="utf-8").read()
        assert 'JOB = "emb_full"' in code and f'PART = "{i}/2"' in code


def test_parts_are_contiguous_blocks_so_part0_is_the_priority_head():
    assert ej.my_shards(100, 10, "0/3", "0/1") == [0, 1, 2]
    assert ej.my_shards(100, 10, "2/3", "0/1") == [6, 7, 8, 9]
    assert ej.my_shards(100, 10, "0/3", "1/2") == [1]


def test_priority_docs_first_and_order_kept(tmp_path, monkeypatch):
    import scripts.kaggle_jobs as kj
    c = tmp_path / "c.jsonl"
    write_jsonl(c, [{"id": 1, "query": "q", "corrected_query": "q"}])
    monkeypatch.setattr(kj, "CORRECTED", str(c))
    docs = tmp_path / "d.jsonl"
    write_jsonl(docs, [{"doc_id": i, "language": "vi", "text": "t"} for i in (5, 6, 7, 8)])
    pool = tmp_path / "pool.jsonl"
    write_jsonl(pool, [{"id": 1, "candidates": [{"doc_id": "8"}, {"doc_id": "6"}]}])
    zpath, n, _ = ke.write_job("j", str(docs), {"p1": "A"}, dataset_dir=str(tmp_path), priority=ke.priority_ids(str(pool)),
                               shard_size=2)
    with zipfile.ZipFile(zpath) as z:
        ids = [json.loads(x)["doc_id"] for x in z.read("docs.jsonl").decode().splitlines()]
        meta = json.loads(z.read("meta.json"))
    assert ids == [6, 8, 5, 7] and meta["n_priority"] == 2 and meta["shard_size"] == 2 and n == 4


def test_embed_skips_shards_done_elsewhere_and_uses_job_shard_size(job):
    tmp, zpath = job
    kaggle_out = tmp / "kaggle"
    run_embed(zpath, kaggle_out, "0/1", done_dirs=[])                 # Kaggle làm hết
    os.remove(kaggle_out / "docs_00002.npy")                           # giả như phiên Kaggle hết giờ trước shard 2
    server = tmp / "server"
    args = ej.main.__globals__["argparse"].Namespace(
        job=zpath, model="m", out=str(server), part="0/1", worker="0/1", device="cpu", dtype="fp16", max_tokens=8,
        batch_tokens=100, shard_size=999, limit_docs=0, max_hours=1.0, done_dirs=[str(kaggle_out)])
    ej.embed(args, enc=FakeEnc())
    assert sorted(f for f in os.listdir(server) if f.endswith(".npy")) == ["docs_00002.ids.npy", "docs_00002.npy"]
    assert not any(f.endswith(".part") for f in os.listdir(server))


def test_local_runner_starts_one_worker_per_gpu_with_done_dirs(tmp_path, monkeypatch):
    import subprocess
    cmds = []

    class P:
        stdout = iter(["ok\n"])

        def __init__(self, cmd, **kw):
            cmds.append(cmd)

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", P)
    codes = ke.main(["local", "--job", "j.zip", "--out", str(tmp_path / "o"), "--done-dirs", "a", "b", "--gpus", "2",
                     "--reverse"])
    assert "--reverse" in cmds[0]
    assert len(cmds) == 2 and cmds[1][cmds[1].index("--worker") + 1] == "1/2"
    assert cmds[0][cmds[0].index("--done-dirs") + 1:] == [os.path.join("a", "q3e4b"), os.path.join("b", "q3e4b")]
    assert cmds[0][cmds[0].index("--model") + 1] == "Qwen/Qwen3-Embedding-4B"


def test_local_runner_supports_vllm_backend_and_custom_python(tmp_path, monkeypatch):
    import subprocess
    cmds = []

    class P:
        stdout = iter(["ok\n"])

        def __init__(self, cmd, **kw):
            cmds.append(cmd)

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", P)
    codes = ke.main(["local", "--job", "j.zip", "--out", str(tmp_path / "o"), "--backend", "vllm",
                     "--python", "/opt/venv/bin/python", "--gpus", "1", "--gpu-memory-utilization", "0.85"])
    assert cmds[0][0] == "/opt/venv/bin/python"
    assert "--backend" in cmds[0] and cmds[0][cmds[0].index("--backend") + 1] == "vllm"
    assert "--gpu-memory-utilization" in cmds[0] and cmds[0][cmds[0].index("--gpu-memory-utilization") + 1] == "0.85"


def test_vllm_encoder_mock_pooling():
    enc = ej.VLLMEncoder.__new__(ej.VLLMEncoder)
    enc.TokensPrompt = None
    enc.vllm_batch_size = 2
    enc.dim = 3

    class MockOutput:
        def __init__(self, vec):
            self.outputs = type("Out", (), {"embedding": vec})()

    class MockLLM:
        def embed(self, prompts, use_tqdm=False):
            return [MockOutput([float(len(p["prompt_token_ids"])), 1.0, 0.0]) for p in prompts]

    enc.llm = MockLLM()
    ids = [[1, 2], [1, 2, 3], [1], [1, 2, 3, 4]]
    v = enc.encode_ids(ids)
    assert v.shape == (4, 3)
    assert v.dtype == np.float16
    assert v[:, 0].tolist() == [2.0, 3.0, 1.0, 4.0]
