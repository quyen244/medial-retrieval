import importlib.util
import json
import os
import zipfile

import pytest

from scripts import kaggle_jobs as kj

_spec = importlib.util.spec_from_file_location("rerank_job", os.path.join(os.path.dirname(__file__), "..", "kaggle", "rerank_job.py"))
rj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rj)


class FakeScorer:
    """Điểm = độ dài text doc / 100: đủ để kiểm thứ tự mà không cần model."""
    dtype = "fake"

    def __init__(self, fixed=None):
        self.fixed = fixed or {}

    def score(self, pairs):
        return [self.fixed.get(d, len(d) / 100) for _, d in pairs]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Thư mục data/kaggle tạm + pool và doc_texts nhỏ."""
    monkeypatch.setattr(kj, "ROOT", str(tmp_path / "kaggle"))
    monkeypatch.setattr(kj, "DATASET_DIR", str(tmp_path / "kaggle" / "dataset"))
    monkeypatch.setattr(kj, "WORK", str(tmp_path / "kaggle" / "work"))
    monkeypatch.setattr(kj, "OUT", str(tmp_path / "kaggle" / "out"))
    monkeypatch.setattr(kj, "fetch_texts", lambda need: {      # thay Qdrant: text lấy từ doc_texts nhỏ dưới đây
        str(r["doc_id"]): r["text"] for r in kj.jsonl(tmp_path / "doc_texts.jsonl") if str(r["doc_id"]) in need})
    pool = tmp_path / "pool.jsonl"
    write_jsonl(pool, [
        {"id": 1, "query": "đau dạ dày", "rr_model": kj.BGE, "rr_len": 1024, "candidates": [
            {"doc_id": "10", "language": "vi", "title": "a", "rrf_score": 0.016, "found_by": ["orig"], "rerank_score": 0.9},
            {"doc_id": "11", "language": "en", "title": "b", "rrf_score": 0.015, "found_by": ["orig"], "rerank_score": 0.5}]},
        {"id": 2, "query": "sốt cao", "rr_model": kj.BGE, "rr_len": 1024, "candidates": [
            {"doc_id": "20", "language": "zh", "title": "c", "rrf_score": 0.016, "found_by": ["orig"], "rerank_score": 0.7}]},
    ])
    texts = tmp_path / "doc_texts.jsonl"
    write_jsonl(texts, [{"doc_id": 10, "language": "vi", "title": "a", "text": "x"},
                        {"doc_id": 11, "language": "en", "title": "b", "text": "xxxxx"},
                        {"doc_id": 20, "language": "zh", "title": "c", "text": "xxx"},
                        {"doc_id": 30, "language": "vi", "title": "d", "text": "xxxxxxxxxx"},
                        {"doc_id": 99, "language": "vi", "title": "z", "text": "unused"}])
    return tmp_path, str(pool), str(texts)


# ------------------------------------------------------------------ rerank_job.py
def make_job_dir(root, name="j", prefix=""):
    d = root / name
    (d / prefix).mkdir(parents=True, exist_ok=True)
    base = d / prefix
    (base / "meta.json").write_text(json.dumps({"name": name, "model": "m", "kind": "cross-encoder"}), encoding="utf-8")
    write_jsonl(base / "queries.jsonl", [{"id": 1, "query": "q1"}, {"id": 2, "query": "q2"}, {"id": 3, "query": "q3"}])
    write_jsonl(base / "docs.jsonl", [{"doc_id": "a", "text": "aa"}, {"doc_id": "b", "text": "bbbb"}])
    write_jsonl(base / "pairs.jsonl", [{"id": 1, "doc_ids": ["a", "b"]}, {"id": 2, "doc_ids": ["b"]}, {"id": 3, "doc_ids": ["a"]}])
    return d


def zip_dir(d, zpath, prefix=""):
    with zipfile.ZipFile(zpath, "w") as z:
        for f in os.listdir(d / prefix if prefix else d):
            z.write(os.path.join(d, prefix, f), arcname=os.path.join(prefix, f) if prefix else f)


@pytest.mark.parametrize("prefix", ["", "j"])
def test_job_reads_same_content_from_dir_and_zip(tmp_path, prefix):
    d = make_job_dir(tmp_path, prefix=prefix)
    z = tmp_path / "j.zip"
    zip_dir(d, z, prefix)
    for path in (str(d), str(z)):
        job = rj.Job(path)
        assert job.meta["name"] == "j"
        assert [r["id"] for r in job.jsonl("pairs.jsonl")] == [1, 2, 3]
        assert not job.has("check.jsonl")


def test_find_job_by_name_zip_extracted_dir_or_renamed_dir(tmp_path):
    root = tmp_path / "input"
    zdir = root / "ds1"
    zdir.mkdir(parents=True)
    src = make_job_dir(tmp_path / "src", "zipjob")
    zip_dir(src, zdir / "zipjob.zip")
    make_job_dir(root / "ds2", "dirjob")
    renamed = make_job_dir(root / "ds3", "whatever")
    meta = json.loads((renamed / "meta.json").read_text(encoding="utf-8"))
    (renamed / "meta.json").write_text(json.dumps({**meta, "name": "renamedjob"}), encoding="utf-8")
    assert rj.find_job("zipjob", (str(root),)).endswith("zipjob.zip")
    assert rj.find_job("dirjob", (str(root),)).endswith("dirjob")
    assert rj.find_job("renamedjob", (str(root),)).endswith("whatever")
    with pytest.raises(SystemExit):
        rj.find_job("nojob", (str(root),))


def test_load_work_shards_by_id_and_skips_done(tmp_path):
    job = rj.Job(str(make_job_dir(tmp_path)))
    _, _, w0 = rj.load_work(job, (0, 2), set())
    _, _, w1 = rj.load_work(job, (1, 2), set())
    assert [q for q, _ in w0] == [2] and [q for q, _ in w1] == [1, 3]
    _, docs, w = rj.load_work(job, (0, 1), {1})
    assert [q for q, _ in w] == [2, 3] and set(docs) == {"a", "b"}


def test_load_work_fails_when_pair_doc_has_no_text(tmp_path):
    d = make_job_dir(tmp_path)
    write_jsonl(d / "pairs.jsonl", [{"id": 1, "doc_ids": ["a", "ghost"]}])
    with pytest.raises(SystemExit):
        rj.load_work(rj.Job(str(d)), (0, 1), set())


def test_read_done_ignores_half_written_last_line(tmp_path):
    out = tmp_path / "s.jsonl"
    out.write_text('{"id": 1, "scores": {}}\n{"id": 2, "sco', encoding="utf-8")
    assert rj.read_done(str(out)) == {1}


def run_fake(monkeypatch, argv, scorer=None):
    monkeypatch.setattr(rj, "make_scorer", lambda meta, args, device: (scorer or FakeScorer(), "fake", "m"))
    return rj.main(argv)


def test_run_writes_scores_resumes_and_honours_skip_done(tmp_path, monkeypatch):
    d = make_job_dir(tmp_path)
    out = tmp_path / "out" / "s0.jsonl"
    run_fake(monkeypatch, ["--job", str(d), "--out", str(out), "--limit-queries", "1"])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": 1, "scores": {"a": 0.02, "b": 0.04}}]
    prev = tmp_path / "prev_scores.jsonl"
    write_jsonl(prev, [{"id": 3, "scores": {"a": 0.5}}])
    run_fake(monkeypatch, ["--job", str(d), "--out", str(out), "--skip-done", str(tmp_path / "prev_*.jsonl")])
    ids = [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert ids == [1, 2]      # câu 1 không chấm lại, câu 3 đã có ở lượt trước


def test_run_stops_when_kaggle_scores_are_on_another_scale(tmp_path, monkeypatch):
    d = make_job_dir(tmp_path)
    write_jsonl(d / "check.jsonl", [{"id": 1, "doc_id": "a", "expected": 0.9}])
    with pytest.raises(SystemExit):
        run_fake(monkeypatch, ["--job", str(d), "--out", str(tmp_path / "o.jsonl")])
    # cùng thang thì chạy tiếp
    run_fake(monkeypatch, ["--job", str(d), "--out", str(tmp_path / "o2.jsonl")], FakeScorer({"aa": 0.9}))
    assert (tmp_path / "o2.jsonl").exists()


def test_run_records_nan_as_null(tmp_path, monkeypatch):
    d = make_job_dir(tmp_path)
    run_fake(monkeypatch, ["--job", str(d), "--out", str(tmp_path / "o.jsonl"), "--limit-queries", "1"],
             FakeScorer({"aa": float("nan")}))
    row = json.loads((tmp_path / "o.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["scores"]["a"] is None


# ------------------------------------------------------------------ kaggle_jobs.py
def test_t14_job_roundtrip_rescore_merge(env, monkeypatch):
    tmp, pool, texts = env
    kj.main(["t14", "--owner", "me", "--pool", pool, "--doc-texts", texts, "--out-pool", str(tmp / "q.jsonl")])
    zpath = os.path.join(kj.DATASET_DIR, "t14_qwen3r4b.zip")
    with zipfile.ZipFile(zpath) as z:
        assert set(z.namelist()) == {"meta.json", "queries.jsonl", "pairs.jsonl", "docs.jsonl"}
        meta = json.loads(z.read("meta.json"))
        docs = [json.loads(line)["doc_id"] for line in z.read("docs.jsonl").decode().splitlines()]
    assert meta["kind"] == "qwen3" and meta["name"] == "t14_qwen3r4b"
    assert sorted(docs) == ["10", "11", "20"]                     # doc 99 không được đưa lên
    assert not os.path.exists(os.path.join(kj.DATASET_DIR, "rerank_job.py"))  # nhúng trong kernel
    ds = json.load(open(os.path.join(kj.DATASET_DIR, "dataset-metadata.json")))
    assert ds["id"] == "me/vibiomir-rerank-jobs"
    km = json.load(open(os.path.join(kj.ROOT, "kernel_t14_qwen3r4b", "kernel-metadata.json")))
    assert km["dataset_sources"] == ["me/vibiomir-rerank-jobs"] and km["enable_gpu"] and km["enable_internet"]
    run_py = open(os.path.join(kj.ROOT, "kernel_t14_qwen3r4b", "run.py"), encoding="utf-8").read()
    assert 'JOB = "t14_qwen3r4b"' in run_py and "NEED_QWEN = True" in run_py
    compile(run_py, "run.py", "exec")
    assert '"--dtype", "fp16"' in run_py and repr(open(kj.RERANK_JOB, encoding="utf-8").read()) in run_py

    # Kaggle chấm (giả) -> tải về out/<job>/run1 -> gộp
    out = os.path.join(kj.OUT, "t14_qwen3r4b", "run1", "scores_t14_qwen3r4b.s0of1.jsonl")
    run_fake(monkeypatch, ["--job", zpath, "--out", out])
    kj.main(["merge", "--name", "t14_qwen3r4b"])
    rows = [json.loads(line) for line in open(tmp / "q.jsonl", encoding="utf-8")]
    q1 = rows[0]
    assert q1["rr_model"] == kj.QWEN4B
    assert [c["doc_id"] for c in q1["candidates"]] == ["11", "10"]        # điểm mới: text dài hơn thắng
    assert q1["candidates"][1]["rerank_score_prev"] == 0.9


def test_merge_refuses_when_scores_missing(env, monkeypatch):
    tmp, pool, texts = env
    kj.main(["t14", "--owner", "me", "--pool", pool, "--doc-texts", texts, "--out-pool", str(tmp / "q.jsonl")])
    out = os.path.join(kj.OUT, "t14_qwen3r4b", "scores_t14_qwen3r4b.s0of2.jsonl")
    run_fake(monkeypatch, ["--job", os.path.join(kj.DATASET_DIR, "t14_qwen3r4b.zip"), "--out", out, "--shard", "0/2"])
    with pytest.raises(SystemExit):
        kj.main(["merge", "--name", "t14_qwen3r4b"])
    kj.main(["merge", "--name", "t14_qwen3r4b", "--allow-missing"])


def test_plan_deep_sends_only_uncached_pairs_and_samples_check_from_cache():
    hits = [(1, "q", ["10", "11", "12"]), (2, "q2", ["20"])]
    cache = {(1, "10"): 0.9, (2, "20"): 0.3}
    pairs, check = kj.plan_deep(hits, cache, n_check=5)
    assert pairs == {1: ["11", "12"], 2: []}
    assert sorted((r["id"], r["doc_id"], r["expected"]) for r in check) == [(1, "10", 0.9), (2, "20", 0.3)]


def test_merge_rows_deep_uses_kaggle_then_cache_and_keeps_retrieval_rank(env):
    tmp, _, texts = env
    cand = tmp / "cand.jsonl"
    write_jsonl(cand, [{"id": 1, "query": "q", "doc_ids": ["10", "30", "11"]}])
    info = {"mode": "deep", "candidates": str(cand), "rr_model": kj.BGE, "rr_len": 1024}
    meta_of = kj.doc_meta(texts, {"10", "30", "11"})
    rows, missing = kj.merge_rows(info, {(1, "30"): 0.95, (1, "11"): 0.1}, {(1, "10"): 0.5}, meta_of)
    assert missing == 0
    cs = rows[0]["candidates"]
    assert [c["doc_id"] for c in cs] == ["30", "10", "11"]
    assert cs[0]["rrf_score"] == round(1 / 62, 5) and cs[0]["language"] == "vi"


def test_load_cache_skips_pools_of_other_reranker(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"id": 1, "rr_model": kj.BGE, "rr_len": 1024, "candidates": [{"doc_id": "1", "rerank_score": 0.4}]}])
    write_jsonl(b, [{"id": 1, "rr_model": "other", "rr_len": 1024, "candidates": [{"doc_id": "2", "rerank_score": 0.9}]}])
    assert kj.load_cache([str(a), str(b), str(tmp_path / "missing.jsonl")]) == {(1, "1"): 0.4}


def test_smoke_kernel_limits_queries_and_has_its_own_slug(env):
    tmp, pool, texts = env
    kj.main(["t14", "--owner", "me", "--pool", pool, "--doc-texts", texts, "--out-pool", str(tmp / "q.jsonl")])
    smoke = os.path.join(kj.ROOT, "kernel_t14_qwen3r4b_smoke")
    km = json.load(open(os.path.join(smoke, "kernel-metadata.json")))
    assert km["id"] == "me/vibiomir-rerank-t14-qwen3r4b-smoke"
    run_py = open(os.path.join(smoke, "run.py"), encoding="utf-8").read()
    assert '"--limit-queries", "20"' in run_py
    compile(run_py, "run.py", "exec")


def test_owner_overrides_stale_dataset_metadata(env):
    old = kj.ensure_dataset_meta("me", "vibiomir-rerank-jobs")
    assert old == "me/vibiomir-rerank-jobs"
    assert kj.ensure_dataset_meta("quyen", "vibiomir-rerank-jobs") == "quyen/vibiomir-rerank-jobs"
    assert kj.ensure_dataset_meta(None, "vibiomir-rerank-jobs") == "quyen/vibiomir-rerank-jobs"


def test_refuses_partial_qdrant_index():
    with pytest.raises(SystemExit):
        kj.check_full_index(256, "medical_docs")
    kj.check_full_index(1_003_391, "medical_docs")


def test_half_dtype_is_fp16_like_local_cache_even_if_bf16_reported():
    class FakeTorch:
        float16, float32, bfloat16 = "fp16", "fp32", "bf16"

        class cuda:
            @staticmethod
            def is_bf16_supported():
                return True
    assert rj.half_dtype(FakeTorch, "cuda:0") == "fp16"
    assert rj.half_dtype(FakeTorch, "cpu") == "fp32"


# ------------------------------------------------------------------ câu sửa (qcorr, retr-pool)
def test_load_queries_corrected_falls_back_to_original_when_empty(tmp_path, monkeypatch):
    q, c = tmp_path / "q.jsonl", tmp_path / "c.jsonl"
    write_jsonl(q, [{"id": 1, "query": "dau dau"}, {"id": 2, "query": "sốt"}])
    write_jsonl(c, [{"id": 1, "query": "dau dau", "corrected_query": "đau đầu"}, {"id": 2, "query": "sốt", "corrected_query": ""}])
    monkeypatch.setattr(kj, "QUERIES", str(q))
    monkeypatch.setattr(kj, "CORRECTED", str(c))
    assert kj.load_queries("orig") == [(1, "dau dau"), (2, "sốt")]
    assert kj.load_queries("corrected") == [(1, "đau đầu"), (2, "sốt")]
    assert kj.load_queries("corrected", limit=1) == [(1, "đau đầu")]


def write_cands(tmp, source="corrected"):
    path = str(tmp / "cands.jsonl")
    kj.write_candidates(path, [(1, "đau dạ dày sửa", ["30", "10", "11"], [0.8, 0.7, 0.6]),
                               (2, "sốt cao sửa", ["20"], [0.5])], "q3e4b", source)
    return path


def test_qcorr_job_scores_every_pair_with_corrected_query_and_merges(env, monkeypatch):
    tmp, _, texts = env
    monkeypatch.setattr(kj, "PHASE1", str(tmp))
    kj.main(["qcorr", "--owner", "me", "--candidates", write_cands(tmp), "--model", kj.QWEN4B, "--doc-texts", texts])
    name = "top300_q3e4b_qwen3r4b_qcorr"
    zpath = os.path.join(kj.DATASET_DIR, f"{name}.zip")
    with zipfile.ZipFile(zpath) as z:
        assert "check.jsonl" not in z.namelist()                              # không cache thì không có gì để kiểm
        meta = json.loads(z.read("meta.json"))
        qs = [json.loads(line) for line in z.read("queries.jsonl").decode().splitlines()]
        pairs = [json.loads(line) for line in z.read("pairs.jsonl").decode().splitlines()]
    assert meta["kind"] == "qwen3" and meta["query_source"] == "corrected"
    assert [q["query"] for q in qs] == ["đau dạ dày sửa", "sốt cao sửa"]
    assert pairs == [{"id": 1, "doc_ids": ["30", "10", "11"]}, {"id": 2, "doc_ids": ["20"]}]
    assert "NEED_QWEN = True" in open(os.path.join(kj.ROOT, f"kernel_{name}", "run.py"), encoding="utf-8").read()

    run_fake(monkeypatch, ["--job", zpath, "--out", os.path.join(kj.OUT, name, f"scores_{name}.s0of1.jsonl")])
    kj.main(["merge", "--name", name, "--doc-texts", texts])
    rows = [json.loads(line) for line in open(tmp / f"doc_{name}_pool.jsonl", encoding="utf-8")]
    cs = rows[0]["candidates"]
    assert rows[0]["query"] == "đau dạ dày sửa" and rows[0]["rr_model"] == kj.QWEN4B
    assert [c["doc_id"] for c in cs] == ["30", "11", "10"]                    # điểm giả = độ dài text
    assert cs[0]["found_by"] == ["q3e4b_qcorr"] and cs[0]["rrf_score"] == round(1 / 61, 5)


def test_qcorr_refuses_candidates_searched_with_original_query(env):
    tmp, _, texts = env
    with pytest.raises(SystemExit, match="corrected"):
        kj.main(["qcorr", "--owner", "me", "--candidates", write_cands(tmp, "orig"), "--doc-texts", texts])


def test_retr_pool_keeps_retrieval_order_and_scores(env, monkeypatch):
    tmp, _, texts = env
    monkeypatch.setattr(kj, "PHASE1", str(tmp))
    kj.main(["retr-pool", "--candidates", write_cands(tmp), "--doc-texts", texts])
    rows = [json.loads(line) for line in open(tmp / "doc_top3_q3e4b_retr_qcorr_pool.jsonl", encoding="utf-8")]
    assert rows[0]["rr_model"] == "retrieval:q3e4b"
    assert [(c["doc_id"], c["rerank_score"], c["language"]) for c in rows[0]["candidates"]] == [
        ("30", 0.8, "vi"), ("10", 0.7, "vi"), ("11", 0.6, "en")]
    kj.main(["retr-pool", "--candidates", write_cands(tmp), "--doc-texts", texts, "--top-k", "2"])
    rows = [json.loads(line) for line in open(tmp / "doc_top2_q3e4b_retr_qcorr_pool.jsonl", encoding="utf-8")]
    assert [c["doc_id"] for c in rows[0]["candidates"]] == ["30", "10"]
