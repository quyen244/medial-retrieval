import json

import pytest

from scripts.fuse_submission import load_chunk_evidence, logit, per_language, rank_docs


def cand(doc_id, language, score):
    return {"doc_id": doc_id, "language": language, "rerank_score": score}


# Hai nhóm ngôn ngữ trên hai thang khác nhau: vi cao, en thấp.
CANDS = [cand("v1", "vi", 0.90), cand("v2", "vi", 0.50), cand("v3", "vi", 0.10),
         cand("e1", "en", 0.40), cand("e2", "en", 0.02), cand("e3", "en", 0.01)]


def test_langmm_maps_each_language_to_its_own_minus_one_to_one():
    key = per_language(CANDS, "langmm")
    assert key["v1"] == key["e1"] == 1.0
    assert key["v3"] == key["e3"] == -1.0
    assert key["v2"] == pytest.approx(0.0)


def test_langmm_puts_best_doc_of_weak_language_level_with_best_doc_of_strong_one():
    order = rank_docs(CANDS, {}, 1, "langmm")
    assert set(order[:2]) == {"v1", "e1"}
    assert rank_docs(CANDS, {}, 1, "orig")[:2] == ["v1", "v2"]


def test_langz_is_zscore_of_logit_within_each_language():
    key = per_language(CANDS, "langz")
    for lang in ("vi", "en"):
        z = [key[c["doc_id"]] for c in CANDS if c["language"] == lang]
        assert sum(z) == pytest.approx(0.0, abs=1e-9)
        assert sum(x * x for x in z) / len(z) == pytest.approx(1.0)
    # logit tách được 0,02 với 0,01 dù xác suất gần nhau
    assert key["e2"] - key["e3"] > 0.3


def test_group_that_cannot_be_normalized_sits_mid_scale():
    cands = [cand("v1", "vi", 0.9), cand("v2", "vi", 0.1), cand("z1", "zh", 0.3),
             cand("e1", "en", 0.2), cand("e2", "en", 0.2)]
    for method in ("langmm", "langz"):
        key = per_language(cands, method)
        assert key["z1"] == 0.0          # nhóm một doc
        assert key["e1"] == key["e2"] == 0.0  # nhóm mọi doc bằng điểm


def test_missing_language_counts_as_vi():
    key = per_language([cand("a", None, 0.9), cand("b", "vi", 0.1)], "langmm")
    assert key == {"a": 1.0, "b": -1.0}


def test_logit_is_finite_at_the_edges():
    assert logit(0.0) < -10 and logit(1.0) > 10


def test_chunkrrf_rewards_docs_found_by_both_branches_and_admits_chunk_only_docs():
    cands = [cand("d1", "vi", 0.9), cand("d2", "vi", 0.8), cand("d3", "vi", 0.7)]
    order = rank_docs(cands, {}, 1, "chunkrrf", chunk_order=["d3", "x9"])
    # d3: 1/63 + 1/61 vượt d1: 1/61; x9 chỉ nhánh chunk thấy nhưng vẫn vào danh sách
    assert order[0] == "d3"
    assert "x9" in order and len(order) == 4


def test_chunkrrf_with_empty_chunk_evidence_equals_orig():
    assert rank_docs(CANDS, {}, 1, "chunkrrf", chunk_order=[]) == rank_docs(CANDS, {}, 1, "orig")


def test_chunkrrf_requires_chunk_evidence():
    with pytest.raises(SystemExit):
        rank_docs(CANDS, {}, 1, "chunkrrf")


def test_chunk_evidence_uses_best_rank_per_branch_then_rrf_across_branches(tmp_path):
    rows = [{"id": 7, "candidates": [
        # A: hạng 1 nhánh vi và hạng 1 nhánh en -> 2/61, đứng đầu
        {"doc_id": "A", "branch": "vi", "retrieval_rank": 1, "language": "vi", "rerank_score": 0.1},
        {"doc_id": "A", "branch": "en", "retrieval_rank": 1, "language": "vi", "rerank_score": 0.1},
        # B: hạng 1 nhánh zh, hai chunk -> MaxP lấy hạng 1 -> 1/61
        {"doc_id": "B", "branch": "zh", "retrieval_rank": 4, "language": "zh", "rerank_score": 0.9},
        {"doc_id": "B", "branch": "zh", "retrieval_rank": 1, "language": "zh", "rerank_score": 0.9},
        # C: hạng 2 nhánh vi -> 1/62; rerank_score cao không được dùng
        {"doc_id": 123, "branch": "vi", "retrieval_rank": 2, "language": "vi", "rerank_score": 0.99},
    ]}]
    path = tmp_path / "dump.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    order, lang = load_chunk_evidence(str(path))
    assert order[7] == ["A", "B", "123"]
    assert lang[(7, "B")] == "zh"
