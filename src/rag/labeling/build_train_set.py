"""Chuyển nhãn bước 5 (judged_kept.jsonl) thành dữ liệu train cho sft-reranker.

    python -m src.rag.labeling.build_train_set
    python -m src.rag.labeling.build_train_set --cross-neg 5 --val-frac 0.05

Input:  data/labeling/judged_kept.jsonl (nhãn), data/labeling/candidates.jsonl (văn bản)
Output: data/labeling/train.jsonl, data/labeling/val.jsonl
        mỗi dòng {"query", "passage", "label"} - đúng 3 cột BinaryCrossEntropyLoss
        của sentence-transformers đọc (thêm cột nào nó cũng coi là input).

CHỈ dùng ứng viên đã được LLM tầng 2 chấm. Ứng viên chưa chấm KHÔNG được suy ra
nhãn: pilot cho thấy ngay cả ứng viên điểm tầng 1 < 0.01 vẫn được LLM chấm >= 2
tới ~23%, và vùng điểm tầng 1 >= 0.9 chưa chấm có ~93% là positive. Coi chúng là
negative thì dạy reranker rằng chunk đúng là sai.

Nhãn = trọng số bước 5 (judge.SCORE_WEIGHT): điểm 3 -> 1.0, 2 -> 0.6, 0 -> 0.0.
Điểm 1 và chấm hỏng (-1) có trọng số None nên bị bỏ.

Negative có hai nguồn:
  - ứng viên LLM chấm 0 của chính query (hard negative, rất ít);
  - "cross-query": chunk được LLM chấm >= 2 cho MỘT QUERY KHÁC, chọn query khác ít
    trùng từ với query này. Đoạn văn y khoa thật, đúng chủ đề của query khác, nên
    gần như chắc chắn sai cho query này - không có negative giả như cách cũ.

Tách val theo QUERY (cùng một query không nằm ở cả hai tập), seed cố định.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹ]+")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def words(text: str) -> set:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 1}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/labeling")
    p.add_argument("--judged", default=None, help="mặc định <dir>/judged_kept.jsonl")
    p.add_argument("--candidates", default=None, help="mặc định <dir>/candidates.jsonl")
    p.add_argument("--out-dir", default=None, help="mặc định <dir>")
    p.add_argument("--cross-neg", type=int, default=5, help="số negative lấy từ query khác, mỗi query")
    p.add_argument("--max-overlap", type=float, default=0.15,
                   help="chỉ lấy cross-query negative từ query có độ trùng từ (Jaccard) dưới ngưỡng này")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    d = Path(a.dir)
    judged_path = Path(a.judged) if a.judged else d / "judged_kept.jsonl"
    cand_path = Path(a.candidates) if a.candidates else d / "candidates.jsonl"
    out_dir = Path(a.out_dir) if a.out_dir else d

    text = {}
    for row in read_jsonl(cand_path):
        for c in row["candidates"]:
            text[(row["qid"], c["chunk_id"])] = c.get("content") or ""

    rows = [r for r in read_jsonl(judged_path) if r.get("kept", True)]
    rng = random.Random(a.seed)
    qids = sorted({r["qid"] for r in rows})
    rng.shuffle(qids)
    val_q = set(qids[: int(round(len(qids) * a.val_frac))])

    # Kho positive theo query, để làm cross-query negative cho query khác.
    qwords = {r["qid"]: words(r["query"]) for r in rows}
    pos_pool = {}
    for r in rows:
        pool = [(c["chunk_id"], text.get((r["qid"], c["chunk_id"])))
                for c in r["candidates"]
                if c.get("t2_judged") and c.get("t2_score", -1) >= 2 and text.get((r["qid"], c["chunk_id"]))]
        if pool:
            pos_pool[r["qid"]] = pool

    out = {"train": [], "val": []}
    st = Counter()
    for r in rows:
        own_ids = {c["chunk_id"] for c in r["candidates"]}
        items = []
        for c in r["candidates"]:
            if not c.get("t2_judged"):
                st["bo_qua_chua_cham"] += 1
                continue
            w = c.get("weight")
            if w is None:
                st["bo_qua_diem_1_hoac_hong"] += 1
                continue
            passage = text.get((r["qid"], c["chunk_id"]))
            if not passage:
                st["thieu_van_ban"] += 1
                continue
            items.append({"query": r["query"], "passage": passage, "label": float(w)})
            if w > 0:
                st[f"pos_{c.get('language')}"] += 1
            else:
                st["neg_llm_cham_0"] += 1

        # Cross-query negative: chỉ lấy trong cùng tập (train/val) để không rò.
        split = "val" if r["qid"] in val_q else "train"
        others = [q for q in pos_pool
                  if q != r["qid"] and ((q in val_q) == (split == "val"))]
        rng.shuffle(others)
        got = 0
        mine = qwords[r["qid"]]
        for q in others:
            if got >= a.cross_neg:
                break
            ow = qwords[q]
            if mine and ow and len(mine & ow) / len(mine | ow) >= a.max_overlap:
                continue
            cid, passage = rng.choice(pos_pool[q])
            if cid in own_ids:
                continue
            items.append({"query": r["query"], "passage": passage, "label": 0.0})
            got += 1
        st["neg_cross_query"] += got
        out[split] += items
        st[f"query_{split}"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in out.items():
        rng.shuffle(items)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    labels = Counter(it["label"] for items in out.values() for it in items)
    print(f"{len(rows)} query ({st['query_train']} train / {st['query_val']} val)")
    print(f"train {len(out['train'])} cặp, val {len(out['val'])} cặp | nhãn {dict(sorted(labels.items()))}")
    print("positive theo ngôn ngữ: " + ", ".join(f"{k[4:]} {v}" for k, v in sorted(st.items()) if k.startswith("pos_")))
    print(f"negative: {st['neg_llm_cham_0']} LLM chấm 0 + {st['neg_cross_query']} từ query khác | "
          f"bỏ qua {st['bo_qua_chua_cham']} ứng viên chưa chấm, {st['bo_qua_diem_1_hoac_hong']} điểm 1 / chấm hỏng, "
          f"{st['thieu_van_ban']} thiếu văn bản")
    print(f"-> {out_dir / 'train.jsonl'}, {out_dir / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
