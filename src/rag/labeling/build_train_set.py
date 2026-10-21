"""Chuyển nhãn bước 5 (judged_kept.jsonl) thành dữ liệu train cho sft-reranker.

    python -m src.rag.labeling.build_train_set
    python -m src.rag.labeling.build_train_set --max-neg 12 --val-frac 0.05

Input:  data/labeling/judged_kept.jsonl (nhãn), data/labeling/candidates.jsonl (văn bản)
Output: data/labeling/train.jsonl, data/labeling/val.jsonl
        mỗi dòng {"query", "passage", "label"} - đúng 3 cột BinaryCrossEntropyLoss
        của sentence-transformers đọc (thêm cột nào nó cũng coi là input).

Nhãn = trọng số của bước 5 (judge.SCORE_WEIGHT): điểm 3 -> 1.0, điểm 2 -> 0.6,
điểm 0 -> 0.0. Điểm 1 và chấm hỏng (-1) có trọng số None nên bị bỏ: 1 điểm là
"cùng chủ đề", ép thành negative là tự tạo negative giả.

Negative mỗi query bị giới hạn (--max-neg): pool 50 có ~35 ứng viên không qua
tầng 2 (gán 0), để nguyên thì negative áp đảo positive. Ưu tiên negative tầng 2
đã chấm 0 (hard negative: tầng 1 thấy giống, LLM nói không), phần còn lại lấy
ngẫu nhiên trong số chưa chấm.

Tách val theo QUERY (cùng một query không nằm ở cả hai tập), seed cố định.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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
    p.add_argument("--max-neg", type=int, default=10, help="số negative tối đa mỗi query")
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

    out = {"train": [], "val": []}
    st = Counter()
    for r in rows:
        pos, hard, easy = [], [], []
        for c in r["candidates"]:
            w = c.get("weight")
            if w is None:
                st["bo_qua_diem_1_hoac_hong"] += 1
                continue
            passage = text.get((r["qid"], c["chunk_id"]))
            if not passage:
                st["thieu_van_ban"] += 1
                continue
            item = {"query": r["query"], "passage": passage, "label": float(w)}
            if w > 0:
                pos.append(item)
                st[f"pos_{c.get('language')}"] += 1
            elif c.get("t2_judged"):
                hard.append(item)
            else:
                easy.append(item)
        rng.shuffle(hard)
        rng.shuffle(easy)
        neg = (hard + easy)[: a.max_neg]
        st["neg_hard"] += min(len(hard), a.max_neg)
        st["neg_easy"] += len(neg) - min(len(hard), a.max_neg)
        split = "val" if r["qid"] in val_q else "train"
        out[split] += pos + neg
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
    print(f"positive theo ngôn ngữ: " + ", ".join(f"{k[4:]} {v}" for k, v in sorted(st.items()) if k.startswith("pos_")))
    print(f"negative: {st['neg_hard']} hard (tầng 2 chấm 0) + {st['neg_easy']} ngẫu nhiên | "
          f"bỏ qua {st['bo_qua_diem_1_hoac_hong']} cặp điểm 1 / chấm hỏng, {st['thieu_van_ban']} cặp thiếu văn bản")
    print(f"-> {out_dir / 'train.jsonl'}, {out_dir / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
