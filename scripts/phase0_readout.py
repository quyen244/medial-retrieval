"""Đọc bảng chấm chi tiết trong `res/` và dựng ngược ground truth.

BTC trả về `res/<tên>.zip` chứa `scores.txt` với đủ P và R hai tầng. Script này
đọc thẳng từ đó, đếm `m` từ chính file đã nộp trong `data/tmp/phase0/`, rồi suy:

    h = P x m        (số trúng)
    g = h / R        (số gold)

QUAN TRỌNG: luôn tính `h = P x m`. ĐỪNG suy `h` ngược từ dạng đóng
`F2 = 5h/(4g+m)` - dạng đóng lệch khoảng 12% vì macro-F2 không bằng F2 của
macro-P/R, và sai số đó đủ để đảo ngược dấu kết luận. Dạng đóng chỉ dùng để
DỰ ĐOÁN khi đổi `m`, không bao giờ để đọc ngược số đã đo.

Dùng:
    python -m scripts.phase0_readout
    python -m scripts.phase0_readout --predict 0.0944   # F2_chunk giữ nguyên, quét m_doc
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES_DIR = "res"
SUB_DIRS = ["data/tmp/phase0", "data/tmp/sub1200"]

# m không đếm được từ file nộp (bài cũ đã xoá hoặc nộp từ nơi khác)
M_FALLBACK = {"D_ft": (35.1, 40.0)}
# tên trong res/ khác tên file nộp
ALIAS = {"E1": "E1_docs44", "E2": "E2_win2500", "E4": "E4_docs25"}

# Bài không có scores.txt trong res/ nhưng đã biết số. Đánh dấu * ở bảng.
# Để rỗng khi mọi bài đều đã có file thật - đừng để số cũ nằm lại đây.
KNOWN = {}


def read_scores(path):
    """scores.txt -> dict. Chấp nhận cả .zip lẫn thư mục đã giải nén."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.endswith("scores.txt")), None)
            if not name:
                return None
            text = z.read(name).decode("utf-8", "replace")
    else:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    out = {}
    for k, v in re.findall(r"^([A-Z_0-9]+)\s*:\s*([-\d.]+)\s*$", text, re.M):
        out[k] = float(v)
    return out or None


def count_m(name):
    for d in SUB_DIRS:
        p = os.path.join(d, name + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                entries = json.load(fh)
            n = len(entries)
            return (sum(len(e["relevant_docs"]) for e in entries) / n,
                    sum(len(e["relevant_chunks"]) for e in entries) / n)
    return M_FALLBACK.get(name)


def collect():
    rows = []
    seen = set()
    for p in sorted(glob.glob(os.path.join(RES_DIR, "*.zip"))
                    + glob.glob(os.path.join(RES_DIR, "*", "scores.txt"))):
        stem = (os.path.basename(p)[:-4] if p.endswith(".zip")
                else os.path.basename(os.path.dirname(p)).removesuffix("_x"))
        name = ALIAS.get(stem, stem)
        if name in seen:
            continue
        s = read_scores(p)
        m = count_m(name)
        if not s or not m:
            print(f"  (bỏ qua {stem}: {'thiếu scores' if not s else 'không đếm được m'})")
            continue
        seen.add(name)
        md, mc = m
        rows.append({
            "name": name, "final": s["FINAL_SCORE"],
            "md": md, "Pd": s["DOCS_PRECISION"], "Rd": s["DOCS_RECALL"], "F2d": s["DOCS_F2MACRO"],
            "mc": mc, "Pc": s["CHUNKS_PRECISION"], "Rc": s["CHUNKS_RECALL"], "F2c": s["CHUNKS_F2MACRO"],
        })
    for name, k in KNOWN.items():
        if name not in seen:
            rows.append({"name": name + "*", **k})
    for r in rows:
        r["hd"], r["gd"] = r["Pd"] * r["md"], r["Pd"] * r["md"] / r["Rd"]
        r["hc"], r["gc"] = r["Pc"] * r["mc"], r["Pc"] * r["mc"] / r["Rc"]
    return sorted(rows, key=lambda r: r["md"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", type=float, metavar="F2_CHUNK",
                    help="quét m_doc, giữ F2_chunk cố định ở giá trị này")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print("không đọc được bài nào trong res/")
        return

    print(f"\n{'bài':14s} {'FINAL':>7s} | {'m_d':>5s} {'h_d':>6s} {'g_d':>5s} {'F2_d':>7s} | "
          f"{'m_c':>5s} {'h_c':>6s} {'g_c':>5s} {'F2_c':>7s} | {'h_c/h_d':>7s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:14s} {r['final']:7.4f} | {r['md']:5.1f} {r['hd']:6.3f} {r['gd']:5.1f} "
              f"{r['F2d']:7.4f} | {r['mc']:5.1f} {r['hc']:6.3f} {r['gc']:5.1f} {r['F2c']:7.4f} | "
              f"{r['hc']/r['hd']:7.2f}")

    g = sum(r["gd"] for r in rows) / len(rows)
    pts = sorted({(round(r["md"], 1), round(r["hd"], 3)) for r in rows})
    if len(pts) >= 2:
        print(f"\nđường cong h_doc (g_doc trung bình = {g:.1f}):")
        for (m1, h1), (m2, h2) in zip(pts, pts[1:]):
            if m2 - m1 < 1.0:
                # Hai mốc m gần như trùng nhau nhưng h khác -> khác nhau ở CÁCH XẾP
                # HẠNG doc, không phải ở số doc. Chia cho delta m bé xíu ra biên vô
                # nghĩa (78%/doc), nên báo riêng thay vì nhét vào đường cong.
                print(f"  m {m1:5.1f} ~ {m2:5.1f}:  h {h1:.3f} vs {h2:.3f}  "
                      f"cùng m, lệch {(h2-h1)/h1*100:+.1f}% -> khác ở cách xếp hạng doc")
                continue
            marg = (h2 - h1) / (m2 - m1)
            thr = h2 / (4 * g + m2)
            b = math.log(h2 / h1) / math.log(m2 / m1)
            verdict = "đáng thêm" if marg > thr else "KHÔNG đáng"
            print(f"  m {m1:5.1f} -> {m2:5.1f}:  h {h1:.3f} -> {h2:.3f}  "
                  f"biên {marg*100:5.2f}%/doc  ngưỡng {thr*100:4.2f}%  b={b:+.3f}  {verdict}")

    best = max(rows, key=lambda r: r["final"])
    print(f"\ntốt nhất: {best['name']} = {best['final']:.4f}  "
          f"(F2_doc {best['F2d']:.4f} + F2_chunk {best['F2c']:.4f})")
    hc_room = best["hd"] - best["hc"]
    print(f"dư địa tầng chunk: h_c/h_d = {best['hc']/best['hd']:.2f}; "
          f"lên 1.00 thì h_c +{hc_room:.2f} -> F2_chunk ~ "
          f"{5*best['hd']/(4*best['gc']+best['mc']):.4f}")

    if args.predict:
        print(f"\nquét m_doc, giữ F2_chunk = {args.predict:.4f} "
              f"(dạng đóng, lệch ~12% nên chỉ dùng để SO SÁNH tương đối):")
        hd = {m: h for m, h in pts}
        for m in sorted(set(list(hd) + [10, 15, 20, 30, 35, 40, 50])):
            h = hd.get(m)
            if h is None:  # nội suy log-log giữa hai mốc gần nhất
                lo = max((p for p in pts if p[0] <= m), default=pts[0])
                hi = min((p for p in pts if p[0] >= m), default=pts[-1])
                if lo[0] == hi[0]:
                    h = lo[1]
                else:
                    b = math.log(hi[1] / lo[1]) / math.log(hi[0] / lo[0])
                    h = lo[1] * (m / lo[0]) ** b
            f2d = 5 * h / (4 * g + m)
            mark = "  <- đã đo" if m in hd else ""
            print(f"   m_doc={m:5.1f}  h={h:5.3f}  F2_doc={f2d:.4f}  "
                  f"FINAL~{(f2d+args.predict)/2:.4f}{mark}")


if __name__ == "__main__":
    main()
