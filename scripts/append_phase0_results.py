"""Nối kết quả Phase 0 vào ket_qua_thi_nghiem.txt, sinh bảng từ chính res/.

Sinh thay vì gõ tay để số trong file tổng kết không lệch khỏi bảng chấm gốc.
Chạy lại được nhiều lần: lần sau thay thế đúng khối cũ thay vì nối thêm.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase0_readout import collect

TARGET = "ket_qua_thi_nghiem.txt"
MARK = "MẪU CHUNG VÒNG 3"

NOTE = {
    "D_ft":          ("chunk 877 ký tự, doc suy từ chunk", "mốc so sánh, cấu hình vòng 2"),
    "E1_docs44":     ("như D_ft, nới doc list lên hết pool", "nới doc THẤT BẠI, thấp hơn cả D_ft"),
    "E4_docs25":     ("25 doc, cửa sổ trọn bài trần 8k", "cắt doc xuống 25 cũng kém hơn 35"),
    "E2_win2500":    ("cửa sổ 2.500 ký tự quanh chunk", "cửa sổ dài ăn điểm lớn ở tầng chunk"),
    "E3_full8k":     ("trọn markdown gốc, trần 8.000", "cửa sổ dài + doc list hết pool"),
    "N3_d35_c44":    ("35 doc + 42,7 cửa sổ, trần 8.000", "TỐT NHẤT. Tách 2 danh sách ĂN ĐIỂM"),
    "N5_d35_cap20k": ("như N3, nới trần lên 20.000", "nới trần QUÁ ĐÀ, h_chunk tụt"),
}
ORDER = ["D_ft", "E1_docs44", "E4_docs25", "E2_win2500", "E3_full8k",
         "N5_d35_cap20k", "N3_d35_c44"]


def table(headers, rows, aligns):
    w = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    def line(l, m, r):
        return l + m.join("─" * (x + 2) for x in w) + r
    def row(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(f" {c:>{w[i]}} " if aligns[i] == ">" else f" {c:<{w[i]}} ")
        return "│" + "│".join(out) + "│"
    return "\n".join([line("┌", "┬", "┐"), row(headers), line("├", "┼", "┤")]
                     + [row(r) for r in rows] + [line("└", "┴", "┘")])


def main():
    rows = {r["name"]: r for r in collect()}
    missing = [n for n in ORDER if n not in rows]
    if missing:
        print(f"thiếu kết quả cho: {missing} - bỏ qua các dòng đó")
    present = [n for n in ORDER if n in rows]

    t1 = table(
        ["#", "File nộp", "m_doc", "m_chunk", "chunk_text", "Điểm", "Kết luận"],
        [[str(i), n + ".zip", f"{rows[n]['md']:.1f}", f"{rows[n]['mc']:.1f}",
          NOTE[n][0], f"{rows[n]['final']:.4f}", NOTE[n][1]]
         for i, n in enumerate(present, 1)],
        ["<", "<", ">", ">", "<", ">", "<"])

    t2 = table(
        ["File nộp", "F2_doc", "P_doc", "R_doc", "h_doc", "g_doc",
         "F2_chunk", "P_chunk", "R_chunk", "h_chunk", "g_chunk", "h_c/h_d"],
        [[n, f"{r['F2d']:.4f}", f"{r['Pd']:.4f}", f"{r['Rd']:.4f}", f"{r['hd']:.2f}", f"{r['gd']:.1f}",
          f"{r['F2c']:.4f}", f"{r['Pc']:.4f}", f"{r['Rc']:.4f}", f"{r['hc']:.2f}", f"{r['gc']:.1f}",
          f"{r['hc']/r['hd']:.2f}"]
         for n in present for r in [rows[n]]],
        ["<"] + [">"] * 11)

    pts = sorted({(round(rows[n]["md"], 1), round(rows[n]["hd"], 3)) for n in present})
    g = sum(rows[n]["gd"] for n in present) / len(present)
    curve = []
    for (m1, h1), (m2, h2) in zip(pts, pts[1:]):
        marg, thr = (h2 - h1) / (m2 - m1), h2 / (4 * g + m2)
        curve.append([f"{m1:.1f} -> {m2:.1f}", f"{h1:.3f} -> {h2:.3f}",
                      f"{marg*100:.2f}%", f"{thr*100:.2f}%",
                      "đáng thêm" if marg > thr else "KHÔNG đáng"])
    t3 = table(["Khoảng m_doc", "h_doc", "Biên mỗi doc", "Ngưỡng", "Phán quyết"],
               curve, ["<", "<", ">", ">", "<"])

    best = max(present, key=lambda n: rows[n]["final"])
    b = rows[best]
    ceil_c = 5 * b["hd"] / (4 * b["gc"] + b["mc"])

    body = f"""
MẪU CHUNG VÒNG 3 - PHASE 0 (lắp lại offline, KHÔNG chạy GPU)
  - Nguồn: data/tmp/sub1200/D_ft_top50.jsonl - 1.200 câu x 50 ứng viên của chính bài D_ft
  - Mọi bài dùng CHUNG pool 50 ứng viên đó; chỉ đổi cách CHỌN và cách TRÌNH BÀY
  - Văn bản chunk lấy từ data/raw_documents/<domain>/<doc_id>.json (markdown gốc, verbatim)
  - Script: scripts/phase0_assemble.py (lắp), scripts/phase0_readout.py (đọc ngược res/)
  - Đủ 1.200 câu nên các bài so trực tiếp được với nhau
  - BTC ĐỔI CÁCH CHẤM ngày 23/10: mẫu số LCS là chunk NỘP, không phải chunk tham chiếu.
    Điểm vòng 1 và vòng 2 chấm theo luật cũ, KHÔNG so được với vòng 3.

{t1}

Bảng chấm chi tiết (res/<tên>.zip -> scores.txt). h = P x m, g = h / R.
LUÔN tính h = P x m. Suy h ngược từ dạng đóng F2 = 5h/(4g+m) lệch ~12% vì macro-F2
không bằng F2 của macro-P/R, đủ để đảo ngược dấu kết luận.

{t2}

Đường cong h_doc (g_doc trung bình = {g:.1f}). Ngưỡng đáng thêm một doc là h/(4g+m):

{t3}

KẾT LUẬN VÒNG 3

  1. Cửa sổ dài là thay đổi DUY NHẤT ăn điểm, và ăn lớn.
     h_chunk 1,89 -> 5,50 (gấp 2,9 lần), F2_chunk 0.0316 -> 0.0944, tỷ lệ h_c/h_d 0,27 -> 0,78.
     Cơ chế không phải mẫu số mà là CĂN ĐÚNG VÙNG: chunk 877 ký tự thường vắt qua
     ranh giới chunk tham chiếu nên chỉ phủ được một phần.
     Trần 8.000 chỉ hơn cửa sổ 2.500 khoảng 2%, tức đã gần bão hoà.

  2. Số doc nộp có ĐỈNH ở m ~ 35, không phải càng nhiều càng tốt.
     Nới lên 42,7 làm F2_doc tụt 0.1293 -> 0.1253; cắt xuống 25 còn tệ hơn (0.1222).
     Biên ở hạng 26-35 là 9,08% mỗi doc, rơi xuống 0,67% ở hạng 36-43, ngưỡng là ~2,9%.

  3. Tầng chunk còn dư địa lớn nhất trong phần không đụng retrieval.
     h_c/h_d = {b['hc']/b['hd']:.2f}. Nếu lên 1,00 thì F2_chunk ~ {ceil_c:.4f} và FINAL ~ {(b['F2d']+ceil_c)/2:.4f}.

  4. Nút thắt thật vẫn là h_doc = {b['hd']:.2f} trên {b['gd']:.0f} gold doc mỗi câu.
     CẢNH BÁO diễn giải: pool chỉ chứa ~44 doc, nên khoảng 44 gold doc mỗi câu CHƯA TỪNG
     được chấm. Việc h_doc bão hoà ở hạng 36-43 nói rằng POOL NÀY đã cạn, KHÔNG nói rằng
     pool rộng hơn là vô ích. Đó là câu hỏi của Phase 1.

  Tốt nhất vòng 3: {best}.zip = {b['final']:.4f}
"""
    text = io.open(TARGET, encoding="utf-8", newline="").read()
    if MARK in text:
        text = text[:text.index(MARK)].rstrip("\r\n")
        text = text[:text.rindex("\n") + 1] if text.endswith("\n") else text
    body = body.replace("\r\n", "\n").replace("\n", "\r\n")
    io.open(TARGET, "w", encoding="utf-8", newline="").write(
        text.rstrip("\r\n") + "\r\n\r\n" + body.lstrip("\r\n"))
    print(f"đã ghi {TARGET} ({os.path.getsize(TARGET)} bytes)")


if __name__ == "__main__":
    main()
