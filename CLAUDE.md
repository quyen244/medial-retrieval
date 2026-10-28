# ViBioMIR - ràng buộc của cuộc thi

Những điều dưới đây là sự thật về bài toán, không suy ra được từ code.
Sai một trong số này là hỏng cả hướng làm, nên đọc trước khi đụng vào pipeline hay định dạng nộp bài.

## Cách chấm điểm

`FINAL = (DOCS_F2 + CHUNKS_F2) / 2`, cả hai tầng đều macro-F2 với `F2 = 5PR/(4P+R)`.

Dạng đóng dùng để suy ngược từ leaderboard: `F2 = 5h/(4g+m)` với `h` = số trúng, `g` = số gold, `m` = số nộp.
Mỗi bài nộp vì thế là một phép đo `h` và `g`, không chỉ là một điểm số.

### Mẫu số của LCS là CHUNK NỘP (cập nhật của BTC, 2026-10-23)

```
LCS_tokens(submitted, reference) / len(submitted_tokens) >= 0.40
```

**Mẫu số là độ dài chunk ta nộp, KHÔNG phải độ dài chunk tham chiếu.**
BTC đã đổi; tài liệu cũ và README trong repo còn ghi theo mẫu số tham chiếu.

Suy diễn từ công thức này nói "nộp dài bị phạt". **Đo thật nói ngược lại.**
Cùng một doc list, chunk mean 3.463 ký tự cho `F2_chunk = 0.0945`, còn chunk 877 ký tự chỉ cho `0.0317`.
Reference của BTC đủ dài để cửa sổ ~3.500 ký tự vẫn qua ngưỡng, trong khi chunk ngắn thường lệch khỏi ranh giới reference.
Yếu tố quyết định là CĂN ĐÚNG VÙNG, không phải độ dài tuyệt đối.

Bài học: với metric này, suy luận từ công thức đã sai hai lần. Chỉ leaderboard mới kết luận được.

## Kết quả đã đo (2026-10-23, luật mới)

Bảng chi tiết nằm ở `res/<tên>.zip`, trong đó có `scores.txt` với đủ P và R hai tầng.
Luôn tính `h = P x m`, ĐỪNG suy `h` từ `F2 = 5h/(4g+m)`: dạng đóng lệch khoảng 12% vì macro-F2 không bằng F2 của macro-P/R, và sai số đó đủ để đảo ngược kết luận.

| Bài | FINAL | m_doc | h_doc | F2_doc | m_chunk | h_chunk | F2_chunk | h_c/h_d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_ft (chunk 877) | 0.0805 | 35,1 | 7,01 | **0.1293** | 40,0 | 2,00 | 0.0317 | 0,29 |
| E1_docs44 | 0.0785 | 42,7 | 7,06 | 0.1253 | 40,0 | 1,89 | 0.0316 | 0,27 |
| E2_win2500 | 0.1090 | 42,7 | 7,06 | 0.1253 | 42,7 | 5,39 | 0.0926 | 0,76 |
| **E3_full8k** | **0.1099** | 42,7 | 7,06 | 0.1253 | 42,7 | **5,50** | **0.0944** | **0,78** |
| E4_docs25 | 0.1071 | 25,0 | 6,09 | 0.1222 | 25,0 | 4,76 | 0.0920 | 0,78 |
| N3_d35_c44 | 0.1112 | 35,0 | 6,93 | 0.1280 | 42,7 | 5,50 | 0.0944 | 0,79 |
| chunk_d35_c44_full8k_plain | 0.1117 | 35,0 | 6,93 | 0.1280 | 42,7 | 5,55 | 0.0955 | 0,80 |
| doc_d35_c44_top60_bge (2026-10-24) | 0.1894 | 35,0 | 11,63 | 0.1972 | 44,0 | 11,89 | 0.1815 | 1,02 |
| doc_d50_c55_top60_bge (2026-10-24) | 0.2090 | 50,0 | 14,90 | 0.2251 | 55,0 | **13,67** | **0.1930** | 0,92 |
| doc_d35_c44_top300_bge_vcorr_enzh (nộp nhầm zip mặc định, 2026-10-25) | 0.1836 | 35,0 | 10,97 | 0.1914 | 44,0 | 11,13 | 0.1758 | 1,01 |
| **doc_d80_c55_top300_bge** (2026-10-25) | **0.2230** | 80,0 | **18,95** | **0.2574** | 55,0 | 12,60 | 0.1887 | 0,66 |
| doc_d80_c55_top300_bge_vcorr_enzh (2026-10-25) | 0.2216 | 80,0 | 18,96 | 0.2520 | 55,0 | 12,99 | 0.1913 | 0,69 |
| doc_d80_c55_top300_bge_vcorr_enzh_asp (2026-10-25) | 0.2013 | 80,0 | 17,15 | 0.2276 | 55,0 | 11,98 | 0.1749 | 0,70 |
| **doc_d110_c55_top300_bge** (2026-10-25) | **0.2316** | 110,0 | **22,83** | **0.2745** | 55,0 | 12,60 | 0.1887 | 0,55 |

Bài cuối là bài đầu tiên tìm trên index cấp document (`medical_docs`, 1.003.391 điểm,
title + description + heading + phần đầu bài ≤1.500 ký tự), lấy 60 doc rồi rerank.
Nó đổi HAI thứ so với bài trên: biểu diễn doc và reranker (`BAAI/bge-reranker-v2-m3`
thay `models/medical-reranker`). Chưa tách được phần nào do cái nào.
Chi tiết: `docs/phat-hien-tang-diem.md` mục 13.

`g_doc` ổn định quanh 50-51, `g_chunk` quanh 53-54.

### Đường cong h_doc và điểm tối ưu

```
m=25.0  -> h=6.093
m=35.1  -> h=7.013     biên 9.11% mỗi doc (hạng 26-35)
m=42.7  -> h=7.058     biên 0.60% mỗi doc (hạng 36-43)
```

Ngưỡng đáng thêm một doc là `h/(4g+m)` = 2.9%.
Hạng 26-35 vượt xa ngưỡng, hạng 36-43 nằm dưới hẳn, nên **đỉnh F2_doc ở m ≈ 35**.
Không nới lên 43 (E1 tụt), cũng không cắt xuống 25 (E4 tụt).

Tầng chunk ngược chiều: biên 4.21% mỗi chunk so với ngưỡng 2.15%, nên giữ đủ 44.
Hai chiều trái nhau chính là chỗ tách `relevant_docs` khỏi `relevant_chunks` sinh lợi.

Con số `b = 0.3` và `m* = 88 doc` trong `docs/chan-doan-he-retrieval.md` là ước lượng ĐÃ BỊ PHỦ ĐỊNH.

### Cảnh báo diễn giải

`h_doc` bão hoà ở hạng 36-43 KHÔNG chứng minh phễu 6.000 doc là vô ích.
Pool hiện tại chỉ có 44 doc trong khi gold có 51, tức khoảng 44 gold doc CHƯA TỪNG được chấm lần nào.
Chúng nằm ngoài pool chứ không phải bị xếp hạng thấp. Đó là câu hỏi khác, chỉ Phase 1 trả lời được.

## Giới hạn nộp bài

- File nộp phải **dưới 100 MB**.
- ZIP chứa đúng một file `.json`, không nằm trong thư mục con.
- Đủ cả 1.200 id trong mọi bài nộp; thiếu một id là loại cả bài.
- `doc_id` kiểu số nguyên; `chunk_text` phải trích nguyên văn từ tài liệu gốc.

## Quy mô ground truth (suy từ leaderboard D_ft, luật cũ)

| Tầng | Nộp m | Trúng h | Gold g |
|---|---:|---:|---:|
| Document | 35,1 | 7,0 | **51,3** |
| Chunk | 40,0 | 2,0 | **54,5** |

Gold có khoảng 51 document mỗi câu, tức khoảng một chunk gold cho mỗi doc gold.
Đây là bài toán vét recall trên một cụm tài liệu cùng chủ đề, không phải bài toán precision.

Candidate pool hiện tại là 65 ứng viên mỗi câu, **nhỏ hơn gold set**.
Không thể tìm 51 tài liệu đúng khi chỉ nhìn 65 ứng viên, nên mọi thí nghiệm rerank hay augment bên trong pool đó đều cho delta gần bằng 0.

## Dữ liệu tại chỗ

- `data/raw_documents/<domain>/<doc_id>.json` chứa markdown gốc đã crawl, trường `content.markdown`.
- `data/raw_documents_index.json` ánh xạ doc_id sang domain, sinh bằng `scripts/phase0_assemble.py --build-index`.
- 1.003.740 tài liệu; 100% doc_id trong dump top-50 có bản gốc.
- Tài liệu được retrieve có median 5.250 ký tự, p90 8.354, dài hơn median corpus (2.257).
- `data/labeling/dev_set.jsonl` CHƯA được gán nhãn (`human_score` toàn None), không dùng để đo được.
- Không có ground truth tại chỗ. Leaderboard là công cụ đo duy nhất.

## Câu hỏi dùng cho truy xuất và rerank (quyết định 2026-10-27)

Từ sau T12, mọi thí nghiệm mới dùng `corrected_query` cho CẢ truy xuất LẪN rerank, thay cho câu gốc.
Nguồn: `data/spelling/corrected_queries.jsonl`, trường `corrected_query`; 224/1.200 câu khác câu gốc.
Bài nộp và mọi pool đến hết T12 (kể cả T12) dùng câu gốc từ `data/queries/real_queries.jsonl`.

Hệ quả cần nhớ:

- KHÔNG dùng lại điểm rerank cache của các pool cũ (chấm theo câu gốc): thí nghiệm dùng câu sửa chấm lại toàn bộ cặp từ đầu.
- Text doc đưa vào reranker phải là payload `text` của Qdrant `medical_docs`, không phải `data/doc_texts.jsonl` (khác dấu ` | `, điểm lệch tới 0,24).
- Đổi sang câu sửa là một biến số riêng: đo nó một mình so với SOTA trước khi gộp với thay đổi khác.
- Bằng chứng tới nay: câu sửa mới chỉ được dùng như MỘT trong các lượt tìm của pool B (gộp RRF), chưa từng được đo khi làm câu duy nhất cho cả tìm lẫn rerank.
- Pool hay bài nộp dùng câu sửa phải ghi rõ trong tên, ví dụ hậu tố `_qcorr`.

## Đặt tên bài nộp

Mọi mã viết tắt (`E1`, `N3`, `S2`, `D_ft`...) tra ở `docs/ky-hieu-bai-nop.md`.
Khi nhắc một mã lần đầu, LUÔN viết kèm nghĩa trong ngoặc: "N3 (35 doc + 43 cửa sổ, trần 8.000)".
Tên mới phải tự mô tả, chứa luôn tham số: `doc_d35_c44_full8k`, không dùng chữ cái tuỳ ý.

## Phát hiện đã đo

Sổ ghi đầy đủ các kết luận có số đo thật, kèm bằng chứng và phán quyết:
`docs/phat-hien-tang-diem.md`. Đọc trước khi đề xuất bất kỳ thay đổi nào về độ dài đoạn,
số lượng nộp, hay cách xếp hạng - phần lớn đã thử rồi.

## Công cụ

- `scripts/phase0_assemble.py` lắp lại bài nộp từ dump top-50, không cần GPU.
- `scripts/phase0_readout.py` suy ngược `h`, `g`, độ dốc `b` của `h(m)`, và `m*` từ một dòng leaderboard.

## Chạy script: smoke trước, tối ưu bằng số đo thật

- Mọi script nặng (embed, rerank, tìm trên toàn corpus...) phải chạy smoke trước: vài phút hoặc vài trăm mẫu (`--max-hours 0.1`, `--limit-*`).
- Từ smoke, đọc số đo thật (doc/s, cặp/s, VRAM, thời gian còn lại) rồi mới chỉnh tham số (batch, số GPU, cách chia part) và chạy thật.
- Tối ưu ở mức vừa đủ: chỉ đổi những gì số đo cho thấy là nút thắt rõ ràng, không thử nghiệm vòng vo hay ước lượng lý thuyết khi đã có số thật.
- Mọi script chạy lâu phải in log tiến độ định kỳ (khoảng mỗi phút): đã làm bao nhiêu, tốc độ, thời gian còn lại. Không được im lặng tới khi xong một khối lớn.
