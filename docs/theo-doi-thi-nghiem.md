# Theo dõi thí nghiệm

Sổ các thí nghiệm ĐANG CHỜ đo.
Kết quả đã đo và phán quyết chuyển sang [phat-hien-tang-diem.md](phat-hien-tang-diem.md), rồi xoá dòng ở đây.
Nguồn chính để quản lý trạng thái: `docs/experiments.json` (hiển thị dạng cây: https://claude.ai/artifact/QSfyHf8JECHQ8mjbGmFwGP). File này giữ giải thích chi tiết.
Giải thích kỹ thuật và ví dụ số: báo cáo "Ghép danh sách xếp hạng" (artifact, 2026-10-25).

Quy tắc:
- Mỗi thí nghiệm đổi ĐÚNG MỘT biến so với một bài đối chứng đã có điểm thật.
- So tầng bị đổi (`DOCS_F2` hoặc `CHUNKS_F2`), không so FINAL, trừ khi tầng kia giống hệt đối chứng.
- Tầng không bị đổi phải ra ĐÚNG số của đối chứng. Lệch là bài dựng sai, bỏ kết quả.
- Cột kết quả để trống tới khi có `scores.txt`. Không điền số ước tính.
- File điểm tải về phải đổi tên theo bài ngay (xem mục 22 sổ phát hiện).

## Tổng quan theo yếu tố

Bài tốt nhất: FINAL 0.2562 = `fuse_d160_c100_top300_orig_ctop60+top300_cap2700` (mục 21), `DOCS_F2` 0.2849, `CHUNKS_F2` 0.2275.

Pipeline có năm chặng, mỗi thí nghiệm đổi một yếu tố ở đúng một chặng.
"Gộp" (fusion) không phải một yếu tố riêng: nó là cách ráp các nguồn lại ở từng chặng, ghi ở cột "Cách gộp".

```
câu hỏi ──> (A) chuỗi đem đi tìm ──> (B) truy xuất, ra pool ──> (C) rerank ──> (D) bằng chứng doc + chunk ──> (E) hình dạng bài nộp
```

### A. Chuỗi đem đi tìm (query dịch, aspects, HyDE)

| ID | Câu hỏi | Thay đổi duy nhất | Cách gộp | So với | Trạng thái |
|---|---|---|---|---|---|
| - | Thêm câu sửa chính tả + dịch en/zh | pool top300 -> pool B | RRF 4 danh sách | `DOCS_F2` 0.2574 | **xong: không lợi**, 0.2520 (mục 17) |
| - | Thêm aspects | pool B -> pool C | RRF | `DOCS_F2` 0.2520 | **xong: thua**, 0.2276 (mục 18) |
| T11 | Doc HyDE tìm thêm có trúng không | pool top300 -> hợp top300 + Bhyde | hợp theo doc_id, xếp theo điểm rerank | `DOCS_F2` 0.2849 | có code, chưa dựng |
| T10 | Điểm 0.2327 là của bài nào | tải lại file điểm | - | - | **xong**: mix0.3 d110; HyDE ra 0.2225 (mục 25) |

### B. Truy xuất (pool)

| ID | Câu hỏi | Thay đổi duy nhất | Cách gộp | So với | Trạng thái |
|---|---|---|---|---|---|
| T12 | Pool 300 có cắt mất doc tốt không | pool top300 -> top1000 | - | `DOCS_F2` 0.2849 | có lệnh, cần GPU |
| T13 | Hybrid trên index doc có tăng recall không | dense -> dense + sparse, cùng độ sâu với T12 | RRF dense/sparse trong Qdrant | `DOCS_F2` của T12 | chờ T12 |
| C0 | (lượt dữ liệu, không nộp) dùng index chunk 3,9M | tìm hybrid trên `medical_documents`, top 1.000 chunk/câu, rerank từng chunk bằng bge | - | - | chưa có code, cần GPU |
| T26a | Qwen3-Embedding-4B xếp pool top1000 của chính câu có đưa gold lên top300 nhiều hơn bge-m3 không | xếp bằng cosine q3e4b (câu sửa) trong pool top1000 riêng | - | `DOCS_RECALL` 0.489 (R@300) | **đã đo 2026-10-29**: R@300 0.5023 nhưng h 33,75 < 34,89, FINAL 0.2436 |
| T26b | Tìm trên hợp pool của mọi câu có lôi thêm gold không | own1000 -> union424k | - | `DOCS_RECALL` của T26a | **đã đo 2026-10-29**: R@300 0.4599 < 0.489 bge, FINAL 0.2313 |
| T26c | Trần recall trên 424k có vượt top1000 của bge không | như T26b, nộp 1.000 doc | - | `DOCS_RECALL` 0.5914 (R@1000) | **đã đo 2026-10-29**: R@1000 0.5744 < 0.5914, FINAL 0.1838 |
| T28a | Gộp RRF hạng tìm bge + q3e4b có phủ hơn từng cái không | xếp pool top1000 theo RRF w=0,5 | RRF k=60 | h 34,89 (bge) | **đã đo 2026-10-29**: h 36,48, R@300 0.5285, FINAL 0.2525 |
| T28b | Hạ điểm doc en/zh trong RRF có lợi không | T28a + en/zh x0,8 | RRF k=60 | h của T28a | **đã đo 2026-10-29**: h 36,57 (+0,09), hoà |
| T29 | Pool RRF (câu sửa, vi 90%) + bge rerank có thắng SOTA không | pool top300 bge -> top300 RRF w=0,6 trong top1000 bge câu sửa, en/zh x0,4, rerank bge câu sửa, d160 | RRF k=60 | `DOCS_F2` 0.2849 | **đã đo 2026-10-29 (B3)**: `DOCS_F2` 0.2866, h_doc 27,60 vs 27,84: hoà |
| T30 | Qwen3-Reranker-0.6B thay bge trên cùng pool T29 | bge-reranker -> Qwen3-Reranker-0.6B (vLLM, instruction y khoa đa ngữ) | - | `DOCS_F2` 0.2866 (T29) | đang chạy; nộp chung chunk SOTA đầu bài 2.700 (B4) |
| T27 | Pool q3e4b + bge rerank có thắng SOTA không | pool top300 bge-m3 -> pool top300 q3e4b, rerank câu sửa | - | `DOCS_F2` 0.2849 | chờ T26a/b, cần GPU |

### C. Rerank (model mới, query dịch để chấm, chuẩn hoá theo ngôn ngữ)

| ID | Câu hỏi | Thay đổi duy nhất | Cách gộp | So với | Trạng thái |
|---|---|---|---|---|---|
| T14 | Reranker mạnh hơn có xếp top 160 tốt hơn không | bge-reranker-v2-m3 -> Qwen3-Reranker-4B | - | `DOCS_F2` 0.2849 | có code, cần GPU |
| T05 | Qwen3-4B có xoá hình phạt ngôn ngữ không | đo trung vị điểm vi/en/zh | - | không nộp | chạy cùng T14 |
| T01 | Chấm doc bằng câu cùng ngôn ngữ | `orig` -> `matched` | thay điểm | `DOCS_F2` 0.2520 | **xong: thua**, 0.2051 (mục 26) |
| T02 | Như T01 nhưng gộp thứ hạng | `orig` -> `rrf(s_o, s_m)` | RRF hai bảng điểm | `DOCS_F2` 0.2520 | có lệnh |
| T03 | Chuẩn hoá theo ngôn ngữ, nhẹ | `orig` -> min-max [-1, 1] mỗi ngôn ngữ | chuẩn hoá rồi xếp chung | `DOCS_F2` 0.2520 | **xong: thua**, 0.2238 (mục 26) |
| T19 | mix0.3 có cộng vào bài SOTA không | `orig` -> `mix0.3` trên pool top300, d160 | trộn điểm | `DOCS_F2` 0.2849 | có lệnh, không cần GPU |
| T04 | Chuẩn hoá theo ngôn ngữ, mạnh | `orig` -> z-score logit mỗi ngôn ngữ | chuẩn hoá rồi xếp chung | `DOCS_F2` 0.2520 | **xong: thua**, 0.1631 (mục 26) |

Mục 26: trên pool B, mọi cách đẩy doc en/zh lên đều thua theo mức đẩy (mix0.3, mix0.5, matched), nên T02-T04 nhiều khả năng thua.
T01-T04 đo trên cấu hình cũ (pool B, d80). Thắng thì phải dựng lại trên cấu hình d160 mới biết có cộng vào 0.2562 không.

### D. Bằng chứng từ nhánh chunk cho tầng doc

| ID | Câu hỏi | Thay đổi duy nhất | Cách gộp | So với | Trạng thái |
|---|---|---|---|---|---|
| T06 | Doc nhánh chunk (dump cũ, top 50) có trúng không | doc list -> RRF(doc list, chunk list) | RRF k=60, nhận mọi doc nhánh chunk | `DOCS_F2` 0.2849 | **xong: thua**, 0.2743 (mục 23) |
| T17 | Đoạn tốt nhất trong thân bài có kéo doc gold vào top 160 không | điểm doc -> max(điểm doc, điểm chunk tốt nhất) trên pool hợp top300 + doc của C0 | MaxP, cùng một reranker | `DOCS_F2` 0.2849 | chờ C0 |

### E. Hình dạng bài nộp

| ID | Câu hỏi | Thay đổi duy nhất | Cách gộp | So với | Trạng thái |
|---|---|---|---|---|---|
| T08 | Đỉnh `m_doc` | 160 -> 220 | - | `DOCS_F2` 0.2849 | **xong: thua**, 0.2810 (mục 24) |
| T09a | Thêm chunk ngắn hơn theo luật kích thước | c100 trần 2.700 -> c120 trần 2.250, vẫn đầu bài | - | `CHUNKS_F2` 0.2275 | có lệnh |
| T15 | Nén zip mạnh hơn có nhét thêm chunk không | `compresslevel` 6 -> 9 | - | đo dung lượng offline trước | chưa có code |
| T18a | Cửa sổ đặt ở đoạn liên quan nhất | cửa sổ đầu bài -> cửa sổ lai ~2.100 ký tự sạch điểm bge cao nhất trong doc | bước 630 (overlap 70%) | `CHUNKS_F2` 0.2275 | **đã đo 2026-10-29 (B3)**: 0.2079, h_chunk 17,53 vs 19,16: thua, NHƯNG chunk ngắn hơn SOTA 17% (mean 1.781 vs 2.158) nên chưa kết luận về vị trí |
| T18b | Chọn doc cho chunk list theo bằng chứng đoạn | 100 doc đầu chunk list -> 100/200 doc điểm cửa sổ cao nhất | - | `CHUNKS_F2` của bài thắng T18a | không nộp: B4 đổi sang chunk SOTA để đo riêng T30 |
| T16 | Chunk list theo thứ tự doc của bài thắng T11/T12/T14 | `--chunk-pool` -> pool thắng | - | `CHUNKS_F2` 0.2275 | chờ T11/T12/T14 |

T07 (cửa sổ căn theo dump chunk cũ) được thay bằng T18a, vì C0 cho bằng chứng tươi trên 1.000 chunk thay vì 50.
T09 cũ (c120, trần 2.000, cửa sổ giữa bài) đổi ba biến cùng lúc, thay bằng T09a.

### Ghép hai thí nghiệm vào một bài nộp

Hai tầng chấm độc lập và `scores.txt` ghi riêng `DOCS_F2`, `CHUNKS_F2`, nên một bài nộp đo được hai thí nghiệm:
một thí nghiệm CHỈ đổi tầng doc ghép với một thí nghiệm CHỈ đổi tầng chunk, đọc mỗi tầng so với đối chứng của riêng nó.
Không ghép hai thí nghiệm cùng tầng. Cái giá: mất phép kiểm "tầng không đổi phải ra đúng số đối chứng", nên phải kiểm file nộp trước khi nộp.
Danh sách bài ghép nằm ở khoá `batches` trong `docs/experiments.json`; node thuộc bài ghép có trường `batch`.

| Bài | Ghép | Đọc kết quả | Kiểm trước khi nộp |
|---|---|---|---|
| B1 | T19 (doc: mix0.3) + T09a (chunk: c120 trần 2.250) | `DOCS_F2` vs 0.2849 là T19; `CHUNKS_F2` vs 0.2275 là T09a | 160 doc và 120 chunk mỗi câu, zip dưới 100 MB |
| B2 | T20 (doc: tiếng Việt trước) + T21 (chunk: tiếng Việt trước) | `DOCS_F2` vs 0.2849 là T20; `CHUNKS_F2` vs 0.2275 là T21 | 160 doc và 100 chunk mỗi câu, zip dưới 100 MB |

Lệnh B2: `python -m scripts.fuse_submission --doc-pool top300 --methods vifirst --m-doc 160 --m-chunk 100 --cap 2700 --chunk-order vifirst --out-dir data --chunk-pool top60 top300`

Ba thí nghiệm không cần GPU thêm ngày 2026-10-26 (chi tiết và số offline trong `docs/experiments.json`):

| ID | Câu hỏi | Thay đổi duy nhất | So với | Offline | Kết quả (2026-10-27, mục 27) |
|---|---|---|---|---|---|
| T20 | Hạ doc en/zh ở tầng doc có lợi không (ngược T03/T04) | `orig` -> `vifirst` | `DOCS_F2` 0.2849 | thay 16,8/160 doc, top 160 thành 91% vi | **thua**: 0.2813 (−0,0036) |
| T21 | Như T20 ở tầng chunk | chunk list: doc vi lên đầu | `CHUNKS_F2` 0.2275 | 17,5/100 chunk hiện là doc en/zh | **thua**: 0.2262 (−0,0013) |
| T22 | Gộp HyDE nhưng chỉ nhận doc vi (bản an toàn của T11) | pool top300 -> hợp top300 + doc vi của Bhyde | `DOCS_F2` 0.2849 | 10,0 doc mới/câu lọt top 160 | **hoà**: 0.2850 (+0,0001), FINAL 0.2562 |

Lệnh T22: `python -m scripts.fuse_submission --doc-pool top300 Bhyde --extra-lang vi --methods orig --m-doc 160 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300`

Lệnh B1: `python -m scripts.fuse_submission --doc-pool top300 --methods mix0.3 --m-doc 160 --m-chunk 120 --cap 2250 --out-dir data --chunk-pool top60 top300`
File: `data/fuse_d160_c120_top300_mix0.3_ctop60+top300_cap2250.zip`.

### Thứ tự nộp đề xuất

1. T12, T14, T11 - cùng nhắm đổi doc bên trong top 160 (mục 24: thêm đuôi đã hết lợi).
2. B1 (T19 + T09a) - một bài đo hai thí nghiệm, dựng được ngay không cần GPU.
3. C0, rồi T17 và T18a. Trước khi nộp, đo offline: số doc chỉ C0 thấy mà điểm vượt doc hạng 160, và tỉ lệ chunk tốt nhất nằm ngoài 2.700 ký tự đầu. Hai số gần 0 thì bỏ, không tốn lượt.
4. T03, T04, T01, T02 khi còn dư lượt.

## Nhóm 1 - doc tiếng Anh và Trung bị reranker chấm thấp oan

Hiện tượng gốc đo trên pool B (360.000 cặp câu-doc), điểm `bge-reranker-v2-m3`:

```
vi  median 0.2258   p90 0.8912   p99 0.9924
en  median 0.0093   p90 0.2165   p99 0.7112
zh  median 0.0112   p90 0.1964   p99 0.7871
```

Pool B có 63% doc en/zh, top 80 sau rerank chỉ còn 24% (mục 16-17).
Gold có bao nhiêu doc en/zh thì CHƯA BIẾT; nếu gold chủ yếu là vi thì mọi thí nghiệm nhóm này sẽ thua.

Cả nhóm dùng chung đối chứng `doc_d80_c55_top300_bge_vcorr_enzh` (pool B, `orig`, 80/55): `DOCS_F2` 0.2520, `h_doc` 18,96.
Các bài nhóm này lấy chunk từ pool top60, khác chunk của đối chứng.
Vì vậy KHÔNG so FINAL với 0.2216; so `DOCS_F2` với 0.2520.
Kiểm tra bài dựng đúng: `CHUNKS_F2` phải ra đúng 0.1930 (chunk list của mục 14).

### T01 - cách A (`matched`)

- Giả thuyết: chấm mỗi doc bằng câu cùng ngôn ngữ (vi <- câu sửa chính tả, en <- câu dịch Anh, zh <- câu dịch Trung) xếp hạng tốt hơn chấm mọi doc bằng câu gốc.
- File: `data/tmp/phase1/fuse_d80_c55_B_matched_ctop60.zip`, đã dựng, chưa nộp.
- Chẩn đoán: thay 28,3/80 doc; top 80 thành vi 50%, en 19%, zh 31%.
- Kết quả: `DOCS_F2` = ___ , `CHUNKS_F2` = ___ (phải là 0.1930)
- Phán quyết: ___

### T02 - `rrf` của hai bảng điểm

- Giả thuyết: tín hiệu của cách A có thật nhưng hai bảng điểm không cùng thang; gộp theo thứ hạng tránh được lỗi đó.
- Lệnh: `python -m scripts.fuse_submission --methods rrf --doc-pool B --m-doc 80 --m-chunk 55 --out-dir data --chunk-pool top60`
- Chẩn đoán: thay 14,9/80 doc; top 80 thành vi 63%, en 15%, zh 22%.
- Đọc cùng T01: cả hai thắng thì chắc; chỉ `matched` thắng thì đáng ngờ (thắng nhờ độ lớn điểm).
- Kết quả: `DOCS_F2` = ___
- Phán quyết: ___

### T03, T04 - chuẩn hoá điểm theo từng ngôn ngữ

- Giả thuyết: trong mỗi câu, chuẩn hoá điểm riêng trong nhóm vi, en, zh rồi mới xếp chung thì bỏ được độ lệch thang giữa các ngôn ngữ.
- Cơ sở: Lawrie et al., ECIR 2023 bác bỏ thống kê giả thuyết điểm doc liên quan cùng phân phối giữa các ngôn ngữ.
- Code: `--methods langmm` (T03) và `--methods langz` (T04) trong `scripts/fuse_submission.py`, hàm `per_language()`.
  Nhóm không chuẩn hoá được (một doc, hoặc mọi doc bằng điểm) nhận 0, tức giữa thang.
- File: `data/fuse_d80_c55_B_langmm_ctop60.zip` (T03, 85,6 MB) và `data/fuse_d80_c55_B_langz_ctop60.zip` (T04, 85,7 MB).
  Đã kiểm offline: `relevant_chunks` trùng 1200/1200 câu với bài T01 (cùng chunk top60), nên `CHUNKS_F2` phải ra 0.1930.
- Chẩn đoán trên pool B, top 80 (tính offline 2026-10-25):

```
cách            trùng orig   vi    en    zh
orig               80,0      76%   12%   12%
T03 min-max        65,8      60%   22%   18%
T04 z(logit)       47,6      37%   33%   30%     <- trùng tỉ lệ pool 36,8 / 33,0 / 30,1
```

- Rủi ro đã thấy trên dữ liệu: T04 biến top 80 thành đúng tỉ lệ ngôn ngữ của pool, tức hạn ngạch theo ngôn ngữ.
  Ở câu 42 nó đẩy bài en "Perforated appendicitis during near-term pregnancy" (điểm gốc 0,39) lên hạng 1, trên bài vi đúng chủ đề điểm 0,90.
- Kết quả T03: `DOCS_F2` = ___
- Kết quả T04: `DOCS_F2` = ___
- Phán quyết: ___

### T05 - chẩn đoán Qwen3-Reranker-4B (không tốn lượt nộp)

- Giả thuyết: hình phạt ngôn ngữ là do reranker yếu khi chấm chéo ngôn ngữ, model mạnh hơn sẽ xoá phần lớn.
- Cơ sở: báo cáo Qwen3 Embedding (arXiv 2506.05176, chưa bình duyệt) cho MMTEB-R 72,74 so với 58,36 của bge-reranker-v2-m3.
- Cách làm: chấm lại pool B bằng Qwen3-Reranker-4B với câu gốc, đo lại trung vị điểm theo ngôn ngữ.
- Cách đọc:
  - khoảng cách vi / en,zh co lại rõ -> vấn đề nằm ở model, dùng Qwen3-4B thay thẳng, nộp như T01;
  - khoảng cách vẫn còn -> vấn đề là thang điểm, ghép Qwen3-4B với T03 hoặc T04.
- Kết quả: trung vị vi = ___ , en = ___ , zh = ___
- Phán quyết: ___

### T14 - đổi reranker sang Qwen3-Reranker-4B trên bài tốt nhất

- Giả thuyết: reranker mạnh hơn đổi được doc nằm TRONG top 160 (mục 24: thêm đuôi đã hết lợi).
- Khác T05: T05 chỉ đo phân phối điểm theo ngôn ngữ offline; T14 là bài nộp.
- Cách làm: chấm lại pool top300 (300 doc/câu, câu gốc) bằng Qwen3-Reranker-4B, xếp lại, nộp 160 doc đầu.
- Biến duy nhất: reranker. Giữ pool top300, `m_doc` 160, chunk list mục 21. Kiểm tra: `CHUNKS_F2` phải ra 0.2275.
- Code: lớp `Qwen3Reranker` trong `src/rag/pipeline/components/reranking.py` (LM nhân quả, điểm = P(yes) / (P(yes) + P(no)), template theo model card),
  `make_reranker()` chọn lớp theo tên model; `scripts/rerank_pool.py` chấm lại một pool, giữ nguyên tập doc.
- Lệnh chấm (GPU, cần ~9 GB VRAM ở bf16): `python -m scripts.rerank_pool` (mặc định pool top300 -> `data/tmp/phase1/doc_top300_qwen3r4b_pool.jsonl`). Chạy dở thì chạy lại cùng lệnh.
- Cuối lượt script in trung vị điểm theo ngôn ngữ, tức kết quả T05.
- Dựng bài: `python -m scripts.fuse_submission --doc-pool top300q4b --methods orig --m-doc 160 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300`
- Chunk list lấy từ pool top60 và top300 (điểm bge) nên vẫn đúng mục 21.
- Kết quả: `DOCS_F2` = ___
- Phán quyết: ___

## Nhóm 2 - dùng index chunk 3,9M (`medical_documents`)

Hiện trạng: từ khi chuyển sang index doc, cả hai tầng lấy từ cùng một danh sách doc đã rerank; tầng chunk là 2.700 ký tự đầu bài.
Index chunk chưa được dùng thật: T06 chỉ dùng dump Phase 0 (`data/tmp/sub1200/D_ft_top50.jsonl`, top 50 chunk, chấm bằng `models/medical-reranker` đã loại),
còn `scripts/phase1_deep_retrieve.py` chưa từng chạy và cũng dùng reranker đã loại.

Vì sao đáng dùng: index doc chỉ thấy 1.500 ký tự đầu mỗi bài; index chunk phủ cả thân bài (~3,9 chunk mỗi doc, 560-1.000 ký tự mỗi chunk, có dense và sparse).
Doc gold có phần liên quan nằm sâu thì index doc không thấy, và cửa sổ đầu bài trượt ngưỡng LCS.
Cơ sở: Callan, SIGIR 1994 (bằng chứng cấp đoạn cho xếp hạng doc); Dai & Callan, SIGIR 2019 (MaxP: điểm doc = điểm đoạn tốt nhất).

### C0 - lượt dữ liệu: chunk pool

- Tìm hybrid dense + sparse trên `medical_documents` bằng câu gốc, top 1.000 chunk/câu, rerank từng chunk bằng bge-reranker-v2-m3.
- Ra một chunk pool: mỗi chunk có `doc_id`, điểm rerank, và vị trí của chunk trong markdown gốc (để căn cửa sổ).
- Chẩn đoán offline trước khi nộp gì:
  - số doc chỉ C0 thấy mà điểm chunk tốt nhất vượt điểm doc hạng 160;
  - tỉ lệ doc trong chunk list có chunk tốt nhất nằm ngoài 2.700 ký tự đầu (trần dư địa của T18a).
  Hai số gần 0 thì bỏ T17/T18, không tốn lượt.
- Code: chưa có.

### T17 - MaxP cho tầng doc

- Giả thuyết: doc có đoạn trong thân bài khớp mạnh là gold mà index doc bỏ sót hoặc chấm thấp vì chỉ thấy phần đầu.
- Cách xếp: pool hợp top300 + doc của C0; điểm doc = max(điểm rerank cấp doc, điểm rerank chunk tốt nhất); nộp 160 doc đầu.
- Khác T06: doc mới phải thắng bằng điểm của cùng reranker, không được nhận chỉ vì nhánh chunk có thấy; bằng chứng chấm bằng bge, 1.000 chunk thay vì 50.
- Biến duy nhất: điểm doc. Chunk list mục 21. Kiểm tra: `CHUNKS_F2` phải ra 0.2275.
- Kết quả: `DOCS_F2` = ___
- Phán quyết: ___

### T18a - cửa sổ lai tốt nhất trong doc (`scripts/chunk_windows.py`, file ở `data/T18_chunk_window/`)

- Giả thuyết: cửa sổ đặt ở đoạn liên quan nhất của doc qua ngưỡng LCS nhiều hơn cửa sổ đầu bài.
- Biến duy nhất: vị trí cửa sổ; giữ 100 doc của chunk list mục 21 (top60 + top300). Không cần C0: chấm thẳng .md gốc.
- Cách cắt: tiền xử lý CẢ bài (strip_markdown + submission_chunk_text) rồi mới cắt. Cửa sổ lai ~2.100 ký tự sạch
  (≈ 2.078 ký tự sạch thực nộp của SOTA cắt 2.700 thô, giữ dung lượng zip), bước 630 = overlap 70%,
  đầu lùi về đầu dòng, cuối lùi về hết câu. Chấm bằng bge-reranker-v2-m3, câu sửa, max_length 1024.
- Đo trên mẫu 3.000 doc: 3,93 cửa sổ/doc, độ dài median 2.045 (p10 1.911), 100% cửa sổ kết thúc ở hết câu/dòng, phủ 100% bài.
- Nộp chung T29 ở phía doc (B3). Đối chứng: `CHUNKS_F2` 0.2275.
- **Kết quả (2026-10-29)**: FINAL 0.2473, `CHUNKS_F2` 0.2079 (P 0,1753, R 0,2597), h_chunk 17,53 vs 19,16. 40,1/100 chunk dời khỏi đầu bài.
  Nhiễu: chunk SOTA thực đo mean 2.158 / median 2.637 ký tự sạch (zip 98,9 MB), không phải ~2.078; cửa sổ T18a mean 1.781 / median 2.017 (zip 84,2 MB).
  Thua có thể do vị trí, do ngắn hơn 17%, hoặc cả hai. Đo lại cho công bằng: cap ~2.700 sạch, ~1M cặp bge (~5 giờ GPU).
- Kết quả: `CHUNKS_F2` = ___
- Phán quyết: ___

### T18b - chunk list chọn doc theo bằng chứng đoạn

- Chấm 200 doc đầu chunk list, chọn 100 doc có điểm cửa sổ tốt nhất cao nhất (MaxP thuần); cửa sổ như T18a.
- Dùng chung file cửa sổ `bge_n200_cap2100_st630_windows.jsonl` với T18a; nộp chung T29 (B4), `DOCS_F2` phải trùng B3.
- Đối chứng: `CHUNKS_F2` của bài thắng T18a (hoặc 0.2275 nếu T18a thua).
- Kết quả: `CHUNKS_F2` = ___
- Phán quyết: ___

## Nhóm 3 - hình dạng bài nộp

Ở trần 100 MB, số chunk và trần ký tự gắn với nhau qua luật kích thước (`m_chunk` x ký tự TB ≈ 215.000, mục 21), nên chúng là MỘT núm vặn.

### T09a - c120, trần 2.250, cửa sổ đầu bài

- Giả thuyết: biên 12,2% mỗi chunk ở hạng 56-100 (mục 21) còn đủ cao ở hạng 101-120, và cửa sổ 2.250 ký tự chưa trượt ngưỡng LCS nhiều.
- Rủi ro: cửa sổ càng ngắn càng dễ trượt (chunk 877 ký tự từng ra 0.0317), nên chỉ bước nhỏ.
- Lệnh: `python -m scripts.fuse_submission --doc-pool top300 --methods orig --m-doc 160 --m-chunk 120 --cap 2250 --out-dir data --chunk-pool top60 top300`
  Vượt 100 MB thì hạ `--cap` 2.200.
- Đối chứng: `CHUNKS_F2` 0.2275. Kiểm tra: `DOCS_F2` phải ra đúng 0.2849.
- Kết quả: `CHUNKS_F2` = ___
- Phán quyết: ___

### T15 - nén zip mức 9

- `zip_submission()` dùng `ZIP_DEFLATED` mức mặc định (6). Mức 9 cho zip nhỏ hơn với cùng nội dung, phần dư đổi thành thêm chunk.
- Đo dung lượng offline trước; chênh không đáng kể thì bỏ, không tốn lượt.
- Code: chưa có.

### T16 - chunk list theo bài doc thắng

- Sau khi T11, T12 hoặc T14 thắng, lấy chunk list theo thứ tự doc của pool thắng.
- Phải đo riêng: ở cùng 55 chunk, pool top300 từng ra ít trúng hơn pool top60 (12,60 so với 13,67, mục 15).

## Nhóm 4 - chiều sâu và độ phủ của pool doc

Sau T08 (mục 24 sổ phát hiện): nới `m_doc` 160 -> 220 làm tụt `DOCS_F2`, nên thêm doc vào ĐUÔI danh sách không còn lợi.
Đòn bẩy còn lại là đổi doc NÀO nằm trong top 160, tức pool phải chứa nhiều doc tốt hơn.

Bằng chứng pool 300 bị cắt (đo offline trên `doc_top300_bge_pool.jsonl`, hạng retrieval suy từ `rrf_score` = 1/(60+hạng)):
số doc mỗi câu lọt top 160 sau rerank, chia theo hạng retrieval trước rerank:

```
hạng   1-50 : 42,3      151-200 : 22,0
      51-100 : 32,7      201-250 : 19,3
     101-150 : 26,7      251-300 : 17,0   <- vẫn 17 doc/câu ngay chỗ cắt
```

Đường giảm chậm, nên doc hạng >300 nhiều khả năng còn vào được top 160.
Doc đó có trúng gold không thì chưa biết, chỉ leaderboard trả lời.

Điểm rerank của cùng một doc giống hệt nhau ở mọi pool (218.663 cặp khớp, 0 lệch), vì rerank luôn dùng câu gốc.
Nên gộp nhiều pool rồi xếp theo điểm rerank là hợp lệ, không phải chấm lại.

### T26 - Qwen3-Embedding-4B trên 424k doc ưu tiên (chưa đủ 1M)

424.120 doc đầu của job embed là HỢP pool top1000 bge-m3 của cả 1.200 câu (`priority_ids` trong `scripts/kaggle_embed.py`).
Khi shard 0-106 có vector, `kaggle_embed search --pool data/tmp/phase1/doc_top1000_bge_pool.jsonl` cho hai cách tìm từ cùng một bộ vector:

- `--pool-mode own` (T26a): câu q chỉ thấy pool top1000 của chính q. Cùng tập ứng viên với bge, chỉ đổi cách xếp, trần 0.5914.
- `--pool-mode union` (T26b): câu q thấy cả 424k. Doc gold bge bỏ sót cho q có thể nằm trong pool của câu khác cùng chủ đề, nên trần >= 0.5914 và CHƯA đo được.

Chênh T26b - T26a là lượng gold lấy được từ pool của câu khác. Chênh lớn thì đáng chạy đủ 1M doc.
Kết quả trên 424k KHÔNG phải cận dưới của kết quả trên 1M: 580k doc còn lại vừa thêm gold vừa thêm nhiễu.
`search` in tỉ lệ doc của pool đã có vector; dưới 100% là chạy sớm, bỏ kết quả.

Cả T26a/b đổi hai thứ so với R300_top300: embedding và câu sửa (bge-m3 tìm bằng câu gốc).
Muốn tách thì cần R@300 của pool `qc_retr` (bge-m3 + câu sửa) làm đối chứng.

Lệnh đầy đủ nằm ở trường `command` của node T26a/b/c trong `docs/experiments.json`.

Kết quả T26b (union 424k, 2026-10-29): `DOCS_RECALL` 0.4599, `DOCS_P` 0.1014, `CHUNKS_F2` 0.2275 (khớp đối chứng).
h_doc = 0,1014 x 300 = 30,42 so với 34,89 của bge-m3 top300: q3e4b + câu sửa mất 4,47 gold/câu (-13%) dù được thấy cả 424k doc.

Kết quả T26a (own1000): `DOCS_RECALL` 0.5023 > 0.489 nhưng `DOCS_P` 0.1125, h_doc = 33,75 < 34,89.
Macro-R tăng vì q3e thắng ở câu ít gold, tổng số hit lại giảm; `DOCS_F2` 0.2598 < 0.2627. Luôn đọc h = P x m, không đọc riêng R.
Kết quả T26c (union, 1.000 doc): `DOCS_RECALL` 0.5744 < 0.5914, h_doc 40,3 vs 42,8.

Phán quyết T26: q3e4b (+ câu sửa) không phải retriever tốt hơn bge-m3 ở độ sâu nào đã đo; tìm rộng (union) còn thêm nhiễu (h 30,42 vs 33,75 của own).
Điều còn mở: q3e xếp KHÁC bge (thắng ở câu ít gold), nên gộp RRF bge + q3e trong pool top1000 có thể hơn từng cái.

### T28 - gộp RRF hạng tìm bge-m3 + q3e4b (`scripts/fuse_retrieval.py`, file ở `data/T28_rrf_bge_q3e/`)

Điểm = w/(60 + hạng_q3e) + (1-w)/(60 + hạng_bge) trong pool top1000 bge của chính câu; không rerank, nộp 300 doc.
Tên file chỉ ghi biến: `w0.5` = chia đều, `_en0.8_zh0.8` = điểm doc en/zh nhân 0,8.

| bài | R@300 | h_doc (P x 300) | FINAL |
|---|---:|---:|---:|
| R300_top300 (bge) | 0.489 | 34,89 | 0.2451 |
| T26a (q3e) | 0.5023 | 33,75 | 0.2436 |
| **T28a w0.5** | **0.5285** | **36,48** | 0.2525 |
| T28b w0.5_en0.8_zh0.8 | 0.5300 | 36,57 | 0.2529 |

Hai retriever chỉ trùng 159/300 doc, nên gộp phủ hơn cả hai (+1,59 gold/câu so với bge). Trọng số ngôn ngữ hoà.

T29: w=0,6, top300 RRF xếp lại bằng điểm bge-reranker CÓ SẴN của pool top1000 (câu gốc, trùng hẳn điểm pool SOTA:
359.865 cặp, lệch 0), nộp d160. Chỉ đổi pool so với SOTA; top160 trùng SOTA 123,6/160, vi 83%.
Lệch quy tắc câu sửa 2026-10-27 (cố ý, để giữ một biến). Kết quả: `DOCS_F2` = ___

### T11 - gộp pool top300 + HyDE

- Giả thuyết: doc HyDE tìm thêm mà reranker xếp cao hơn đuôi top 160 hiện tại thì trúng nhiều hơn đuôi đó.
- Khác T06: doc mới thắng bằng chính điểm rerank, không phải được nhận vì nhánh khác có thấy.
- Chẩn đoán offline: pool hợp 418 doc/câu; 19,7 doc mới lọt top 160 (vi 9,1 / zh 5,6 / en 5,0).
  Gộp cả top300 + B + C + HyDE: 559 doc/câu, 27,9 doc mới.
- Rủi ro: một nửa doc mới là en/zh, đúng nhóm mà pool B từng thua (0.2520 so với 0.2574 ở d80). Kết quả T03/T04 trả lời một phần rủi ro này.
- Biến duy nhất: pool doc. `m_doc` 160, chunk list mục 21. Kiểm tra: `CHUNKS_F2` phải ra 0.2275.
- Code: `--doc-pool` nhận nhiều pool, hợp theo `doc_id` (`merge_pools()`); doc lệch điểm giữa các pool thì dừng.
- Lệnh: `python -m scripts.fuse_submission --doc-pool top300 Bhyde --methods orig --m-doc 160 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300`
- File (sau khi chạy lệnh): `data/fuse_d160_c100_top300+Bhyde_orig_ctop60+top300_cap2700.zip`.
- Kết quả: `DOCS_F2` = ___
- Phán quyết: ___

### T12 - pool top1000

- Giả thuyết: bảng hạng retrieval ở trên; doc hạng 301-1000 lọt top 160 và trúng nhiều hơn doc chúng đẩy ra.
- Lệnh truy xuất + rerank (GPU), dùng lại điểm 300 doc đã chấm: `python scripts/phase1_doc_retrieve.py --name doc_top1000_bge --set retrieval.doc.top_docs=1000 --set "retrieval.doc.rerank_cache=[data/tmp/phase1/doc_top300_bge_pool.jsonl]"`
- Dựng bài: `python -m scripts.fuse_submission --doc-pool top1000 --methods orig --m-doc 160 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300`
- Biến duy nhất: độ sâu pool (cùng câu gốc, cùng reranker). Kiểm tra: `CHUNKS_F2` phải ra 0.2275.
- Kết quả: `DOCS_F2` = ___
- Phán quyết: ___

### T13 - tìm hybrid dense + sparse

- Giả thuyết: index `medical_docs` hiện chỉ tìm dense; BGE-M3 (Chen et al., Findings ACL 2024) cho dense + sparse cao hơn dense riêng trên MIRACL.
- Audit ngày nạp index ghi `medical_docs` có vector `sparse` (`docs/audit-dich-query-va-retrieval.md` mục 5.1); kiểm lại trên Qdrant trước khi chạy.
- So cùng độ sâu pool với T12 để chỉ đổi một biến.

## Việc dọn dẹp

### T10 - chốt danh tính điểm 0.2327

`res/fuse_d110_c55_top300_mix0.3_ctop300.zip` và `res/doc_top300_bge_vcorr_enzh_hyde.zip` trùng từng byte.
Tải lại file điểm của một trong hai bài để biết điểm 0.2327 là của bài nào (mục 22 sổ phát hiện).
