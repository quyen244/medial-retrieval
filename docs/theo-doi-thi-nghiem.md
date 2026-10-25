# Theo dõi thí nghiệm

Sổ các thí nghiệm ĐANG CHỜ đo.
Kết quả đã đo và phán quyết chuyển sang [phat-hien-tang-diem.md](phat-hien-tang-diem.md), rồi xoá dòng ở đây.
Giải thích kỹ thuật và ví dụ số: báo cáo "Ghép danh sách xếp hạng" (artifact, 2026-10-25).

Quy tắc:
- Mỗi thí nghiệm đổi ĐÚNG MỘT biến so với một bài đối chứng đã có điểm thật.
- So tầng bị đổi (`DOCS_F2` hoặc `CHUNKS_F2`), không so FINAL, trừ khi tầng kia giống hệt đối chứng.
- Tầng không bị đổi phải ra ĐÚNG số của đối chứng. Lệch là bài dựng sai, bỏ kết quả.
- Cột kết quả để trống tới khi có `scores.txt`. Không điền số ước tính.
- File điểm tải về phải đổi tên theo bài ngay (xem mục 22 sổ phát hiện).

## Tổng quan

| ID | Câu hỏi | Biến đổi | So với | Trạng thái |
|---|---|---|---|---|
| T01 | Chấm doc bằng câu cùng ngôn ngữ (cách A) có lợi không | `orig` -> `matched` | `DOCS_F2` 0.2520 | đã dựng, chờ nộp |
| T02 | Cách A gộp theo thứ hạng thay vì theo điểm | `orig` -> `rrf` (s_o, s_m) | `DOCS_F2` 0.2520 | có lệnh, chưa dựng |
| T03 | Chuẩn hoá điểm theo ngôn ngữ, mức nhẹ | `orig` -> min-max [-1, 1] mỗi ngôn ngữ | `DOCS_F2` 0.2520 | đã dựng, chờ nộp |
| T04 | Chuẩn hoá điểm theo ngôn ngữ, mức mạnh | `orig` -> z-score trên logit mỗi ngôn ngữ | `DOCS_F2` 0.2520 | đã dựng, chờ nộp |
| T05 | Qwen3-Reranker-4B có xoá hình phạt ngôn ngữ không | reranker bge-v2-m3 -> Qwen3-4B | trung vị điểm vi/en/zh | chưa chạy (offline) |
| T06 | Bằng chứng từ nhánh chunk có thêm doc gold không | doc list -> RRF(doc list, chunk list) | `DOCS_F2` 0.2849 | **thua: 0.2743** |
| T07 | Căn cửa sổ vào đoạn liên quan có lợi không | cửa sổ `head` -> cửa sổ quanh chunk tốt nhất | `CHUNKS_F2` 0.2275 | chưa có code |
| T08 | Đỉnh `m_doc` nằm ở đâu | `m_doc` 160 -> 220 | `DOCS_F2` 0.2849 | có lệnh, chưa dựng |
| T09 | Nhiều cửa sổ ngắn hơn nữa | c100 cap2700 head -> c120 cap2000 middle | `CHUNKS_F2` 0.2275 | có lệnh, NHIỄU BA BIẾN |
| T10 | Điểm 0.2327 là của bài nào | tải lại file điểm | - | việc dọn dẹp |

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

## Nhóm 2 - tận dụng nhánh chunk

Đo trên `data/tmp/sub1200/D_ft_top50.jsonl` (top 50 chunk mỗi câu, index chunk `medical_documents`):
nhánh chunk cho 44,4 doc mỗi câu, trong đó 20,3 doc NẰM NGOÀI pool doc top300.
Điểm `rerank_score` trong file đó do `models/medical-reranker` chấm, model đã bị loại; chỉ dùng `retrieval_rank`.

### T06 - RRF doc + chunk ở tầng doc

- Giả thuyết: doc được cả hai nhánh tìm thấy, hoặc chỉ nhánh chunk tìm thấy, trúng nhiều hơn doc ở đuôi danh sách doc.
- Cách xếp: danh sách 1 = pool top300 theo điểm rerank; danh sách 2 = doc theo bằng chứng chunk;
  điểm doc = `Σ 1/(60 + hạng)` trên hai danh sách; nộp 160 doc đầu.
- Danh sách 2 dựng thế nào: `retrieval_rank` trong dump được đánh RIÊNG theo từng branch (vi ~26, en ~16, zh ~8 chunk mỗi câu),
  nên không so thẳng được giữa các branch.
  Mỗi doc lấy hạng chunk tốt nhất trong từng branch (MaxP), rồi gộp các branch bằng RRF (`load_chunk_evidence()`).
- Code: `--methods chunkrrf` trong `scripts/fuse_submission.py`, bằng chứng chunk qua `--chunk-evidence`.
- File: `data/fuse_d160_c100_top300_chunkrrf_ctop60+top300_cap2700.zip` (98,9 MB).
  Đã kiểm offline: bản `orig` dựng cùng lệnh trùng nguyên 1200/1200 entry với bài 0.2562 đã chấm,
  và `relevant_chunks` của T06 trùng 1200/1200 câu với bài đó; mỗi câu khác 26,5 doc.
- Biến duy nhất: cách chọn 160 doc. `relevant_chunks` giữ đúng chunk list của mục 21 (`fuse_d160_c100_top300_orig_ctop60+top300_cap2700`).
- Đối chứng: `DOCS_F2` 0.2849 (mục 20 và 21 có doc list giống hệt nhau).
- Kiểm tra bài dựng đúng: `CHUNKS_F2` phải ra đúng 0.2275.
- Chẩn đoán (offline): 26,5/160 doc bị thay; 20,3 trong đó là doc chỉ nhánh chunk thấy, còn lại là doc hạng doc >160 được nhánh chunk kéo lên.
  Doc bị đẩy ra là đuôi hạng ~135-160, nơi biên trúng đo được khoảng 10% mỗi doc (mục 20).
- Lưu ý trọng số: nhánh chunk chỉ dài ~44 doc nên doc tệ nhất của nó (`1/104`) vẫn thắng hạng 160 của nhánh doc (`1/220`).
  Mọi doc nhánh chunk đều lọt vào; nếu T06 thua, thử RRF có trọng số trước khi kết luận nhánh chunk vô ích.
- Kết quả (`res/fuse_d160_c100_top300_chunkrrf_ctop60+top300_cap2700.zip`): FINAL 0.2509, `DOCS_F2` 0.2743, `CHUNKS_F2` 0.2275 (khớp, bài dựng đúng).
  `DOCS_P` 0.1646 -> `h_doc` = 26,34 so với 27,84 của đối chứng; `DOCS_R` 0.4105 so với 0.4168.
- Phán quyết: **KHÔNG được.** 26,5 doc thay vào mỗi câu trúng ít hơn 26,5 doc bị đẩy ra tổng cộng 1,50 doc.
  RRF k=60 không trọng số đưa toàn bộ doc nhánh chunk vào danh sách, và tập đó kém đuôi hạng 135-160 của nhánh doc.
  Chưa loại nhánh chunk hẳn: chưa thử bản chỉ dùng nhánh chunk để xếp lại doc ĐÃ có trong pool (không nhận doc ngoài pool), hay RRF có trọng số.

### T07 - cửa sổ căn theo chunk tốt nhất

- Giả thuyết: ở trần ngắn, cửa sổ đặt quanh đoạn mà nhánh chunk tìm thấy qua ngưỡng LCS nhiều hơn cửa sổ đầu bài.
- Biến duy nhất: vị trí cửa sổ; giữ c100, trần 2.700 như mục 21.
- Đối chứng: `CHUNKS_F2` 0.2275 (mục 21). Kiểm tra: `DOCS_F2` phải ra đúng 0.2849.
- Giới hạn: chỉ doc có trong nhánh chunk mới căn được; doc khác giữ cửa sổ đầu bài.
- Code: chưa có.
- Kết quả: `CHUNKS_F2` = ___
- Phán quyết: ___

## Nhóm 3 - số lượng nộp

### T08 - `m_doc` 220

- Giả thuyết: biên hạng 111-160 là 10,0% mỗi doc, vẫn trên ngưỡng, nên nới tiếp còn lợi.
- Lệnh: `python -m scripts.fuse_submission --methods orig --doc-pool top300 --m-doc 220 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300`
- Đối chứng: `DOCS_F2` 0.2849. Kiểm tra: `CHUNKS_F2` phải ra đúng 0.2275.
- Kết quả: `DOCS_F2` = ___
- Phán quyết: ___

### T09 - c120, trần 2.000, cửa sổ giữa bài

- Đổi BA biến cùng lúc so với mục 21 (số chunk, trần, vị trí cửa sổ), nên thắng hay thua cũng không quy được cho biến nào.
- Lệnh: `python -m scripts.fuse_submission --methods orig --doc-pool top300 --m-doc 160 --m-chunk 120 --cap 2000 --window middle --out-dir data --chunk-pool top60 top300`
- Có thể bị script chặn vì vượt 100 MB; nếu vậy hạ `--cap`.
- Kết quả: `CHUNKS_F2` = ___
- Phán quyết: ___

## Việc dọn dẹp

### T10 - chốt danh tính điểm 0.2327

`res/fuse_d110_c55_top300_mix0.3_ctop300.zip` và `res/doc_top300_bge_vcorr_enzh_hyde.zip` trùng từng byte.
Tải lại file điểm của một trong hai bài để biết điểm 0.2327 là của bài nào (mục 22 sổ phát hiện).
