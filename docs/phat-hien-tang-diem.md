# Phát hiện ảnh hưởng tới điểm

Sổ ghi các kết luận ĐÃ ĐO, kèm bằng chứng và phán quyết.
Chỉ ghi vào đây thứ có số đo thật; suy luận chưa kiểm chứng để ở [ke-hoach-phase-1-2.md](ke-hoach-phase-1-2.md).
Nghĩa của mọi mã viết tắt: [ky-hieu-bai-nop.md](ky-hieu-bai-nop.md).

Mỗi mục ghi: phát hiện, bằng chứng, phán quyết, hệ quả.

---

## 1. Hai tầng chấm độc lập, nên phải tách hai danh sách

`relevant_docs` và `relevant_chunks` được chấm bằng hai công thức `F2 = 5h/(4g+m)` riêng biệt.
Trước Phase 0, danh sách doc được suy ra từ danh sách chunk, nên hai tầng dùng chung một số lượng.

Bằng chứng: `E3_full8k` (42,7 doc + 42,7 đoạn) cho 0.1099, `N3_d35_c44` (35 doc + 42,7 đoạn) cho 0.1112.
Đúng một biến thay đổi.

**Phán quyết: có tác dụng.** Đây là nguồn tăng điểm thứ hai của Phase 0.

Hệ quả: mọi cấu hình sau này phải chọn `m_doc` và `m_chunk` riêng, không bao giờ gộp.

## 2. Đỉnh của `m_doc` ở khoảng 35, hai chiều đều tụt

Đường cong `h_doc` đo từ ba bài nộp:

```
m = 25,0  ->  h = 6,093
m = 35,1  ->  h = 7,013     biên 9,11% mỗi doc (hạng 26-35)
m = 42,7  ->  h = 7,058     biên 0,60% mỗi doc (hạng 36-43)
```

Ngưỡng đáng thêm một doc là `h/(4g+m)` = 2,9%.
Hạng 26-35 vượt xa ngưỡng, hạng 36-43 nằm dưới hẳn.

**Phán quyết: đỉnh ở m ≈ 35.** Nới lên 43 thì `E1_docs44` tụt còn 0.0785; cắt xuống 25 thì `E4_docs25` tụt còn 0.1071.

Hệ quả: khi `h_doc` tăng (Phase 1), `m*` sẽ dịch theo, phải dò lại chứ không giữ cứng 35.

## 3. Độ dài đoạn nộp: có ngưỡng ở khoảng 2.500 ký tự, vượt qua rồi thì vô ích

Đây là phát hiện quan trọng nhất về tầng chunk.

```
877   ->  2.500 ký tự :  h_chunk 2,00 -> 5,39    +170%
2.500 ->  8.000 ký tự :  h_chunk 5,39 -> 5,50      +2%
8.000 -> 20.000 ký tự :  đứng yên, FINAL tụt nhẹ (0.1112 -> 0.1101)
```

**Phán quyết: toàn bộ mức tăng nằm ở bước đầu.** Không phải "càng dài càng tốt".

Lý do không phải độ dài tuyệt đối mà là CĂN ĐÚNG VÙNG:
chunk 877 ký tự thường lệch khỏi ranh giới đoạn reference, còn cửa sổ ~2.500 đủ bao trọn nó.

Hệ quả: đừng tiêu thêm công vào việc nới trần. Trần 8.000 đã chốt.

## 4. `win2500` và `full8k` gần như là một bài

So từng đoạn giữa hai bài nộp, 51.262 đoạn:

```
giống hệt từng ký tự : 26.208  (51,1%)
full8k dài hơn       : 25.054  (48,9%)
```

Hơn nửa tài liệu ngắn hơn 2.500 ký tự nên cả hai chế độ đều trả về trọn bài.
Điểm cũng trùng: 0.1090 so với 0.1099.

**Phán quyết: hai chế độ tương đương.** Dùng `full8k` vì ĐƠN GIẢN HƠN, không phải vì dài hơn.
`full8k` chỉ cần `raw[:8000]`, khỏi chạy `locate()` dò vị trí và khỏi lo 2,5% trường hợp dò trượt.

## 5. Suy diễn từ công thức chấm đã sai hai lần

Lần 1: luật mới đổi mẫu số LCS sang độ dài chunk NỘP.
Suy diễn nói "nộp dài bị phạt". Đo thật: chunk 3.463 ký tự cho `F2_chunk` 0.0945, chunk 877 ký tự chỉ cho 0.0317.
Ngược hẳn.

Lần 2: suy `h` từ dạng đóng `F2 = 5h/(4g+m)` thay vì `h = P x m`.
Dạng đóng lệch khoảng 12% vì macro-F2 không bằng F2 của macro-P/R.
Sai số đó đủ để đảo dấu kết luận: nó cho ra "cắt doc list xuống 12-25" trong khi sự thật là đỉnh ở 35.
Hai bài `N1_d25_c44` và `N2_d12_c44` đã dựng theo kết luận sai rồi phải xoá.

**Phán quyết: chỉ leaderboard mới kết luận được.** Luôn tính `h = P x m` từ `scores.txt`.

## 6. Trần của pool đã đo, không phải suy đoán

Dump top-50 chunk mỗi câu cho 44,4 doc duy nhất (min 30, max 50).

`E1_docs44` nộp hết pool, trung bình 42,7 doc, chỉ trúng 7,06.
Gold có khoảng 51 doc mỗi câu.

**Phán quyết: pool chứa đúng ~7 gold doc; 44 trong 51 gold nằm hoàn toàn ngoài pool.**
Recall bị chặn cứng ở 13,9% bất kể rerank thế nào.

Hệ quả: mọi thí nghiệm rerank, augment, đổi prompt bên trong pool 65 ứng viên đều cho delta gần 0.
Đó là lý do vòng 1 và 2 có 21 thí nghiệm mà không rút ra được gì.

## 7. Xếp hạng doc theo bằng chứng gộp

Trước: thứ tự doc là "chunk nào xuất hiện trước thì doc đó đứng trước".
Cách đó coi một doc có 3 chunk mạnh ngang một doc có 1 chunk ở hạng 48.

Sau: điểm của một doc = điểm rerank cao nhất trong các chunk của nó (`doc_rank()` ở `scripts/phase0_assemble.py`).

**Phán quyết: đã áp dụng ở mọi bài vòng 3.** Chưa tách riêng được đóng góp của nó vì nó đi kèm thay đổi 1.

## 8. Quy mô ground truth dựng ngược được từ leaderboard

`g_doc ≈ 51`, `g_chunk ≈ 53`, ổn định qua mọi cấu hình.

Hai con số xấp xỉ nhau nghĩa là **mỗi bài gold có đúng khoảng một đoạn gold**.

Hệ quả: tỉ số `h_chunk / h_doc` đo được chất lượng của văn bản ta nộp, tách khỏi chất lượng tìm kiếm.
Nếu tìm đúng bài mà đoạn không qua ngưỡng LCS thì tỉ số tụt.

## 9. Tỉ số `h_chunk / h_doc` đứng yên ở 0,79

Qua `E2`, `E3`, `E4`, `N3`, `N5` - năm cấu hình khác nhau - tỉ số luôn là 0,76-0,79.

Nghĩa là trong 6,93 bài gold tìm được, chỉ 5,50 bài có đoạn qua ngưỡng LCS.
Mất 21%.

Nghi phạm 1, cắt ở 8.000 ký tự làm mất đoạn gold: **đã bác**, vì `N5_d35_cap20k` nới trần lên 20.000 không tăng.

Nghi phạm 2, trafilatura cắt nội dung khác cách BTC cắt: **còn lại, chưa bác được**.

Hệ quả: phục hồi hết 21% đáng +0,012 FINAL.
Có thật nhưng nhỏ hơn hẳn bài toán recall tầng doc, nên không ưu tiên.

## 10. Cú pháp markdown chỉ chiếm 1% mẫu số

`content.markdown` do `trafilatura.extract(output_format="markdown")` sinh ra, không phải định dạng gốc của trang.
Dấu `#`, `**`, `|`, `>` là sản phẩm của bước chuyển đổi của ta.
Dưới luật mới, mẫu số là độ dài chunk NỘP nên những token đó làm phình mẫu số mà không bao giờ khớp.

Đo trên 17.140 đoạn của `N3_d35_c44`:

```
token cú pháp markdown : 1,06%
ký tự cú pháp          : 1,09%
token link/URL         : 0,00%   (trafilatura bỏ link mặc định)
```

**Phán quyết: ĐÃ NỘP, TĂNG. Reference của BTC gần text thuần hơn markdown.**

```
                 FINAL    DOCS_F2   CHUNKS_F2   CHUNKS_P   h_chunk
N3_d35_c44       0.1112   0.128     0.0944      0.1288     5,500
..._plain        0.1117   0.128     0.0955      0.1300     5,551
```

`DOCS_F2`, `DOCS_PRECISION` và `DOCS_RECALL` giống nhau từng chữ số, đúng như thiết kế,
nên toàn bộ delta thuộc về tầng chunk.
Bài `chunk_d35_c44_full8k_plain` (như `N3_d35_c44` nhưng bóc sạch cú pháp) đã dựng, chờ nộp.
Cô lập hoàn hảo: `relevant_docs` giống hệt 1200/1200 câu, nên `F2_doc` buộc phải giữ nguyên 0.1253.
Token giảm 1,70%, ký tự giảm 1,31%, 93,5% số đoạn bị thay đổi, không đoạn nào rỗng.

Hạn chế đã lường trước: phép thử KHÔNG đối xứng.
Nếu reference là text thuần thì bỏ cú pháp chỉ giảm mẫu số, điểm tăng.
Nếu reference là markdown thì bỏ đi giảm cả tử lẫn mẫu, điểm gần như đứng yên.
Nên kết quả 'tăng rõ' kết luận được, còn 'đứng yên' thì mơ hồ.
Nó đã tăng, nên kết luận chắc.

## 11. FINAL quy về một đại lượng duy nhất

`FINAL ≈ 0,016 x h_doc`, hiệu chỉnh từ năm bài nộp vòng 3.

```
h_doc 6,93 (nay)  ->  FINAL 0.111
h_doc 15          ->  FINAL ~0.24
h_doc 25          ->  FINAL ~0.40
```

Hệ quả: mỗi tài liệu gold tìm thêm được đáng +0,016 điểm.
Mọi việc khác đã chốt; chỉ còn `h_doc` cần tối ưu.

## 12. Hai cơ chế khác nhau, đừng gộp: làm sạch token rác ≠ cắt ngắn văn bản

Bản đầu của mục này gộp hai thứ vào một cột "% ký tự → % trúng" như thể là một hiện tượng
đo hai lần. Sai. Chúng khác cơ chế:

| Thay đổi | Token bị động tới | Khớp reference được không |
|---|---|---|
| Bóc cú pháp (`chunk_d35_c44_full8k_plain`) | ký hiệu `#`, `**`, `>` và dấu gạch bảng | **không bao giờ** |
| Nới trần 8k → 20k (`N5_d35_cap20k`) | chữ thật ở phần đuôi tài liệu | khớp một phần |

Bỏ token không bao giờ khớp là LÀM SẠCH: mẫu số giảm, tử số giữ nguyên, lợi thuần.
Bỏ chữ thật thì tử số giảm theo. Không đặt chung thước đo được.

**Phán quyết về làm sạch: có tác dụng, +0,93% h_chunk.** Xem mục 10.

**Phán quyết về độ dài: trần 8.000 là đỉnh, cả hai phía đều tụt.**

```
E2_win2500   trung bình 1.967 ký tự  ->  CHUNKS_F2  0.0927
E3_full8k    trung bình 3.463 ký tự  ->  CHUNKS_F2  0.0944
N5_cap20k    trung bình 3.731 ký tự  ->  CHUNKS_F2  0.0922
```

Ngắn hơn thua, dài hơn cũng thua. Hợp với mục 3: có ngưỡng căn vùng ở khoảng 2.500 ký tự,
dưới đó mất căn vùng, trên 8.000 thì phần thêm vào loãng hơn phần đã có.

Lưu ý `E2` so với `E3` có nhiễu: E2 là cửa sổ căn vào chunk tốt nhất, E3 lấy phần đầu
tài liệu, nên khác cả vị trí lẫn độ dài. Phép so sạch về độ dài là `E3` với `N5`
(cùng lấy phần đầu, chỉ khác trần), và nó cho dài hơn thì tụt.

Bài `chunk_d35_c44_full4k_plain` (trần 4.000, trung bình 2.570 ký tự) đã dựng để đóng
nốt phía ngắn trong cùng họ "lấy phần đầu". Không kỳ vọng nó tăng; mục đích là loại
hẳn biến độ dài khỏi danh sách việc cần làm.

## 13. Index cấp document: h_doc 6,93 -> 11,63, FINAL 0.1117 -> 0.1894

Bài `doc_d35_c44_top60_bge` (2026-10-24), đo bằng `h = P x m`:

| | `chunk_d35_c44_full8k_plain` | `doc_d35_c44_top60_bge` |
|---|---:|---:|
| FINAL | 0.1117 | **0.1894** |
| DOCS_F2 / P / R | 0.1280 / 0.1982 / 0.1364 | **0.1972 / 0.3323 / 0.2009** |
| CHUNKS_F2 / P / R | 0.0955 / 0.1300 / 0.1043 | **0.1815 / 0.2703 / 0.1893** |
| h_doc (m_doc 35) | 6,93 | **11,63** |
| h_chunk | 5,55 (m 42,7) | **11,89** (m 44) |
| h_chunk / h_doc | 0,80 | 1,02 |

Cách làm: tìm thẳng trên `medical_docs` (một điểm mỗi tài liệu, text = title |
description | heading | phần đầu, ≤1.500 ký tự, đã làm sạch bằng
`scripts/denoise_doc_texts.py`), dense BGE-M3, KHÔNG lọc ngôn ngữ, lấy 60 doc,
rerank bằng câu gốc, nộp 35 doc và 44 cửa sổ markdown gốc trần 8.000 ký tự đã bóc
cú pháp. Index chunk `medical_documents` không được dùng.

**Phán quyết: đổi biểu diễn là bước nhảy lớn nhất từ trước tới nay**, đúng chẩn đoán
`docs/chan-doan-he-retrieval.md`: nút thắt là pool và biểu diễn, không phải reranker hay
độ dài chunk (mọi bài Phase 0 nằm trong 0.107-0.112).

Ba điểm phải đọc đúng:

- **Lẫn hai biến.** Bài đổi cả biểu diễn lẫn reranker (BGE thay bản fine-tune của
  N3). Muốn tách cần bài đối chứng `chunk_N3_bge` (N3 với reranker BGE).
- **h_chunk / h_doc vượt 1** vì tầng chunk nộp 44 doc còn tầng doc nộp 35, và pool
  mới có đủ 60 doc khác nhau. Pool chunk cũ chỉ ra ~44 doc nên nhiều chunk chung doc.
  Tỉ số 0,79 "đứng yên" ở mục 9 là tính chất của pool chunk, không phải hằng số.
- **Mục 11 đứng vững:** `0,016 x 11,63 = 0,186`, thực đo 0.1894.

Doc nộp lên: vi 87% / en 7% / zh 5% (câu hỏi tiếng Việt, chưa dịch).

Hệ quả: `m_doc = 35` là tối ưu cho h = 7. Ngưỡng biên `h/(4g+m)` giờ là
11,6/(4 x 51 + 35) ≈ 4,9%, trong khi precision trung bình 33%, nên m* nhiều khả năng
cao hơn 35. Phải đo, không suy (mục 5).

## 14. Nộp dài hơn từ cùng pool top60: FINAL 0.1894 -> 0.2090

Bài `doc_d50_c55_top60_bge`: lắp offline từ đúng pool đã rerank của mục 13
(`scripts/assemble_doc_submission.py`), chỉ đổi số nộp.

| | m_doc | h_doc | DOCS_F2 | m_chunk | h_chunk | CHUNKS_F2 | FINAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| d35_c44 | 35 | 11,63 | 0.1972 | 44 | 11,89 | 0.1815 | 0.1894 |
| d50_c55 | 50 | 14,90 | 0.2251 | 55 | 13,67 | 0.1930 | **0.2090** |

Biên đo được:
- doc hạng 36-50: +3,27 trúng / 15 = **21,8% mỗi doc**, ngưỡng `h/(4g+m)` = 4,9%
- chunk hạng 45-55: +1,77 trúng / 11 = **16,1% mỗi chunk**, ngưỡng 4,6%

**Phán quyết: còn xa đỉnh ở cả hai tầng.** Khác hẳn pool chunk cũ (mục 2: biên hạng
36-43 chỉ 0,6%). Pool doc-level đủ tốt để doc hạng 50 vẫn trúng 1/5.

Hai giới hạn từ đây:
- Tầng doc bị chặn bởi kích thước pool (top60 chỉ có 60 doc). Muốn nới phải dùng pool sâu hơn.
- Tầng chunk bị chặn bởi **giới hạn 100 MB**: ~1,58 MB mỗi chunk của 1.200 câu ở trần
  8.000, nên m_chunk tối đa ~58-60. Nới tiếp buộc phải hạ trần ký tự. Trần 8.000 là đỉnh
  đo ở mục 12 khi CHƯA bị chặn bởi kích thước file, nên đánh đổi m_chunk và trần chưa đo.

## 15. Pool sâu top300: tầng doc tăng, tầng chunk GIẢM - FINAL 0.2090 -> 0.2230

Bài `doc_d80_c55_top300_bge`: dense lấy 300 doc, rerank BGE cả 300 bằng câu gốc, nộp
80 doc / 55 chunk (trần 8.000). So với mục 14, tầng chunk giữ nguyên m = 55 để cô lập
hiệu ứng của chiều sâu.

| | m_doc | h_doc | DOCS_F2 | m_chunk | h_chunk | CHUNKS_F2 | FINAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| d50_c55_top60 | 50 | 14,90 | 0.2251 | 55 | 13,67 | 0.1930 | 0.2090 |
| d80_c55_top300 | 80 | **18,95** | **0.2574** | 55 | 12,60 | 0.1887 | **0.2230** |

Đo offline trên hai file nộp (không tốn lượt):
- Top 50 doc của hai bài chỉ trùng **26,3 doc** mỗi câu: reranker kéo khoảng nửa top 50 từ
  hạng dense 61-300 lên.
- 55 chunk trùng **28,1**: khoảng 27 chunk mỗi câu bị thay. Chunk mới vào dài hơn
  (TB 3.653 ký tự so với 3.224 của chunk bị đẩy ra).

**Tầng doc:** biên hạng 51-80 khoảng (18,95 - 14,90)/30 = 13,5% mỗi doc, ngưỡng
`h/(4g+m)` = 18,95/284 = 6,7%. Còn đáng nới thêm. (Con số biên lẫn cả việc top 50 đã đổi,
chỉ dùng để định hướng, không phải biên thuần.)

**Tầng chunk:** cùng m = 55 mà h_chunk giảm 1,07. Doc mà reranker kéo từ sâu lên trúng ở
tầng doc nhưng ít trúng chunk hơn doc chúng đẩy ra. Hai giả thuyết, CHƯA tách được:
1. Chunk dài hơn → mẫu số LCS (độ dài chunk nộp) lớn hơn → dễ trượt ngưỡng 0,40.
2. Thứ hạng dense mang thông tin mà reranker (chỉ đọc 1.500 ký tự đầu) bỏ qua; pool nông
   ngầm giữ ràng buộc "dense cũng thấy liên quan".

**Phán quyết:** chiều sâu có lợi cho `relevant_docs`, có hại nhẹ cho `relevant_chunks`.
Hai tầng không nhất thiết phải lấy từ cùng một thứ hạng; lấy chunk từ pool top60 và doc từ
pool top300 ước tính được FINAL ≈ (0.2574 + 0.1930)/2 = 0.225 (chưa nộp).

## 16. Pool B (orig + sửa chính tả + en + zh) nộp nhầm ở 35/44: FINAL 0.1836

File nộp là zip script retrieve TỰ SINH theo mặc định config (35 doc / 44 chunk), không
phải bản lắp offline 80/55. Không so thẳng được với 0.2230 (khác m), nhưng so được với
`doc_d35_c44_top60_bge` (cùng m).

| | pool | m_doc | h_doc | DOCS_F2 | m_chunk | h_chunk | CHUNKS_F2 | FINAL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d35_c44_top60 | top60, orig | 35 | **11,63** | **0.1972** | 44 | **11,89** | **0.1815** | **0.1894** |
| B 35/44 | top300, orig+corr+en+zh | 35 | 10,97 | 0.1914 | 44 | 11,13 | 0.1758 | 0.1836 |

Đo offline trên các pool:
- Top 35 của B trùng **28,0 / 35** doc với top 35 của pool top300 chỉ-orig; chỉ 5% doc
  trong top 35 là doc mà list orig không tìm ra. Ở hạng đầu, B gần như = top300 chỉ-orig.
- Reranker (chấm bằng câu tiếng Việt) dồn doc en/zh xuống: pool B có 63% doc en/zh nhưng
  top 80 sau rerank chỉ còn 24% (top 35: 15%). Điểm rerank trung vị doc vi 0,23, en/zh 0,01.

**Phán quyết (có điều kiện):** ở m nhỏ, pool sâu làm TỤT hạng đầu cả hai tầng (h_doc -0,66,
h_chunk -0,76). Vì B ≈ top300-orig ở top 35, phần tụt này nhiều khả năng do chiều sâu chứ
không do biến thể - khớp với mục 15 (tầng chunk tụt khi đổi top60 -> top300). Chiều sâu chỉ
có lợi vì cho phép nộp nhiều doc hơn. Hiệu ứng riêng của biến thể dịch CHƯA đo được: chúng
kéo doc en/zh vào pool nhưng reranker loại phần lớn trước khi tới phần nộp.

Bài học quy trình: script retrieve luôn sinh kèm một zip theo `submission.*` của config;
bản nộp phải là bản `assemble_doc_submission` có `d<m>_c<m>` trong tên.

## 17. Pool B ở 80/55: biến thể dịch gần như trung tính - FINAL 0.2216

| | pool | h_doc | DOCS_F2 | DOCS_R | h_chunk | CHUNKS_F2 | FINAL |
|---|---|---:|---:|---:|---:|---:|---:|
| d80_c55_top300 | chỉ orig | 18,95 | **0.2574** | **0.3117** | 12,60 | 0.1887 | **0.2230** |
| d80_c55_top300_vcorr_enzh | orig+corr+en+zh | 18,96 | 0.2520 | 0.3029 | 12,99 | 0.1913 | 0.2216 |

- Tầng doc: tổng số trúng NHƯ NHAU (18,95 vs 18,96) nhưng DOCS_F2 thấp hơn 0.0054 vì recall
  macro thấp hơn: trúng dồn vào ít câu hơn / câu có nhiều gold hơn. Macro-F2 thưởng cho trải
  đều trên các câu, không chỉ tổng số trúng.
- Tầng chunk: +0,39 trúng (+0.0026 F2). Vẫn dưới pool top60 (13,67).

**Phán quyết:** với rerank bằng câu tiếng Việt, thêm bản dịch vào pool KHÔNG có lợi ròng.
Khớp mục 16: reranker loại phần lớn doc en/zh mà bản dịch kéo vào. Muốn bản dịch đóng góp
phải đổi cách rerank (`scripts/rescore_pools.py`, rerank theo ngôn ngữ doc).

## 18. Thêm aspects (pool C) làm tụt cả hai tầng - FINAL 0.2013

| | pool | h_doc | DOCS_F2 | h_chunk | CHUNKS_F2 | FINAL |
|---|---|---:|---:|---:|---:|---:|
| B 80/55 | orig+corr+en+zh | **18,96** | **0.2520** | **12,99** | **0.1913** | **0.2216** |
| C 80/55 | B + aspects (TB 17,8 lượt tìm mỗi câu) | 17,15 | 0.2276 | 11,98 | 0.1749 | 0.2013 |

- Pool C: 179/300 doc là doc câu gốc không tìm ra. Top 80 của C trùng B 55/80; 9% top 80
  là doc CHỈ aspects tìm ra.
- Top 80 của C có nhiều doc en/zh hơn B (31% so với 24%) mà vẫn kém hơn.

**Phán quyết:** aspects HẠI (-1,8 doc, -1,0 chunk trúng). Mỗi aspect là một khía cạnh hẹp,
17,8 list RRF lấn át list câu gốc, kéo vào doc đúng chủ đề con nhưng lệch nhu cầu chính.
Lưu ý cho hướng rerank theo ngôn ngữ: đẩy thêm doc en/zh vào top 80 KHÔNG tự động có lợi
(ở đây lẫn với hiệu ứng aspects nên không kết luận chắc).

## 19. Nới m_doc 80 -> 110 trên pool top300: FINAL 0.2230 -> 0.2316

Bài `doc_d110_c55_top300_bge`: y hệt `doc_d80_c55_top300_bge`, chỉ đổi m_doc.

| m_doc | h_doc | DOCS_P | DOCS_R | DOCS_F2 | FINAL |
|---:|---:|---:|---:|---:|---:|
| 80 | 18,95 | 0.2369 | 0.3117 | 0.2574 | 0.2230 |
| 110 | **22,83** | 0.2075 | 0.3598 | **0.2745** | **0.2316** |

Tầng chunk không đổi (0.1887), đúng như kỳ vọng: hai tầng chấm độc lập.

Biên hạng 81-110: 3,88 / 30 = **12,9% mỗi doc**, ngưỡng `h/(4g+m)` = 22,83/314 = 7,3%.
Đường biên giảm rất chậm: 21,8% (36-50) -> 13,5% (51-80) -> 12,9% (81-110).

**Phán quyết:** vẫn còn xa đỉnh. Tầng doc KHÔNG bị giới hạn dung lượng (doc_id rất nhỏ),
nên nới tiếp lên 150-200 chỉ tốn lượt nộp. Pool top300 cho phép tới 300.

## Nguyên tắc rút ra về cách thí nghiệm

Vòng 1 và 2 có 21 bài nộp đổi nhiều thứ cùng lúc và không rút ra được gì.
Phase 0 đi được vì mỗi bài chỉ đổi đúng một biến.

Không có dev set (`data/labeling/dev_set.jsonl` có `human_score` toàn None), leaderboard là thước đo duy nhất.
Mỗi lượt nộp vì thế là tài nguyên khan hiếm và phải mang đúng một phép đo.
