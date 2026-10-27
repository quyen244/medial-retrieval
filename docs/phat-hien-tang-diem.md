# Phát hiện ảnh hưởng tới điểm

Sổ ghi các kết luận ĐÃ ĐO, kèm bằng chứng và phán quyết.
Chỉ ghi vào đây thứ có số đo thật; suy luận chưa kiểm chứng để ở [ke-hoach-phase-1-2.md](ke-hoach-phase-1-2.md).
Nghĩa của mọi mã viết tắt: [ky-hieu-bai-nop.md](ky-hieu-bai-nop.md).
Thí nghiệm đang chờ đo: [theo-doi-thi-nghiem.md](theo-doi-thi-nghiem.md).

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

## 11. Đường cong FINAL theo `h_doc` là lõm, đừng fit đường thẳng

Mọi điểm đã đo, mỗi dòng là một bài nộp:

```
h_doc  6,93  ->  FINAL 0.1117      h_doc 18,95  ->  FINAL 0.2230
h_doc 11,63  ->  FINAL 0.1894      h_doc 22,83  ->  FINAL 0.2316
h_doc 14,90  ->  FINAL 0.2090      h_doc 27,84  ->  FINAL 0.2562
```

`h_doc` gấp 4 lần nhưng FINAL chỉ gấp 2,3 lần.
Lý do nằm trong công thức: ta nâng `h` bằng cách nâng `m`, mà `F2 = 5h/(4g+m)` có `m` ở mẫu số.
Mỗi doc thêm vào vừa cộng tử số vừa cộng mẫu số, nên lợi biên giảm dần.

**Phán quyết: không ngoại suy FINAL, chỉ nội suy trong bảng trên.**
Tính thẳng `F2 = 5h/(4g+m)` cũng không chốt được vì `g` không còn xác định: từ bài d160,
`h/R` cho `g_doc ≈ 67` còn dạng đóng cho `≈ 82`, trong khi mục 8 ghi 51.
Ba số này lệch nhau quá xa để dùng dự đoán (lý do ở mục 5: macro-F2 không bằng F2 của macro-P/R).

May là mọi phán quyết về biên ở mục 20-21 KHÔNG đổi dấu trong cả khoảng `g` 51-82,
vì `g` chỉ vào ngưỡng chứ không vào biên đo được:

```
g_doc = 51  ->  ngưỡng tầng doc 7,6%
g_doc = 67  ->  ngưỡng tầng doc 6,5%
g_doc = 82  ->  ngưỡng tầng doc 5,7%
biên tầng doc đo được ở hạng 111-160:  10,0%
```

Tầng chunk cũng vậy: biên 12,2% so với ngưỡng 4,5-6,3% trong cả khoảng `g_chunk` 53-82.
Nên "còn đáng nới thêm" ở mục 20-21 kết luận được; còn "đỉnh ở đâu" thì không.

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
- **Tầng doc và tầng chunk tăng gần bằng nhau** (h_doc +4,70, h_chunk +6,34), nên không
  thể quy mức tăng cho riêng tầng nào.

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

## 20. Tách hai tầng lấy từ hai pool khác nhau, m_doc 110 -> 160: FINAL 0.2316 -> 0.2389

Bài `fuse_d160_c55_top300_orig_ctop60`: `relevant_docs` lấy 160 doc từ pool top300, `relevant_chunks` lấy 55 chunk từ pool top60.
Đúng bài mà mục 15 đề xuất ở dòng "lấy chunk từ pool top60 và doc từ pool top300", dự đoán FINAL ≈ 0.225.

| | m_doc | h_doc | DOCS_F2 | m_chunk | h_chunk | CHUNKS_F2 | FINAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19. d110_c55_top300 | 110 | 22,83 | 0.2745 | 55 | 12,60 | 0.1887 | 0.2316 |
| 20. d160_c55 (chunk top60) | 160 | **27,84** | **0.2849** | 55 | **13,67** | **0.1930** | **0.2389** |

Hai thay đổi cùng lúc, nhưng cả hai đều kiểm chứng được riêng:
- Tầng chunk khớp mục 14 từng chữ số (`h_chunk` 13,67, `CHUNKS_F2` 0.1930), đúng như kỳ vọng vì dùng lại chunk list của pool top60.
- Tầng doc chỉ đổi `m` trên cùng pool top300.

Biên hạng 111-160: `(27,84 - 22,83)/50` = **10,0% mỗi doc**, ngưỡng `h/(4g+m)` = 7,6% nếu `g_doc` = 51.

**Phán quyết: vẫn trên ngưỡng, nhưng khoảng cách đang hẹp lại từ hai đầu.**
Biên giảm 21,8% -> 13,5% -> 12,9% -> 10,0% trong khi ngưỡng tăng 4,9% -> 6,7% -> 7,3% -> 7,6%, vì `m` nằm ở mẫu số.
Phán quyết này không phụ thuộc giá trị `g` (xem mục 11), nhưng vị trí giao điểm thì phụ thuộc, nên KHÔNG ngoại suy ra `m*`.
Muốn biết đỉnh thì nộp `m_doc` = 220 - tầng doc không tốn dung lượng nên chỉ tốn một lượt.

Hệ quả: khẳng định mục 15 - hai tầng không cần lấy từ cùng một thứ hạng.
Tầng doc muốn pool sâu, tầng chunk muốn pool nông.

## 21. Nhiều cửa sổ ngắn thắng ít cửa sổ dài khi trần 100 MB đã chặn: FINAL 0.2389 -> 0.2562

Bài `fuse_d160_c100_top300_orig_ctop60+top300_cap2700`: chunk nối hai pool (hết 60 doc của top60 thì lấy tiếp top300), trần ký tự hạ từ 8.000 xuống 2.700.
`relevant_docs` giống hệt mục 20 ở cả 1200 câu, nên `DOCS_F2` buộc phải giữ nguyên 0.2849 - cô lập hoàn hảo, toàn bộ delta thuộc tầng chunk.

| | m_chunk | mean ký tự | h_chunk | CHUNKS_P | CHUNKS_F2 | FINAL | zip |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20. c55, trần 8.000 | 55 | 3.540 | 13,67 | 0.2485 | 0.1930 | 0.2389 | 86,1 MB |
| 21. c100, trần 2.700 | 100 | 2.158 | **19,16** | 0.1916 | **0.2275** | **0.2562** | **98,9 MB** |

**Phán quyết: đổi độ dài lấy số lượng là có lợi, +0,0345 `CHUNKS_F2` = +0,017 FINAL.**

Đây là câu trả lời cho câu hỏi để mở ở mục 14 ("trần 8.000 là đỉnh đo ở mục 12 khi CHƯA bị chặn bởi kích thước file, nên đánh đổi m_chunk và trần chưa đo").
Mục 3 và 12 vẫn đúng trong phạm vi của chúng: khi dung lượng còn thừa thì nới trần không hại, cắt ngắn thì hại.
Khi dung lượng đã chặn, bài toán đổi từ "trần bao nhiêu" sang "chia 100 MB thế nào", và đỉnh dịch hẳn sang phía nhiều cửa sổ ngắn.

Ràng buộc đo được, dùng để lập kế hoạch:

```
byte/ký tự trong JSON = 1,37        tỉ lệ nén zip = 0,277
=> biên giới 100 MB:  m_chunk x mean_ký_tự ≈ 215.000
   c=100 @ 2.158 (đang ở đây, 98,9 MB)   c=120 @ 1.790
   c=140 @ 1.535                          c=160 @ 1.345
```

Biên hạng 56-100: `(19,16 - 13,67)/45` = **12,2% mỗi chunk**, ngưỡng `h/(4g+m)` = `19,16/312` = 6,1%.
Còn gấp đôi ngưỡng, nhưng mọi bước tiếp phải trả bằng độ dài.

Hệ quả và rủi ro: mục 3 đo `877 ký tự -> h_chunk 2,00` (sụp hẳn) còn `E2_win2500` ở 1.967 ký tự vẫn cho 0.0927.
Vách sụp nằm đâu đó giữa 877 và 1.967, CHƯA đo.
Đi tới `c=140 @ 1.535` là bước vào đúng vùng chưa biết đó.

## 22. Bài học quy trình: file điểm tải về phải đổi tên ngay

Thư mục `res/` có 7 zip cho 5 phép đo: hai cặp trùng từng byte (cùng cả `elapsedTime` lẻ 13 chữ số thập phân, tức cùng một lần chấm tải về hai lần).

| File | Thực chất |
|---|---|
| `fuse_d160_c55_top300_orig_ctop60.zip` | đúng tên, mục 20 |
| `doc_d110_c55_top300_bge.zip` | **bản sao của trên**; d110 thật là 0.2316 (mục 19) |
| `fuse_d160_c100_..._ctop60t.zip` | đúng tên, mục 21 |
| `scoring_result.zip` | **bản sao của trên**, chưa đổi tên sau khi tải |
| `fuse_d110_c55_top300_mix0.3_ctop300.zip` | FINAL 0.2327, nhưng chủ của điểm này còn SUY LUẬN |
| `doc_top300_bge_vcorr_enzh_hyde.zip` | **bản sao của trên**; điểm thật của HyDE là 0.2225 (mục 25) |

Cặp cuối ĐÃ CHỐT (2026-10-25): bài HyDE được chấm lại ra 0.2225 (mục 25), nên 0.2327 là của `fuse_d110_c55_top300_mix0.3_ctop300`.
| `doc_d80_c55_top300_bge_vcorr_enzh_a.zip` | đúng tên, mục 18 (`a` = aspects) |

Hai bài `fuse_d160_*` có `relevant_docs` giống hệt nhau cả 1200 câu, nên tầng doc của chúng chấm ra số y hệt.
Đó là thứ làm việc gán nhãn sai khó phát hiện: nhìn `DOCS_P`/`DOCS_R` trùng khít thì dễ kết luận sai rằng hai bài dùng chung `m_doc`.
Cách chữa là đọc thẳng `len(relevant_docs)` trong file nộp, không suy từ `scores.txt`.

**Phán quyết: `scoring_result.zip` phải được đổi tên thành tên bài ngay khi tải, trước khi tải bài kế tiếp.**
Và trước khi ghi một dòng vào file này, đối chiếu `m_doc`/`m_chunk` với file nộp thật.

## 23. RRF doc list với doc từ nhánh chunk: FINAL 0.2562 -> 0.2509 (thua)

Bài `fuse_d160_c100_top300_chunkrrf_ctop60+top300_cap2700` (thí nghiệm T06 trong [theo-doi-thi-nghiem.md](theo-doi-thi-nghiem.md)).
Chỉ đổi cách chọn 160 doc: RRF k=60 giữa thứ hạng rerank của pool top300 và thứ hạng doc từ dump chunk `D_ft_top50`.
`relevant_chunks` giữ nguyên mục 21, nên `CHUNKS_F2` ra đúng 0.2275.

| | `DOCS_P` | `DOCS_R` | `h_doc` | `DOCS_F2` | FINAL |
|---|---:|---:|---:|---:|---:|
| Mục 21 (orig) | 0.1740 | 0.4168 | 27,84 | 0.2849 | 0.2562 |
| T06 chunkrrf | 0.1646 | 0.4105 | 26,34 | 0.2743 | 0.2509 |

Mỗi câu thay 26,5/160 doc (đo offline), trong đó 20,3 doc chỉ nhánh chunk thấy.
Tập thay vào trúng ít hơn tập bị đẩy ra 1,50 doc mỗi câu.

**Phán quyết: RRF không trọng số với nhánh chunk làm tụt tầng doc. Không dùng.**
Còn để ngỏ: dùng nhánh chunk chỉ để xếp lại doc đã có trong pool, chưa đo.

## 24. Nới m_doc 160 -> 220: FINAL 0.2562 -> 0.2542 (thua)

Bài `fuse_d220_c100_top300_orig_ctop60+top300_cap2700` (T08).
Đã kiểm file nộp: 220 doc mỗi câu, 160 doc đầu và `relevant_chunks` trùng 1200/1200 câu với mục 21; `CHUNKS_F2` ra đúng 0.2275.

| | `DOCS_P` | `DOCS_R` | `h_doc` | `DOCS_F2` | FINAL |
|---|---:|---:|---:|---:|---:|
| m_doc 160 (mục 21) | 0.1740 | 0.4168 | 27,84 | 0.2849 | 0.2562 |
| m_doc 220 | 0.1450 | 0.4601 | 31,90 | 0.2810 | 0.2542 |

60 doc hạng 161-220 trúng thêm 4,06 doc mỗi câu, tức biên 6,8% mỗi doc.
Biên theo hạng đã đo: 81-110 là 12,9%, 111-160 là 10,0%, 161-220 là 6,8%.

**Phán quyết: 6,8% đã dưới ngưỡng, F2 tụt. Giữ m_doc 160.**
Recall 0.46 ở 220 doc không phải trần của pool: muốn tăng điểm phải đổi doc nằm TRONG top 160, không phải nộp thêm đuôi.

## 25. HyDE và mix0.3: HyDE gỡ lại phần doc mà pool B mất, mix0.3 thắng nhẹ trên pool top300

Hai bài, cùng ngày, đã chốt danh tính điểm 0.2327 (mục 22).

**HyDE** (`doc_top300_bge_vcorr_enzh_hyde`, 80 doc / 55 chunk, pool Bhyde = RRF 5 lượt tìm: câu gốc, câu sửa, en, zh, đoạn HyDE):

| | pool | `DOCS_P` | `DOCS_R` | `h_doc` | `DOCS_F2` | `CHUNKS_F2` | FINAL |
|---|---|---:|---:|---:|---:|---:|---:|
| d80 top300 (mục 15) | chỉ câu gốc | 0.2369 | 0.3117 | 18,95 | **0.2574** | 0.1887 | **0.2230** |
| d80 B (mục 17) | + câu sửa + en + zh | 0.2370 | 0.3029 | 18,96 | 0.2520 | 0.1913 | 0.2216 |
| d80 Bhyde | + HyDE | 0.2364 | 0.3121 | 18,91 | 0.2563 | 0.1888 | 0.2225 |

**Phán quyết: trung tính.** HyDE kéo `DOCS_F2` của pool B lên +0,0043 nhưng vẫn dưới pool chỉ câu gốc; `h_doc` gần như không đổi.

**mix0.3 trên pool top300** (`fuse_d110_c55_top300_mix0.3_ctop300`, 110 doc / 55 chunk, chunk list top300 như mục 19):

| | `DOCS_P` | `DOCS_R` | `h_doc` | `DOCS_F2` | FINAL |
|---|---:|---:|---:|---:|---:|
| d110 orig (mục 19) | 0.2075 | - | 22,83 | 0.2745 | 0.2316 |
| d110 mix0.3 | 0.2093 | 0.3636 | 23,02 | **0.2766** | **0.2327** |

**Phán quyết: thắng nhẹ, +0,0021 `DOCS_F2`.** Pool top300 khoảng 80% doc vi nên điểm câu cùng ngôn ngữ ít khi khác điểm câu gốc; hiệu ứng nhỏ.
Chưa thử trên bài SOTA d160 (T19 trong `docs/experiments.json`).

## 26. Cách A trên pool B: đẩy điểm doc en/zh lên càng mạnh càng thua

Cùng pool B, 80 doc, cùng chunk list top60 (`CHUNKS_F2` ra đúng 0.1930 ở cả ba bài, nên bài dựng đúng). Chỉ đổi trọng số λ của điểm câu cùng ngôn ngữ (`s_m`) trong `λ·s_m + (1-λ)·s_o`.

| λ | bài | `DOCS_P` | `DOCS_R` | `h_doc` | `DOCS_F2` | Δ so với orig |
|---:|---|---:|---:|---:|---:|---:|
| 0 | orig (mục 17) | 0.2370 | 0.3029 | 18,96 | 0.2520 | - |
| 0,3 | `fuse_d80_c55_B_mix0.3_ctop60` | 0.2288 | 0.2928 | 18,30 | 0.2424 | −0,0096 |
| 0,5 | `fuse_d80_c55_B_mix0.5_ctop60` | 0.2205 | 0.2809 | 17,64 | 0.2332 | −0,0188 |
| 1 | `fuse_d80_c55_B_matched_ctop60` (T01) | 0.1890 | 0.2509 | 15,12 | 0.2051 | −0,0469 |

Chuẩn hoá điểm theo từng ngôn ngữ (T03, T04, nộp 2026-10-26, cùng pool B, cùng chunk list, `CHUNKS_F2` 0.1930 ở cả hai):

| cách | bài | en+zh trong top 80 | `DOCS_P` | `DOCS_R` | `h_doc` | `DOCS_F2` | Δ so với orig |
|---|---|---:|---:|---:|---:|---:|---:|
| orig | (mục 17) | 24% | 0.2370 | 0.3029 | 18,96 | 0.2520 | - |
| min-max [-1, 1] (T03) | `fuse_d80_c55_B_langmm_ctop60` | 40% | 0.2121 | 0.2688 | 16,97 | 0.2238 | −0,0282 |
| z-score logit (T04) | `fuse_d80_c55_B_langz_ctop60` | 63% | 0.1477 | 0.2001 | 11,82 | 0.1631 | −0,0889 |

**Phán quyết: thua, đơn điệu theo λ.** Trên pool B (63% doc en/zh), chấm bằng câu cùng ngôn ngữ đưa doc en/zh lên, và chúng trúng ít hơn doc vi bị đẩy ra.
Suy luận, chưa đo: hoặc gold chủ yếu là doc vi, hoặc điểm của reranker với câu dịch thổi phồng doc en/zh. T03 và T04 đã xác nhận: chuẩn hoá theo ngôn ngữ càng đẩy en/zh lên càng thua (T04 thua nặng nhất). T02 và phần en/zh của T11 cùng hướng rủi ro này.

## 27. Hạ doc en/zh cũng thua; gộp doc vi của HyDE hoà (T20, T21, T22, nộp 2026-10-27)

Đối chứng là SOTA `fuse_d160_c100_cap2700` (160 doc, 100 chunk, trần 2.700): `DOCS_F2` 0.2849, `CHUNKS_F2` 0.2275.
`h = P × m`, với m = 160 doc và 100 chunk.

T20 và T21 đi chung bài B2 (`fuse_d160_c100_top300_vifirst_ctop60+top300_cap2700_cvi`): T20 đọc tầng doc, T21 đọc tầng chunk.

| thí nghiệm | tầng | `P` | `R` | `h` | `F2` | Δ |
|---|---|---:|---:|---:|---:|---:|
| SOTA | doc | 0.1740 | 0.4168 | 27,84 | 0.2849 | - |
| T20 (doc vi trước) | doc | 0.1716 | 0.4117 | 27,46 | 0.2813 | −0,0036 |
| T22 (+ doc vi của Bhyde) | doc | 0.1731 | 0.4209 | 27,70 | 0.2850 | +0,0001 |
| SOTA | chunk | 0.1916 | 0.2835 | 19,16 | 0.2275 | - |
| T21 (chunk doc vi trước) | chunk | 0.1896 | 0.2823 | 18,96 | 0.2262 | −0,0013 |

T22 ra đúng `CHUNKS_F2` 0.2275 ở tầng không đổi, nên bài dựng đúng.

**Phán quyết:**

- T20 thua, T21 thua. Cùng với mục 26, thứ tự của reranker (câu gốc) đã đúng chỗ theo cả hai chiều: đẩy doc en/zh lên thua, hạ xuống cũng thua. Ngôn ngữ của doc không phải đòn bẩy để sắp lại.
- T22 hoà. `R` tăng 0.0041 nhưng `P × m` giảm 0,14; khác biệt nằm trong nhiễu. Doc vi mà HyDE tìm thêm không thắng doc hạng thấp của pool top300.

## 28. Nộp cả pool: top300 chỉ chứa 48,9% gold, pool B còn ít hơn (chẩn đoán, 2026-10-27)

Nộp toàn bộ 300 doc của pool thì thứ hạng không còn ảnh hưởng, nên `DOCS_RECALL` là trần recall của pool.
Tầng chunk giữ như SOTA và ra đúng 0.2275 ở cả hai bài, nên bài dựng đúng.

| pool | `DOCS_P` | `DOCS_R` (= phủ) | `h` (P × 300) |
|---|---:|---:|---:|
| top300 (chỉ câu gốc) | 0.1163 | **0.4890** | 34,89 |
| B (RRF câu gốc + câu sửa + en + zh) | 0.0883 | 0.3891 | 26,49 |

Đối chiếu với SOTA trên cùng pool top300 (R@160 = 0.4168, h 27,84):

- Ở 160 doc, reranker đã lấy được 0.4168 / 0.4890 = **85%** số gold có trong pool. Hạng 161-300 chỉ trúng (34,89 − 27,84) / 140 = 5,0% mỗi doc.
- **51% gold không có trong pool top300.** Không reranker hay cách xếp lại nào với tới phần này.
- Pool B phủ ÍT hơn top300 (0.3891 so với 0.4890): các lượt tìm bằng bản dịch en/zh chiếm chỗ của doc gold chứ không mở rộng phủ. Đây là lý do thật của mục 17, không phải do xếp hạng doc en/zh.

**Phán quyết:** nghẽn chính là PHỦ, không phải xếp hạng.
Ưu tiên những gì đưa thêm gold vào pool: pool sâu hơn (T12), embedding mới với text dài (`scripts/embed_q3e.py`), PRF theo cụm (T24).
Những gì chỉ xếp lại trong pool (reranker mới, T23, cắt theo từng câu) chỉ còn tranh phần 15% gold đã có trong pool mà nằm dưới hạng 160.
Lưu ý: corpus đã crawl được 82,2% (vi 88,7%), nên một phần gold có thể không bao giờ tìm được dù pool sâu tới đâu.

## 29. Pool top1000 (T12): có thêm gold nhưng reranker kéo nhầm doc sâu lên, thua (2026-10-27)

Pool top1000: câu gốc, dense bge-m3, `hnsw_ef` 1000, rerank bge-reranker-v2-m3 (740.429 cặp chấm trên Kaggle, 459.571 cặp dùng lại cache).

| bài | m_doc | `DOCS_P` | `DOCS_R` | `h` | `DOCS_F2` |
|---|---:|---:|---:|---:|---:|
| SOTA (pool top300) | 160 | 0.1740 | 0.4168 | 27,84 | 0.2849 |
| **T12** (pool top1000) | 160 | 0.1563 | 0.3919 | 25,01 | 0.2606 |
| chẩn đoán R@300 (mục 28) | 300 | 0.1163 | 0.4890 | 34,89 | - |
| chẩn đoán R@1000 | 1.000 | 0.0428 | 0.5914 | 42,80 | - |

- Hạng truy xuất 301-1000 có thêm 42,80 − 34,89 = 7,9 gold mỗi câu, trên 700 doc: 1,1% mỗi doc.
- Reranker đưa trung bình 55,5 doc mới (đo offline) vào top 160, và nhóm này trúng kém nhóm bị đẩy ra khoảng 5 điểm % (h giảm 2,83).
- Ở 160 doc, T12 chỉ lấy được 0.3919 / 0.5914 = 66% số gold có trong pool, so với 85% của pool top300.
- 41% gold vẫn nằm ngoài top 1.000 của bge-m3.

**Phán quyết: T12 thua.** Đào sâu bằng cùng retriever và cùng reranker làm loãng: pool sâu có thêm gold nhưng thưa (1,1%), và bge-reranker không lọc được.
Hai hướng còn lại: (1) giữ thông tin thứ hạng truy xuất khi xếp (doc hạng sâu phải có điểm rerank rất cao mới được vào), làm offline được trên pool top1000; (2) đổi tầng đầu (embedding mới) để gold lên hạng cao ngay từ khi tìm.

Phụ: bài R@1000 nộp 90 chunk, `h_chunk` 0.2013 × 90 = 18,12 so với 19,16 ở 100 chunk: chunk hạng 91-100 vẫn trúng 10,4% mỗi chunk.

## Nguyên tắc rút ra về cách thí nghiệm

Vòng 1 và 2 có 21 bài nộp đổi nhiều thứ cùng lúc và không rút ra được gì.
Phase 0 đi được vì mỗi bài chỉ đổi đúng một biến.

Không có dev set (`data/labeling/dev_set.jsonl` có `human_score` toàn None), leaderboard là thước đo duy nhất.
Mỗi lượt nộp vì thế là tài nguyên khan hiếm và phải mang đúng một phép đo.
