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

**Phán quyết: chi phí mẫu số không đáng kể.**
Bài `chunk_d35_c44_full8k_plain` (như `N3_d35_c44` nhưng bóc sạch cú pháp) đã dựng, chờ nộp.
Cô lập hoàn hảo: `relevant_docs` giống hệt 1200/1200 câu, nên `F2_doc` buộc phải giữ nguyên 0.1253.
Token giảm 1,70%, ký tự giảm 1,31%, 93,5% số đoạn bị thay đổi, không đoạn nào rỗng.

Hạn chế của phép thử, phải biết trước khi tiêu một lượt nộp: nó KHÔNG đối xứng.
Nếu reference là text thuần thì bỏ cú pháp chỉ giảm mẫu số, điểm tăng.
Nếu reference là markdown thì bỏ đi giảm cả tử lẫn mẫu, điểm gần như đứng yên.
Nên kết quả 'tăng rõ' kết luận được, còn 'đứng yên' thì mơ hồ.
Giá trị thông tin thấp hơn hai bài Phase 1, xếp sau nếu quota nộp eo hẹp.

## 11. FINAL quy về một đại lượng duy nhất

`FINAL ≈ 0,016 x h_doc`, hiệu chỉnh từ năm bài nộp vòng 3.

```
h_doc 6,93 (nay)  ->  FINAL 0.111
h_doc 15          ->  FINAL ~0.24
h_doc 25          ->  FINAL ~0.40
```

Hệ quả: mỗi tài liệu gold tìm thêm được đáng +0,016 điểm.
Mọi việc khác đã chốt; chỉ còn `h_doc` cần tối ưu.

---

## Nguyên tắc rút ra về cách thí nghiệm

Vòng 1 và 2 có 21 bài nộp đổi nhiều thứ cùng lúc và không rút ra được gì.
Phase 0 đi được vì mỗi bài chỉ đổi đúng một biến.

Không có dev set (`data/labeling/dev_set.jsonl` có `human_score` toàn None), leaderboard là thước đo duy nhất.
Mỗi lượt nộp vì thế là tài nguyên khan hiếm và phải mang đúng một phép đo.
