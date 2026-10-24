# Ký hiệu bài nộp - bảng tra nghĩa

Mọi mã viết tắt từng dùng trong `ket_qua_thi_nghiem.txt`, các doc kế hoạch, và trong trao đổi.
Chữ cái đầu chỉ là **đợt làm việc**, không mang nghĩa kỹ thuật gì - đó là một lựa chọn tồi, xem mục Quy ước mới ở cuối.

## Quy ước cũ (đang tồn tại, phải giữ để còn đọc lại được kết quả)

| Chữ đầu | Nghĩa | Đợt |
|---|---|---|
| `A` | bài chuẩn, câu hỏi gốc không augment | vòng 1 |
| `B`, `Bp` | có augment; `Bp` = B' là bản sửa prompt | vòng 1 |
| `D` | có phân rã câu dài thành sub-query | vòng 2 |
| `_ft` | dùng reranker đã fine-tune | vòng 2 |
| `E` | Phase 0 đợt 1 - thử tách doc/chunk và cửa sổ dài | vòng 3 |
| `N` | Phase 0 đợt 2 - chốt m_doc tại đỉnh | vòng 3 |
| `S` | Phase 1 - tìm ở cấp document (chưa chạy) | vòng 4 |
| `P` | phép thăm dò, không nhằm ăn điểm (chưa chạy) | vòng 4 |

## Các tham số lặp lại trong tên

| Ký hiệu | Nghĩa |
|---|---|
| `m_doc` | số document nộp trong `relevant_docs` |
| `m_chunk` | số đoạn nộp trong `relevant_chunks` |
| `d35` | m_doc = 35 |
| `c44` | m_chunk = 44 |
| `k40` | nộp 40 chunk (cách viết cũ của vòng 1-2) |
| `win2500` | mỗi đoạn nộp là cửa sổ 2.500 ký tự cắt từ markdown gốc |
| `full8k` | mỗi đoạn nộp là trọn markdown gốc, cắt ở 8.000 ký tự |
| `docs44` | lấy hết doc có trong pool (~44) |
| `hyb` | bật nhánh sparse, tìm hybrid dense + sparse |
| `noslot` | tắt cơ chế giữ chỗ cho sub-query |
| `plain` | bóc cú pháp markdown khỏi `chunk_text` (`scripts/md_strip.py`) |

## Bảng đầy đủ các bài ĐÃ NỘP

Điểm vòng 1 và 2 chấm theo luật chunk cũ, vòng 3 theo luật mới - **không so chéo được**.

### Vòng 3 (Phase 0) - luật chấm hiện hành

| Mã | Nghĩa đầy đủ | m_doc | m_chunk | Văn bản đoạn nộp | FINAL |
|---|---|---:|---:|---|---:|
| `D_ft` | bài gốc vòng 2 dùng làm mốc | 35,1 | 40 | chunk 877 ký tự | 0.0805 |
| `E1_docs44` | như D_ft nhưng nới doc list hết pool | 42,7 | 40 | chunk 877 ký tự | 0.0785 |
| `E4_docs25` | cắt doc list xuống 25 | 25 | 25 | trọn bài, trần 8.000 | 0.1071 |
| `E2_win2500` | cửa sổ 2.500 ký tự quanh chunk tốt nhất | 42,7 | 42,7 | cửa sổ 2.500 | 0.1090 |
| `E3_full8k` | trọn markdown gốc | 42,7 | 42,7 | trọn bài, trần 8.000 | 0.1099 |
| `N5_d35_cap20k` | như N3 nhưng nới trần lên 20.000 | 35 | 42,7 | trọn bài, trần 20.000 | 0.1101 |
| **`N3_d35_c44`** | **35 doc + 43 cửa sổ, trần 8.000 - TỐT NHẤT** | 35 | 42,7 | trọn bài, trần 8.000 | **0.1112** |

Hai bài đã dựng rồi xoá vì dựa trên kết luận sai (tưởng doc list càng ngắn càng tốt):
`N1_d25_c44` (25 doc) và `N2_d12_c44` (12 doc).

### Vòng 1 - 200 câu đầu, luật chunk cũ

| Mã | Nghĩa | Điểm |
|---|---|---:|
| `A` | câu gốc, không augment, không phân rã, nộp 20 chunk | 0.0112 |
| `B` | thêm augment bản đầu + phân rã + giữ chỗ | 0.0104 |
| `Bp` | augment bản sửa prompt (B') | 0.0103 |
| `Bp_hyb` | như Bp, thêm nhánh sparse | 0.0104 |
| `Bp_noslot` | như Bp, tắt giữ chỗ sub-query | 0.0105 |
| `Bp_sqrt` | như Bp, quota ngôn ngữ theo căn bậc hai cỡ corpus | 0.0105 |
| `mix_Ashort_Bplong` | câu ngắn lấy từ A, câu dài lấy từ Bp | 0.0110 |
| `mix_Bpshort_Along` | ngược lại | 0.0106 |
| `A_v1k10` / `A_v1k50` | A nhưng nộp 10 / 50 chunk | 0.007 / 0.0133 |

### Vòng 2 - đủ 1200 câu, reranker fine-tune

Chỉ bài `D_ft` được chấm (0.0805, theo luật mới). Các bài còn lại chưa chạy:
`A_ft` (bỏ phân rã), `A_base` (reranker gốc), `D_ft_noslot`, `A_ft_hyb`, `Bp_ft`,
`D_ft_k30`, `D_ft_k50`, `mix_Dshort_Bplong`, `D_ft_adaptive`.

## Các bài DỰ ĐỊNH, chưa chạy

| Mã | Nghĩa đầy đủ | Để đo gì |
|---|---|---|
| `S1_doc60` | tìm ở cấp document, lấy 60 doc rồi rerank | so với N3 → hiệu ứng của **biểu diễn mới** |
| `S2_doc1000` | y hệt S1 nhưng lấy 1000 doc | so với S1 → hiệu ứng của **chiều sâu** |
| `P1_enzh_only` | chỉ nộp doc tiếng Anh và tiếng Trung trong pool | 7 gold doc hiện tìm được thuộc ngôn ngữ nào |
| `P2_vi_only` | chỉ nộp doc tiếng Việt | như trên, phía còn lại |
| `chunk_d35_c44_full8k_plain` | như `N3_d35_c44` nhưng bóc sạch cú pháp markdown | reference của BTC là markdown hay text thuần (ĐÃ DỰNG, chờ nộp) |

`C65` và `T2800` là tên nội bộ trong `scripts/phase1_deep_retrieve.py`
(đối chứng 65 chunk và thí nghiệm 2800 chunk), không phải tên file nộp.

## Quy ước mới, áp dụng từ giờ

Tên phải **tự mô tả**, chứa luôn tham số, không dùng chữ cái tuỳ ý:

```
<tầng>_<m_doc>_<m_chunk>_<văn bản đoạn>_<biến thể>

doc_d35_c44_full8k            tìm cấp document, 35 doc, 44 đoạn, trọn bài trần 8k
doc_d35_c44_full8k_hyb        thêm nhánh sparse
chunk_d35_c44_full8k          tìm cấp chunk (cách cũ), cùng cách nộp
```

`<tầng>` là `doc` hay `chunk` - nói rõ tìm kiếm diễn ra ở cấp nào. Đây là phân biệt
quan trọng nhất của Phase 1 và nó hoàn toàn vắng mặt trong cách đặt tên cũ.

Mỗi lần nhắc một mã lần đầu trong một câu trả lời hay một tài liệu, viết kèm nghĩa
trong ngoặc. Mỗi bảng so sánh phải có một cột mô tả cấu hình, không chỉ có cột tên.
