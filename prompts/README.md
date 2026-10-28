# Prompt cho pipeline sinh nhãn

File `.txt` trong thư mục này là **nguồn sự thật**. Bạn sửa file, chạy lại là ăn ngay.
MLflow Prompt Registry chỉ đóng vai sổ ghi version: mỗi lần nội dung đổi, `prompts.py` đăng ký
một version mới và ghi URI của nó vào run, để sau này truy ngược được run nào dùng prompt nào.

Cú pháp biến là **hai ngoặc nhọn**: `{{ten_bien}}`. Đây là cú pháp của MLflow Prompt Registry,
không phải f-string Python.

Code sẽ **từ chối chạy** nếu prompt thiếu biến bắt buộc, kèm thông báo nói rõ thiếu biến nào.
Thà hỏng ngay còn hơn chạy 5 tiếng rồi mới phát hiện.

---

## `gen_query_short.txt`

Sinh câu hỏi ngắn gọn, 8-25 từ. Khớp với 66,3% query thật dưới 30 từ.

| Biến | Nội dung |
|---|---|
| `{{chunk}}` | Nội dung chunk nguồn. Có thể là tiếng Anh, Việt, hoặc Trung |
| `{{chunk_language}}` | `en` / `vi` / `zh` |
| `{{few_shots}}` | Vài query thật cùng dạng ngắn, mỗi dòng một câu |
| `{{vi_terms}}` | Thuật ngữ tiếng Việt tra từ bảng ICD song ngữ, phân cách bằng dấu phẩy. Rỗng nếu không tra được |

**Output phải là JSON** đúng dạng:

```json
{"query_1": "...", "query_2": "..."}
```

## `gen_query_narrative.txt`

Sinh đoạn bệnh nhân kể bệnh, 40-150 từ, có chi tiết thừa. Khớp với 33,8% query thật trên 30 từ.
Cùng bộ biến và cùng dạng output như trên.

---

## Bốn ràng buộc nên có trong cả hai prompt

Đây là các quyết định thiết kế từ `docs/pipeline-sinh-nhan-reranker.md`, không phải gợi ý văn phong.
Thiếu một trong số này thì tập nhãn hỏng theo cách khó phát hiện.

1. **Luôn sinh ra tiếng Việt**, kể cả khi `{{chunk}}` là tiếng Anh hoặc tiếng Trung.
   Query của cuộc thi 100% tiếng Việt, và cặp vi→en là thứ reranker cần học nhất.

2. **Dùng từ bệnh nhân hay dùng, không bê nguyên thuật ngữ lâm sàng trong chunk.**
   Đây vừa là chuyện khớp văn phong, vừa là cơ chế chống rò rỉ xuất xứ: nếu query mượn
   nguyên thuật ngữ của chunk nguồn thì chunk đó luôn thắng chỉ nhờ trùng từ, và reranker
   học thành máy đếm từ trùng. `{{vi_terms}}` có để phục vụ việc này.

3. **Không nhắc tới tài liệu.** Cấm "theo đoạn trên", "nghiên cứu này", "tài liệu cho thấy".
   Bệnh nhân thật không biết có tài liệu nào tồn tại.

4. **Giữ nguyên thực thể Latin** nếu chunk có tên thuốc, xét nghiệm, vắc xin, hội chứng
   (Dapoxetine, HBsAg, AstraZeneca, Basedow...). Chúng gần như bất biến qua ngôn ngữ nên là
   mỏ neo truy xuất chính xác cao.

---

## Đổi prompt rồi thì làm gì

Không cần làm gì thêm. Chạy lại `run_generate.py`, nó tự so nội dung với version mới nhất
trong registry; khác thì đăng ký version mới, giống thì dùng lại version cũ.

Xem lịch sử:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## `q3e_instruct_p1.txt`, `q3e_instruct_p2.txt`

Instruction cho câu hỏi khi tìm bằng Qwen3-Embedding-4B (`scripts/embed_q3e.py search --instruction-file`).
Không có biến; code gộp xuống dòng thành dấu cách và ghép thành `Instruct: <file>\nQuery: <câu sửa>`.
Hậu tố tên file (`p1`, `p2`) đi vào tên file ứng viên và tên pool, nên mỗi bản instruction một file riêng, đừng sửa đè.
