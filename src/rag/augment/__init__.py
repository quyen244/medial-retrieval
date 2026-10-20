"""Sinh câu truy vấn đã augment, chạy OFFLINE trước khi retrieval.

Mỗi query tốn 4-7 lời gọi LLM. Nếu làm việc này bên trong pipeline retrieval
thì mỗi lần đổi một tham số retrieval là trả lại toàn bộ chi phí đó. Tách ra
chạy một lần, ghi `augmented.jsonl`, rồi mọi thí nghiệm retrieval đọc lại cùng
một file: rẻ hơn, soi tay được, và lặp lại được.
"""
