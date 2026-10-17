from sentence_transformers import CrossEncoder



model = CrossEncoder(
     "cross-encoder/ms-marco-MiniLM-L-6-v2", 
     cache_folder = 'models'
)

query =  "Chính sách hoàn tiền trong 7 ngày là gì?"

documents = [
    "Khách hàng được hoàn tiền trong vòng 7 ngày.",
    "Cách thay đổi mật khẩu tài khoản.",
    "Khách hàng được đổi sản phẩm trong vòng 7 ngày.",
]


pairs = [(query , doc) for doc in documents]

scores = model.predict(pairs)

results = sorted(
    zip(documents, scores),
    key=lambda x : x[1],
    reverse=True
)

print('query : ' , query)
for doc, score in results:
    print(f"{score:.4f} | {doc}")


