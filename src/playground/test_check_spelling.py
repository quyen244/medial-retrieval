from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
import requests

def spelling_generator(query):
    url = 'http://localhost:8000/v1/chat/completions'
    
    res = requests.post(url=url,
                json={
                    'model' : "Qwen/Qwen3-1.7B",
                    'messages' : [
                          {
                "role": "system",
                "content": """
                    Bạn là bộ sửa lỗi nhận dạng giọng nói tiếng Việt.
                    Chỉ sửa các từ bị nhận dạng sai dựa trên ngữ cảnh.
                    Giữ nguyên ý nghĩa và cấu trúc câu.
                    Không tự ý diễn đạt lại.
                    Chỉ trả về câu đã sửa.
                    """
            },
            {
                "role": "user",
                "content": (
                    'Câu: "Tôi thường xuân đi học mỗi ngày."\n'
                    "Hãy sửa lỗi nhận dạng giọng nói."
                )
            }
                    ],
                "temperature": 0,
                "max_tokens": 512,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
                  })
    
    return res

if __name__ == '__main__':
    query = "Đau họng, khó ngủ có nên ngậm viên ho lotus chiết xuất cao thường xuân không?"
    res =  spelling_generator(query)

    res.raise_for_status()
    
    print(res.json()["choices"][0]["message"]["content"])
            
            
            




