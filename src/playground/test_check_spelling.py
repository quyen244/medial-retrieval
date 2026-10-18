from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
import requests

def spelling_generator(query):
    url = 'http://localhost:8000/v1/chat/completions'
    
    prompt = f"""    
    Bạn là chuyên gia tiếng việt, kiểm tra 1 câu query có sai chính tả hay không ? nếu sai thì chỉnh lại cho đúng + bằng chứng sai.

    query : {query}

    output : 
        trả về định dạng JSON theo format.

        {{
            'wrong_spelling' : <có sai chính tả không ?> thuộc [true, false],
            'final_query' : <câu query cuối cùng, nếu sai thì sửa không thì giữ nguyên câu query>,
            'evidence' : <danh sách các từ vựng sai>
        }}
        
    ví dụ : 
    query = "Bạn em khôn đau răng mà vẫn bị đau âm ỉ miện"
    {{
            'wrong_spelling' : True,
            'final_query' : "Bạn em không đau răng mà vẫn bị đau âm ỉ miệng",
            'evidence' :['khôn' , 'miện']
    }}
    """
    
    res = requests.post(url=url,
                json={
                    'model' : "Qwen/Qwen3-1.7B",
                    'messages' : [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                "temperature": 0.5,
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
            
            
            




