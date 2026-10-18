from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
import requests

def router_generator(queries):
    url = 'http://localhost:8000/v1/chat/completions'
    
    prompt = f"""    
    bạn là 1 query router hãy đánh giá và phân loại query thành các mức độ easy, medium , và hard, 
    dựa trên mức độ khó, có đòi hỏi suy luận nhiều không ? , có cần nhiều thông tin không để trả lười được không ? 

    Đây là danh sách queries:
        
          {queries}
        

        output : 
            trả về định dạng JSON theo format.

            [
            {{
                'id' : <mã định danh của query>,
                'rank' : <mức độ khó của query> thuộc ['easy' , 'medium', 'hard']
            }}
            ]
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
    queries = """
    [
        {
            'id" : "1",
            'query': 'Em bị dị ứng một loại thuốc giảm đau, khi uống thuốc có thành phần này chỉ bị sưng mắt, nếu ngưng uống triệu chứng tự hết. Vậy em có thể tiêm vắc xin Covid-19 được không?'
        } ,
        {
            'id" : "2",
            'query': 'Đau họng, khó ngủ có nên ngậm viên ho lotus chiết xuất cao thường xuân không?'
        } 
    ]
    """
        
    res = router_generator(queries)

    res.raise_for_status()
    
    print(res.json()["choices"][0]["message"]["content"])
            
            
            
