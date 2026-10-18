from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
import requests

encoder = SentenceTransformer("BAAI/bge-m3")
vectorstore = VectorStore(encoder, url = 'http://localhost:6333', collection_name='query_documents')


def get_few_shot(chunk):
    results = vectorstore.retrieve(query=chunk, top_k = 2, verbose=True)
    print(results)
    return [{'query' : res.text} for res in results]

def query_generator(chunk , few_shots):
    url = 'http://localhost:8000/v1/chat/completions'
    
    prompt = f"""    
    role : Bạn là chuyên gia về ngôn ngữ việt nam về y tế. 
    
    Task: 
    Đưa ra 2 câu hỏi (query) dựa trên 1 đoạn ngắn (chunk) của tài liệu. 
    Nội dung chunk :
        {chunk}
        
    Instructions:
    step 1 : đọc nội dung đoạn cắt (chunk) của tài liệu .
    step 2 : phân tích và phân rã đoạn cắt thành các câu hỏi.
    step 3 : kiểm tra lại câu hỏi.
    
    Requirements: 
        - Ngôn ngữ trả về tiếng việt 
        - Giọng văn đơn giản
        - Câu hỏi không được leak thông tin 
        - Câu hỏi phải giống bệnh nhân hỏi bác sĩ. 
    Ví dụ các câu hỏi đạt chuẩn: 
    {few_shots}   
    OUTPUT FORMAT:
    Return in JSON
    {
        'query_1' : "first generated query",
        'query_2' : "second generated query",
    }
    """
    
    res = requests.post(url=url,
                data={
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
                    "enable_thinking": false
                }
                  })
    
    return res

if __name__ == '__main__':
    chunk = """Các nhà khoa học Mỹ cho rằng việc ghi thời lượng tập thể dục
        cần thiết để tiêu hao phần năng lượng có trong một sản phẩm ăn uống, 
        ví dụ “1 lon nước ngọt = 1 giờ chạy bộ”, hiệu quả hơn việc công bố hàm lượng 
        calorie trên nhãn sản phẩm. Nhóm nghiên cứu thuộc
        Đại học Johns Hopkins đã thử đưa 3 thông tin khác nhau lên nhãn các 
        chai nước ngọt cùng loại tại các cửa hàng bán lẻ. Ở nhóm chai thứ nhất, 
        họ ghi lượng calorie có trong chai nước ngọt. 
        Ở nhóm chai thứ 2, các chuyên gia ghi lượng calorie đó chiếm bao nhiêu 
        phần trăm trong chế độ ăn trung bình một ngày của một người. 
        Còn nhóm chai thứ 3, họ quy đổi lượng calorie của chai nước ngọt 
        ra số giờ mà người dùng phải tập thể dục để đốt cháy nó. 
        Kết quả, việc cung cấp thông tin liên quan đến hàm lượng calorie 
        đã khiến doanh số mặt hàng này giảm 40%, nhưng thông tin người dùng phải chạy bộ đến 
        50 phút để tiêu hao 250 calorie trong lon nước ngọt khiến doanh số bán hàng cho các bạn trẻ 
        giảm đến 50%."""
        
    few_shots = get_few_shot(chunk)
    res = query_generator(chunk, few_shot=few_shots)
    
    print(res)
    
    if  hasattr(res , 'content'):
        print(res.content)
        
    if  hasattr(res , 'text'):
            print(res.text)
            
            
            
