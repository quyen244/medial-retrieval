from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    MarkdownHeaderTextSplitter
)

headers = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


REGISTRY_SPLITTERS = {
    'markdown' : MarkdownHeaderTextSplitter,
    'recursive' : RecursiveCharacterTextSplitter
}

SAMPLE_1 = """

# AWS Lambda

AWS Lambda cho phép chạy code mà không cần quản lý server.

## Configuration

### Memory
Có thể cấu hình bộ nhớ từ 128 MB trở lên.
Với hai chiến lược cụ thể nhắm vào mục tiêu đích là nhóm người bệnh trẻ, hiện đại cần chủ động thời gian và nhóm người bệnh có điều kiện kinh tế nhưng không muốn xa quê hương mà vẫn được điều trị bệnh với chất lượng quốc tế ngay tại quê nhà trong sự chăm sóc và tình yêu thương của gia đình, người thân thì có thể nói đây là những mục tiêu mang đầy tính nhân văn và thiết thực trong thời đại công nghệ số của Bệnh viện Bạch Mai."


### Timeout
Timeout xác định thời gian tối đa một lần chạy.
Mặc định là 3 giây.

## Pricing

Bạn trả phí dựa trên số lần gọi và thời gian chạy.

"""
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len,
)


def run(name : str, **kwargs):
    print('=' * 50)
    print(f"{name} splitter")
    print('=' * 50)
    
    splitter = REGISTRY_SPLITTERS[name](
         **kwargs
    )
    sections = splitter.split_text(SAMPLE_1)
    
    chunks = recursive_splitter.split_documents(sections)

    for i, chunk in enumerate(chunks):
        print(f"\n{'=' * 20} CHUNK {i + 1} {'=' * 20}")
        print("Metadata:", chunk.metadata)
        print("Content:", chunk.page_content)
if __name__ == "__main__":
    run('markdown' , headers_to_split_on=headers,
            strip_headers=False)
    
  



