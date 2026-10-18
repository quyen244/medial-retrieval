import os 
from pathlib import Path
import json 

ROOT = Path(__file__).parent.parent
SAVE_DIR = ROOT / 'data' / 'testset' / 'chunking_domain_sample.json'
if not SAVE_DIR.exists():
    os.makedirs(ROOT / 'data' / 'testset' , exist_ok=True)
    

def main():
    DATA_DIR = ROOT / 'data' / 'raw_documents'
    res = []
    
    for domain in os.listdir(DATA_DIR):
        if domain == '_failures':
            continue
            
        domain_path = DATA_DIR / domain 
        list_samples = sorted(os.listdir(domain_path), 
                            key = lambda x : int(x.split('.json')[0]))
        src_first_sample = domain_path / str(list_samples[0])
       
        with open(src_first_sample , 'r' , encoding='utf-8') as f:
            first_sample = json.load(f)
            
        res.append(first_sample)
        
    with open(SAVE_DIR , 'w' , encoding='utf-8') as f:
        f.write(json.dumps(res, indent=4, ensure_ascii=False))
        
if __name__ == "__main__":
    main()
        


