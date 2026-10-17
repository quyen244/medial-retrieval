from datasets import load_dataset
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / 'cache'

def get_dataset(name : str):
    """get dataset 
    name : Literal['corpus' , 'query']
    """
    ds = load_dataset(
        "AIGuruTinix/ViBioMIR", name,
        cache_dir=CACHE_DIR
    )
    
    return ds 
