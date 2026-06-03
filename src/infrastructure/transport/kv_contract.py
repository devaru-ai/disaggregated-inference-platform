from pydantic import BaseModel
from typing import List

class KVCacheMetadata(BaseModel):
    request_id: str
    shape: List[int]
    dtype: str
    block_size: int
    num_tokens: int  
    compression_flag: str = "none"
    shared_memory_pointer: str