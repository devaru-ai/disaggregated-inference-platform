import torch
from queue import SimpleQueue

class KVArena:
    def __init__(
        self,
        device,
        config,
        num_pages=2048,   # Default to a larger page count
        page_size=32,     # Standard page chunk size
        chunk_size=256,   # Pages per chunk for 7D indexing
        dtype=torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.page_size = page_size
        self.num_pages = num_pages
        self.chunk_size = chunk_size
        
        # Calculate chunks for 7D indexing
        self.num_chunks = (num_pages + chunk_size - 1) // chunk_size

        self.layers = config.num_hidden_layers
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 7D Memory layout: [chunks, layers, chunk_size, 2, kv_heads, page_size, head_dim]
        # This matches the indexing in p2p_engine.py
        self.memory = torch.zeros(
            (
                self.num_chunks,
                self.layers,
                self.chunk_size,
                2,
                self.kv_heads,
                self.page_size,
                self.head_dim,
            ),
            dtype=dtype,
            device=self.device,
        )

        self.free_page_ids = SimpleQueue()
        for i in range(num_pages):
            self.free_page_ids.put(i)

    @property
    def free_page_count(self): # Changed from free_pages
        return self.free_page_ids.qsize()

    def free_pages(self, page_ids): 
        for pid in page_ids:
            self.free_page_ids.put(pid)
            
    def allocate_pages(self, count):
        allocated = []
        for _ in range(count):
            if self.free_page_ids.empty():
                raise RuntimeError(f"KV arena exhausted: requested {count}, but queue is empty")
            allocated.append(self.free_page_ids.get())
        return allocated

    def release_pages(self, page_ids):
        for pid in page_ids:
            self.free_page_ids.put(pid)
    
    def get_page(self, page_id, layer_idx, kv_type):
        """
        Maps a 1D page_id to the 7D memory structure.
        kv_type: 'k' or 'v' -> maps to index 0 or 1 in the '2' dimension.
        """
        chunk_idx = page_id // self.chunk_size
        offset_in_chunk = page_id % self.chunk_size
        kv_idx = 0 if kv_type == 'k' else 1
        
        # Access the 7D tensor slice
        # Layout: [chunks, layers, chunk_size, 2, kv_heads, page_size, head_dim]
        return self.memory[chunk_idx, layer_idx, offset_in_chunk, kv_idx, :, :, :]