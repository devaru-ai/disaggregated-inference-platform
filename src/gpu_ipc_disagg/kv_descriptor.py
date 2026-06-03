from dataclasses import dataclass
from typing import List
import torch

@dataclass(frozen=True)
class KVDescriptor:
    request_id: str
    page_ids: List[int]
    seq_len: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    def validate(self):
        """Ensures the descriptor state is consistent."""
        if not self.page_ids:
            raise ValueError("KVDescriptor: page_ids is empty.")
        if self.seq_len <= 0:
            raise ValueError(f"KVDescriptor: invalid seq_len {self.seq_len}")
