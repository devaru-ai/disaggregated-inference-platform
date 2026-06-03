import hashlib
from typing import List

class KVAwareRouter:
    def __init__(self, workers: List[str]):
        if not workers:
            raise ValueError("KVAwareRouter requires at least one worker URL.")
        self.workers = workers

    def get_target_worker(self, system_prompt: str) -> str:
        """Deterministically hashes the system prefix to ensure KV cache locality."""
        h = int(hashlib.md5(system_prompt.encode()).hexdigest(), 16)
        return self.workers[h % len(self.workers)]

    def update_pool(self, workers: List[str]):
        self.workers = workers