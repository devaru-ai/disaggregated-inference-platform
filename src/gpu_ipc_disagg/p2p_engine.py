import torch

class TransferHandle:
    def __init__(self, descriptor, event):
        self.descriptor = descriptor
        self.event = event


class P2PEngine:
    def __init__(self, src_device, dst_device, arena):
        self.src_device = torch.device(src_device)
        self.dst_device = torch.device(dst_device)
        self.arena = arena
        self.transfer_stream = torch.cuda.Stream(device=self.src_device)

    def transfer_paged_kv(self, past_key_values, descriptor):
        event = torch.cuda.Event()

        with torch.cuda.stream(self.transfer_stream):

            for layer_idx, (k, v) in enumerate(past_key_values):
                if k.dim() != 4:
                    raise ValueError(f"Unexpected KV shape: {k.shape}")

                if k.shape[0] != 1:
                    raise ValueError(
                        f"Only batch=1 supported, got batch={k.shape[0]}"
                    )

                k = k[0]  # [heads, seq_len, head_dim]
                v = v[0]

                num_heads, seq_len, head_dim = k.shape

                k_heads = num_heads  
                seq_len_int = seq_len

                for page_idx, page_id in enumerate(descriptor.page_ids):

                    start = page_idx * self.arena.page_size
                    end = min(start + self.arena.page_size, seq_len_int)

                    if start >= seq_len_int:
                        continue

                    chunk = end - start
                    if chunk <= 0:
                        continue

                    chunk_idx = page_id // self.arena.chunk_size
                    offset = page_id % self.arena.chunk_size

                    k_slice = k[:, start:end, :]
                    v_slice = v[:, start:end, :]

                    self.arena.memory[
                        chunk_idx,
                        layer_idx,
                        offset,
                        0,
                        :,
                        :chunk,
                        :,
                    ].copy_(k_slice, non_blocking=True)

                    self.arena.memory[
                        chunk_idx,
                        layer_idx,
                        offset,
                        1,
                        :,
                        :chunk,
                        :,
                    ].copy_(v_slice, non_blocking=True)

            event.record(self.transfer_stream)

        return TransferHandle(descriptor, event)