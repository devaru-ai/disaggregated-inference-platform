import time
import torch
from transformers import AutoModelForCausalLM

class DecodeWorker:
    def __init__(self, model_id, tokenizer, device, arena):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.arena = arena

        self.decode_stream = torch.cuda.Stream(device=self.device)

        print(f"[Decode {self.device}] Loading model...")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        
        self.eos_token_tensor = torch.tensor(
            [[self.model.config.eos_token_id]], 
            device=self.device
        )

    def reconstruct_hf_cache(self, descriptor):
        """Restores the HF-compatible KV cache from the 7D arena memory."""
        logical = []

        for layer_idx in range(descriptor.num_layers):
            k_pages = []
            v_pages = []

            for page_idx, page_id in enumerate(descriptor.page_ids):
                start = page_idx * self.arena.page_size
                end = min(start + self.arena.page_size, descriptor.seq_len)
                chunk = end - start

                chunk_idx = page_id // self.arena.chunk_size
                offset = page_id % self.arena.chunk_size

                # Index using the 7D structure: 
                # [chunk_idx, layer_idx, offset, kv_idx, heads, page_size, head_dim]
                k_pages.append(self.arena.memory[chunk_idx, layer_idx, offset, 0, :, :chunk, :])
                v_pages.append(self.arena.memory[chunk_idx, layer_idx, offset, 1, :, :chunk, :])

            # Concatenate chunks to restore [1, num_kv_heads, total_seq_len, head_dim]
            k = torch.cat(k_pages, dim=1).unsqueeze(0)
            v = torch.cat(v_pages, dim=1).unsqueeze(0)

            logical.append((k, v))

        return tuple(logical)

    @torch.inference_mode()
    def execute_decode(self, handle, last_token, max_new_tokens=128):
        t0 = time.perf_counter()

        with torch.cuda.stream(self.decode_stream):
            self.decode_stream.wait_event(handle.event)
            
            if hasattr(handle.descriptor, "validate"):
                handle.descriptor.validate()
            
            past_key_values = self.reconstruct_hf_cache(handle.descriptor)
            
            transfer_ms = (time.perf_counter() - t0) * 1000

            token = last_token.to(self.device)
            generated = []

            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(int(next_token))

                if torch.equal(next_token, self.eos_token_tensor):
                    break

                token = next_token
                past_key_values = outputs.past_key_values

        self.arena.release_pages(handle.descriptor.page_ids)

        return {
            "generated_tokens": len(generated),
            "text": self.tokenizer.decode(generated),
            "ipc_transfer_ms": transfer_ms,
        }