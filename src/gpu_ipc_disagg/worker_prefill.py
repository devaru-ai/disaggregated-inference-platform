import math
import uuid
import torch
from transformers import AutoModelForCausalLM
from kv_descriptor import KVDescriptor


class PrefillWorker:
    def __init__(self, model_id, tokenizer, device, p2p_engine):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.base_p2p = p2p_engine
        self.compute_stream = torch.cuda.Stream(device=self.device)

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def execute_prefill(self, prompt_text, p2p_engine=None):
        engine = p2p_engine or self.base_p2p

        enc = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        seq_len = input_ids.shape[-1]

        pages_needed = math.ceil(seq_len / engine.arena.page_size)
        page_ids = engine.arena.allocate_pages(pages_needed)

        with torch.cuda.stream(self.compute_stream):
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )

        engine.transfer_stream.wait_stream(self.compute_stream)

        past_kv = outputs.past_key_values  # Tuple[layer] -> (K, V)

        if past_kv is None:
            raise RuntimeError("Model did not return past_key_values")

        num_layers = len(past_kv)

        k0, v0 = past_kv[0]

        # Expected shapes:
        # k0: [batch, num_heads, seq, head_dim]
        num_kv_heads = k0.shape[1]
        head_dim = k0.shape[-1]

        descriptor = KVDescriptor(
            request_id=str(uuid.uuid4()),
            page_ids=page_ids,
            seq_len=seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=k0.dtype,
        )

        handle = engine.transfer_paged_kv(
            past_kv,
            descriptor,
        )

        last_token = input_ids[:, -1:].cpu()

        return handle, last_token