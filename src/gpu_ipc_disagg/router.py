import itertools

class DisaggregatedRouter:
    def __init__(self, prefill_worker, decode_workers, p2p_engines):
        self.prefill_worker = prefill_worker
        self.routing_pool = list(zip(decode_workers, p2p_engines))
        self.route_iterator = itertools.cycle(self.routing_pool)

    def process_request(self, prompt_text: str, max_new_tokens: int = 128):

        decode_worker, target_p2p_engine = next(self.route_iterator)

        handle, last_token = self.prefill_worker.execute_prefill(
            prompt_text,
            p2p_engine=target_p2p_engine,
        )

        return decode_worker.execute_decode(
            handle=handle,
            last_token=last_token,
            max_new_tokens=max_new_tokens,
        )