import time
import logging

logger = logging.getLogger("Backpressure")

class BackpressureController:

    def __init__(self, max_concurrent_seqs=64, heartbeat_ttl_sec=5.0):

        self.max = max_concurrent_seqs
        self.ttl = heartbeat_ttl_sec

        self.remote_active = 0

        self.local_reserved = 0

        self.last_heartbeat = 0.0

        # hysteresis thresholds
        self.open_threshold = max_concurrent_seqs
        self.close_threshold = int(max_concurrent_seqs * 0.85)

        self.is_open = False

    def update_remote_state(self, active_sequences: int):

        self.remote_active = active_sequences
        self.last_heartbeat = time.time()

        # hysteresis reset logic
        total = self.remote_active + self.local_reserved

        if self.is_open and total <= self.close_threshold:
            self.is_open = False

    def _stale(self) -> bool:
        return (time.time() - self.last_heartbeat) > self.ttl

    def try_acquire(self) -> bool:

        if self._stale():
            self.is_open = True
            logger.error("stale heartbeat -> open circuit")
            return False

        total = self.remote_active + self.local_reserved

        if total >= self.open_threshold:
            self.is_open = True
            logger.warning("over capacity -> open circuit")
            return False

        self.local_reserved += 1
        return True

    def release(self):

        if self.local_reserved > 0:
            self.local_reserved -= 1