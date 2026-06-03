import time
import requests
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Heartbeat-Worker")

class EngineHeartbeat:
    def __init__(self, gateway_url, node_id, local_vllm_url="http://localhost:8000"):
        self.gateway_url = gateway_url
        self.node_id = node_id
        self.local_vllm_url = local_vllm_url

    def get_queue_depth(self):
        """Pulls actual scheduler pressure from vLLM's prometheus metrics on the A100."""
        try:
            resp = requests.get(f"{self.local_vllm_url}/metrics", timeout=0.2)
            if resp.status_code == 200:
                # Parse the exact number of running requests inside the vLLM engine
                for line in resp.text.split('\n'):
                    if line.startswith('vllm:num_requests_running'):
                        return int(float(line.split()))
            return 0
        except:
            return -1 # Indicates engine is dead or unreachable

    def send_pulse(self):
        active_seqs = self.get_queue_depth()
        
        payload = {
            "node_id": self.node_id, 
            "active_sequences": active_seqs,
            "timestamp": time.time()
        }
        try:
            requests.post(f"{self.gateway_url}/heartbeat", json=payload, timeout=0.5)
        except requests.exceptions.RequestException:
            pass # Suppress network spam if the CPU gateway is temporarily down

    def start(self):
        logger.info(f"Starting A100 heartbeat monitor -> {self.gateway_url}")
        while True:
            self.send_pulse()
            time.sleep(1.0) # Pulse every second

if __name__ == "__main__":
    TARGET_GATEWAY = os.getenv("GATEWAY_URL", "http://127.0.0.1:8080")
    WORKER_ID = os.getenv("NODE_ID", "Worker_A100_Core")
    
    LOCAL_VLLM = os.getenv("LOCAL_VLLM_PORT", "http://localhost:8000")
    
    monitor = EngineHeartbeat(TARGET_GATEWAY, WORKER_ID, local_vllm_url=LOCAL_VLLM)
    monitor.start()