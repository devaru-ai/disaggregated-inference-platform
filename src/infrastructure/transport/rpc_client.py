import torch
import logging
import json
import requests
from .kv_contract import KVCacheMetadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Transport-RPC")

class RPCClient:
    def __init__(self, node_id):
        self.node_id = node_id

    def send_kv_cache(self, target_url: str, request_id: str, tensor: torch.Tensor):
        # 1. Simulate writing the heavy tensor to a high-speed shared memory store (Data Plane)
        shm_key = f"shm_kv_{request_id}"
        
        # 2. Cast the PyTorch size object to a standard Python integer
        try:
            seq_len_int = int(tensor.shape)
        except IndexError:
            # Fallback for unexpected 1D tensor layouts
            seq_len_int = int(tensor.numel())

        # 3. Build the explicit control-plane contract
        metadata = KVCacheMetadata(
            request_id=request_id,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            block_size=16, 
            num_tokens=seq_len_int,
            shared_memory_pointer=shm_key
        )
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Convert Pydantic model to dictionary safely (V2 vs V1 compatibility)
        payload = metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict()
        
        try:
            # If control plane routing takes > 500ms, the network is fundamentally bottlenecked.
            response = requests.post(
                f"{target_url}/receive_kv", 
                json=payload, 
                headers=headers,
                timeout=0.5 
            )
            
            if response.status_code == 200:
                logger.info(json.dumps({"event": "KV_METADATA_TRANSFER_SUCCESS", "target": target_url, "request_id": request_id}))
                return True
            else:
                logger.error(json.dumps({"event": "KV_METADATA_TRANSFER_FAILED", "status_code": response.status_code}))
                return False
                
        except requests.exceptions.Timeout:
            logger.error("KV Transfer Timeout. Network budget exceeded.")
            return False
        except Exception as e:
            logger.error(f"KV Transfer Error: {str(e)}")
            return False