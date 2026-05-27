# DIP: Disaggregated Inference Platform

Different AI workloads have distinct hardware requirements aligned with their serving characteristics. Model performance optimization co-designs model execution and serving infrastructure around production constraints such as tail-latency targets (p95/p99), concurrency profiles, request burstiness, sequence length distributions, and hardware limits, while maintaining predictable, cost-efficient inference at scale.

This project studies model serving performance under heterogeneous system constraints, focusing on how architecture and data movement strategies impact inference performance.

### Evaluated Architectures

* **Monolithic (Coupled) Inference**
    * Distributed A100 clusters  
    * Distributed H100 clusters  

* **Phase-Disaggregated Inference**
    * A100–H100 split execution (prefill and decode separation)  
    * KV-cache transfer between prefill and decode stages (H100 clusters)  
    * KV-cache transfer between prefill and decode stages (A100 clusters)  

* **Hybrid Compute**
    * CPU–GPU offloading architectures for memory-bound workloads  
