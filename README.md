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

* **Role-Based Disaggregated Inference**
    * Control-plane / data-plane separation for LLM serving 






# Monolithic (Coupled) Inference
### 1. Monolithic (A100) — Llama-3-8B-Instruct
| Context | Concurrency | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | P95 (ms) | KV Util (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **512** | 8 | 501 | 17.9 | 345 | 502 | 1.44 |
| **512** | 32 | 1333 | 28.9 | 742 | 1794 | 5.67 |
| **512** | 64 | 2829 | 42.1 | 882 | 3527 | 11.34 |
| **512** | 128 | 4004 | 58.9 | 902 | 10640 | 17.72 |
| **512** | 256 | 11315 | 56.5 | 976 | 20874 | 18.05 |

Throughput improves with concurrency up to moderate batch sizes, after which it saturates. This saturation coincides with steep increases in TTFT and P95 latency, indicating a transition into a queueing- and memory-pressure-dominated regime.

# Phase-Disaggregated Inference
### 1. KV-cache transfer between prefill and decode stages (A100 clusters) — Llama-3-8B-Instruct 
| Context | Concurrency | Throughput (tok/s) | TPOT (ms/tok) | IPC Transfer (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **128** | 1 | 16.93 | 59.06 | 11.45 |
| **128** | 2 | 28.20 | 70.91 | 7.28 |
| **128** | 4 | 21.86 | 182.48 | 20.68 |
| **128** | 8 | 22.25 | 358.80 | 46.64 |
| **512** | 1 | 19.09 | 52.39 | 19.63 |
| **512** | 2 | 21.42 | 92.89 | 23.70 |
| **512** | 4 | 19.25 | 207.64 | 51.68 |
| **512** | 8 | 21.16 | 377.20 | 108.21 |

Throughput remains largely flat across concurrency, while TPOT and IPC transfer increase sharply with load and context size, indicating that inter-stage communication dominates system cost under scaling.

# Role-Based Disaggregated Inference
Admission-controlled routing with circuit breaking separates request intake from GPU execution, enforcing backpressure via semaphore and GPU admission gates to shed load under burst concurrency.

| Total Requests | Successful (200) | Rejected (503) | Throughput (req/s) |
|---------------:|-----------------:|---------------:|-------------------:|
| 150            | 65               | 85              | 44.7              |

Bursty high-concurrency inference workloads can cause GPU queues to grow unbounded, resulting in severe TTFT inflation, tail-latency amplification (P95), and starvation of long-running requests. 

The hybrid role-based architecture mitigates this by enforcing admission control and circuit breaking, converting overload into early rejection rather than allowing cascading latency collapse. 

This design prioritizes predictable latency and bounded GPU execution over unrestricted throughput, enabling stable serving behavior under sustained overload conditions.

