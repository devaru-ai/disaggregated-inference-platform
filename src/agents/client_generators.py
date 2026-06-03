class CodingAgent:
    def build_payload(self, code_snippet: str, max_tokens: int = 512):
        return {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "prompt": f"Analyze and optimize the Big-O efficiency of this code:\n{code_snippet}",
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": True
        }

class ResearchAgent:
    def build_payload(self, query: str, context: str, max_tokens: int = 256):
        return {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "prompt": f"Synthesize this context:\n{context}\n\nQuery: {query}",
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": True
        }