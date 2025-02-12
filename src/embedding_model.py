# src/embedding_model.py
# Removed langgraph dependency and implemented a custom LocalEmbeddingNode.
from vllm import LLM 

MAX_TOKENS = 512  # Adjust based on model token limit
TOKEN_OVERHEAD = 8  # Reserve extra tokens (adjust if needed)
EFFECTIVE_MAX = MAX_TOKENS - TOKEN_OVERHEAD

# New helper that uses the engine's tokenizer to compute token count and split text
### TODO fix problem with chunking the incoming text that make overflow the token length ###
def split_text_into_chunks(text: str, tokenizer, max_tokens: int = MAX_TOKENS):
    tokens = tokenizer.encode(text)
    # Change "<" to "<=" so that a prompt with exactly max_tokens is not split.
    if len(tokens) <= EFFECTIVE_MAX:
        return [text]
    chunks = []
    for i in range(0, len(tokens), EFFECTIVE_MAX):
        chunk_tokens = tokens[i:i+EFFECTIVE_MAX]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

class LocalEmbeddingNode:
    def __init__(self, model_path: str):
        self.engine = LLM(model_path, task="embed", trust_remote_code=True)
    
    def run(self, text: str):
        # Generate output using vllm's embed method
        outputs = self.engine.embed(text)
        return [output.outputs.embedding for output in outputs] 

class EmbeddingModel:
    def __init__(self, model_path: str):
        # Instantiate the local embedding node using the given model_path.
        self.model = LocalEmbeddingNode(model_path=model_path)

    def embed_text(self, text: str):
        """
        Generate embeddings for the given text by splitting based on token count.
        Returns a list of embeddings (one per chunk).
        """
        # Use the engine's tokenizer to split the text
        tokenizer = self.model.engine.get_tokenizer()
        chunks = split_text_into_chunks(text, tokenizer, max_tokens=MAX_TOKENS)
        embeddings = [self.model.run(chunk) for chunk in chunks]
        return embeddings
