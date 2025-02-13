# src/qa_node.py

import json
import numpy as  np
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import subprocess

from embedding_model import EmbeddingModel

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_index(index_file: str) -> List[Dict]:
    """
    Load the precomputed index from a JSON file.
    """
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)
    return index

# New helper function to load the index from a SQLite DB.
def load_index_sqlite(db_file: str) -> List[Dict]:
    import sqlite3, json
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT source_type, source, content, embedding FROM embeddings")
    rows = cursor.fetchall()
    index = []
    for row in rows:
        source_type, source, content, embedding_str = row
        embedding = json.loads(embedding_str)
        if source_type == "url":
            index.append({"url": source, "content": content, "embedding": embedding})
        else:
            index.append({"file": source, "content": content, "embedding": embedding})
    conn.close()
    return index

def retrieve_relevant_documents(query_embedding, index: List[Dict], top_k: int = 2, min_similarity: float = 0.6) -> List[Dict]:
    """
    Compute cosine similarity for each document, discard those below min_similarity,
    and return the top_k matches.
    """
    scored_docs = []
    for doc in index:
        embedding_data = doc["embedding"]
        if isinstance(embedding_data[0], list):
            sims = [cosine_similarity(query_embedding, np.expand_dims(vec, axis=0)) for vec in embedding_data]
            sim = max(sims)
        else:
            sim = cosine_similarity(query_embedding, embedding_data)
        if sim >= min_similarity:
            scored_docs.append((sim, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

# New helper function to generate a summary of the context.
def summarize_context(context: str, llm, tokenizer, max_tokens: int = 3000) -> str:
    tokens = tokenizer.encode(context)
    if len(tokens) <= max_tokens:
        return context
    summary_prompt = (
        "The following is a long context extracted from several documents. "
        "Summarize the key points and important details concisely:\n\n" + context
    )
    summary = llm.chat(summary_prompt)
    return summary

# New helper to chunk text based on the model's maximum token length.
def chunk_text(text: str, tokenizer, max_tokens: int) -> list:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
    return chunks

# Updated QAClass to use the .chat method and chunk overly long prompts.
class QAClass:
    def __init__(self, index_file: str, embedding_model_path: str, llm_model_path: str, engine: str = "vllm"):
        self.index_file = index_file
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.engine = engine
        from embedding_model import EmbeddingModel
        self.embedding_model = EmbeddingModel(model_path=embedding_model_path)
        from llm_model import LLMModel
        # Instantiate local LLM instance.
        self.llm = LLMModel(llm_model_path)
        # Initialize conversation context.

    def answer(self, query: str) -> str:
        # Load the index from the SQLite DB.
        index = load_index_sqlite(self.index_file)
        # Generate embedding for the query.
        query_embedding = self.embedding_model.embed_text(query)
        # Retrieve relevant documents.
        relevant_docs = retrieve_relevant_documents(query_embedding, index)
        context = "\n".join([
            f"URL: {doc['url']}\nContent: {doc['content']}..."
            for doc in relevant_docs if "url" in doc
        ])
        # Obtain tokenizer from the embedding node and summarize context if too long.
        tokenizer = self.embedding_model.model.engine.get_tokenizer()
        context = summarize_context(context, self, tokenizer, max_tokens=8000)
        prompt = (
            f"Using the following summarized context, answer the question:\n\n"
            f"Summary:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        # Define a max prompt tokens limit (adjust based on model capabilities).
        MAX_PROMPT_TOKENS = 8000
        
        prompt_chunks = chunk_text(prompt, tokenizer, MAX_PROMPT_TOKENS)
        #prompt_chunks = prompt
        responses = []
        for chunk in prompt_chunks:
            # Pass chunk instead of the full list
            responses.append(self.llm.chat(chunk))
        flattened_responses = []
        for r in responses:
            if isinstance(r, list):
                flattened_responses.append(" ".join(r))
            elif isinstance(r, str):
                flattened_responses.append(r)
            else:
                continue
        response = " ".join(flattened_responses)
        return response

# New interactive chat function
def interactive_chat(index_file: str, embedding_model_path: str, llm_model_path: str, engine: str = "vllm"):
    print("Interactive Chat (type 'exit' to quit)")
    qa = QAClass(index_file, embedding_model_path, llm_model_path, engine)
    while True:
        query = input("You: ")
        if query.lower().strip() == "exit":
            print("Exiting chat.")
            break
        answer = qa.answer(query)
        print("AI:", answer)

# Modify run_qa to use QAClass.
def run_qa(query: str, index_file: str, embedding_model_path: str, llm_model_path: str, engine: str = "vllm"):
    qa = QAClass(index_file, embedding_model_path, llm_model_path, engine)
    return qa.answer(query)

# # Modify main block to use interactive chat.
# if __name__ == "__main__":
#     # Example usage:
#     index_file = "../data/index.json"               # Adjust if needed.
#     embedding_model_path = "../models/embedding_model"  # Adjust if needed.
#     llm_model_path = "../models/llm_model"            # Adjust if needed.
#     interactive_chat(index_file, embedding_model_path, llm_model_path, engine="vllm")
