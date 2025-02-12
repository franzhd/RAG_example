# src/embedding_node.py
import os
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from embedding_model import EmbeddingModel
import sqlite3  # added import for SQLite
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add a global embedding model to reuse across indexing processes.
global_embedding_model = None

def fetch_web_content(url: str) -> str:
    """
    Fetch the webpage content from the given URL.
    Uses requests and BeautifulSoup to extract text.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove unwanted elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"Failed to fetch content from {url}: {e}")
        return ""

def read_links_from_folder(folder_path: str) -> List[str]:
    """
    Read all .txt files in the folder and extract URLs (one per line).
    """
    links = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                file_links = [line.strip() for line in f if line.strip()]
                links.extend(file_links)
    return links

def read_local_files(folder_path: str) -> List[Dict]:
    """
    Recursively read all files in the folder (excluding the 'links' subfolder) and return a list of dicts containing file paths and their content.
    """
    local_files = []
    for root, dirs, files in os.walk(folder_path):
        if os.path.basename(root) == "links":
            continue
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                local_files.append({"file": file_path, "content": content})
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
    return local_files

def create_index(links: List[str], embedding_model: EmbeddingModel) -> List[Dict]:
    """
    For each URL, fetch its content, generate an embedding, and record the metadata.
    """
    index = []
    for url in links:
        print(f"Processing URL: {url}")
        content = fetch_web_content(url)
        print(content[:200])  # Display the first 200 characters of the content
        if not content:
            break
        embedding = embedding_model.embed_text(content)
        index.append({
            "url": url,
            "content": content,
            "embedding": embedding
        })
    return index

def run_indexing(data_folder: str, index_output_file: str, embedding_model_path: str):
    global global_embedding_model
    # Read URLs and local files
    links = read_links_from_folder(data_folder)
    print(f"Found {len(links)} links in {data_folder}")
    
    # Reuse global embedding model if available.
    from embedding_model import EmbeddingModel
    if global_embedding_model is None:
        global_embedding_model = EmbeddingModel(model_path=embedding_model_path)
    embedding_model = global_embedding_model

    index = create_index(links, embedding_model)
    
    # Derive local files folder as the parent of the links folder if applicable
    local_files_folder = os.path.abspath(os.path.join(data_folder, os.pardir))
    local_files = read_local_files(local_files_folder)
    print(f"Found {len(local_files)} local files in {local_files_folder} (excluding 'links' subfolder)")
    
    for file_info in local_files:
        print(f"Processing file: {file_info['file']}")
        embedding = embedding_model.embed_text(file_info["content"])
        index.append({
            "file": file_info["file"],
            "content": file_info["content"],
            "embedding": embedding
        })
    
    # Replace JSON saving with SQLite saving policy
    conn = sqlite3.connect(index_output_file)  # using the index_output_file as the SQLite DB file
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            source TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    for entry in index:
        if "url" in entry:
            source_type = "url"
            source = entry["url"]
        else:
            source_type = "file"
            source = entry["file"]
        # Check if this source is already indexed
        cursor.execute("SELECT COUNT(*) FROM embeddings WHERE source = ?", (source,))
        if cursor.fetchone()[0] > 0:
            print(f"Skipping already indexed source: {source}")
            continue
        cursor.execute("""
            INSERT INTO embeddings (source_type, source, content, embedding)
            VALUES (?, ?, ?, ?)
        """, (source_type, source, entry["content"], json.dumps(entry["embedding"])))
    conn.commit()
    conn.close()
    print(f"Index saved to SQLite DB at {index_output_file}")

    # Optionally, do not deallocate the global model to allow reuse in future indexing runs.
    # ...existing code...
