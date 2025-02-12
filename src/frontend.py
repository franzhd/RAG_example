from flask import Flask, request, redirect, url_for, flash, render_template_string, session
import os
from embedding_node import run_indexing
import requests  # added import for checking URL accessibility
import sqlite3  # already imported in this file's context if needed
import threading  # added import for background processing
from qa_node import run_qa  # now importing only run_qa
import uuid  # to create a unique session id
import json  # to save embeddings as JSON strings

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

# Paths configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LINKS_FILE = os.path.join(BASE_DIR, "data", "links", "example_links.txt")
INDEX_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "index.db")
#EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "embedding_model", "UAE-large-V1-quant.onnx")
EMBEDDING_MODEL_PATH = "WhereIsAI/UAE-Large-V1"
LLM_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def check_url_accessible(url: str) -> bool:
    try:
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except Exception:
        return False

def get_folder_tree(folder: str) -> str:
    tree_lines = []
    for root, dirs, files in os.walk(folder):
        level = root.replace(folder, '').count(os.sep)
        indent = " " * 4 * level
        tree_lines.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for file in files:
            tree_lines.append(f"{subindent}{file}")
    return "\n".join(tree_lines)

@app.before_request
def assign_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

def ensure_chat_history_table():
    conn = sqlite3.connect(INDEX_OUTPUT_FILE)
    cursor = conn.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_answer TEXT,
            user_embedding TEXT,
            bot_embedding TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

@app.route("/", methods=["GET"])
def index():
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        links_content = f.read()
    # Updated HTML with navigation links to swap interfaces.
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Embedding Indexer</title>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
      </head>
      <body>
        <nav>
          <div class="nav-wrapper teal">
            <ul id="nav-mobile" class="left">
              <li><a href="/">Indexer</a></li>
            </ul>
            <ul id="nav-mobile" class="right">
              <li><a href="/chat">Chatbot</a></li>
            </ul>
            <div class="center"><span class="brand-logo">Embedding Indexer</span></div>
          </div>
        </nav>
        <div class="container" style="margin-top: 20px;">
          <div class="row">
            <!-- Form for updating the links list -->
            <div class="col s12 m6">
              <div class="card">
                <div class="card-content">
                  <span class="card-title">Edit Links</span>
                  <form action="/update" method="post">
                    <div class="input-field">
                      <textarea name="links" class="materialize-textarea">{{ links_content }}</textarea>
                      <label for="links">Links</label>
                    </div>
                    <button type="submit" class="btn waves-effect waves-light teal">Update Links</button>
                  </form>
                </div>
              </div>
            </div>
            <!-- Form for running the indexing process -->
            <div class="col s12 m6">
              <div class="card">
                <div class="card-content">
                  <span class="card-title">Run Indexing</span>
                  <form action="/run-indexing" method="post">
                    <button type="submit" class="btn waves-effect waves-light teal">Run Indexing</button>
                  </form>
                </div>
              </div>
            </div>
          </div>
          <!-- Existing file upload form -->
          <div class="row">
            <div class="col s12">
              <div class="card">
                <div class="card-content">
                  <span class="card-title">Upload Files</span>
                  <form action="/upload" method="post" enctype="multipart/form-data">
                    <div class="file-field input-field">
                      <div class="btn teal">
                        <span>File</span>
                        <input type="file" name="files" multiple>
                      </div>
                      <div class="file-path-wrapper">
                        <input class="file-path validate" type="text">
                      </div>
                    </div>
                    <button type="submit" class="btn waves-effect waves-light teal">Upload Files</button>
                  </form>
                </div>
              </div>
            </div>
          </div>
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              <div class="row">
                <div class="col s12">
                  {% for message in messages %}
                    <div class="card-panel teal lighten-4">{{ message }}</div>
                  {% endfor %}
                </div>
              </div>
            {% endif %}
          {% endwith %}
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
      </body>
    </html>
    """
    return render_template_string(html, links_content=links_content)

@app.route("/update", methods=["POST"])
def update():
    new_links = request.form.get("links")
    # Split links by newline and remove surrounding whitespace
    links_list = [line.strip() for line in new_links.splitlines() if line.strip()]
    with open(LINKS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(links_list))
    flash("Links updated successfully.")
    return redirect(url_for("index"))

@app.route("/run-indexing", methods=["POST"])
def run_indexing_route():
    # Run indexing in a background thread
    def background_indexing():
        run_indexing(os.path.dirname(LINKS_FILE), INDEX_OUTPUT_FILE, EMBEDDING_MODEL_PATH)
    thread = threading.Thread(target=background_indexing)
    thread.start()
    # Render a progress page with a spinner and auto-redirect to the status page
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Indexing In Progress</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
        <style>
          .spinner {
            margin: 50px auto;
            width: 50px;
            height: 50px;
            border: 5px solid #ccc;
            border-top-color: #1abc9c;
            border-radius: 100%;
            animation: spin 1s infinite linear;
          }
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        </style>
      </head>
      <body>
        <nav>
          <div class="nav-wrapper teal">
            <a href="#" class="brand-logo center">Indexing Status</a>
          </div>
        </nav>
        <div class="container center-align" style="margin-top:50px;">
          <h5>Indexing process started. Please wait...</h5>
          <div class="spinner"></div>
          <p>This may take a few moments.</p>
        </div>
        <script>
          // Redirect to status page after 5 seconds (adjust delay as needed)
          setTimeout(function(){
             window.location.href = "/status";
          }, 5000);
        </script>
      </body>
    </html>
    """
    return html

@app.route("/upload", methods=["POST"])
def upload():
    if 'files' not in request.files:
        flash("No file part in the request")
        return redirect(url_for("index"))
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash("No files selected")
        return redirect(url_for("index"))
    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
    flash(f"Uploaded {len(files)} file(s) successfully.")
    return redirect(url_for("index"))

@app.route("/status", methods=["GET"])
def status():
    # Retrieve indexed websites from the SQLite DB
    indexed = []
    try:
        conn = sqlite3.connect(INDEX_OUTPUT_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT source_type, source FROM embeddings")
        indexed = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error reading index: {e}")
    
    # Build file structure for the data/uploaded folder only
    tree = get_folder_tree(UPLOAD_FOLDER)
    
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Index Status</title>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
      </head>
      <body>
        <nav>
          <div class="nav-wrapper teal">
            <a href="#" class="brand-logo center">Index Status</a>
          </div>
        </nav>
        <div class="container" style="margin-top:20px;">
          <h5>Indexed Websites</h5>
          <ul class="collection">
            {% for typ, src in indexed %}
              <li class="collection-item">{{ typ }}: {{ src }}</li>
            {% else %}
              <li class="collection-item">No indexed websites found.</li>
            {% endfor %}
          </ul>
          <h5>Uploaded Files Tree Structure</h5>
          <pre>{{ tree }}</pre>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
      </body>
    </html>
    """
    return render_template_string(html, indexed=indexed, tree=tree)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    ensure_chat_history_table()
    if request.method == "POST":
        query = request.form.get("query", "")
        if not query:
            flash("Please enter a query.")
            return redirect(url_for("chat"))
        
        # Use run_qa to generate an answer.
        answer = run_qa(query, INDEX_OUTPUT_FILE, EMBEDDING_MODEL_PATH, LLM_MODEL_PATH)
        # Record chat history
        from embedding_model import EmbeddingModel
        emb_model = EmbeddingModel(model_path=EMBEDDING_MODEL_PATH)
        user_embed = emb_model.embed_text(query)
        bot_embed = emb_model.embed_text(answer)
        conn = sqlite3.connect(INDEX_OUTPUT_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (session_id, user_message, bot_answer, user_embedding, bot_embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session["session_id"],
            query,
            answer,
            json.dumps(user_embed),
            json.dumps(bot_embed)
        ))
        conn.commit()
        conn.close()
        return render_template_string("""
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
            <title>Chatbot Response</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
          </head>
          <body>
            <nav>
              <div class="nav-wrapper teal">
                <ul id="nav-mobile" class="left">
                  <li><a href="/">Indexer</a></li>
                </ul>
                <ul id="nav-mobile" class="right">
                  <li><a href="/chat">Chatbot</a></li>
                </ul>
                <div class="center"><span class="brand-logo">Chatbot Interface</span></div>
              </div>
            </nav>
            <div class="container" style="margin-top:50px;">
              <h5>Chatbot Query:</h5>
              <p><strong>Question:</strong> {{ query }}</p>
              <h5>Answer:</h5>
              <p>{{ answer }}</p>
              <a href="/chat" class="btn waves-effect waves-light">Back</a>
            </div>
          </body>
        </html>
        """, query=query, answer=answer)
    else:
        return render_template_string("""
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
            <title>Chat with Chatbot</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
          </head>
          <body>
            <nav>
              <div class="nav-wrapper teal">
                <ul id="nav-mobile" class="left">
                  <li><a href="/">Indexer</a></li>
                </ul>
                <ul id="nav-mobile" class="right">
                  <li><a href="/chat">Chatbot</a></li>
                </ul>
                <div class="center"><span class="brand-logo">Chatbot Interface</span></div>
              </div>
            </nav>
            <div class="container" style="margin-top:50px;">
              <h5>Chat with Chatbot</h5>
              <form method="post">
                <div class="input-field">
                  <textarea name="query" class="materialize-textarea" placeholder="Type your question here..."></textarea>
                  <label for="query">Your Question</label>
                </div>
                <button type="submit" class="btn waves-effect waves-light">Submit</button>
              </form>
            </div>
            <!-- JS to clear history on unload -->
            <script>
              window.addEventListener("beforeunload", function(){
                  navigator.sendBeacon("/delete-chat-history");
              });
            </script>
          </body>
        </html>
        """)

@app.route("/delete-chat-history", methods=["POST", "GET"])
def delete_chat_history():
    if "session_id" in session:
        conn = sqlite3.connect(INDEX_OUTPUT_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session["session_id"],))
        conn.commit()
        conn.close()
    return ("", 204)

if __name__ == "__main__":
    app.run(debug=True)