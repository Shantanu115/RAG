
import os
import re
import faiss
import numpy as np
import re
import requests
from flask import Flask, render_template_string, request
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


DOCS_FOLDER =os.path.join(os.path.dirname(__file__), "RAG_assignment")
CHUNK_SIZE = 100
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TOP_K = 3

def load_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

class VectorIndex:
    def __init__(self, docs_folder):
        self.chunks, self.chunk_origins = [], []
        documents = load_documents(docs_folder)
        for fname, text in documents.items():
            for chunk in chunk_text(text):
                self.chunks.append(chunk)
                self.chunk_origins.append(fname)
        self.model = SentenceTransformer(MODEL_NAME)
        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=False
        )

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, top_k=TOP_K):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(np.array(query_vec), k=top_k)
        return [self.chunks[i] for i in indices[0]]

def calculate_expression(query):
    expression = re.sub(r'[^0-9+\-*/(). ]', '', query)
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"

def define_word(query):
    word = re.sub(r'[^a-zA-Z0-9 ]', '', query.replace("define", "")).strip().lower()
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        

        definition = data[0]['meanings'][0]['definitions'][0]['definition']
        return f"{word.capitalize()}: {definition}"
    
    except requests.exceptions.HTTPError:
        return f"No definition found for '{word}'."
    except Exception as e:
        return f"Error: {str(e)}"


os.environ["OPENROUTER_API_KEY"] = "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"

class DeepSeekLLM:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-prover-v2:free",
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=os.environ["OPENROUTER_API_KEY"]
        )

    def generate_answer(self, query: str, context: str) -> str:
        """
        Sends a combined prompt (context + query) to DeepSeek and returns the answer text.
        """
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        return resp.content


class QAAgent:
    def __init__(self):
        self.index = VectorIndex(DOCS_FOLDER)
        self.llm = DeepSeekLLM()

    def handle_query(self, query):
        logs = [f"Query: {query}"]
        if "calculate" in query.lower():
            logs.append("→ Routed to Calculator")
            result = calculate_expression(query)
            return "Calculator", logs, [], result
        elif "define" in query.lower():
            logs.append("→ Routed to Dictionary")
            result = define_word(query)
            return "Dictionary", logs, [], result
        else:
            logs.append("→ Routed to RAG")
            snippets = self.index.retrieve(query)
            for i, s in enumerate(snippets, 1):
                logs.append(f"Context {i}: {s[:60]}...")
            context = " ".join(snippets)
            result = self.llm.generate_answer(query, context)
            return "RAG ", logs, snippets, result


app = Flask(__name__)
agent = QAAgent()

HTML_TEMPLATE = '''
<!DOCTYPE html><html><head><title>RAG QA Assistant</title>
<style>body{font-family:sans-serif;padding:2em;}input,button{padding:0.5em;}textarea{width:100%;height:100px;}</style>
</head><body>
<h2>RAG Multi-Agent Q&A Assistant</h2>
<form method="post">
<input name="query" type="text" size="60" placeholder="Ask something..." value="{{ query or '' }}" required>
<button type="submit">Submit</button>
</form>

{% if branch %}
<h3>Branch Used: {{ branch }}</h3>
<h4>Logs:</h4><ul>{% for log in logs %}<li>{{ log }}</li>{% endfor %}</ul>

{% if snippets %}
<h4>Context Snippets:</h4><ul>{% for s in snippets %}<li>{{ s }}</li>{% endfor %}</ul>
{% endif %}
<h4>Answer:</h4><p>{{ result }}</p>
{% endif %}
</body></html>
'''

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "")
        branch, logs, snippets, result = agent.handle_query(query)
        return render_template_string(HTML_TEMPLATE, query=query, branch=branch, logs=logs, snippets=snippets, result=result)
    return render_template_string(HTML_TEMPLATE)
import psutil


if __name__ == "__main__":
    app.run(debug=True)
