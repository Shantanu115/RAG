
RAG Multi-Agent Q&A Assistant

This is a Flask-based multi-agent question-answering application. It intelligently routes queries to one of three specialized agents based on keyword detection:

-  Calculator Agent – Triggered by queries containing the word "calculate".
-  Dictionary Agent – Triggered by queries containing the word "define".
-  RAG Agent – Handles all other queries using a Retrieval-Augmented Generation (RAG) pipeline.

 Usage Tip:
To use the calculator, include the word "calculate" in your query.
To get a definition, include the word "define" in your query.
All other queries will be handled by the RAG agent.

 Architecture Overview

                +----------------+
                |    User UI     |
                +--------+-------+
                         |
                         v
                  [ Flask Server ]
                         |
         +---------------+--------------+
         |               |              |
         v               v              v
    [ Calculator ]   [ Dictionary ]   [ RAG Module ]
                         |               |
                         |               v
                         |     - Loads and chunks `.txt` files
                         |     - Encodes chunks with SentenceTransformer
                         |     - Retrieves top-k context using FAISS
                         |     - Sends context + query to DeepSeek LLM
                         |
                   External API: DictionaryAPI.dev
                   External API: OpenRouter.ai (LLM)

this model is Rag is trained for Faq and product specs of Samsung Galaxy S23, iPhone 15 Max, Tesla Model 3, SONYWH1000XM5 and DellXPS13 
 Key Design Choices

- Multi-Agent Routing: Simple rule-based logic to dispatch queries to relevant agents.
- Embedding-Based Retrieval: Uses sentence-transformers/paraphrase-MiniLM-L3-v2 to vectorize document chunks.
- Fast Similarity Search: Powered by FAISS for nearest-neighbor search over document vectors.
- LLM-Powered QA: Uses Deep Seek Prover LLM through Open Router for coherent and context-aware answers.
- Seamless UX: Lightweight HTML interface built using Flask.

Folder Structure

.
├── app.py                 # Main Flask application
├── RAG_assignment/        # Folder containing `.txt` documents for RAG
│   └── *.txt              # Raw text documents

 How to Run

 Prerequisites

- Python 3.8+
- Install dependencies:

pip install flask faiss-cpu sentence-transformers langchain langchain-openai requests

 Set API Key

You must set your OpenRouter (https://openrouter.ai/) API key:

export OPENROUTER_API_KEY="your_api_key_here"

Or edit the following line in app.py:

os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

 Run the App

python app.py

Then open your browser and go to: http://localhost:10000

 Notes

- Place `.txt` files inside the RAG_assignment/ folder for the RAG agent to access.
- Only keyword-based routing is implemented — no NLP-based intent detection.
- You can customize chunk size, top-k retrieval, and model in the config variables at the top of app.py.


