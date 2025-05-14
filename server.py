import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# ── Load .env and Environment Variables ─────────────────────────────────
load_dotenv()  # reads .env into os.environ

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.environ.get("INDEX_NAME", "index1")
PDF_PATH         = os.environ.get("PDF_PATH", "FAQ_Chariot_Toyota.pdf")

# ── Static assets & keywords ─────────────────────────────────────────────
MODEL_IMAGES = {
    "8fd15": "assets/8FD15.jpg",
    "8fd18": "assets/8FD18.jpeg",
    "8fd25": "assets/8FD25.jpg",
    "8fd30": "assets/8FD30.jpg",
}
FICHE_IMAGES = {
    "8fd15": "assets/fiche_8FD15.jpg",
    "8fd18": "assets/fiche_8FD18.jpeg",
    "8fd25": "assets/fiche_8FD25.jpg",
    "8fd30": "assets/fiche_8FD30.jpg",
}
IMAGE_KEYWORDS = ["image", "photo", "montrer", "affiche", "voir"]

def init_services():
    """Initialize OpenAI & Pinecone clients, create/populate index if needed."""
    openai = OpenAI(api_key=OPENAI_API_KEY)
    pc     = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            vector_type="dense",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()

    # Populate the index from the PDF on first run
    if stats.get("total_vector_count", 0) == 0:
        loader   = PyPDFLoader(PDF_PATH)
        docs     = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks   = splitter.split_documents(docs)

        upserts = []
        for i, doc in enumerate(chunks):
            emb = openai.embeddings.create(
                model="text-embedding-3-small",
                input=doc.page_content
            ).data[0].embedding

            meta = {
                "source": "FAQ_Chariot_Toyota.pdf",
                "page":   doc.metadata.get("page"),
                "chunk":  i,
                "text":   doc.page_content
            }
            upserts.append((f"doc-{i}", emb, meta))

        index.upsert(vectors=upserts)

    return openai, index

# ── Flask App Setup ───────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
openai, index = init_services()

@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data     = request.get_json()
    question = data.get('question', '')
    low      = question.lower()

    # 1) Image or fiche requests
    for code in MODEL_IMAGES:
        if code in low:
            if any(k in low for k in IMAGE_KEYWORDS):
                return jsonify({
                    'type': 'image',
                    'url':  MODEL_IMAGES[code],
                    'caption': f"Chariot élévateur {code.upper()}"
                })
            if 'fiche' in low:
                return jsonify({
                    'type': 'image',
                    'url':  FICHE_IMAGES[code],
                    'caption': f"Fiche technique {code.upper()}"
                })

    # 2) Standard RAG flow
    emb = openai.embeddings.create(
        model='text-embedding-3-small',
        input=question
    ).data[0].embedding

    resp = index.query(vector=emb, top_k=3, include_metadata=True)
    frags = [m.metadata.get('text','') for m in resp.matches]
    context = "\n\n".join(frags)

    chat = openai.chat.completions.create(
        model='o4-mini-2025-04-16',
        messages=[
            {'role':'system', 'content':'You are a helpful assistant.'},
            {'role':'user',   'content':f"Answer based only on the context below:\n\n{context}\n\nQuestion: {question}"}
        ]
    )
    answer = chat.choices[0].message.content
    return jsonify({ 'type':'text', 'answer': answer })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
