import gradio as gr
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Load data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunking
def split_into_chunks(text, chunk_size=400):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

chunks = split_into_chunks(text)

# Embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embeddings = embedding_model.encode(chunks, show_progress_bar=True)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = [chunks[i] for i in indices[0]]
    return results

def ask(question):
    results = search(question)
    context = "\n\n".join(results)
    prompt = f"""أنت مساعد ذكي يجيب على الأسئلة بناءً على النصوص التاريخية المتوفرة فقط.

النصوص:
{context}

السؤال: {question}

أجب بدقة بناءً على المعلومات الموجودة في النصوص أعلاه فقط. إذا لم تجد إجابة واضحة، قل ذلك بصدق."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=3, placeholder="اسأل سؤالاً عن الكتابات...", label="السؤال"),
    outputs=gr.Textbox(label="الإجابة"),
    title="Islamic Conquest of Egypt",
)

demo.launch()