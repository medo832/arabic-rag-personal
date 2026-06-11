import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()


# Load the data and open it

with open('data.txt', 'r',encoding= "utf-8") as f:
    text = f.read()

print(text[:100])  # Print the first 100 characters to check the data

# Split the text into chunks 

def split_into_chuncks(text, chunk_size= 200):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words: 
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space
        if current_length >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

    # Append any remaining words as a final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

chunks = split_into_chuncks(text)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0]}")  # Print the chunk to check the output


# Embeded model 

empedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Embedding the chunks... wait until finish")
embeddiings = empedding_model.encode(chunks, show_progress_bar=True)
print (f"Embeddings shape: {embeddiings.shape}")

# Store the embedding in a FAISS 

import faiss
import numpy as np

dimension = embeddiings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddiings))
print(f"Number of vectors in the index: {index.ntotal}")

def search(query, k=3):
    query_embedding = empedding_model.encode([query])
    distances, indicies = index.search(np.array(query_embedding), k)
    result = [chunks[i] for i in indicies[0]]
    return result

# Example query
query = "ماذا قال النص عن القمر؟"
results = search(query)

for i, result in enumerate(results):
    print(f"\nResult{i+1}:")
    print(result)


# Generate answer using Gimini
# Step 6: Generate answer using llama 

# Step 6: Generate answer using Groq
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(question):
    results = search(question)
    context = "\n\n".join(results)
    
    prompt = f"""أنت مساعد نفسي يفهم الكتابة الفلسفية العربية و يحلل شخصية الكاتب بكل ذكاء.
    
بناءً على هذه النصوص فقط:
{context}

أجب على هذا السؤال: {question}

إذا لم تجد إجابة واضحة في النصوص، قل ذلك بصدق."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )
    return response.choices[0].message.content

question = "ما هي مميزات شخصية الكاتب؟"
answer = ask(question)
print(answer)
