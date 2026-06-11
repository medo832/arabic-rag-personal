Arabic RAG Personal

An Arabic Retrieval-Augmented Generation (RAG) system designed to analyze personal writings and answer questions based on the author’s texts.

Overview

Arabic RAG Personal is a lightweight NLP project that combines semantic search and large language models to retrieve relevant passages from Arabic texts and generate contextual answers.

The system is specialized for philosophical, reflective, and personal writings, allowing users to explore themes, personality traits, and ideas expressed by an author.

Features

* Arabic text processing
* Semantic chunking and retrieval
* Multilingual sentence embeddings
* FAISS vector search
* Context-aware question answering
* Personality-oriented analysis
* Groq + Llama 3.3 integration

Pipeline

1. Load Arabic text data
2. Split text into chunks
3. Generate embeddings using Sentence Transformers
4. Store vectors in FAISS
5. Retrieve top-k relevant chunks
6. Build contextual prompt
7. Generate answer using Llama 3.3

Tech Stack

* Python
* Sentence Transformers
* FAISS
* NumPy
* Groq API
* Llama 3.3 70B

Example Questions

* What are the author’s personality strengths?
* What does the writer think about fear?
* What did the text mention about the moon?
* What recurring themes appear in the writings?

Future Improvements

* PDF support
* Semantic chunking
* Persistent vector database
* Streamlit web interface
* Hybrid retrieval (BM25 + Vector Search)
* Citation-based answers
* Advanced personality profiling

Project Goal

To explore Arabic Retrieval-Augmented Generation systems and build intelligent tools capable of understanding and analyzing personal writings in Arabic.
