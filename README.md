# Islamic Egypt History — Arabic RAG System

An Arabic Retrieval-Augmented Generation (RAG) system that answers questions about the history of Islamic Egypt, grounded in a curated Arabic corpus. Large language models often hallucinate when asked about specialized Arabic historical topics — dates get invented, names get confused, and sources get fabricated. This project addresses that by retrieving relevant passages from a verified Arabic Wikipedia corpus on the Islamic conquest of Egypt and instructing the model to answer **only** from those passages, admitting honestly when the answer is not in the corpus.

**🔴 Live demo:** https://medo9090-islamic-egypt-history.hf.space/

## Architecture

```
data.txt (Arabic corpus)
   │
   ▼
Chunking ── whitespace word split, accumulated into ~400-character chunks
   │
   ▼
Embeddings ── sentence-transformers/paraphrase-multilingual-mpnet-base-v2
   │
   ▼
FAISS IndexFlatL2 ── exact L2 vector index, built in memory at startup
   │
   ▼
Retrieval ── query embedded with the same model, top-5 nearest chunks
   │
   ▼
Prompt construction ── Arabic instruction: answer ONLY from the retrieved
   │                   passages; if no clear answer exists, say so honestly
   ▼
Generation ── Groq API, llama-3.3-70b-versatile, temperature 0.3
   │
   ▼
Gradio UI ── question textbox in, answer textbox out
```

All steps run inside `app.py` (the deployed Hugging Face Space). `main.py` is the earlier development script used to prototype the pipeline (smaller chunks, top-3 retrieval) and runs the same core stages from the command line.

## Tech Stack

| Component | Choice |
|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers) |
| Vector store | FAISS (`IndexFlatL2`) |
| LLM | Llama 3.3 70B via Groq API |
| UI | Gradio |
| Language | Python |

## Installation

```bash
git clone https://github.com/medo832/islamic-egypt-history.git
cd islamic-egypt-history
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key (free at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_key_here
```

## Usage

```bash
python app.py
```

Open the local Gradio URL printed in the terminal, type a question in Arabic, and press Submit. The first launch downloads the embedding model and builds the FAISS index, which takes a minute or two.

## Example Questions (verified against the live demo)

These were tested on the deployed Space and answered correctly:

- **من هو القائد المسلم الذي فتح مصر؟** → عمرو بن العاص
- **متى سقطت الإسكندرية في يد المسلمين؟** → سنة 21هـ الموافقة لسنة 642م
- **متى سقطت مدينة الفرما وكم استمر حصارها؟** → 19 محرم 19هـ / 20 يناير 640م، بعد حصار شهر — the system even cited the variant account of Yaqut al-Hamawi (two months)
- **لماذا فضل الأقباط حكم المسلمين على حكم الروم؟** → a grounded answer covering Byzantine religious persecution, taxation, and Muslim tolerance

Out-of-scope questions are refused honestly rather than hallucinated (also verified):

- **من بنى الأهرامات؟** → "لا توجد إجابة واضحة في النصوص الموجودة..."
- **من هو رئيس مصر الحالي؟** → "لا يوجد معلومات عن رئيس مصر الحالي في النصوص..."

## Data Source

The corpus (`data.txt`) is compiled from the Arabic Wikipedia article on Islamic Egypt:
[مصر الإسلامية — ويكيبيديا](https://ar.wikipedia.org/wiki/مصر_الإسلامية), used under the
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Limitations

These are real failure modes observed while testing the live demo — not generic RAG caveats:

1. **Broad questions can miss obvious answers.** Asking «متى فتح المسلمون مصر؟» — the single most natural question for this corpus — returned "no clear answer found" twice, even though the corpus opens with the answer (640–642م). General phrasings embed far from the specific chunk containing the summary fact, so top-5 L2 retrieval pulls adjacent narrative chunks instead. Ironically, more specific questions (the fall of al-Farama or Alexandria) work better than the broadest one.

2. **Less famous proper names can fail completely.** Asking about أرمانوسة (daughter of al-Muqawqis, present in the corpus) returned "no information about Armanusa **or Amr ibn al-As** in the texts" — a doubly wrong answer, since both appear in the corpus. When retrieval misses, the model's honesty instruction makes it deny information that actually exists.

3. **Fixed-size chunking splits facts across boundaries.** Chunks are cut every ~400 characters with no respect for sentences or topics, so a date can end up separated from the event it belongs to, weakening retrieval for exact-fact questions.

4. **Single-article corpus.** Coverage is limited to the conquest-era narrative of one Wikipedia article; later periods of Islamic Egypt (Tulunid, Fatimid, Ayyubid, Mamluk) are thin or absent.

5. **No citations and no persistence.** Answers do not indicate which chunk they came from, and the FAISS index is rebuilt from scratch on every startup (no saved index), which slows cold starts on the free Hugging Face tier.

Planned improvements that follow directly from these failures: sentence-aware/semantic chunking, hybrid retrieval (BM25 + vectors) to catch proper names and broad phrasings, a persisted vector index, and chunk-level citations in answers.

## Author

**Mohamed Elkenawi** — [github.com/medo832](https://github.com/medo832)
