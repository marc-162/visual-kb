# Visual KB — Cross-Modal Visual Knowledge Base

A multimodal semantic search tool powered by **Gemini Embedding 2** (`gemini-embedding-2-preview`). Search a local photo library using natural language queries or by uploading a reference image — no captions, no tags, no preprocessing required.

---

## What It Does

Visual KB embeds images and text into the same high-dimensional vector space using Google's Gemini Embedding 2 model. This enables:

- **Text → Image search** — type a natural-language query and find visually matching photos.
- **Image → Image search** — upload a photo and find structurally similar images.
- **AI Diagnosis** — Gemini Flash analyses matches against your query and produces a short expert summary.
- **Private local storage** — all vectors persist locally via ChromaDB.

## Tech Stack

| Component        | Technology                                                     |
| ---------------- | -------------------------------------------------------------- |
| Embeddings       | Gemini Embedding 2 (`gemini-embedding-2-preview`) via `google-genai` |
| Vector Database  | ChromaDB (cosine distance, local persistent storage)           |
| API Backend      | FastAPI + Uvicorn                                              |
| Streamlit UI     | Streamlit (glassmorphism dark theme)                           |
| HTML Frontend    | Vanilla HTML/CSS/JS (served by the API at `/static`)           |
| Image Processing | Pillow                                                         |

## Project Structure

```
visual-kb/
├── src/
│   ├── __init__.py
│   ├── api.py           # FastAPI backend
│   ├── app.py           # Streamlit frontend
│   ├── constants.py     # All named constants & thresholds
│   ├── embedder.py      # Gemini embedding helpers
│   ├── utils.py         # File-system utilities
│   └── vector_store.py  # ChromaDB wrapper
├── static/
│   └── index.html       # Standalone HTML frontend
├── sample_photos/       # Demo images (add your own)
├── .env.example         # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/<your-username>/visual-kb.git
cd visual-kb
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and replace "your_api_key_here" with your Gemini API key
# Get a free key at https://aistudio.google.com/
```

### 3. Add sample photos

Place images (`.jpg`, `.png`, `.webp`) into the `sample_photos/` directory.

## Running

### FastAPI backend (serves the HTML frontend)

```bash
python -m src.api
# → http://localhost:8000
# → HTML UI at http://localhost:8000/static/index.html
```

### Streamlit UI (alternative frontend)

```bash
streamlit run src/app.py
# → http://localhost:8501
```

## Environment Variables

| Variable         | Required | Description                          |
| ---------------- | -------- | ------------------------------------ |
| `GEMINI_API_KEY` | Yes      | Google Gemini API key from AI Studio |

## API Endpoints

| Method | Path               | Description                            |
| ------ | ------------------ | -------------------------------------- |
| GET    | `/status`          | Library status (count, model, dims)    |
| POST   | `/index`           | Wipe and re-index `sample_photos/`     |
| POST   | `/search/text`     | Text-to-image search + AI diagnosis    |
| POST   | `/search/image`    | Image-to-image search + AI diagnosis   |
| GET    | `/photo/{filename}`| Serve a photo from the library         |
| POST   | `/add`             | Upload & index new photos              |

### Example API calls

```bash
# Check status
curl http://localhost:8000/status

# Re-index photos
curl -X POST http://localhost:8000/index

# Text search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "ice buildup on compressor", "n_results": 3}'

# Image search
curl -X POST http://localhost:8000/search/image \
  -F "file=@photo.jpg"

# Upload new photos
curl -X POST http://localhost:8000/add \
  -F "files=@new_photo.jpg"
```

## License

MIT
