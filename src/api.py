"""
Visual KB — FastAPI backend.

Serves the HTML frontend and exposes endpoints for indexing,
text search, image search, and photo management.
"""

import logging
import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from src.constants import (
    DIAGNOSIS_MAX_IMAGES,
    DIAGNOSIS_MODEL,
    EMBEDDING_DIMENSIONS,
    IMAGE_TO_IMAGE_SIM_HIGH,
    IMAGE_TO_IMAGE_SIM_LOW,
    MIN_DISPLAY_SCORE,
    SAMPLE_PHOTOS_DIR,
    TEXT_TO_IMAGE_SIM_HIGH,
    TEXT_TO_IMAGE_SIM_LOW,
)
from src.embedder import client
from src.utils import get_sample_photos
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Visual KB API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_DIR = os.path.join(_PROJECT_ROOT, SAMPLE_PHOTOS_DIR)
os.makedirs(PHOTO_DIR, exist_ok=True)

vs = VectorStore()

# Serve the HTML frontend as a static file
_STATIC_DIR = os.path.join(_PROJECT_ROOT, "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    n_results: int = Field(default=3, ge=1, le=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine_display_score(distance: float, low: float, high: float) -> float:
    """Convert a ChromaDB cosine *distance* to a [0, 1] display score.

    ChromaDB cosine distance = 1 - cosine_similarity.
    raw_sim = 1 - distance  →  rescale [low, high] → [0, 1], clamped.
    """
    raw_sim = 1.0 - distance
    return max(0.0, min(1.0, (raw_sim - low) / (high - low)))


def _generate_diagnosis(query_text: str, result_files: list[str]) -> str:
    """Use Gemini Flash to produce a 2-3 sentence expert diagnosis."""
    if not client:
        return "AI Diagnosis unavailable: API key not configured."

    images: list[Image.Image] = []
    for filename in result_files[:DIAGNOSIS_MAX_IMAGES]:
        path = os.path.join(PHOTO_DIR, filename)
        if os.path.exists(path):
            try:
                images.append(Image.open(path))
            except Exception:
                pass

    prompt = (
        f"You are an expert technical inspector. Review the user's "
        f"current focus ('{query_text}') against these matching records "
        f"from the historical visual knowledge base (past jobs). "
        f"Provide a 2-3 sentence concise, insightful diagnosis. Connect "
        f"the visual patterns in these past cases to the current search, "
        f"highlighting what an inspector should focus on based on these "
        f"historical precedents."
    )

    try:
        response = client.models.generate_content(
            model=DIAGNOSIS_MODEL,
            contents=[prompt, *images] if images else [prompt, query_text],
        )
        return response.text.strip()
    except Exception as exc:
        logger.exception("Diagnosis generation failed")
        return f"AI Diagnosis error: {exc}"


def _parse_results(
    results: dict,
    sim_low: float,
    sim_high: float,
) -> list[dict]:
    """Filter and score raw ChromaDB results."""
    parsed: list[dict] = []
    if not results or not results["ids"] or not results["ids"][0]:
        return parsed

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for i, dist in enumerate(distances):
        score = _cosine_display_score(dist, sim_low, sim_high)
        if score >= MIN_DISPLAY_SCORE:
            parsed.append(
                {
                    "filename": os.path.basename(metadatas[i]["path"]),
                    "score": score,
                }
            )
    return parsed


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/status")
async def get_status():
    """Return current library status."""
    return {
        "photos_indexed": vs.count(),
        "dimensions": EMBEDDING_DIMENSIONS,
        "model": "Gemini Embedding 2",
        "status": "ready",
    }


@app.post("/index")
async def reindex():
    """Wipe and re-index all photos in the sample directory."""
    vs.clear_collection()
    photos = get_sample_photos(SAMPLE_PHOTOS_DIR)

    count = 0
    for i, photo_path in enumerate(photos):
        try:
            filename = os.path.basename(photo_path)
            vs.index_image(
                photo_path,
                f"img_{i}_{filename}",
                metadata={"path": photo_path},
            )
            count += 1
        except Exception:
            logger.exception("Error indexing %s", photo_path)

    return {"success": True, "count": count}


@app.post("/search/text")
async def search_text(search: SearchQuery):
    """Text-to-image semantic search with AI diagnosis."""
    try:
        results = vs.search_by_text(search.query, n_results=9)
        parsed = _parse_results(
            results, TEXT_TO_IMAGE_SIM_LOW, TEXT_TO_IMAGE_SIM_HIGH
        )
        filenames = [r["filename"] for r in parsed]
        diagnosis = _generate_diagnosis(search.query, filenames)
        return {"results": parsed, "diagnosis": diagnosis}

    except Exception:
        logger.exception("Text search failed")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    """Image-to-image search with AI diagnosis."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg"
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        results = vs.search_by_image(tmp_path, n_results=9)
        parsed = _parse_results(
            results, IMAGE_TO_IMAGE_SIM_LOW, IMAGE_TO_IMAGE_SIM_HIGH
        )
        filenames = [r["filename"] for r in parsed]
        diagnosis = _generate_diagnosis("visual search", filenames)
        return {"results": parsed, "diagnosis": diagnosis}

    except Exception:
        logger.exception("Image search failed")
        raise HTTPException(status_code=500, detail="Search failed")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/photo/{filename}")
async def get_photo(filename: str):
    """Serve an image file from the photo directory."""
    # Prevent path-traversal attacks
    if os.sep in filename or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = os.path.join(PHOTO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(path)


@app.post("/add")
async def add_photos(files: List[UploadFile] = File(...)):
    """Upload and index one or more images."""
    count = 0
    for upload in files:
        if not upload.filename:
            continue
        # Prevent path-traversal via crafted filename
        safe_name = os.path.basename(upload.filename)
        try:
            save_path = os.path.join(PHOTO_DIR, safe_name)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            count += 1
        except Exception:
            logger.exception("Error saving %s", safe_name)

    # Auto re-index after adding
    await reindex()
    return {"success": True, "count": count}


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
