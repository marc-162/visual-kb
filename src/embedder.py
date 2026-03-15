import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from src.constants import EMBEDDING_MODEL, EMBEDDING_TASK_TYPE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------
load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")

client: genai.Client | None = None
if _api_key and _api_key != "your_api_key_here":
    client = genai.Client(api_key=_api_key)


def _require_client() -> genai.Client:
    """Return the initialised client or raise with a helpful message."""
    if client is None:
        raise ValueError(
            "GenAI Client not initialised. "
            "Set GEMINI_API_KEY in your .env file."
        )
    return client


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def embed_image(image_path: str) -> list[float]:
    """Generate an embedding vector for an image file."""
    c = _require_client()
    img = Image.open(image_path)
    result = c.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[img],
        config=types.EmbedContentConfig(task_type=EMBEDDING_TASK_TYPE),
    )
    return result.embeddings[0].values


def embed_text(query: str) -> list[float]:
    """Generate an embedding vector for a text query."""
    c = _require_client()
    result = c.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type=EMBEDDING_TASK_TYPE),
    )
    return result.embeddings[0].values


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing google-genai embedder…")
    if not _api_key:
        logger.error("GEMINI_API_KEY not found in .env")
    else:
        try:
            vec = embed_text("test query")
            logger.info("Text embedding dimension: %d", len(vec))
            logger.info("Successfully connected to Gemini Embedding 2.")
        except Exception as exc:
            logger.error("Error: %s", exc)
