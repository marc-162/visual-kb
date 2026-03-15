"""
Named constants for similarity scoring and configuration.

ChromaDB is configured with cosine distance (hnsw:space = "cosine").
Cosine distance = 1 - cosine_similarity, so raw_similarity = 1 - distance.

The raw cosine similarity from Gemini Embedding 2 clusters in a
narrow band that depends on modality:
  - Text-to-Image queries:  similarities typically fall in [0.2, 0.4]
  - Image-to-Image queries: similarities typically fall in [0.6, 1.0]

We linearly rescale those bands to a [0, 1] display score so that a
"50 %" threshold is intuitive for end-users.

Formula:  display_score = clamp((raw_sim - LOW) / (HIGH - LOW), 0, 1)
"""

# --- Distance / similarity ---------------------------------------------------
# ChromaDB distance metric (configured on the collection).
CHROMA_DISTANCE_METRIC = "cosine"

# Raw-similarity rescaling bands (cosine similarity = 1 - cosine distance).
TEXT_TO_IMAGE_SIM_LOW = 0.2
TEXT_TO_IMAGE_SIM_HIGH = 0.4

IMAGE_TO_IMAGE_SIM_LOW = 0.6
IMAGE_TO_IMAGE_SIM_HIGH = 1.0

# Minimum display-score (after rescaling) to include in results.
MIN_DISPLAY_SCORE = 0.5

# --- Embedding model ----------------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 3072
EMBEDDING_TASK_TYPE = "SEMANTIC_SIMILARITY"

# --- Diagnosis model ----------------------------------------------------------
DIAGNOSIS_MODEL = "gemini-3.1-flash-lite-preview"
DIAGNOSIS_MAX_IMAGES = 3

# --- Paths --------------------------------------------------------------------
CHROMA_DB_DIR = "chroma_db"
SAMPLE_PHOTOS_DIR = "sample_photos"
COLLECTION_NAME = "visual_kb"
