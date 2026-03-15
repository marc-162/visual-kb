"""
Visual KB — Streamlit frontend.

A glassmorphism-styled UI for multimodal semantic search
powered by Gemini Embedding 2 and ChromaDB.
"""

import logging
import os
import tempfile

import streamlit as st
from PIL import Image

from src.constants import (
    EMBEDDING_DIMENSIONS,
    IMAGE_TO_IMAGE_SIM_HIGH,
    IMAGE_TO_IMAGE_SIM_LOW,
    MIN_DISPLAY_SCORE,
    TEXT_TO_IMAGE_SIM_HIGH,
    TEXT_TO_IMAGE_SIM_LOW,
)
from src.utils import get_sample_photos
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="Visual KB", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS for liquid-glass aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700;800&display=swap');

/* Orbs */
.bg-orbs {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -999; background: #08080F; overflow: hidden;
}
.orb { position: absolute; border-radius: 50%; filter: blur(120px); opacity: 0.55; }
.orb-tl { top: -10%; left: -10%; width: 50vw; height: 50vw; background: #a78bfa; }
.orb-tr { top: -10%; right: -10%; width: 40vw; height: 40vw; background: #2dd4bf; }
.orb-bc { bottom: -20%; left: 30%; width: 50vw; height: 50vw; background: #f472b6; }
.orb-br { bottom: -10%; right: -10%; width: 45vw; height: 45vw; background: #38bdf8; }

* { font-family: 'Syne', sans-serif !important; }
.dm-mono { font-family: 'DM Mono', monospace !important; }

/* Main App Wrappers */
.stApp { background: transparent !important; }
.stAppHeader { background: transparent !important; }

/* Main Content Area */
[data-testid="stAppViewContainer"] > section:nth-child(2), [data-testid="stMain"] {
    background: rgba(0,0,0,0.2) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.55) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6, p, span, label, strong {
    color: #ffffff;
}

/* Sidebar Nav/Buttons */
[data-testid="stSidebar"] button[kind="secondary"] {
    color: rgba(255,255,255,0.75) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] button[kind="secondary"] p {
    color: inherit !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover, [data-testid="stSidebar"] button[kind="secondary"]:active {
    color: #ffffff !important;
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}

/* Primary Search Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #a78bfa, #38bdf8) !important;
    box-shadow: 0 4px 15px rgba(167, 139, 250, 0.4) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
}
button[kind="primary"] p { color: #ffffff !important; }

/* Input Styles */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    caret-color: #ffffff !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.08) !important;
    border: 1px dashed rgba(255,255,255,0.15) !important;
}
[data-testid="stFileUploader"] section * { color: #ffffff !important; }

/* Tabs */
[data-testid="stTabs"] button {
    background: transparent !important;
    color: rgba(255,255,255,0.75) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button p {
    color: rgba(255,255,255,0.75) !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid #a78bfa !important;
}
[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #ffffff !important;
}

/* Result Cards */
.result-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}
.card-accent-0:hover { box-shadow: 0 4px 20px rgba(167,139,250,0.5); border-color: rgba(167,139,250,0.5); }
.card-accent-1:hover { box-shadow: 0 4px 20px rgba(56,189,248,0.5); border-color: rgba(56,189,248,0.5); }
.card-accent-2:hover { box-shadow: 0 4px 20px rgba(52,211,153,0.5); border-color: rgba(52,211,153,0.5); }

/* Score Bars */
.score-bar-container { height: 2px; width: 100%; background: rgba(255,255,255,0.1); margin-top: 8px; margin-bottom: 12px; border-radius: 2px; overflow: hidden; }
.score-bar { height: 100%; transition: width 0.5s ease; }
.score-bar-0 { background: linear-gradient(90deg, #a78bfa, transparent); }
.score-bar-1 { background: linear-gradient(90deg, #38bdf8, transparent); }
.score-bar-2 { background: linear-gradient(90deg, #34d399, transparent); }

/* Diagnosis Section */
.diagnosis-section {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 8px;
    padding: 12px;
    color: rgba(255,255,255,0.85);
    font-size: 0.85rem;
}
.diagnosis-section strong { color: #ffffff; font-weight: 600; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="bg-orbs">
        <div class="orb orb-tl"></div>
        <div class="orb orb-tr"></div>
        <div class="orb orb-bc"></div>
        <div class="orb orb-br"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Vector store (cached across Streamlit reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_vector_store() -> VectorStore:
    return VectorStore()


vs = _get_vector_store()


# ---------------------------------------------------------------------------
# Render result cards
# ---------------------------------------------------------------------------
def render_results(results: dict, mode: str = "text") -> None:
    if not results or not results["ids"] or len(results["ids"][0]) == 0:
        st.info("No matching photos found.")
        return

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    low, high = (
        (TEXT_TO_IMAGE_SIM_LOW, TEXT_TO_IMAGE_SIM_HIGH)
        if mode == "text"
        else (IMAGE_TO_IMAGE_SIM_LOW, IMAGE_TO_IMAGE_SIM_HIGH)
    )

    valid_results: list[dict] = []
    for i in range(len(ids)):
        raw_sim = 1.0 - distances[i]
        similarity = (raw_sim - low) / (high - low)
        sim_pct = int(max(0.0, min(1.0, similarity)) * 100)

        if sim_pct >= int(MIN_DISPLAY_SCORE * 100):
            valid_results.append(
                {
                    "path": metadatas[i]["path"],
                    "filename": os.path.basename(metadatas[i]["path"]),
                    "sim_pct": sim_pct,
                    "reason": metadatas[i].get(
                        "reason", "No inspection notes available."
                    ),
                }
            )

    if not valid_results:
        st.info("No photos met the 50% match threshold.")
        return

    st.markdown(
        f'<p style="color:rgba(255,255,255,0.4); font-size:0.9rem; '
        f'margin-bottom:24px;">Showing {len(valid_results)} matches '
        f"above 50%</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, res in enumerate(valid_results):
        col = cols[i % 3]
        accent_idx = i % 3

        with col:
            st.markdown(
                f'<div class="result-card card-accent-{accent_idx}">',
                unsafe_allow_html=True,
            )
            try:
                img = Image.open(res["path"])
                st.image(img, use_container_width=True)

                info_html = f"""
                <div style="margin-top:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="dm-mono" style="font-size:0.75rem; color:rgba(255,255,255,0.6); text-transform:uppercase; letter-spacing:0.05em;">{res['filename']}</span>
                        <span class="dm-mono" style="font-size:0.75rem; color:#ffffff;">{res['sim_pct']}% Match</span>
                    </div>
                    <div class="score-bar-container">
                        <div class="score-bar score-bar-{accent_idx}" style="width:{res['sim_pct']}%;"></div>
                    </div>
                    <div class="diagnosis-section">
                        <strong>DIAGNOSIS</strong><br>
                        {res['reason']}
                    </div>
                </div>
                """
                st.markdown(info_html, unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Error: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_html = """
    <div class="logo-container">
        <div class="logo-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="black" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round">
                <path d="m8 3 4 8 5-5 5 15H2L8 3z"/>
            </svg>
        </div>
        <h2 style="margin:0; font-size:1.4rem;">Visual KB</h2>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)

    num_indexed = vs.count()

    if st.button("Index Photos"):
        photos = get_sample_photos()
        if not photos:
            st.error("No sample photos found in ./sample_photos/")
        else:
            metadata_map = {
                "ice_1.jpg": "Severe ice slab on the low-pressure service valve. Low charge suspected.",
                "ice_2.jpg": "Frost migration from evaporator to compressor. Filter or blower motor check required.",
                "ice_3.jpg": "Ice forming under insulation. High humidity and low airflow condition.",
                "ice_4.jpg": "Solid block of ice on the A-coil. Unit must be thawed before leak testing.",
                "ice_5.jpg": "Iced up service port. Cannot connect manifold gauges safely until thawed.",
                "ice_6.jpg": "Light frost on exterior line. Potential beginning of airflow restriction.",
                "water_1.jpg": "Water stains on drywall. Possible slow leak from the tank bottom or drain valve.",
                "water_2.jpg": "Standing water in the utility closet. Condensate pump failure confirmed.",
                "water_3.jpg": "Corrosion at the tank base. Anode rod likely exhausted, tank integrity compromised.",
                "water_4.jpg": "Buckling floor boards around the riser. Subfloor moisture levels high.",
                "water_5.jpg": "Mold growth on the baseboard. Chronic moisture from the drain line.",
                "water_6.jpg": "Scaling and water marks on flooring. Primary pan was overflowing.",
                "elec_1.jpg": "Heavy green oxidation on the contactor terminals. High voltage drop detected.",
                "elec_2.jpg": "Rust inside wire nuts. Moisture intrusion in the outdoor disconnect box.",
                "elec_3.jpg": "Thermal damage to main breaker terminal. Likely caused by loose connection or over-current.",
                "elec_4.jpg": "Melted insulation on the start capacitor lead. Component overheated due to high ESR.",
                "elec_5.jpg": "Pitting on the contact surfaces. Suggest replacement of the disconnect assembly.",
                "elec_6.jpg": "Surface corrosion on the ground bar. Poor grounding path; high resistance to fault.",
                "sample_00.jpg": "Standard bathroom orientation. No visible leak detected.",
                "sample_01.jpg": "Modern kitchen setup. Visual inspection of cabinets normal.",
                "sample_02.jpg": "Close-up of bathroom sink plumbing. Joints appear dry.",
                "sample_04.jpg": "Outdoor HVAC condensing unit. Fans operational during inspection.",
                "flow-chart3.png": "Technical process flow for moisture detection and resolution.",
                "org-vt-confu.png": "Diagnostic matrix for electrical fault categorization.",
                "aug-vt-confu.png": "Augmented dataset matrix for training improved corrosion detection.",
            }

            progress_text = "Indexing photos with Gemini 2…"
            my_bar = st.progress(0, text=progress_text)

            for i, photo in enumerate(photos):
                fname = os.path.basename(photo)
                reason = metadata_map.get(fname, "Generic sample photo.")
                idx = f"img_{i}_{fname}"
                try:
                    vs.index_image(
                        photo,
                        idx,
                        metadata={"path": photo, "reason": reason},
                    )
                    my_bar.progress(
                        (i + 1) / len(photos), text=f"Indexed {fname}"
                    )
                except Exception as exc:
                    st.warning(f"Skipped {fname}: {exc}")
                    continue

            my_bar.empty()
            st.success(f"Successfully (re)indexed {len(photos)} photos!")
            st.rerun()

    if st.button(
        "Reset Database",
        type="secondary",
        help="Wipe all stored vectors",
    ):
        vs.clear_collection()
        st.warning("Database cleared.")
        st.rerun()

    st.markdown(
        '<div class="dm-mono" style="margin: 32px 0 12px 0; font-size:0.75rem; '
        'color:rgba(255,255,255,0.6); text-transform:uppercase; '
        'letter-spacing:0.1em;">LIBRARY STATUS</div>',
        unsafe_allow_html=True,
    )
    stats_html = f"""
    <div style="background:transparent; border:none; padding:0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span class="dm-mono" style="color:rgba(255,255,255,0.7); font-size:0.85rem;">PHOTOS</span>
            <span class="dm-mono" style="color:#FFFFFF; font-weight:500; font-size:0.85rem;">{num_indexed}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span class="dm-mono" style="color:rgba(255,255,255,0.7); font-size:0.85rem;">DIMENSIONS</span>
            <span class="dm-mono" style="color:#FFFFFF; font-weight:500; font-size:0.85rem;">{EMBEDDING_DIMENSIONS}</span>
        </div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if num_indexed == 0:
    empty_state_html = """
    <div style="text-align:center; margin-top:120px; opacity:0.8;">
        <div style="margin-bottom:24px;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none"
                 stroke="rgba(255,255,255,0.2)" stroke-width="1.5"
                 stroke-linecap="round" stroke-linejoin="round">
                <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                <circle cx="9" cy="9" r="2"/>
                <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
            </svg>
        </div>
        <h2 style="font-weight:500; color:rgba(255,255,255,0.9);">
            Begin by indexing your collection
        </h2>
        <p style="color:rgba(255,255,255,0.4); max-width:400px; margin:0 auto;">
            Connect your local visual knowledge base to Gemini 2 to enable
            semantic multimodal search.
        </p>
    </div>
    """
    st.markdown(empty_state_html, unsafe_allow_html=True)
else:
    tab1, tab2, tab3 = st.tabs(["Text Search", "Image Match", "Add to Library"])

    with tab1:
        st.markdown(
            '<h1 style="font-size:2.5rem; margin-top:20px; margin-bottom:8px;">'
            "Semantic Search</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:1.1rem; margin-bottom:32px; '
            'color:rgba(255,255,255,0.5);">Query your visual data '
            "using natural language.</p>",
            unsafe_allow_html=True,
        )

        query = st.text_input(
            "Search",
            placeholder="e.g. 'cracked tile near drain' or 'modern kitchen'",
            label_visibility="collapsed",
            key="text_search",
        )

        if st.button("Search", type="primary", key="btn_text"):
            if query:
                with st.spinner("Searching embedding space…"):
                    try:
                        results = vs.search_by_text(query, n_results=9)
                        st.markdown("<br>", unsafe_allow_html=True)
                        render_results(results, mode="text")
                    except Exception as exc:
                        st.error(f"Error searching: {exc}")
            else:
                st.warning("Please enter a query.")

    with tab2:
        st.markdown(
            '<h1 style="font-size:2.5rem; margin-top:20px; margin-bottom:8px;">'
            "Visual Match</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:1.1rem; margin-bottom:32px; '
            'color:rgba(255,255,255,0.5);">Find similar photos by '
            "uploading an image.</p>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload reference",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="img_search",
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Reference Image", width=300)

            if st.button("Find Similar", type="primary", key="btn_img"):
                with st.spinner("Embedding and searching…"):
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".jpg"
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        results = vs.search_by_image(tmp_path, n_results=9)
                        st.markdown("<br>", unsafe_allow_html=True)
                        render_results(results, mode="image")
                    except Exception as exc:
                        st.error(f"Error searching: {exc}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)

    with tab3:
        st.markdown(
            '<h1 style="font-size:2.5rem; margin-top:20px; margin-bottom:8px;">'
            "Add to Library</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:1.1rem; margin-bottom:32px; '
            'color:rgba(255,255,255,0.5);">Upload new photos to your '
            "visual knowledge base.</p>",
            unsafe_allow_html=True,
        )

        new_photos = st.file_uploader(
            "Upload photos to index",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="lib_upload",
        )

        if new_photos:
            reason_txt = st.text_area(
                "Inspection Notes (optional)",
                placeholder="e.g. Broken tile in primary bathroom…",
                help="This text will be associated with all uploaded photos.",
            )

            if st.button("Upload and Index", type="primary"):
                user_dir = os.path.join(os.getcwd(), "user_uploads")
                os.makedirs(user_dir, exist_ok=True)

                prog = st.progress(0, text="Saving and indexing…")
                success_count = 0

                for i, pf in enumerate(new_photos):
                    save_path = os.path.abspath(
                        os.path.join(user_dir, pf.name)
                    )
                    with open(save_path, "wb") as f:
                        f.write(pf.getbuffer())

                    try:
                        idx = f"user_{pf.name}_{i}"
                        vs.index_image(
                            save_path,
                            idx,
                            metadata={
                                "path": save_path,
                                "reason": reason_txt
                                or "User uploaded photo.",
                            },
                        )
                        success_count += 1
                        prog.progress(
                            (i + 1) / len(new_photos),
                            text=f"Indexed {pf.name}",
                        )
                    except Exception as exc:
                        st.error(f"Failed to index {pf.name}: {exc}")

                prog.empty()
                if success_count > 0:
                    st.success(
                        f"Successfully added {success_count} photos to "
                        f"your library!"
                    )
                    st.rerun()
