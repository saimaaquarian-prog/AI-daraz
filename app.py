import os
import json
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- Config ----------------
FAISS_INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "metadata.json")
CONFIG_PATH = os.path.join(FAISS_INDEX_DIR, "config.json")

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

DEPARTMENTS = ["returns", "delivery", "refunds", "seller", "payments", "customer_support"]
DEPARTMENT_LABELS = {
    "returns": "Returns",
    "delivery": "Delivery",
    "refunds": "Refunds",
    "seller": "Seller",
    "payments": "Payments",
    "customer_support": "Customer Support",
}

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Daraz Support Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #FAFAFA;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {DARAZ_DARK};
    }}
    section[data-testid="stSidebar"] * {{
        color: #F5F5F5 !important;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        color: #F5F5F5 !important;
    }}
    div[data-baseweb="radio"] label {{
        color: #F5F5F5 !important;
    }}
    .daraz-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 20px;
        background: linear-gradient(90deg, {DARAZ_ORANGE}, #FF8A3D);
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .daraz-header h1 {{
        color: white;
        font-size: 24px;
        margin: 0;
    }}
    .daraz-header p {{
        color: #FFE8DA;
        margin: 0;
        font-size: 13px;
    }}
    .source-tag {{
        display: inline-block;
        background-color: #FFF1E8;
        color: {DARAZ_ORANGE};
        border: 1px solid {DARAZ_ORANGE};
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        margin: 2px 4px 2px 0;
    }}
    .stButton > button {{
        background-color: {DARAZ_ORANGE};
        color: white;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: #D64A00;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="daraz-header">
        <div style="font-size: 32px;">🛍️</div>
        <div>
            <h1>Daraz Customer Support Assistant</h1>
            <p>Internal knowledge assistant for policy &amp; operations questions</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Load resources (cached, no re-embedding of PDFs) ----------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None

    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embed_model_name = "all-MiniLM-L6-v2"
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
            embed_model_name = cfg.get("embed_model", embed_model_name)

    return index, metadata, embed_model_name


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


index, metadata, embed_model_name = load_index_and_metadata()

if index is None:
    st.error(
        f"Could not find a pre-built FAISS index at `{FAISS_INDEX_DIR}/`. "
        "Run your ingestion script first to build `index.faiss` and `metadata.json`."
    )
    st.stop()

embedder = load_embedder(embed_model_name)
groq_client = load_groq_client()

if groq_client is None:
    st.error(
        "No Groq API key found. Add `GROQ_API_KEY` to your Streamlit secrets "
        "(Settings → Secrets) to enable answer generation."
    )
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.markdown("## 📂 Knowledge Base Sections")
st.sidebar.markdown("Restrict search to a specific section, or search all.")

section_choice = st.sidebar.radio(
    label="Search scope",
    options=["All sections"] + [DEPARTMENT_LABELS[d] for d in DEPARTMENTS],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(
    "This assistant answers questions using Daraz's internal policy "
    "documents (returns, delivery, refunds, seller, payments, and "
    "customer support). Answers are generated only from retrieved "
    "policy content."
)

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Map label back to department key
label_to_dept = {v: k for k, v in DEPARTMENT_LABELS.items()}
selected_department = None if section_choice == "All sections" else label_to_dept[section_choice]

# ---------------- Retrieval ----------------
def retrieve_chunks(query: str, department: str = None, top_k: int = TOP_K):
    query_vec = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    # Over-fetch when filtering by department, then trim, since the flat
    # index has no native metadata filter.
    fetch_k = top_k * 8 if department else top_k
    fetch_k = min(fetch_k, index.ntotal)

    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx]
        if department and entry.get("department") != department:
            continue
        results.append({**entry, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def build_prompt(query: str, chunks: list):
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source_file']} | Department: {c['department']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant policy content found."

    system_prompt = (
        "You are the Daraz Customer Support Operation Assistant. "
        "Answer the user's question using ONLY the policy context provided below. "
        "Be concise, accurate, and practical, as if briefing a support agent. "
        "If the context does not contain enough information to answer confidently, "
        "say so clearly and suggest escalating to the relevant department rather than guessing. "
        "Do not invent policy details that are not in the context."
    )

    user_prompt = f"Context from Daraz policy documents:\n\n{context}\n\nQuestion: {query}"

    return system_prompt, user_prompt


def generate_answer(query: str, chunks: list):
    system_prompt, user_prompt = build_prompt(query, chunks)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            tags = "".join(
                f'<span class="source-tag">{s["source_file"]} · {DEPARTMENT_LABELS.get(s["department"], s["department"])}</span>'
                for s in msg["sources"]
            )
            st.markdown(tags, unsafe_allow_html=True)

# ---------------- Chat input ----------------
placeholder_text = (
    "Ask about Daraz policies..."
    if selected_department is None
    else f"Ask about {DEPARTMENT_LABELS[selected_department]} policies..."
)

user_input = st.chat_input(placeholder_text)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(user_input, department=selected_department)

        if not chunks:
            answer = (
                "I couldn't find anything relevant in the selected section. "
                "Try switching to **All sections**, or rephrase your question."
            )
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(user_input, chunks)

        st.markdown(answer)

        if chunks:
            tags = "".join(
                f'<span class="source-tag">{c["source_file"]} · {DEPARTMENT_LABELS.get(c["department"], c["department"])}</span>'
                for c in chunks
            )
            st.markdown(tags, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks if chunks else []}
    )
