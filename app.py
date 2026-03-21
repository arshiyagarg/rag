import streamlit as st
import html as html_lib
from src.pipeline      import hint
from src.retriever     import retrieve
from src.reranker      import rerank
from src.prompt_builder import build_messages
from src.generator     import stream_generator, StreamResult
from src.config        import (
    RETRIEVAL_TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
    JINA_RERANKER_TOP_N,
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="DSA Hint Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

/* Root tokens */
:root {
    --bg:        #0d0f12;
    --bg2:       #13161b;
    --bg3:       #1a1e26;
    --border:    #252a35;
    --accent:    #7c6af5;
    --accent2:   #4ecdc4;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --so-color:  #f48024;
    --cp-color:  #4ecdc4;
    --pass:      #22c55e;
    --fail:      #ef4444;
}

/* Global */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.dsa-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.dsa-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #7c6af5 0%, #4ecdc4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.dsa-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 400;
}

/* Pattern badge */
.pattern-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(124,106,245,0.15), rgba(78,205,196,0.15));
    border: 1px solid rgba(124,106,245,0.4);
    border-radius: 8px;
    padding: 10px 18px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #a89df7;
    margin-bottom: 1rem;
}
.pattern-icon { font-size: 1.2rem; }

/* Source cards */
.source-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.source-card:hover { border-color: var(--accent); }
.source-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.source-badge {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-so  { background: rgba(244,128,36,0.15); color: var(--so-color); border: 1px solid rgba(244,128,36,0.3); }
.badge-cp  { background: rgba(78,205,196,0.15);  color: var(--cp-color);  border: 1px solid rgba(78,205,196,0.3); }
.source-title {
    font-size: 0.82rem;
    color: var(--text);
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 260px;
}
.source-url {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
}
.score-pill {
    margin-left: auto;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    background: rgba(124,106,245,0.12);
    color: #a89df7;
    border: 1px solid rgba(124,106,245,0.2);
    white-space: nowrap;
}

/* Hint output box */
.hint-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,245,0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c6af5, #4ecdc4) !important;
    color: #0d0f12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Selectbox / slider */
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div { color: var(--text) !important; }

/* Labels */
label, .stTextArea label, .stTextInput label, .stSelectbox label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-family: 'Syne', sans-serif !important;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Stat numbers */
.stat-row {
    display: flex;
    gap: 20px;
    margin-top: 0.5rem;
}
.stat {
    text-align: center;
    padding: 8px 14px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent2);
}
.stat-lbl {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Spinner */
.stSpinner { color: var(--accent) !important; }

/* Remove streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    source_filter = st.selectbox(
        "Source filter",
        options=["All sources", "Stack Overflow only", "CP-Algorithms only"],
        index=0,
    )
    source_map = {
        "All sources":           None,
        "Stack Overflow only":   "stackoverflow",
        "CP-Algorithms only":    "cp_algorithms",
    }
    selected_source = source_map[source_filter]

    threshold = st.slider(
        "Retrieval threshold",
        min_value=0.1,
        max_value=0.9,
        value=RETRIEVAL_SCORE_THRESHOLD,
        step=0.05,
        help="Min similarity score. Lower = more results but noisier.",
    )

    top_k = st.slider(
        "Candidates to retrieve",
        min_value=5,
        max_value=30,
        value=RETRIEVAL_TOP_K,
        step=5,
        help="How many chunks to fetch before reranking.",
    )

    top_n = st.slider(
        "Results after reranking",
        min_value=1,
        max_value=10,
        value=JINA_RERANKER_TOP_N,
        step=1,
        help="Final chunks passed to the LLM.",
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#6b7280;line-height:1.6'>"
        "Sources: Stack Overflow Q&A + CP-Algorithms articles<br>"
        "Embeddings: Gemini text-embedding-004<br>"
        "Reranker: Jina v2 multilingual<br>"
        "Generation: Groq llama-3.3-70b"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="dsa-header" style="text-align:center">
    <p class="dsa-title" style="font-size:3.2rem;justify-content:center">Recurse</p>
    <p class="dsa-sub" style="font-size:1.05rem">Paste a problem + your stuck code → get a targeted hint, not the answer</p>
</div>
""", unsafe_allow_html=True)


# ── Input area ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    problem = st.text_area(
        "Problem description",
        height=130,
    )

with col_right:
    code = st.text_area(
        "Your current code (optional)",
        height=130,
    )

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    submitted = st.button("⚡ Get Hint", use_container_width=True)


# ── Run pipeline ───────────────────────────────────────────────
if submitted:
    if not problem.strip():
        st.warning("Please enter a problem description.")
        st.stop()

    st.divider()
    out_left, out_right = st.columns([3, 2], gap="large")

    with out_left:
        # ── Retrieve + rerank ──────────────────────────────────
        with st.spinner("Retrieving relevant references..."):
            retrieval_query = (
                f"{problem.strip()}\n\n{code.strip()}"
                if code.strip() else problem.strip()
            )
            chunks = retrieve(
                query=retrieval_query,
                top_k=top_k,
                score_threshold=threshold,
                source_filter=selected_source,
                rerank=False,   # rerank separately so we can show before/after
            )
            if chunks:
                chunks = rerank(retrieval_query, chunks, top_n=top_n)

        if not chunks:
            st.error(
                "No relevant chunks found. Try lowering the threshold "
                "or broadening the source filter."
            )
            st.stop()

        # ── Detect mode + build prompt ────────────────────────
        from src.pipeline import _detect_mode
        mode = _detect_mode(problem.strip())
        mode_label = "🛠 Code Fix Mode" if mode == "code_fix" else "💡 Hint Mode"
        st.caption(mode_label)

        messages = build_messages(
            chunks=chunks,
            problem=problem.strip(),
            code_snippet=code.strip() if code.strip() else None,
            mode=mode,
        )

        sr = StreamResult()

        st.markdown("#### 💡 Hint")
        response_placeholder = st.empty()

        # Stream tokens into the placeholder
        full_text = ""
        with st.spinner(""):
            for token in stream_generator(messages, sr):
                full_text += token
                response_placeholder.markdown(full_text)

        # ── Parse and render hint as structured sections ───────
        import re

        def render_hint_native(text: str) -> None:
            """
            Render hint using native Streamlit components.
            Splits on numbered points and renders each as a
            separate st.markdown() — Streamlit handles code
            blocks natively with proper syntax highlighting.
            """
            icons = {
                "1": "🔍", "2": "💭",
                "3": "🛠", "4": "📖",
            }
            colors = {
                "1": "#a89df7", "2": "#4ecdc4",
                "3": "#f9c74f", "4": "#90e0ef",
            }

            parts = re.split(r'(?=^\d+\.\s)', text.strip(), flags=re.MULTILINE)

            if len(parts) <= 1:
                # No numbered structure — render as-is
                st.markdown(text)
                return

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                num_match = re.match(r'^(\d+)\.\s', part)
                if num_match:
                    num   = num_match.group(1)
                    body  = part[num_match.end():]
                    icon  = icons.get(num, "•")
                    color = colors.get(num, "#e8eaf0")

                    # Split label from body — label ends at first \n
                    # but keep entire body together for st.markdown so
                    # code blocks are never split across calls
                    lines      = body.split("\n")
                    label_line = lines[0].strip()
                    body_rest  = "\n".join(lines[1:]).strip()

                    # Colored section header
                    st.markdown(
                        f'<div style="border-left:3px solid {color};'
                        f'padding:6px 12px;margin:10px 0 4px;'
                        f'font-weight:600;font-size:0.9rem;color:{color}">'
                        f'{icon}&nbsp;&nbsp;{label_line}</div>',
                        unsafe_allow_html=True,
                    )
                    # Body — full native st.markdown so code blocks
                    # with ``` get proper syntax highlighting
                    if body_rest:
                        st.markdown(body_rest)
                    elif not body_rest and "```" in label_line:
                        # Edge case: entire body is a code block on same line
                        st.markdown(body)
                else:
                    st.markdown(part)

        # Clear the streaming placeholder and render structured output
        response_placeholder.empty()
        render_hint_native(sr.explanation)

        # ── Pattern badge ──────────────────────────────────────
        pattern_match = re.search(
            r"pattern identified[:\s*_]*(.+?)(?:\n|$)",
            sr.explanation,
            re.IGNORECASE,
        )
        pattern = pattern_match.group(1).strip().strip("*_") if pattern_match else None

        if pattern and pattern.lower() != "unknown":
            st.markdown(
                f'<div class="pattern-badge">'
                f'<span class="pattern-icon">🧩</span>'
                f'Pattern: {pattern}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Stats row ──────────────────────────────────────────
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat"><div class="stat-val">{sr.input_tokens}</div><div class="stat-lbl">Input tokens</div></div>'
            f'<div class="stat"><div class="stat-val">{sr.output_tokens}</div><div class="stat-lbl">Output tokens</div></div>'
            f'<div class="stat"><div class="stat-val">{len(chunks)}</div><div class="stat-lbl">Chunks used</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Sources panel ──────────────────────────────────────────
    with out_right:
        st.markdown("#### 📚 Sources")

        # Deduplicate chunks by URL — keep highest rerank score per URL
        cited_urls  = set(sr.sources)
        seen_urls: set[str] = set()
        deduped: list[dict] = []
        for c in sorted(chunks, key=lambda x: x.get("rerank_score", x.get("score", 0)), reverse=True):
            url = c.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(c)

        # Cited sources first, then uncited — both already deduplicated
        cited   = [c for c in deduped if c.get("url", "") in cited_urls]
        uncited = [c for c in deduped if c.get("url", "") not in cited_urls]
        ordered = cited + uncited

        if not ordered:
            st.markdown(
                "<div style='color:#6b7280;font-size:0.85rem'>"
                "No sources to display."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for chunk in ordered:
                src   = chunk.get("source_type", "unknown")
                title = chunk.get("title", "Untitled")
                url   = chunk.get("url", "")
                score = chunk.get("rerank_score", chunk.get("score", 0))
                cited_marker = "✦ " if url in cited_urls else ""

                if src == "stackoverflow":
                    badge    = '<span class="source-badge badge-so">SO</span>'
                    so_score = chunk.get("so_score")
                    accepted = chunk.get("is_accepted", False)
                    extra    = ""
                    if so_score is not None:
                        acc_mark = " ✓" if accepted else ""
                        extra = (
                            f'<span style="font-size:0.65rem;color:#f48024;'
                            f'font-family:JetBrains Mono,monospace">↑{so_score}{acc_mark}</span>'
                        )
                elif src == "cp_algorithms":
                    badge = '<span class="source-badge badge-cp">CP</span>'
                    topic = chunk.get("topic", "")
                    extra = (
                        f'<span style="font-size:0.65rem;color:#4ecdc4;'
                        f'font-family:JetBrains Mono,monospace">{topic}</span>'
                        if topic else ""
                    )
                else:
                    badge = '<span class="source-badge" style="background:rgba(107,114,128,0.15);color:#9ca3af;border:1px solid rgba(107,114,128,0.3)">DOC</span>'
                    extra = ""

                st.markdown(
                    f'<div class="source-card">'
                    f'  <div class="source-header">'
                    f'    {badge}'
                    f'    <span class="source-title" title="{title}">{cited_marker}{title}</span>'
                    f'    <span class="score-pill">{score:.3f}</span>'
                    f'  </div>'
                    f'  <div style="display:flex;align-items:center;gap:8px;margin-top:2px">'
                    f'    {extra}'
                    f'    <a href="{url}" target="_blank" class="source-url">{url}</a>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Retrieval debug expander ───────────────────────────
        # with st.expander("🔍 Retrieval details", expanded=False):
        #     st.markdown(
        #         f"**Query:** `{retrieval_query[:120]}{'...' if len(retrieval_query) > 120 else ''}`\n\n"
        #         f"**Threshold:** `{threshold}`  **Top-K:** `{top_k}`  **Top-N:** `{top_n}`\n\n"
        #         f"**Chunks retrieved:** {len(chunks)}"
        #     )
        #     for i, c in enumerate(chunks, 1):
        #         rs       = c.get("rerank_score", None)
        #         vs       = c.get("score", 0.0)
        #         rs_str   = f"{rs:.4f}" if isinstance(rs, float) else "—"
        #         vs_str   = f"{vs:.3f}" if isinstance(vs, float) else "—"
        #         src      = c.get("source_type", "?")
        #         cid      = c.get("chunk_id", "")
        #         st.markdown(
        #             f"{i}. **{src}** — {cid}  "
        #             f"vec={vs_str}  rerank={rs_str}"
        #         )