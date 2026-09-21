"""
app.py - the chat page (what the visitor sees). Run it with:  streamlit run app.py
"""

import base64
import os

import streamlit as st

from rag import build_chatbot, friendly_error

HERE = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(HERE, "data", "sweet_crumbs_knowledge.txt")
MAX_QUESTIONS = 15  # limit per visitor session, so a public demo cannot use up your free quota

SAMPLE_QUESTIONS = [
    "What are your opening hours?",
    "Do you deliver on Sundays?",
    "Do you have vegan options?",
    "How do I order a custom cake?",
]

st.set_page_config(page_title="Sweet Crumbs Assistant", page_icon="🧁")

# ---------------------------------------------------------------------------
# Logo pictures. They are drawn as small SVG images, so there is nothing to download.
# Want a real photo instead? Put a file called logo.png (or logo.jpg) in a folder
# named "assets" next to app.py and it will be used automatically.
# ---------------------------------------------------------------------------
CUPCAKE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<polygon points="14,32 50,32 45,57 19,57" fill="#D98F5C"/>
<line x1="24" y1="33" x2="26" y2="56" stroke="#B9743F" stroke-width="2"/>
<line x1="32" y1="33" x2="32" y2="56" stroke="#B9743F" stroke-width="2"/>
<line x1="40" y1="33" x2="38" y2="56" stroke="#B9743F" stroke-width="2"/>
<ellipse cx="32" cy="30" rx="20" ry="8" fill="#F06292"/>
<ellipse cx="32" cy="22" rx="15" ry="7" fill="#F48FB1"/>
<ellipse cx="32" cy="15" rx="9" ry="6" fill="#F8BBD0"/>
<circle cx="32" cy="8" r="4" fill="#C62828"/>
</svg>"""

COOKIE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="27" fill="#D9A066"/>
<circle cx="32" cy="32" r="27" fill="none" stroke="#C48A4E" stroke-width="3"/>
<ellipse cx="22" cy="22" rx="5" ry="4" fill="#5D3A1A"/>
<ellipse cx="41" cy="20" rx="4" ry="4" fill="#5D3A1A"/>
<ellipse cx="46" cy="36" rx="5" ry="4" fill="#5D3A1A"/>
<ellipse cx="30" cy="38" rx="5" ry="4" fill="#5D3A1A"/>
<ellipse cx="19" cy="41" rx="4" ry="3" fill="#5D3A1A"/>
<ellipse cx="36" cy="49" rx="4" ry="3" fill="#5D3A1A"/>
</svg>"""


def svg_to_img(svg, size):
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{encoded}" width="{size}" height="{size}" alt="">'


def logo_html():
    """Use assets/logo.png or assets/logo.jpg if it exists, otherwise the drawn cupcake."""
    for name, mime in (("logo.png", "image/png"), ("logo.jpg", "image/jpeg")):
        path = os.path.join(HERE, "assets", name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            return (
                f'<img src="data:{mime};base64,{encoded}" width="72" height="72" '
                'style="border-radius:50%; object-fit:cover;" alt="">'
            )
    return svg_to_img(CUPCAKE_SVG, 60)


# ---------------------------------------------------------------------------
# Look and feel (colors, banner, chips)
# ---------------------------------------------------------------------------
STYLE = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #FFF8F0 !important; }
[data-testid="stSidebar"] { background-color: #FCE7D6; }
[data-testid="stToolbar"], footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 820px; }

.hero { display: flex; align-items: center; gap: 18px; padding: 22px 26px; border-radius: 18px;
        background: linear-gradient(120deg, #C8567B 0%, #E58AA5 55%, #F5B79A 100%);
        box-shadow: 0 6px 18px rgba(200, 86, 123, 0.25); }
.hero-logo { background: white; border-radius: 50%; width: 84px; height: 84px; flex: none;
             display: flex; align-items: center; justify-content: center; }
.hero-text { flex: 1; }
.hero-title { margin: 0; color: white; font-family: Georgia, serif; font-size: 2rem; font-weight: 700; line-height: 1.2; }
.hero-sub { margin: 4px 0 0 0; color: #FFF1E6; font-size: 0.95rem; }
.hero-cookie { flex: none; opacity: 0.95; }

.chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 6px 0; }
.chip { background: white; border: 1px solid #F0C9D4; color: #7A3B4D; border-radius: 999px;
        padding: 6px 14px; font-size: 0.85rem; }

.section-label { margin: 18px 0 8px 0; color: #9A5A6B; font-size: 0.78rem; font-weight: 700;
                 letter-spacing: 0.08em; text-transform: uppercase; }
.visit-card { background: white; border: 1px solid #F0C9D4; border-radius: 14px; padding: 14px 16px;
              color: #4A2C2A; font-size: 0.88rem; line-height: 1.55; }
.visit-card h4 { margin: 0 0 6px 0; color: #C8567B; font-family: Georgia, serif; font-size: 1.05rem; }
.note { color: #8B6B72; font-size: 0.8rem; line-height: 1.45; }
[data-testid="stChatInput"] { border: 1px solid #E8A0B4; border-radius: 26px; background-color: white; }
.stButton > button { border: 1px solid #C8567B; color: #C8567B; border-radius: 20px; background-color: white; }
.stButton > button:hover { background-color: #C8567B; color: white; border-color: #C8567B; }
[data-testid="stChatMessage"] { background-color: #FCE7D6; border-radius: 14px; }
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

HERO = f"""<div class="hero">
<div class="hero-logo">{logo_html()}</div>
<div class="hero-text">
<p class="hero-title">Sweet Crumbs Bakery</p>
<p class="hero-sub">Virtual assistant &middot; ask about our menu, prices, delivery and allergies</p>
</div>
<div class="hero-cookie">{svg_to_img(COOKIE_SVG, 56)}</div>
</div>
<div class="chips">
<span class="chip">Open 7 days a week</span>
<span class="chip">Free delivery over $50</span>
<span class="chip">Custom cakes: order 5 days ahead</span>
</div>"""

st.markdown(HERO, unsafe_allow_html=True)


def safe_text(text):
    """Streamlit treats $...$ as math. Escape the $ so prices such as $6 show correctly."""
    return text.replace("$", "\\$")


def get_api_key():
    """Look for the key in Streamlit secrets first, then in an environment variable."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets file yet
        return os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner="Reading the Sweet Crumbs document...")
def load_bot(api_key):
    # cache_resource = this runs once, not on every click
    return build_chatbot(api_key, DOCUMENT_PATH)


# ---------- Session memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "sources", "search_query"}
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0


def questions_left_text():
    left = MAX_QUESTIONS - st.session_state.questions_asked
    return f"Questions left in this session: **{left}**"


# ---------- Sidebar ----------
VISIT_CARD = """<div class="visit-card">
<h4>Visit us</h4>
42 Maple Street, Brightwater<br>
Mon-Fri 7:00am-6:00pm<br>
Sat 8:00am-5:00pm<br>
Sun 8:00am-1:00pm<br>
(555) 013-2244
</div>"""

with st.sidebar:
    st.markdown(VISIT_CARD, unsafe_allow_html=True)
    st.markdown('<p class="section-label">About this demo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="note">A portfolio project: an AI assistant that answers only from the '
        "bakery's information sheet (RAG with the Gemini API). Sweet Crumbs is a fictional "
        "bakery. For allergies, always confirm with the shop.</p>",
        unsafe_allow_html=True,
    )
    show_how = st.toggle("Show how it works", value=False)
    counter_box = st.empty()
    counter_box.write(questions_left_text())
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.questions_asked = 0
        st.rerun()

# ---------- API key check ----------
api_key = get_api_key()
if not api_key:
    st.error(
        "No Gemini API key found. Create the file `.streamlit/secrets.toml` with the line "
        '`GEMINI_API_KEY = "your-key-here"` (see HOW_TO_RUN.md), then restart the app.'
    )
    st.stop()


def show_sources(sources, search_query):
    """The 'how it works' panel: which document pieces were found for this answer."""
    with st.expander("How this answer was found"):
        st.write(f"**Search text used:** {search_query}")
        st.write("**Document pieces retrieved (best match first):**")
        for chunk, score in sources:
            st.markdown(f"- **{chunk.title}** (similarity {score:.2f})")
            st.text(chunk.text)


# ---------- Sample question buttons (only before the first message) ----------
typed = st.chat_input("Ask about Sweet Crumbs...")
clicked = None
sample_area = st.empty()
if not st.session_state.messages:
    with sample_area.container():
        st.markdown('<p class="section-label">Popular questions</p>', unsafe_allow_html=True)
        columns = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            with columns[i % 2]:
                if st.button(q, key=f"sample_{q}", use_container_width=True):
                    clicked = q
question = typed or clicked
if question:
    sample_area.empty()

# ---------- Show earlier messages ----------
for message in st.session_state.messages:
    avatar = "🧁" if message["role"] == "assistant" else "🙂"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(safe_text(message["content"]))
        if show_how and message.get("sources"):
            show_sources(message["sources"], message["search_query"])

# ---------- Handle a new question ----------
if question:
    with st.chat_message("user", avatar="🙂"):
        st.markdown(safe_text(question))

    if st.session_state.questions_asked >= MAX_QUESTIONS:
        with st.chat_message("assistant", avatar="🧁"):
            st.info("You have reached the limit of 15 questions for this demo session. Thanks for trying it!")
    else:
        history = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]
        with st.chat_message("assistant", avatar="🧁"):
            try:
                with st.spinner("Thinking..."):
                    bot = load_bot(api_key)
                    result = bot.ask(question, history)
                st.markdown(safe_text(result.text))
                if show_how:
                    show_sources(result.sources, result.search_query)
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.text,
                        "sources": result.sources,
                        "search_query": result.search_query,
                    }
                )
                st.session_state.questions_asked += 1
                counter_box.write(questions_left_text())
            except Exception as error:  # show a friendly message, never a crash
                st.error(friendly_error(error))
                print("Error:", repr(error))  # the real error appears in your terminal
