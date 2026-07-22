"""
app.py -- Streamlit Web Interface for Spam Mail Detection System.

Run with:
    streamlit run app.py

Features:
  - Large email text area
  - Real-time spam/ham prediction
  - Spam probability gauge
  - Color-coded result (Red=Spam, Green=Ham)
  - Sample email loader
  - Modern glassmorphism-inspired dark UI
"""

import sys
import time
from pathlib import Path

import streamlit as st

#  Path setup 
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# 
# Page configuration (MUST be first Streamlit call)
# 
st.set_page_config(
    page_title  = "Spam Mail Detector",
    page_icon   = "?",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# 
# Custom CSS -- Dark glassmorphism theme
# 
CUSTOM_CSS = """
<style>
  /*  Google Font  */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /*  Root overrides  */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /*  Background  */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
  }

  /*  Main container  */
  .main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
  }

  /*  Header card  */
  .header-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .header-card h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }
  .header-card p {
    color: rgba(255,255,255,0.6);
    font-size: 1rem;
  }

  /*  Input card  */
  .input-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
  }

  /*  Text area  */
  .stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 14px !important;
    resize: vertical;
    transition: border-color 0.3s ease;
  }
  .stTextArea textarea:focus {
    border-color: rgba(167, 139, 250, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
  }

  /*  Primary button  */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
  }
  .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.55) !important;
  }
  .stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
  }

  /*  Secondary button  */
  .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.12) !important;
  }

  /*  Result cards  */
  .result-spam {
    background: linear-gradient(135deg, rgba(239,68,68,0.18) 0%, rgba(185,28,28,0.12) 100%);
    border: 1.5px solid rgba(239,68,68,0.45);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: slideIn 0.4s ease;
  }
  .result-ham {
    background: linear-gradient(135deg, rgba(34,197,94,0.18) 0%, rgba(21,128,61,0.12) 100%);
    border: 1.5px solid rgba(34,197,94,0.45);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: slideIn 0.4s ease;
  }
  .result-title-spam { font-size: 2rem; font-weight: 700; color: #f87171; margin: 0; }
  .result-title-ham  { font-size: 2rem; font-weight: 700; color: #4ade80; margin: 0; }
  .result-sub        { color: rgba(255,255,255,0.55); font-size: 0.9rem; margin-top: 0.3rem; }

  /*  Metric cards  */
  .metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-label { color: rgba(255,255,255,0.5); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #f0f0f0; }

  /*  Progress bar styling  */
  .stProgress > div > div > div > div {
    border-radius: 99px !important;
  }

  /*  Sidebar  */
  [data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  [data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.75); }

  /*  Labels / headings  */
  .stMarkdown h3 { color: rgba(255,255,255,0.9); }
  label { color: rgba(255,255,255,0.7) !important; }

  /*  Divider  */
  hr { border-color: rgba(255,255,255,0.08); }

  /*  Animations  */
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0);    }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }
  .pulse { animation: pulse 1.5s infinite; }

  /*  Sample pills  */
  .sample-tag {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
    margin: 3px;
    cursor: pointer;
  }

  /*  Footer  */
  .footer {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
  }
</style>
"""

# 
# Sample emails
# 
SAMPLE_SPAM = [
    "CONGRATULATIONS! You've won a FREE iPhone 15 Pro. Click here to claim your prize before it expires!",
    "URGENT: Your bank account has been suspended. Call 0800-XXXX immediately to avoid permanent closure.",
    "You are a WINNER! Reply WIN to claim your GBP1,000 cash prize. Limited time offer!",
    "FREE entry! Win tickets to the FA Cup Final. Text WIN to 85023. T&C apply.",
]

SAMPLE_HAM = [
    "Hey, are you joining us for lunch today? We're going to the new Italian place downtown.",
    "The quarterly report is ready for review. Could you take a look before Thursday's meeting?",
    "Happy birthday! Hope you have a wonderful day. Let's celebrate this weekend!",
    "Reminder: Your dentist appointment is scheduled for tomorrow at 2:30 PM.",
]


# 
# Model loader (cached)
# 

@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Load model and vectorizer, caching them across re-runs."""
    from train_model import load_model
    return load_model()


# 
# UI helpers
# 

def render_header() -> None:
    st.markdown("""
    <div class="header-card">
      <h1>? Spam Mail Detector</h1>
      <p>AI-powered email classification . Multinomial Naive Bayes . TF-IDF Features</p>
    </div>
    """, unsafe_allow_html=True)


def render_result(result: dict) -> None:
    """Render the prediction result card."""
    is_spam   = result["label"] == 1
    card_cls  = "result-spam" if is_spam else "result-ham"
    title_cls = "result-title-spam" if is_spam else "result-title-ham"
    icon      = "?" if is_spam else "?"
    label     = "SPAM" if is_spam else "NOT SPAM"
    sub       = "This email appears to be unsolicited / spam." if is_spam \
                else "This email looks legitimate."

    st.markdown(f"""
    <div class="{card_cls}">
      <p class="{title_cls}">{icon} &nbsp; {label}</p>
      <p class="result-sub">{sub}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability meters
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Spam Probability</div>
          <div class="metric-value" style="color: #f87171;">{result['spam_prob']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Ham Probability</div>
          <div class="metric-value" style="color: #4ade80;">{result['ham_prob']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Spam probability bar
    st.markdown("**Spam Probability Gauge**")
    spam_frac = result["spam_prob"] / 100
    st.progress(spam_frac)
    st.caption(
        f"{'? High spam confidence' if spam_frac > 0.8 else '? Low spam confidence' if spam_frac < 0.3 else '? Moderate confidence'}"
    )


def render_sidebar(model_loaded: bool) -> None:
    """Render the sidebar with model status and info."""
    with st.sidebar:
        st.markdown("### ? Spam Detector")
        st.markdown("---")

        # Model status
        if model_loaded:
            st.success("? Model loaded")
        else:
            st.error("? Model not found")
            st.info("Run `python main.py` first to train and save the model.")

        st.markdown("---")
        st.markdown("### ? About")
        st.markdown("""
        This tool uses **Multinomial Naive Bayes** trained on the SMS Spam Collection dataset.

        **Pipeline:**
        - TF-IDF Vectorisation (5,000 features)
        - Bigram + Unigram features
        - PorterStemmer text cleaning
        - Stratified 80/20 split

        **Typical Performance:**
        - Accuracy: ~98%
        - F1 Score: ~97%
        """)

        st.markdown("---")
        st.markdown("### ? Quick Stats")
        st.metric("Algorithm", "Multinomial NB")
        st.metric("Features",  "5,000 TF-IDF")
        st.metric("Dataset",   "SMS Spam Collection")

        st.markdown("---")
        st.markdown(
            "<div style='color:rgba(255,255,255,0.3);font-size:0.75rem;'>"
            "Spam Mail Detection System<br>College Mini Project . 2024</div>",
            unsafe_allow_html=True
        )


# 
# Main app
# 

def main() -> None:
    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    #  Load model 
    model_loaded = False
    model, vectorizer = None, None
    try:
        model, vectorizer = load_model_cached()
        model_loaded = True
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"? Could not load model: {e}")

    #  Sidebar 
    render_sidebar(model_loaded)

    #  Header 
    render_header()

    #  Model not found banner 
    if not model_loaded:
        st.error(
            "? **Model not found.** Please run `python main.py` in your terminal "
            "to train and save the model first, then refresh this page."
        )
        st.stop()

    #  Sample email loader 
    with st.expander("? Load a Sample Email", expanded=False):
        col_s, col_h = st.columns(2)

        with col_s:
            st.markdown("**? Spam Samples**")
            for i, sample in enumerate(SAMPLE_SPAM):
                if st.button(f"Spam #{i+1}", key=f"spam_{i}", use_container_width=True):
                    st.session_state["email_text"] = sample

        with col_h:
            st.markdown("**? Ham Samples**")
            for i, sample in enumerate(SAMPLE_HAM):
                if st.button(f"Ham #{i+1}", key=f"ham_{i}", use_container_width=True):
                    st.session_state["email_text"] = sample

    #  Main input area 
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    default_text = st.session_state.get("email_text", "")
    email_text = st.text_area(
        "? Enter Email / SMS Text",
        value=default_text,
        height=200,
        placeholder=(
            "Paste or type the email / SMS message you want to analyse here...\n\n"
            "Example: 'Congratulations! You've won a free holiday. Click here to claim.'"
        ),
        key="email_input",
        label_visibility="visible",
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        predict_clicked = st.button(
            "? Analyse Email",
            type="primary",
            use_container_width=True,
            disabled=not email_text.strip()
        )
    with col_clear:
        if st.button("? Clear", use_container_width=True):
            st.session_state["email_text"] = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    #  Prediction 
    if predict_clicked and email_text.strip():
        with st.spinner("? Analysing message ..."):
            time.sleep(0.4)   # brief UX pause

            try:
                from predict import predict_email

                result = predict_email(email_text, model=model, vectorizer=vectorizer)

                st.markdown("---")
                st.markdown("### ? Prediction Result")
                render_result(result)

                # Cleaned text expander
                with st.expander("? View Cleaned Text", expanded=False):
                    st.code(result["clean_text"] or "(empty after cleaning)", language=None)

            except Exception as exc:
                st.error(f"? Prediction failed: {exc}")

    #  History (session state) 
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if predict_clicked and email_text.strip():
        try:
            from predict import predict_email
            result = predict_email(email_text, model=model, vectorizer=vectorizer)
            st.session_state["history"].append({
                "text"       : email_text[:80] + ("..." if len(email_text) > 80 else ""),
                "prediction" : result["prediction"],
                "spam_prob"  : result["spam_prob"],
            })
        except Exception:
            pass

    if st.session_state["history"]:
        with st.expander(f"? Prediction History ({len(st.session_state['history'])} items)", expanded=False):
            for i, entry in enumerate(reversed(st.session_state["history"][-10:]), 1):
                icon  = "?" if entry["prediction"] == "Spam" else "?"
                color = "#f87171" if entry["prediction"] == "Spam" else "#4ade80"
                st.markdown(
                    f"**{icon} {entry['prediction']}** "
                    f"<span style='color:{color};font-size:0.85rem;'>({entry['spam_prob']:.1f}% spam)</span>"
                    f"  --  {entry['text']}",
                    unsafe_allow_html=True
                )

    #  Footer 
    st.markdown(
        "<div class='footer'>? Spam Mail Detection System &nbsp;.&nbsp; "
        "Multinomial Naive Bayes &nbsp;.&nbsp; Built with Streamlit</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
