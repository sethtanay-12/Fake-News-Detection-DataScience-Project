import streamlit as st
import pickle
import re

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Bharat NEWS Check",
    page_icon="🇮🇳",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model = pickle.load(open('rf_model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# ── Text cleaner (same as notebook) ──────────────────────
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ── Predictor ────────────────────────────────────────────
def predict(text):
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    verdict = "REAL" if prediction == 1 else "FAKE"
    confidence = round(max(probability) * 100, 1)
    return verdict, confidence

# ── UI ───────────────────────────────────────────────────
st.title("🇮🇳 Bharat NEWS Check")
st.caption("Multilingual Fake News Detector — Hindi · English · Hinglish")
st.divider()

# Sample buttons
st.markdown("**Try a sample:**")
col1, col2, col3 = st.columns(3)

sample_fake_hindi  = "Modi ji ne kaha petrol bilkul free hoga next month! Share karo jaldi!"
sample_real        = "The Election Commission announced UP 2027 voting date as March 5."
sample_fake_eng    = "Vaccines cause autism, government hiding the truth!! 100% confirmed!"

if col1.button("Fake news (Hindi)"):
    st.session_state['user_input'] = sample_fake_hindi
if col2.button("Real news (English)"):
    st.session_state['user_input'] = sample_real
if col3.button("Fake news (English)"):
    st.session_state['user_input'] = sample_fake_eng

# Text input box
user_text = st.text_area(
    "Paste your news, tweet, or WhatsApp message here:",
    value=st.session_state.get('user_input', ''),
    height=140,
    placeholder="Type in Hindi, English, or Hinglish..."
)

# Analyze button
if st.button("Analyze", type="primary", use_container_width=True):
    if user_text.strip():
        with st.spinner("Analyzing..."):
            verdict, confidence = predict(user_text)

        st.divider()

        # Verdict display
        col_v, col_c = st.columns(2)
        if verdict == "FAKE":
            col_v.error(f"VERDICT: {verdict}")
        else:
            col_v.success(f"VERDICT: {verdict}")
        col_c.metric("Confidence", f"{confidence}%")

        # Confidence bar
        st.progress(confidence / 100)

        # Message
        if verdict == "FAKE":
            st.warning("This content shows signs of misinformation. Please verify with official sources before sharing.")
        else:
            st.info("This content appears credible. Always cross-check important news with official portals.")

        # Stats
        st.divider()
        st.markdown("**Text statistics:**")
        word_count   = len(user_text.split())
        excl_count   = user_text.count('!')
        hindi_chars  = len(re.findall(r'[\u0900-\u097F]', user_text))
        lang         = "Hindi" if hindi_chars > 10 else ("Hinglish" if hindi_chars > 0 else "English")

        c1, c2, c3 = st.columns(3)
        c1.metric("Words", word_count)
        c2.metric("Language", lang)
        c3.metric("Exclamations (!)", excl_count)

    else:
        st.warning("Please enter some text first.")

st.divider()
st.caption("Project by : Tanay Seth | Roll No: 25SCS1003004674 | Data Science Project 2025")