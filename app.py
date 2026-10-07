import streamlit as st
import numpy as np
from joblib import load

from src.robots_classifier.io import fetch_robots_txt

st.set_page_config(page_title="Robots.txt Ad-hoc Classifier", layout="centered")

st.title("Robots.txt Ad-hoc Webcontent Classifier")
st.caption("LinearSVC + char TF-IDF (trained on dataset_2500, 15 classes)")

MODEL_PATH_DEFAULT = "models/final_best_2500.joblib"

@st.cache_resource
def load_bundle(path: str):
    bundle = load(path)
    return bundle["vectorizer"], bundle["model"], bundle.get("meta", {})

def predict_topk(vec, clf, text: str, topk: int = 5):
    X = vec.transform([text])
    pred = clf.predict(X)[0]
    top = []
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)[0]
        idx_sorted = np.argsort(scores)[::-1]
        for i in idx_sorted[:topk]:
            top.append((clf.classes_[i], float(scores[i])))
    return pred, top

# Sidebar: model path
st.sidebar.header("Model")
model_path = st.sidebar.text_input("Model path", value=MODEL_PATH_DEFAULT)
topk = st.sidebar.slider("Top-k", min_value=3, max_value=10, value=5)

try:
    vec, clf, meta = load_bundle(model_path)
    st.sidebar.success("Model loaded")
except Exception as e:
    st.sidebar.error(f"Cannot load model: {e}")
    st.stop()

def robots_txt_url_from_input(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return "https://your-site.com/robots.txt"

    # Si l'utilisateur a déjà mis /robots.txt, on garde tel quel
    if u.endswith("/robots.txt"):
        if u.startswith("http://") or u.startswith("https://"):
            return u
        return "https://" + u

    # Ajoute un schéma si absent
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "https://" + u

    # Enlève trailing slash
    u = u.rstrip("/")

    return u + "/robots.txt"

tab1, tab2 = st.tabs(["From URL", "From robots.txt text"])

with tab1:
    st.subheader("Predict from a website URL")
    url = st.text_input("Website (e.g. https://lemonde.fr)")
    if st.button("Fetch and Predict", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                robots = fetch_robots_txt(url, timeout=10)
                st.text_area("Fetched robots.txt", value=robots, height=250)
                pred, top = predict_topk(vec, clf, robots, topk=topk)
                st.success(f"Predicted category: {pred}")
                if top:
                    st.write("Top scores (SVM decision scores):")
                    st.table([{"rank": i+1, "category": c, "score": s} for i, (c, s) in enumerate(top)])

            except Exception as e:
                robots_url = robots_txt_url_from_input(url)

                st.error("Unable to fetch robots.txt automatically (the website may block non-browser requests).")
                st.info(f"Open this in your browser, copy the content, then paste it in the 'From robots.txt text' tab:\n\n{robots_url}")

                # Optionnel: afficher l'erreur technique en petit
                with st.expander("Technical details"):
                    st.write(str(e))

with tab2:
    st.subheader("Predict from pasted robots.txt")
    robots_text = st.text_area("Paste robots.txt content here", height=250)
    if st.button("Predict from text"):
        if not robots_text.strip():
            st.warning("Paste some robots.txt content.")
        else:
            pred, top = predict_topk(vec, clf, robots_text, topk=topk)
            st.success(f"Predicted category: {pred}")
            if top:
                st.write("Top scores (SVM decision scores):")
                st.table([{"rank": i+1, "category": c, "score": s} for i, (c, s) in enumerate(top)])

with st.expander("Model metadata"):
    st.json(meta)