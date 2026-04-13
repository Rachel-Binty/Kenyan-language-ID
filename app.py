import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Kenyan Language ID", page_icon="🇰🇪", layout="centered")

st.title("🌍 Language Identification System")
st.markdown("""
**Course:** CSC423 – Special Topics  
**Student:** Rachel Mugisha  
**Project:** Identifying English, Swahili, Sheng, Luo, and Mixed code language.
""")
st.divider()

# --- 2. TEXT PREPROCESSING ---
def clean_text(text):
    text = text.lower()                          # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces
    return text

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_all_models():
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found. Make sure it is in the same folder as app.py.")
        return None

    df['Clean_Text'] = df['Text'].apply(clean_text)

    # Character n-gram TF-IDF (highly effective for language ID)
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X = tfidf.fit_transform(df['Clean_Text'])
    y = df['Language']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naïve Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=42),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "model": clf,
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    labels = sorted(df['Language'].unique())
    return tfidf, results, labels

# --- LOAD MODELS ---
output = train_all_models()
if output is None:
    st.stop()

tfidf, results, labels = output

# --- 4. MODEL SELECTOR ---
st.subheader("🤖 Select a Model")
selected_model_name = st.selectbox(
    "Choose which trained model to use for prediction:",
    list(results.keys())
)
selected_model = results[selected_model_name]["model"]

st.divider()

# --- 5. INTERACTIVE PREDICTION ---
st.subheader("📝 Test the Model")
user_input = st.text_input("Enter a short sentence (English, Swahili, Sheng, Luo, or Mixed):")

if st.button("Identify Language"):
    if user_input:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = selected_model.predict(vectorized)[0]
        st.success(f"**{selected_model_name}** detected: **{prediction}**")
        st.balloons()
    else:
        st.warning("Please type a sentence first.")

st.divider()

# --- 6. MODEL COMPARISON TABLE ---
with st.expander("📊 Model Comparison & Evaluation Metrics"):

    # --- Accuracy Comparison ---
    st.write("### Accuracy Comparison")
    comparison_data = {
        "Model": list(results.keys()),
        "Accuracy (%)": [f"{v['accuracy']*100:.2f}%" for v in results.values()],
    }
    st.table(pd.DataFrame(comparison_data))

    # --- Bar Chart ---
    fig_bar, ax = plt.subplots(figsize=(7, 4))
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in model_names]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    bars = ax.bar(model_names, accuracies, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.2f}%", ha='center', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig_bar)

    st.divider()

    # --- Per-model confusion matrix + report ---
    for name, data in results.items():
        st.write(f"### {name}")
        st.write(f"**Accuracy:** {data['accuracy']*100:.2f}%")

        fig_cm, ax_cm = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            data["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix – {name}")
        st.pyplot(fig_cm)

        st.write("**Classification Report:**")
        report_df = pd.DataFrame(data["report"]).transpose()
        st.table(report_df.style.format(precision=2))
        st.divider()

# --- 7. FOOTER ---
st.caption("Developed for ANU BBIT Special Topics Course © 2026")
