import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. PAGE CONFIGURATION (Frontend UI) 
st.set_page_config(page_title="Kenyan Language ID", page_icon="🇰🇪", layout="centered")

st.title("🌍 Language Identification System")
st.markdown("""
**Course:** CSC423 – Special Topics  
**Student:** Rachel Mugisha  
**Project:** Identifying English, Swahili, Sheng, and Luo (Dholuo).
""")
st.divider()

# 2. DATA PREPROCESSING FUNCTION 
def clean_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'[^\w\s]', '', text) # Removing punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Removing extra spaces
    return text

# 3. MODEL TRAINING & EVALUATION (The "Brain") 
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found. Please make sure the file is in the same folder as app.py")
        return None, None, None, None, None, None

    
    df['Clean_Text'] = df['Text'].apply(clean_text)
    
    
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
    X = tfidf.fit_transform(df['Clean_Text'])
    y = df['Language']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    labels = sorted(df['Language'].unique())

    return tfidf, model, acc, cm, report, labels


tfidf, model, acc, cm, report, labels = train_model()

# 4. INTERACTIVE PREDICTION SECTION 
st.subheader("📝 Test the AI")
user_input = st.text_input("Enter a short sentence (English, Swahili, Sheng, or Luo):")

if st.button("Identify Language"):
    if user_input:
        
        cleaned_input = clean_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        
        
        prediction = model.predict(vectorized_input)[0]
        
        
        st.success(f"Detected Language: **{prediction}**")
        st.balloons() 
    else:
        st.warning("Please type a sentence first.")

st.divider()

# 5. EVALUATION METRICS (For the 80 Marks) 
with st.expander("📊 View Model Performance & Metrics (Lecturer Review)"):
    if acc:
        st.write(f"### Model Accuracy: {acc*100:.2f}%")
        
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        st.write("#### Detailed Classification Report")
        st.table(pd.DataFrame(report).transpose())
    else:
        st.info("Performance metrics will appear once data is loaded correctly.")

# 6. FOOTER 
st.caption("Developed for ANU BBIT Special Topics Course © 2026")
