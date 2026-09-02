import streamlit as st
import joblib
from features import extract_linguistic_score

# Cache the models so they don't reload on every button click
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

st.title("Clickbait Intensity Analyzer")
st.write("Analyze how clickbait a headline is based on NLP and linguistic rules.")

headline = st.text_input("Enter headline:", "You won't believe what happened next!")

if st.button("Analyze"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        # 1. Base ML Probability
        vec_input = vectorizer.transform([headline])
        ml_prob = model.predict_proba(vec_input)[0][1] # Probability of class 1 (Clickbait)

        # 2. Linguistic Score
        ling_score = extract_linguistic_score(headline)

        # 3. Hybrid Intensity Calculation (70% ML, 30% Linguistic)
        intensity_score = (ml_prob * 0.7) + (ling_score * 0.3)
        intensity_100 = round(intensity_score * 100)

        # Determine Category
        if intensity_100 <= 20: 
            category = "Very Low"
        elif intensity_100 <= 40: 
            category = "Low"
        elif intensity_100 <= 60: 
            category = "Moderate"
        elif intensity_100 <= 80: 
            category = "High"
        else: 
            category = "Very High"

        # Display Results
        st.subheader("Analysis Result")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Intensity Score", f"{intensity_100} / 100")
        col2.metric("Category", category)
        col3.metric("ML Probability", f"{round(ml_prob * 100)}%")

        st.markdown("---")
        st.write("### Scoring Breakdown")
        st.write(f"- **Machine Learning Confidence:** {round(ml_prob * 100)}% (Based on TF-IDF vocabulary patterns)")
        st.write(f"- **Linguistic Triggers Score:** {round(ling_score * 100)} / 100 (Based on punctuation, caps, and trigger words)")

        if intensity_100 > 60:
            st.error("Conclusion: This headline exhibits strong clickbait characteristics.")
        elif intensity_100 > 40:
            st.warning("Conclusion: This headline uses some attention-grabbing tactics.")
        else:
            st.success("Conclusion: This headline appears informative and neutral.")