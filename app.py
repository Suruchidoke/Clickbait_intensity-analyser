import streamlit as st
import pandas as pd
import joblib
import json
from features import extract_linguistic_features

@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return model, vectorizer, metrics

model, vectorizer, metrics = load_assets()

st.title("MediaLens: Headline Manipulation Analyzer")

# 4-Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Single Analyzer", "Batch Upload", "System Architecture", "Model Performance"])

# TAB 1: Original Single Analyzer
with tab1:
    headline = st.text_input("Enter headline:", "You won't believe what happened next!")
    if st.button("Analyze Single"):
        if headline.strip() == "":
            st.warning("Please enter a headline.")
        else:
            vec_input = vectorizer.transform([headline])
            ml_prob = model.predict_proba(vec_input)[0][1] 
            ling_score, triggers = extract_linguistic_features(headline)

            intensity_score = (ml_prob * 0.7) + (ling_score * 0.3)
            intensity_100 = round(intensity_score * 100)

            st.markdown(f"### MANIPULATION INTENSITY: {intensity_100} / 100")
            
            st.write("#### Scoring Breakdown")
            st.progress(ml_prob, text=f"ML Prediction Confidence: {round(ml_prob * 100)} / 100")
            st.progress(ling_score, text=f"Linguistic Rules Score: {round(ling_score * 100)} / 100")
            
            st.caption("Formula: ML provides 70% weight, Linguistic Rules provide 30%")
            st.markdown("---")

            with st.expander("🔍 View Explainable AI (XAI) Breakdown", expanded=True):
                for rule, reason in triggers.items():
                    if rule == 'Neutral Language':
                        st.markdown(f"🟢 **{rule}:** {reason}")
                    else:
                        st.markdown(f"🔴 **{rule}:** {reason}")
                
                st.markdown(f"🟢 **ML Model:** {round(ml_prob*100)}% confident based on historical vocabulary patterns.")

# TAB 2: Batch Processing (Now with Excel & CSV)
with tab2:
    st.header("Batch CSV/Excel Manipulation Analysis")
    st.write("Upload a file (`.csv`, `.xlsx`, `.xls`) containing a column named `headline`.")
    
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)
                
            if 'headline' not in batch_df.columns:
                st.error("Error: File must contain a column named 'headline'.")
            else:
                results = []
                with st.spinner("Processing headlines..."):
                    for text in batch_df['headline']:
                        if pd.isna(text):
                            continue
                        
                        vec_input = vectorizer.transform([str(text)])
                        ml_prob = model.predict_proba(vec_input)[0][1]
                        ling_score, _ = extract_linguistic_features(str(text))
                        
                        score = round(((ml_prob * 0.7) + (ling_score * 0.3)) * 100)
                        
                        if score <= 20: cat = "Very Low"
                        elif score <= 40: cat = "Low"
                        elif score <= 60: cat = "Moderate"
                        elif score <= 80: cat = "High"
                        else: cat = "Very High"
                        
                        results.append({"Headline": text, "Manipulation Intensity": score, "Intensity Category": cat})
                
                results_df = pd.DataFrame(results)
                st.success(f"Successfully analyzed {len(results_df)} headlines.")
                st.dataframe(results_df, use_container_width=True)
                
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Analyzed Data",
                    data=csv_export,
                    file_name='medialens_batch_results.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Failed to read file. Error: {str(e)}")

# TAB 3: NLP Pipeline Architecture
with tab3:
    st.header("System Architecture")
    st.markdown("""
    ### Dual-Engine Pipeline
    This application grades text using a hybrid approach, ensuring resilience against both known linguistic tropes and statistical vocabulary patterns.
    
    **1. Machine Learning Pipeline (70% Weight)**
    * `Headline` → `Text Cleaning` → `Tokenization` → `Stopword Removal` → `TF-IDF Vectorization` → `Logistic Regression Classifier` → `Probability Score`
    
    **2. Linguistic Rule Engine (30% Weight)**
    * `Headline` → `Feature Extraction (Regex)` → `Pattern Matching (Format, Structure, Vocab)` → `Bounded Heuristic Score`
    
    **3. Fusion Engine**
    * Computes weighted average and maps explanation triggers for Explainable AI (XAI) output.
    """)

# TAB 4: Model Performance
with tab4:
    st.header("Dataset & Model Evaluation")
    
    st.subheader("Algorithm Comparison")
    comp_df = pd.DataFrame.from_dict(metrics["model_comparison"], orient="index")
    st.dataframe(comp_df.style.highlight_max(axis=0))
    st.info("Logistic Regression was selected for production because it provided the strongest balance between performance, interpretability, and computational efficiency during text vectorization.")
    
    st.markdown("---")
    
    st.subheader("Production Model Diagnostics (Logistic Regression)")
    st.metric("Global Accuracy", f"{round(metrics['accuracy'] * 100, 2)}%")
    st.write(f"**Test Set Size:** {metrics['test_samples']} samples")
    
    st.subheader("Confusion Matrix")
    cm = metrics["confusion_matrix"]
    st.code(f"""
                Predicted
              Neutral    Manipulative
Actual Neut   {cm[0][0]}        {cm[0][1]}
Actual Manip  {cm[1][0]}        {cm[1][1]}
    """)
    
    st.subheader("Classification Report")
    st.json(metrics['classification_report'])