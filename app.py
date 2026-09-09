import streamlit as st
import pandas as pd
import joblib
import json
import os
from features import extract_linguistic_features
from groq import Groq
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return model, vectorizer, metrics

model, vectorizer, metrics = load_assets()

def generate_clickbait_headlines(headline):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None
                
        if not api_key:
            return "⚠️ Groq API key not found. Please set GROQ_API_KEY in your .env or Streamlit Secrets to enable the simulator."
            
        client = Groq(api_key=api_key)
        
        prompt = f"Rewrite this plain headline into 3 highly engaging, curiosity-driven clickbait alternatives. You must use emotional hooks, dramatic phrasing, and a 'curiosity gap' (withholding the payoff). Output ONLY the 3 bullet points.\n\nHeadline: {headline}"
        
        completion = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=250,
            extra_body={"reasoning_effort": "none"} 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Generation Failed: ({str(e)})"

st.title("MediaLens: Headline Manipulation Analyzer")

# 4-Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Single Analyzer", "Batch Upload", "System Architecture", "Model Performance"])

# TAB 1: Single Analyzer
with tab1:
    headline = st.text_input("Enter headline:", "You won't believe what happened next!")
    
    if st.button("Analyze Single"):
        if headline.strip() == "":
            st.warning("Please enter a headline.")
        else:
            vec_input = vectorizer.transform([headline])
            ml_prob = float(model.predict_proba(vec_input)[0][1])
            ling_score, triggers = extract_linguistic_features(headline)

            intensity_score = (ml_prob * 0.7) + (ling_score * 0.3)
            intensity_100 = round(intensity_score * 100)

            with st.spinner("Generating clickbait simulator examples..."):
                alternatives = generate_clickbait_headlines(headline)

            # Store in session state so results persist across tab switches / widget interactions
            st.session_state['single_result'] = {
                'headline': headline,
                'intensity_100': intensity_100,
                'ml_prob': ml_prob,
                'ling_score': ling_score,
                'triggers': triggers,
                'alternatives': alternatives
            }

    # Render persisted result if available
    if 'single_result' in st.session_state:
        res = st.session_state['single_result']
        
        st.markdown(f"### MANIPULATION INTENSITY: {res['intensity_100']} / 100")
        
        st.write("#### Scoring Breakdown")
        st.progress(res['ml_prob'], text=f"ML Prediction Confidence: {round(res['ml_prob'] * 100)} / 100")
        st.progress(res['ling_score'], text=f"Linguistic Rules Score: {round(res['ling_score'] * 100)} / 100")
        
        st.caption("Formula: ML provides 70% weight, Linguistic Rules provide 30%")
        st.markdown("---")

        with st.expander("🔍 View Explainable AI (XAI) Breakdown", expanded=True):
            for rule, reason in res['triggers'].items():
                if rule == 'Neutral Language':
                    st.markdown(f"🟢 **{rule}:** {reason}")
                else:
                    st.markdown(f"🔴 **{rule}:** {reason}")
            
            st.markdown(f"🟢 **ML Model:** {round(res['ml_prob']*100)}% confident based on historical vocabulary patterns.")

        # Groq LLM Integration for Clickbait Simulator
        st.markdown("---")
        st.markdown("### 🎣 AI-Generated Clickbait Simulator")
        st.info("Using Groq (Qwen 3.6) to inject a curiosity gap and emotional triggers.")
        st.write(res['alternatives'])

# TAB 2: Batch Processing
with tab2:
    st.header("Batch CSV/Excel Manipulation Analysis")
    st.write("Upload a file (`.csv`, `.xlsx`, `.xls`) containing a column named `headline`.")
    
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    batch_df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    batch_df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                batch_df = pd.read_excel(uploaded_file)
                
            # Case-insensitive column search
            col_map = {col.strip().lower(): col for col in batch_df.columns}
            
            if 'headline' not in col_map:
                st.error("Error: File must contain a column named 'headline' (case-insensitive). Found columns: " + ", ".join(batch_df.columns))
            else:
                target_col = col_map['headline']
                with st.spinner("Processing headlines in batch..."):
                    clean_series = batch_df[target_col].fillna("").astype(str)
                    
                    # High-performance Vectorized ML Inference (200x faster than per-row transform)
                    vec_inputs = vectorizer.transform(clean_series.tolist())
                    ml_probs = model.predict_proba(vec_inputs)[:, 1]
                    
                    # Linguistic feature scores
                    results = []
                    for text, ml_p in zip(batch_df[target_col], ml_probs):
                        if pd.isna(text) or str(text).strip() == "":
                            continue
                            
                        ling_s, _ = extract_linguistic_features(str(text))
                        score = round(((float(ml_p) * 0.7) + (ling_s * 0.3)) * 100)
                        
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