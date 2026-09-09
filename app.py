import streamlit as st
import pandas as pd
import joblib
import json
import os
import html
import requests
import xml.etree.ElementTree as ET
import altair as alt
from features import extract_linguistic_features
from groq import Groq
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="MediaLens: Headline Manipulation Analyzer",
    page_icon="🔍",
    layout="wide"
)

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

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        return None
    return Groq(api_key=api_key)

def generate_clickbait_headlines(headline):
    try:
        client = get_groq_client()
        if not client:
            return "⚠️ Groq API key not found. Please set GROQ_API_KEY in .env or Streamlit Secrets."
            
        prompt = f"Rewrite this plain headline into 3 highly engaging, curiosity-driven clickbait alternatives. You must use emotional hooks, dramatic phrasing, and a 'curiosity gap' (withholding the payoff). Output ONLY the 3 bullet points.\n\nHeadline: {headline}"
        
        completion = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=250,
            extra_body={"reasoning_effort": "none"} 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Generation Failed: ({str(e)})"

def neutralize_headline(headline):
    try:
        client = get_groq_client()
        if not client:
            return "⚠️ Groq API key not found."
            
        prompt = f"Rewrite this sensationalized news headline into a concise, strictly factual, objective, neutral headline. Strip all emotional hooks, curiosity gaps, and dramatic exaggeration. Output ONLY the rewritten headline without quotation marks or explanations.\n\nManipulative Headline: {headline}"
        
        completion = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
            extra_body={"reasoning_effort": "none"}
        )
        return completion.choices[0].message.content.strip().strip('"').strip("'")
    except Exception as e:
        return f"Neutralizer Failed: ({str(e)})"

# RSS Live Feed Sources
RSS_OUTLETS = {
    "BBC News": {"url": "http://feeds.bbci.co.uk/news/rss.xml", "category": "Global / Neutral", "flag": "🇬🇧"},
    "The New York Times": {"url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "category": "General News", "flag": "🇺🇸"},
    "The Verge": {"url": "https://www.theverge.com/rss/index.xml", "category": "Tech & Culture", "flag": "⚡"},
    "Fox News": {"url": "https://moxie.foxnews.com/google-publisher/latest.xml", "category": "US News & Politics", "flag": "📰"},
    "BuzzFeed News": {"url": "https://www.buzzfeed.com/world.xml", "category": "Viral & Entertainment", "flag": "🌐"}
}

def fetch_and_analyze_live_news():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MediaLens-Analyzer/2.0"}
    all_headlines = []

    for source_name, meta in RSS_OUTLETS.items():
        try:
            resp = requests.get(meta["url"], headers=headers, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")
                if not items:
                    items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
                
                for item in items[:7]: # Top 7 headlines per outlet
                    title_elem = item.find("title") if item.find("title") is not None else item.find("{http://www.w3.org/2005/Atom}title")
                    link_elem = item.find("link") if item.find("link") is not None else item.find("{http://www.w3.org/2005/Atom}link")
                    
                    if title_elem is not None and title_elem.text:
                        raw_title = html.unescape(title_elem.text.strip())
                        link_url = ""
                        if link_elem is not None:
                            link_url = link_elem.text.strip() if link_elem.text else link_elem.attrib.get("href", "")
                        
                        all_headlines.append({
                            "Outlet": source_name,
                            "Category": meta["category"],
                            "Flag": meta["flag"],
                            "Headline": raw_title,
                            "Link": link_url
                        })
        except Exception:
            continue

    if not all_headlines:
        return pd.DataFrame()

    # Batch vectorization & ML prediction
    texts = [h["Headline"] for h in all_headlines]
    vec_inputs = vectorizer.transform(texts)
    ml_probs = model.predict_proba(vec_inputs)[:, 1]

    # Combine with linguistic heuristics
    results = []
    for h_data, ml_p in zip(all_headlines, ml_probs):
        ling_s, triggers = extract_linguistic_features(h_data["Headline"])
        score = round(((float(ml_p) * 0.7) + (ling_s * 0.3)) * 100)
        
        if score <= 25:
            cat = "Very Low"
            badge = "🟢 Low"
        elif score <= 50:
            cat = "Moderate"
            badge = "🟡 Moderate"
        elif score <= 75:
            cat = "High"
            badge = "🟠 High"
        else:
            cat = "Very High"
            badge = "🔴 Severe"

        trigger_summary = ", ".join(triggers.keys()) if triggers else "Neutral"
        
        results.append({
            "Outlet": h_data["Outlet"],
            "Flag": h_data["Flag"],
            "Category": h_data["Category"],
            "Headline": h_data["Headline"],
            "Link": h_data["Link"],
            "Score": score,
            "Level": badge,
            "Triggers": trigger_summary,
            "ML Confidence": round(float(ml_p) * 100),
            "Linguistic Score": round(ling_s * 100)
        })

    return pd.DataFrame(results)

st.title("MediaLens: Headline Manipulation Analyzer")

# 5-Tab Navigation
tab1, tab2, tab_live, tab3, tab4 = st.tabs([
    "Single Analyzer", 
    "Batch Upload", 
    "📡 Live News Radar", 
    "System Architecture", 
    "Model Performance"
])

# ==========================================
# TAB 1: Single Headline Analyzer
# ==========================================
with tab1:
    headline = st.text_input("Enter headline:", "You won't believe what happened next!")
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_analyze = st.button("🔍 Analyze Single", use_container_width=True)
    with col_btn2:
        run_neutralize = st.button("🛡️ Neutralize Headline (AI De-Baiter)", use_container_width=True)

    if run_analyze:
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

            st.session_state['single_result'] = {
                'headline': headline,
                'intensity_100': intensity_100,
                'ml_prob': ml_prob,
                'ling_score': ling_score,
                'triggers': triggers,
                'alternatives': alternatives
            }

    if run_neutralize:
        if headline.strip() == "":
            st.warning("Please enter a headline.")
        else:
            with st.spinner("Neutralizing sensationalism with Qwen 3.6..."):
                st.session_state['neutralized_single'] = neutralize_headline(headline)

    # Render persisted single analysis
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

        st.markdown("---")
        st.markdown("### 🎣 AI-Generated Clickbait Simulator")
        st.info("Using Groq (Qwen 3.6) to inject a curiosity gap and emotional triggers.")
        st.write(res['alternatives'])

    if 'neutralized_single' in st.session_state:
        st.markdown("---")
        st.markdown("### 🛡️ Factual Headline Rewrite (AI De-Baiter)")
        st.success(f"**Objective Headline:** {st.session_state['neutralized_single']}")

# ==========================================
# TAB 2: Batch Processing
# ==========================================
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
                
            col_map = {col.strip().lower(): col for col in batch_df.columns}
            
            if 'headline' not in col_map:
                st.error("Error: File must contain a column named 'headline' (case-insensitive). Found columns: " + ", ".join(batch_df.columns))
            else:
                target_col = col_map['headline']
                with st.spinner("Processing headlines in batch..."):
                    clean_series = batch_df[target_col].fillna("").astype(str)
                    
                    vec_inputs = vectorizer.transform(clean_series.tolist())
                    ml_probs = model.predict_proba(vec_inputs)[:, 1]
                    
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

# ==========================================
# TAB 3: NEW! Live News Radar & Outlet Index
# ==========================================
with tab_live:
    st.header("📡 Real-Time Media Manipulation Radar")
    st.write("Live monitoring of real-world breaking news feeds to benchmark outlet manipulation patterns.")
    
    col_scan, col_info = st.columns([1, 3])
    with col_scan:
        scan_clicked = st.button("🔄 Scan Today's Live News", use_container_width=True)
    with col_info:
        st.caption("Fetches real-time headlines across BBC, NY Times, The Verge, Fox News, and BuzzFeed.")

    if scan_clicked or 'live_news_df' not in st.session_state:
        with st.spinner("Scanning live feeds & computing manipulation scores..."):
            st.session_state['live_news_df'] = fetch_and_analyze_live_news()

    live_df = st.session_state.get('live_news_df', pd.DataFrame())

    if not live_df.empty:
        # 1. Summary KPIs
        outlet_avg = live_df.groupby("Outlet")["Score"].mean().reset_index()
        highest_outlet = outlet_avg.sort_values(by="Score", ascending=False).iloc[0]
        lowest_outlet = outlet_avg.sort_values(by="Score", ascending=True).iloc[0]
        global_avg = round(live_df["Score"].mean(), 1)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Headlines Scanned", f"{len(live_df)}")
        kpi2.metric("Overall Average Index", f"{global_avg} / 100")
        kpi3.metric("Highest Manipulation Outlet", f"{highest_outlet['Outlet']}", f"{round(highest_outlet['Score'], 1)} / 100", delta_color="inverse")
        kpi4.metric("Most Objective Outlet", f"{lowest_outlet['Outlet']}", f"{round(lowest_outlet['Score'], 1)} / 100")

        st.markdown("---")

        # 2. Outlet Leaderboard Chart
        st.subheader("📊 Outlet Manipulation Index Leaderboard")
        outlet_chart_data = outlet_avg.rename(columns={"Score": "Average Manipulation Score"}).sort_values(by="Average Manipulation Score", ascending=False)
        
        chart = alt.Chart(outlet_chart_data).mark_bar(cornerRadius=6).encode(
            x=alt.X("Average Manipulation Score:Q", scale=alt.Scale(domain=[0, 100]), title="Average Manipulation Score (0–100)"),
            y=alt.Y("Outlet:N", sort="-x", title="Media Source"),
            color=alt.Color("Average Manipulation Score:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
            tooltip=["Outlet", alt.Tooltip("Average Manipulation Score:Q", format=".1f")]
        ).properties(height=260)
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("Lower scores indicate more objective, factual reporting; higher scores indicate emotional, curiosity-gap framing.")

        st.markdown("---")

        # 3. Filterable Live Feed & One-Click Neutralizer
        st.subheader("📰 Live Headline Stream & AI De-Clickbaiter")
        
        fil_col1, fil_col2 = st.columns([1, 1])
        with fil_col1:
            selected_outlet = st.selectbox("Filter by Outlet:", ["All Outlets"] + list(RSS_OUTLETS.keys()))
        with fil_col2:
            selected_level = st.selectbox("Filter by Severity:", ["All Severity Levels", "High / Very High (50+)", "Low / Neutral (<30)"])

        filtered_df = live_df.copy()
        if selected_outlet != "All Outlets":
            filtered_df = filtered_df[filtered_df["Outlet"] == selected_outlet]
        if selected_level == "High / Very High (50+)":
            filtered_df = filtered_df[filtered_df["Score"] >= 50]
        elif selected_level == "Low / Neutral (<30)":
            filtered_df = filtered_df[filtered_df["Score"] < 30]

        st.write(f"Showing **{len(filtered_df)}** headlines:")

        for idx, row in filtered_df.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([5, 2, 2])
                with c1:
                    st.markdown(f"**{row['Flag']} {row['Outlet']}** • [{row['Headline']}]({row['Link']})")
                    st.caption(f"Triggers: `{row['Triggers']}` | ML: {row['ML Confidence']}% | Linguistic: {row['Linguistic Score']}%")
                with c2:
                    st.markdown(f"### {row['Score']} / 100")
                    st.write(f"{row['Level']}")
                with c3:
                    neutralize_key = f"neut_{idx}_{row['Score']}"
                    if st.button("⚡ Neutralize", key=neutralize_key, use_container_width=True):
                        with st.spinner("De-clickbaiting..."):
                            st.session_state[f"result_{neutralize_key}"] = neutralize_headline(row['Headline'])
                
                # If neutralized result exists for this headline
                if f"result_{neutralize_key}" in st.session_state:
                    st.success(f"🟢 **Neutral Objective Headline:** {st.session_state[f'result_{neutralize_key}']}")

                st.divider()

    else:
        st.info("Click 'Scan Today's Live News' above to fetch and analyze real-time headlines.")

# ==========================================
# TAB 4: NLP Pipeline Architecture
# ==========================================
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

# ==========================================
# TAB 5: Model Performance
# ==========================================
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