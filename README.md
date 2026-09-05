# MediaLens: Headline Manipulation Analyzer

> **An Explainable Hybrid NLP System for Detecting and Understanding Headline Manipulation**

Everyday digital-news consumers are exposed to hundreds of headlines across news websites, search engines, and social media feeds. Many headlines use engagement-oriented techniques such as sensational language, curiosity gaps, urgency, and excessive formatting to capture attention.

**MediaLens** is a hybrid Natural Language Processing (NLP) system that detects, measures, and explains the **manipulative intensity** of news and media headlines on a 0–100 scale.

The system combines a **TF-IDF + Logistic Regression** machine learning model with a **rule-based linguistic analysis engine** to provide both a manipulation prediction and an interpretable explanation of the linguistic cues contributing to the result.

> **MediaLens does not determine whether a headline is true or false. It helps users understand how the headline is designed to capture attention.**

---

## 🎯 Problem Statement

Digital users consume large numbers of headlines every day, making it difficult to distinguish straightforward information from headlines designed primarily to maximize engagement.

Common techniques include:

* Sensational or emotionally charged vocabulary
* Curiosity gaps and withheld information
* Excessive capitalization
* Repeated punctuation
* Listicle-style framing
* Urgency and imperative language

Traditional machine learning models can learn statistical patterns from historical datasets, but may struggle with newer or previously unseen forms of engagement bait. Purely rule-based systems, on the other hand, can be rigid and unable to capture broader language patterns.

**MediaLens addresses both limitations using a hybrid ML + linguistic-rule architecture.**

---

## 💡 Solution

MediaLens analyzes a headline through two complementary engines:

### 1. Statistical ML Engine — 70%

The machine learning pipeline captures statistical vocabulary and phrase patterns using:

* Text preprocessing and normalization
* TF-IDF vectorization
* Unigrams and bigrams
* Logistic Regression
* Probability-based classification

### 2. Linguistic Heuristic Engine — 30%

The rule-based engine explicitly searches for engagement-oriented linguistic patterns such as:

* Excessive capitalization
* Repeated or excessive punctuation
* Listicle structures
* Curiosity-gap phrases
* Sensational vocabulary
* Urgency and imperative language
* Emotional or fear-oriented framing

### 3. Explainable AI Layer

Instead of returning only a numerical prediction, MediaLens identifies the linguistic signals detected in the headline and presents them in a human-readable format.

For example:

> **"10 SHOCKING Things You Won't Believe About Your Favorite Celebrity!!!"**

MediaLens may identify:

* Listicle framing
* Sensational vocabulary
* Curiosity gap
* Excessive capitalization
* Excessive punctuation

This makes the prediction easier for users to understand and evaluate.

---

## 👥 Target Users

### Primary Users — Everyday Digital-News Consumers

MediaLens is primarily designed for people who regularly encounter headlines across:

* News websites
* Search engines
* Social media
* Content recommendation feeds

The goal is not to tell users **what to believe**, but to increase awareness of the linguistic techniques being used to capture their attention.

### Potential Future Users

* Journalists and editors
* Media-literacy organizations
* Researchers studying online media
* Content creators
* News and browser-platform developers

---

## ✨ Key Features

| Feature                        | Description                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------- |
| 🤖 **ML Classification**       | TF-IDF + Logistic Regression for statistical headline classification          |
| 🔀 **Hybrid Scoring**          | Combines ML prediction and linguistic heuristics using a 70/30 weighting      |
| 🔍 **Explainable AI**          | Shows the linguistic signals contributing to the result                       |
| 📊 **0–100 Intensity Score**   | Converts the combined prediction into an interpretable manipulation intensity |
| 📁 **Batch Analysis**          | Analyze multiple headlines using CSV or Excel files                           |
| 📈 **Performance Dashboard**   | View model metrics and confusion matrices                                     |
| 🧠 **Architecture Visualizer** | Understand the complete NLP processing pipeline                               |
| ✍️ **Neutral Rewrite**         | Provides a less engagement-oriented alternative headline                      |

---

## 🧠 System Architecture

```text
                         User Input
                       (News Headline)
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       ┌──────────────────┐        ┌────────────────────┐
       │ Statistical ML   │        │ Linguistic Rule    │
       │ Engine (70%)     │        │ Engine (30%)       │
       │                  │        │                    │
       │ Text Processing  │        │ Format Analysis    │
       │       ↓          │        │ Structural Cues    │
       │ TF-IDF           │        │ Lexical Triggers   │
       │       ↓          │        │                    │
       │ Logistic         │        │ Rule Score         │
       │ Regression       │        │ 0.0 – 1.0          │
       └────────┬─────────┘        └─────────┬──────────┘
                │                            │
                │ P(Manipulative)            │
                │                            │
                └─────────────┬──────────────┘
                              ▼
                     ┌─────────────────┐
                     │  Fusion Engine  │
                     │                 │
                     │ ML × 0.70       │
                     │ Rule × 0.30     │
                     └────────┬────────┘
                              ▼
                  Manipulation Intensity
                         0 – 100
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       Intensity Category            XAI Explanation
      Very Low → Very High          Detected Linguistic Cues
                │                           │
                └─────────────┬─────────────┘
                              ▼
                     User Interpretation
                              │
                              ▼
                     Neutral Alternative
```

---

## 📚 Dataset & Provenance

The ML pipeline is trained using the **Clickbait Dataset by Aman Anand Rai** from Kaggle.

**Dataset characteristics:**

* Approximately **32,000 headlines**
* Approximately **50% clickbait / 50% non-clickbait**
* Clickbait sources include BuzzFeed, Upworthy, ViralNova, ThatScoop, BoredPanda, and Huffington Post
* Non-clickbait sources include The New York Times, The Guardian, The Hindu, and WikiNews

### Dataset Limitation

The dataset contains strong publisher/source associations. Consequently, some of the learned patterns may reflect **publisher characteristics rather than purely linguistic characteristics**.

MediaLens therefore supplements the statistical model with explicit linguistic heuristics to detect engagement-oriented patterns that may not be adequately represented in the historical training data.

---

## 🔬 NLP Pipeline & Mathematical Design

### 1. Text Preprocessing

The input headline undergoes preprocessing and normalization before being passed to the feature extraction stage.

### 2. TF-IDF Vectorization

MediaLens converts text into numerical feature vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

Configuration:

```text
ngram_range = (1, 2)
max_features = 5000
```

#### Unigrams + Bigrams

Using both unigrams and bigrams allows the model to capture individual words as well as short phrases.

Examples:

```text
"shocking"
"won't believe"
"breaking news"
```

#### Maximum Features

The vocabulary is limited to 5,000 features to reduce dimensionality, computational cost, and the influence of extremely rare terms.

### 3. Logistic Regression

The resulting TF-IDF vectors are provided to a Logistic Regression classifier.

The model produces a probability representing its estimated likelihood that the headline belongs to the manipulative/clickbait class.

### 4. Hybrid Scoring

The final manipulation intensity combines the ML prediction with the rule-engine score:

$$
\text{Intensity Score}
=
\min
\left(
(P_{\text{ML}}\times0.70)
+
(S_{\text{Rules}}\times0.30),
1.0
\right)
\times100
$$

Where:

* $P_{\text{ML}}$ = ML model probability
* $S_{\text{Rules}}$ = normalized linguistic rule score
* 0.70 = ML contribution
* 0.30 = rule-engine contribution

---

## 📊 Manipulation Intensity Categories

|      Score | Category     |
| ---------: | ------------ |
|   **0–20** | 🟢 Very Low  |
|  **21–40** | 🟢 Low       |
|  **41–60** | 🟡 Moderate  |
|  **61–80** | 🟠 High      |
| **81–100** | 🔴 Very High |

The score represents **manipulation intensity**, not the probability that the underlying information is false.

---

## 🔎 Example Analysis

### Input

> **"10 SHOCKING Things You Won't Believe About Your Favorite Celebrity!!!"**

### MediaLens Output

```text
Manipulation Intensity: 94 / 100
Category: Very High

ML Probability: 91%

Linguistic Signals:
✓ Listicle framing
✓ Excessive capitalization
✓ Excessive punctuation
✓ Sensational vocabulary
✓ Curiosity-gap phrase
```

### Detected Signals

| Signal                  | Evidence                               |
| ----------------------- | -------------------------------------- |
| 🔴 Listicle             | Headline begins with a number          |
| 🔴 Capitalization       | Multiple words use excessive uppercase |
| 🔴 Punctuation          | Repeated exclamation marks             |
| 🔴 Sensational language | "SHOCKING"                             |
| 🔴 Curiosity gap        | "You Won't Believe"                    |

---

## 📈 Model Benchmark

Models are evaluated using a stratified **80/20 train-test split**.

| Model                   | Accuracy |  F1-Score |     Inference | Selection    |
| ----------------------- | -------: | --------: | ------------: | ------------ |
| **Logistic Regression** | **~92%** | **~0.91** |     **<5 ms** | **Selected** |
| Naive Bayes             |     ~89% |     ~0.88 | Extremely Low | Not selected |
| Random Forest           |     ~90% |     ~0.89 |        ~45 ms | Not selected |

### Why Logistic Regression?

Logistic Regression provides a strong balance between:

* Classification performance
* Fast inference
* Model simplicity
* Probability-based predictions
* Compatibility with the continuous hybrid scoring system

> **Note:** Benchmark values should be updated if the final training pipeline produces different measurements.

---

## 🖥️ Application Modules

MediaLens provides multiple components through its Streamlit interface:

### 1. Single Headline Analyzer

Enter an individual headline and receive:

* Manipulation intensity
* ML prediction
* Rule score
* Detected linguistic signals
* Explanation
* Neutral alternative

### 2. Batch Analyzer

Upload a CSV or Excel file containing multiple headlines and analyze them simultaneously.

Example:

```text
headline
-----------------------------------------
10 Things You Didn't Know About AI
Scientists Announce New Climate Findings
You Won't BELIEVE What Happened Next!!!
```

### 3. Model Performance Dashboard

Provides model evaluation information such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Model comparison

### 4. Architecture Visualizer

Provides a visual explanation of the MediaLens NLP and hybrid-scoring pipeline.

---

## 🛠️ Technology Stack

| Component            | Technology                           |
| -------------------- | ------------------------------------ |
| Programming Language | Python                               |
| NLP                  | Scikit-learn                         |
| Feature Extraction   | TF-IDF                               |
| ML Model             | Logistic Regression                  |
| Rule Engine          | Python linguistic heuristics / regex |
| XAI                  | Rule-based evidence explanations     |
| Interface            | Streamlit                            |
| Data Processing      | Pandas                               |
| Visualization        | Matplotlib / Streamlit               |
| Model Serialization  | Pickle                               |
| Dataset              | Kaggle Clickbait Dataset             |

---

## 📂 Project Structure

```text
MediaLens/
│
├── data/
│   └── clickbait_data.csv
│
├── 01_baseline.py
├── app.py
├── features.py
│
├── model.pkl
├── vectorizer.pkl
├── metrics.json
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.9+
* `pip`
* `venv`
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/medialens.git
cd medialens
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

### Retraining the Model

To regenerate the trained model and evaluation metrics:

```bash
python 01_baseline.py
```

---

## ⚠️ Limitations

### 1. Factual Verification

MediaLens analyzes **linguistic manipulation**, not factual accuracy.

A headline can be:

* Manipulative but factually correct
* Non-manipulative but factually incorrect

Therefore, MediaLens should **not** be treated as a fake-news detector.

### 2. Dataset Bias

The training dataset contains publisher/source associations, which may introduce domain and publication-style bias.

### 3. English Language Focus

The current NLP pipeline and heuristic rules are primarily designed for English.

Support for languages such as:

* Hindi
* Marathi
* Other Indian languages

would require multilingual datasets, language-specific preprocessing, and additional linguistic rules.

### 4. Context Limitation

The system evaluates the **headline itself** and does not currently analyze the complete article, publisher reputation, or external evidence.

---

## 🚀 Future Scope

### 🌐 Browser Extension

Develop a browser extension that automatically analyzes headlines encountered on news websites and displays the manipulation intensity directly beside them.

### 🌍 Multilingual Analysis

Extend the system to Hindi, Marathi, and other regional languages.

### 📰 Article-Level Analysis

Analyze headlines together with article content to provide a richer linguistic assessment.

### 🔍 Factual Verification

Integrate a separate fact-verification pipeline to distinguish:

```text
Manipulation Analysis
        +
Factual Verification
        ↓
Comprehensive Media Analysis
```

This would keep **linguistic manipulation** and **factual accuracy** as separate, measurable dimensions.

### 📊 Media Trend Analytics

Analyze large collections of headlines to identify:

* Manipulation trends over time
* Common manipulation techniques
* Publisher-level patterns
* Topic-specific engagement strategies

---

## 🔐 Responsible AI Considerations

MediaLens is intended as an **awareness and media-literacy tool**, not an authority that determines whether a publisher or article is trustworthy.

The system should therefore:

* Explain predictions rather than simply labeling content
* Clearly communicate uncertainty
* Avoid equating manipulation with misinformation
* Avoid making claims about publisher intent
* Keep factual verification separate from linguistic analysis

---

## 🎯 Project Goal

MediaLens aims to move beyond simply asking:

> **"Is this headline clickbait?"**

and instead ask:

> **"What linguistic techniques are being used to capture my attention?"**

By combining **machine learning, linguistic rules, and explainable analysis**, MediaLens provides users with a more transparent way to understand the headlines they encounter every day.

---

## 📌 Project Status

**Current Stage:** Final-Year Academic Project

**Core Pipeline:**

```text
Headline
   ↓
Text Preprocessing
   ↓
TF-IDF Features
   ↓
Logistic Regression ──────┐
                          │
Linguistic Rules ─────────┤
                          ↓
                    Hybrid Fusion
                          ↓
              Manipulation Intensity
                          ↓
                 XAI Explanation
                          ↓
                 User Awareness
```

---

## 👩‍💻 Authors

**MediaLens — Headline Manipulation Analyzer**

Developed as a final-year Artificial Intelligence & Data Science project.
