import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import json

# 1. Load and clean
df = pd.read_csv("data/clickbait_data.csv")
df = df.drop_duplicates().dropna()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["headline"], df["clickbait"], test_size=0.2, random_state=42, stratify=df["clickbait"]
)

# 3. Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train & Compare Multiple Models (Feature 6)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

comparison_results = {}
for name, m in models.items():
    m.fit(X_train_vec, y_train)
    preds = m.predict(X_test_vec)
    comparison_results[name] = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "F1 Score": round(f1_score(y_test, preds), 4)
    }

# 5. Select Primary Model for Production
primary_model = models["Logistic Regression"]
y_pred = primary_model.predict(X_test_vec)

# 6. Save artifacts
joblib.dump(primary_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# 7. Save Metrics for Streamlit Dashboard
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

metrics = {
    "accuracy": report["accuracy"],
    "classification_report": report,
    "confusion_matrix": cm,
    "test_samples": len(y_test),
    "model_comparison": comparison_results
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
print("Multi-model training complete. Saved metrics.json successfully.")