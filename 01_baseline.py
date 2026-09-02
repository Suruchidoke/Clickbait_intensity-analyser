import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Load and clean
df = pd.read_csv(r"C:\Users\Suruchi\OneDrive\Desktop\project\Clickbait intensity analyser\clickbait-analyzer\data\clickbait_data.csv")
df = df.drop_duplicates().dropna()

print(f"Cleaned dataset shape: {df.shape}")
print(df["clickbait"].value_counts())

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["headline"], df["clickbait"], test_size=0.2, random_state=42, stratify=df["clickbait"]
)

# 3. Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 6. Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl successfully.")