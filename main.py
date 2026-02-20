import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.tsv", sep="\t", header=None)
df.columns = ['label','message']

# Convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save files
pickle.dump(tfidf, open("vectorizer.pkl","wb"))
pickle.dump(model, open("model.pkl","wb"))

print("✅ Model trained and saved successfully!")
