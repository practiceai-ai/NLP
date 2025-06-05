import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Sample data
data = {
    "review": [
        "I LOVE this product!!",
        "Worst. Purchase. Ever...",
        "It was ok, not great, but not bad.",
        None,
        "I love this product!!",  # Duplicate
        "THE packaging was bad & delivery late."
    ]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)
print("\n")

# Step 1: Handle missing values
df = df.dropna()
print("After removing missing values:")
print(df)
print("\n")

# Step 2: Remove duplicates
df = df.drop_duplicates()
print("After removing duplicates:")
print(df)
print("\n")

# Step 3: Convert text to lowercase
df['cleaned'] = df['review'].str.lower()
print("After converting to lowercase:")
print(df[['review', 'cleaned']])
print("\n")

# Step 4: Remove punctuation
df['cleaned'] = df['cleaned'].str.translate(str.maketrans('', '', string.punctuation))
print("After removing punctuation:")
print(df[['review', 'cleaned']])
print("\n")

# Step 5: Remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])

df['cleaned'] = df['cleaned'].apply(remove_stopwords)
print("After removing stopwords:")
print(df[['review', 'cleaned']])
