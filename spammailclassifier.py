import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1','v2']]
df.columns= ['label','message']
df['label'] = df['label'].map({'ham': 0,'spam':1})
print("Class Distribution")
print(df['label'].value_counts())

def clean_text(text):
    if pd.isnull(text):
        return " "
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text.strip()
df['message'] = df['message'].apply(clean_text)

X = df['message']
y = df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

vectorizer = TfidfVectorizer(
    stop_words = "english",
    max_df = 0.7,
    min_df = 2,
    ngram_range = (1,2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(class_weight = "balanced")
model.fit(X_train_vec,y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

y_pred = model.predict(X_test_vec)

print("\n Classification Report:")
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

top_spam = sorted(zip(weights,feature_names),reverse=True)[:20]
print("\n Top Spam Words")
for weight,word in top_spam:
    print(word)
top_ham = sorted(zip(weights,feature_names))[:20]
print("\n Top Ham Words")
for weight,word in top_ham:
    print(word)

while True:
    text = input("\n Enter your message here or((exit to stop)")
    if text.lower() == 'exit':
        break
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)

    print("SPAM" if pred[0] == 1 else "NOT SPAM")
    print("-" * 30)




