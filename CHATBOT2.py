import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df=pd.read_csv("dataset.csv")

vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(df['question'])

model ={
    "vectorizer":vectorizer,
    "x":x,
    "answers":df['answer'].tolist(),
    "question":df['question'].tolist
}

pickle.dump(model, open("model.pkl","wb"))
print("model trained and save successful")