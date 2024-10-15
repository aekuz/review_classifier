import pickle

with open('softmax_tfidf_model.pkl', 'rb') as f:
    tfidf_model = pickle.load(f)

with open('tfidf_vectoriser.pkl', 'rb') as f:
    tfidf_vectoriser = pickle.load(f)

def classify(review):
    vec = tfidf_vectoriser.transform([review])
    pred = tfidf_model.predict(vec)
    return pred[0]
