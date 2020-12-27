import joblib
from preprocess import Preprocesser
from nltk.corpus import stopwords
import numpy as np


class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("vectorizer.pkl")
        self.model = joblib.load("model.pkl")
        self.stopwords = set(stopwords.words('english'))
        self.preprocesser = Preprocesser(stopwords=self.stopwords)

    def predict_text(self, text):
        try:
            text = self.preprocesser(text)
            vectorized = self.vectorizer.transform([text])
            return np.transpose(np.array(self.model.predict_proba(vectorized))[:, :, 1])[0].tolist()
        except:
            print("prediction error")
            return None
