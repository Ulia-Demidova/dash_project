from nltk.stem import WordNetLemmatizer
import re
from functools import lru_cache


class Preprocesser:
    def __init__(self, stopwords):
        self.bad_symbol_regexp = re.compile('[^а-яёa-z\s]')
        ru_lemmatizer = WordNetLemmatizer()

        @lru_cache(maxsize=10 ** 6)
        def lru_lemmatizer(word):
            return ru_lemmatizer.lemmatize(word)

        self.lemmatizer = lru_lemmatizer
        self.stopwords = stopwords

    def __call__(self, text):
        text = text.lower()
        text = re.sub(self.bad_symbol_regexp, '', text)
        lemmas = [
            self.lemmatizer(token)
            for token in text.split()
        ]
        lemmas = [
            token
            for token in lemmas
            if token not in self.stopwords
        ]

        return ' '.join(lemmas)