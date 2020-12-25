import joblib


class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("toxic_vectorizer_dump.pkl")
        self.model = joblib.load("toxic_model_dump.pkl")
        self.target_names = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']

    def predict_probas(self, text):
        vectorized = self.vectorizer.transform([text])
        return self.model.predict_proba(vectorized)[0]

    def get_result(self, text):
        predictions = self.predict_probas(text)
        if predictions[0] > 0.3:
            return True, dict(zip(self.target_names, predictions))
        else:
            return False, None
