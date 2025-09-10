from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, os

class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.categories = [
            'Invoice','Contract','Report','Email','Legal Document',
            'Technical Manual','Resume','Research Paper','Policy Document','Other'
        ]

    def create_sample_training_data(self):
        texts = [
            "Invoice total payment due amount customer supplier",
            "Agreement terms contract binding obligations",
            "Annual report financial data analysis",
            "Email correspondence subject sender recipient",
            "Court legal document lawsuit attorney",
            "Technical manual installation guide setup",
            "Resume work experience education skills",
            "Research paper methodology results conclusion",
            "Policy guidelines compliance regulations"
        ]
        labels = [
            'Invoice','Contract','Report','Email','Legal Document',
            'Technical Manual','Resume','Research Paper','Policy Document'
        ]
        return texts, labels

    def train_model(self):
        X, y = self.create_sample_training_data()
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        return acc

    def classify_document(self, text):
        if not self.model: return "Other", 0.0
        proba = self.model.predict_proba([text])[0]
        return self.model.predict([text])[0], max(proba)

    def save_model(self, path): joblib.dump(self.model, path)
    def load_model(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path); return True
        return False
