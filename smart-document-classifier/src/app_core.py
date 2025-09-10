from src.processor import DocumentProcessor
from src.classifier import DocumentClassifier
from datetime import datetime

class SmartDocumentClassifierApp:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.classifier = DocumentClassifier()
        self.model_path = "models/document_classifier_model.joblib"
        if not self.classifier.load_model(self.model_path):
            self._initialize_model()

    def _initialize_model(self):
        acc = self.classifier.train_model()
        self.classifier.save_model(self.model_path)
        return acc

    def process_document(self, path, filename, ext):
        text = self.processor.extract_text(path, ext)
        if not text:
            return {"filename": filename, "status": "error", "error": "Could not extract text"}
        category, conf = self.classifier.classify_document(text)
        meta = self.processor.extract_metadata_with_spacy(text)
        return {
            "filename": filename,
            "category": category,
            "confidence": conf,
            "text_preview": text[:500] + ("..." if len(text) > 500 else ""),
            "word_count": meta.get("word_count",0),
            "sentence_count": meta.get("sentence_count",0),
            "keywords": meta.get("keywords",[]),
            "entities": meta.get("entities",{}),
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }
