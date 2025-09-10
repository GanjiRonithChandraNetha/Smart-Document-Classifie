import spacy
import PyPDF2
from docx import Document as DocxDocument
import streamlit as st
from typing import Dict, Any, List

class DocumentProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            st.warning("Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_text_from_pdf(self, path: str) -> str:
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def extract_text_from_docx(self, path: str) -> str:
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs).strip()

    def extract_text(self, path: str, ext: str) -> str:
        if ext.lower() == "pdf":
            return self.extract_text_from_pdf(path)
        if ext.lower() in ["docx", "doc"]:
            return self.extract_text_from_docx(path)
        return ""

    def extract_metadata_with_spacy(self, text: str) -> Dict[str, Any]:
        if not self.nlp or not text:
            return {}
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return {
            "word_count": len([t for t in doc if not t.is_space]),
            "sentence_count": len(list(doc.sents)),
            "entities": entities,
            "keywords": self._extract_keywords(doc)
        }

    def _extract_keywords(self, doc, top_n=10) -> List[str]:
        keywords = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 2]
        from collections import Counter
        return [w for w, _ in Counter(keywords).most_common(top_n)]
