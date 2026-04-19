from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import os

# Mock guidelines for demonstration if no real data is provided
DEFAULT_GUIDELINES = """
1. For patients with LeadTime > 15 days: Send a personal phone call reminder 3 days before.
2. For patients with a history of hypertension or diabetes: Emphasize the importance of regular checkups in the reminder.
3. For patients over 60: Ask if they need assistance with transportation.
4. For high-risk no-show patients (Risk Score > 0.7): Offer a telehealth option if the visit doesn't require physical examination.
5. For patients who missed the last appointment: Acknowledge the missed visit and re-emphasize the care plan.
"""

class GuidelineRetriever:
    def __init__(self):
        # In a real app, you'd load from a file or DB
        self.embeddings = OpenAIEmbeddings() if os.getenv("OPENAI_API_KEY") else None
        self.vectorstore = None
        
    def initialize_mock_data(self):
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        docs = [Document(page_content=x) for x in text_splitter.split_text(DEFAULT_GUIDELINES)]
        
        if self.embeddings:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            # Fallback to simple keyword search if no API key
            self.docs = docs

    def retrieve(self, query):
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=2)
        else:
            # Simple fallback: return all for now or filter
            return [doc.page_content for doc in self.docs]
