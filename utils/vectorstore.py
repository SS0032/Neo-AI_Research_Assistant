from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class VectorStore:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = None

    def add_documents(self, documents):

        self.db = Chroma.from_documents(
            documents,
            self.embedding_model
        )

    def similarity_search(self, query, k=3):

        if self.db is None:
            return []

        docs = self.db.similarity_search(query, k=k)

        results = []

        for d in docs:
            results.append({
                "text": d.page_content,
                "metadata": d.metadata
            })

        return results