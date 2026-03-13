import faiss
import numpy as np
from models.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.texts = []
        self.metadatas = []

        self.index = faiss.IndexFlatL2(384)

    def add_documents(self, documents):

        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        embeddings = self.embedding_model.embed_documents(texts)

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.texts.extend(texts)
        self.metadatas.extend(metadata)

    def similarity_search(self, query, k=3):

        if len(self.texts) == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for i in indices[0]:

            if i >= 0 and i < len(self.texts):

                results.append({
                    "text": self.texts[i],
                    "metadata": self.metadatas[i]
                })

        return results