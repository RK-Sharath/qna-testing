# Chromadb
try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("Could not import chromdb: Please install chromadb package.")
# Langchain
try:
    import langchain.embeddings
    from langchain.vectorstores import Chroma
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")
# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Could not import langchain: Please install sentence-transformers package.")
# Utils
from typing import Optional, Any, Iterable, List


class MiniLML6V2EmbeddingFunctionLangchain(langchain.embeddings.openai.Embeddings):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode(texts).tolist()
    def embed_query(self, query):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode([query]).tolist()[0]

class ChromaWithUpsert(Chroma):
    def upsert_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.
        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            import uuid
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._collection.upsert(
            metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids
        )
        return ids
    def is_empty(self):
        return self._collection.count()==0

