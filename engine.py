
from src.schemas.chroma import ChromaWithUpsert, MiniLML6V2EmbeddingFunctionLangchain
from src.schemas import Configuration
from src.schemas import Parameters
# Utils
import os
import logging
# Langchain
try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.document_loaders import UnstructuredPDFLoader
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")
# Gen AI
from genai.extensions.langchain import LangChainInterface
from genai.model import Credentials
from genai.schemas import GenerateParams

logging.basicConfig(level=logging.INFO)

class Engine:
    
    def __init__(self):
        self._config: Configuration = Configuration()
        self._vector_store = None
        self._llm = None
        self._embeddings = None
        self._filename = None
        self._data = None
        self._texts = None
        self._qa = None
        self.params: Parameters = Parameters()
            
    def _debug_params(self):
        logging.info(f"Chunk size: {self.params.chunk_size}")
        logging.info(f"Chunk overlap: {self.params.chunk_overlap}")
        logging.info(f"Model: {self.params.model}")
        logging.info(f"Temperature: {self.params.temperature}")
        logging.info(f"Top k: {self.params.top_k}")
        logging.info(f"Top p: {self.params.top_p}")
        logging.info(f"Repetition penalty: {self.params.repetition_penalty}")
        logging.info(f"Min new tokens: {self.params.min_new_tokens}")
        logging.info(f"Max new tokens: {self.params.max_new_tokens}")
        logging.info(f"Chain type: {self.params.chain_type}")
        logging.info(f"Search type: {self.params.search_type}")
        logging.info(f"Search k: {self.params.search_k}")
    
    def _load_llm(self):
        self._llm = LangChainInterface(
            model=self.params.model,
            credentials=Credentials(
                self._config.genai_api_key,
                api_endpoint=self._config.genai_endpoint
            ),
            params=GenerateParams(
                decoding_method="sample",
                max_new_tokens=self.params.max_new_tokens,
                min_new_tokens=self.params.min_new_tokens,
                temperature=self.params.temperature,
                repetition_penalty=self.params.repetition_penalty,
                top_k=self.params.top_k,
                top_p=self.params.top_p
            ).dict()
        )
    
    @property
    def filename(self):
        return self._filename
    
    def save_file(self, file):
        self._filename = file.name
        if not os.path.exists(self._filename):
            with open(self._filename, mode='wb') as f:
                f.write(file.getvalue())
    
    def load_data(self):
        loader = UnstructuredPDFLoader(self._filename)
        self._data = loader.load()
        
    def chunk_data(self):
        text_splitter = CharacterTextSplitter(
            chunk_size=self.params.chunk_size,
            chunk_overlap=self.params.chunk_overlap
        )
        self._texts = text_splitter.split_documents(self._data)
        if self._texts is None or len(self._texts) == 0:
            raise Exception("It seems that the document does not contain any text.")
    
    def create_vector_store(self):
        self._vector_store = ChromaWithUpsert(
            collection_name=f"store_minilm6v2",
            embedding_function=MiniLML6V2EmbeddingFunctionLangchain(),  # you can have something here using /embed endpoint
            #persist_directory=".",
        )
        if self._vector_store.is_empty():
            _ = self._vector_store.upsert_texts(
                texts=[doc.page_content for doc in self._texts]
            )
            #self._vector_store.persist()
        
    def is_file_loaded(self):
        return self._filename is not None
        
    def is_vector_store_loaded(self):
        return self._vector_store is not None
    
    def _load_qa(self):
        self._qa = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type=self.params.chain_type,
            retriever=self._vector_store.as_retriever(
                search_type=self.params.search_type,
                search_kwargs={
                    "k": self.params.search_k
                } # 2 chunks of 1000 characters, "stuff" = all fed to the llm
            )
        )
    
    def query(self, q):
        try:
            self._debug_params()
            self._load_llm()
            self._load_qa()
            #q = f""" Given the following text, provide a short and succint summary for: {q}"""
            answer = self._qa.run(q)
            #results = self._qa({"query": q})
            #answer = results["result"]
            #print(results["source_documents"])
        except Exception as e:
            logging.error(e)
            answer = "Sorry, I don't know."
        return answer
        