from genai.schemas import ModelType

class Parameters:
    
    default_chunk_size = 1000
    default_chunk_overlap = 5
    default_model = ModelType.FLAN_UL2
    default_temperature = 0.7
    default_top_k = 50
    default_top_p = 0.15
    default_repetition_penalty = 1.5
    default_min_new_tokens = 100
    default_max_new_tokens = 400
    default_chain_type = "stuff"
    default_search_type = "similarity"
    default_search_k = 3
    
    def __init__(self) -> None:
        # Embeddings
        self.chunk_size = Parameters.default_chunk_size
        self.chunk_overlap = Parameters.default_chunk_overlap
        # Model
        self.model = Parameters.default_model
        self.temperature = Parameters.default_temperature
        self.top_k = Parameters.default_top_k
        self.top_p = Parameters.default_top_p
        self.repetition_penalty = Parameters.default_repetition_penalty
        self.min_new_tokens = Parameters.default_min_new_tokens
        self.max_new_tokens = Parameters.default_max_new_tokens
        # QA
        self.chain_type = Parameters.default_chain_type
        self.search_type = Parameters.default_search_type
        self.search_k = Parameters.default_search_k
    