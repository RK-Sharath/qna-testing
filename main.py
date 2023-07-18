import streamlit as st
from engine import Engine
from params import Parameters
import time
from genai.schemas import ModelType

st.set_page_config(
    initial_sidebar_state="collapsed"
)

if "engine" not in st.session_state:
    st.session_state.engine: Engine = Engine()
if "previous_query" not in st.session_state:
    st.session_state.previous_query: str = None
if "rerun_requested" not in st.session_state:
    st.session_state.rerun_requested: bool = False


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parameters
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
with st.sidebar:
    
    st.header("Embeddings")
    # Chunk size
    st.session_state.engine.params.chunk_size = st.number_input(
        label="Chunk size (in characters)",
        value=Parameters.default_chunk_size,
        min_value=100,
        max_value=1000,
        step=10,
        disabled=(
            st.session_state.engine.is_file_loaded() 
            or st.session_state.engine.is_vector_store_loaded() 
        ),
        help="The size of the text chunks that will be used to create the embeddings."
    )
    # Chunk overlap
    st.session_state.engine.params.chunk_overlap = st.number_input(
        label="Chunk overlap (% of chunk size)",
        value=Parameters.default_chunk_overlap,
        min_value=1,
        max_value=100,
        step=5,
        disabled=(
            st.session_state.engine.is_file_loaded() 
            or st.session_state.engine.is_vector_store_loaded()
        ),
        help="The percentage of overlap between two consecutive chunks."
    )
    
    st.write("")
    st.header("Model")
    # Model
    st.session_state.engine.params.model = st.selectbox(
        label="Model",
        options=[member.value for member in ModelType],
        index=next((idx for idx, member in enumerate(ModelType) if member.value == Parameters.default_model), None),
        format_func=lambda option: next((member.name for member in ModelType if member.value == option), None),
        help="The Large Language Model (LLM) that will be used to answer the question."
    )
    # Temperature
    st.session_state.engine.params.temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=2.0,
        value=Parameters.default_temperature,
        step=0.01,
        help="The higher the temperature, the more random the generated text."
    )
    # Top K
    st.session_state.engine.params.top_k = st.slider(
        label="Top K",
        min_value=1,
        max_value=100,
        value=Parameters.default_top_k,
        step=1,
        help="The number of tokens to sample from."
    )
    # Top P
    st.session_state.engine.params.top_p = st.slider(
        label="Top P",
        min_value=0.0,
        max_value=1.0,
        value=Parameters.default_top_p,
        step=0.01,
        help="The cumulative probability of the most likely tokens to sample from."
    )
    # Repetition penalty
    st.session_state.engine.params.repetition_penalty = st.slider(
        label="Repetition penalty",
        min_value=1.0,
        max_value=2.0,
        value=Parameters.default_repetition_penalty,
        step=0.01,
        help="The penalty for repeating tokens."
    )
    cols = st.columns(2)
    with cols[0]:
        # Min new tokens
        st.session_state.engine.params.min_new_tokens = st.number_input(
            label="Min new tokens",
            value=Parameters.default_min_new_tokens,
            min_value=1,
            max_value=1000,
            step=1,
            help="The minimum number of tokens that will be added to the previous answer."
        )
    with cols[1]:
        # Max new tokens
        st.session_state.engine.params.max_new_tokens = st.number_input(
            label="Max new tokens",
            value=Parameters.default_max_new_tokens,
            min_value=1,
            max_value=1000,
            step=1,
            help="The maximum number of tokens that will be added to the previous answer."
        )
    
    st.write("")
    st.header("QA Chain")
    st.session_state.engine.params.chain_type = st.selectbox(
        label="Chain type",
        options=["stuff", "map_reduce", "refine", "map-rerank"],
        index=next((idx for idx, chain_type in enumerate(["stuff", "map_reduce", "refine", "map-rerank"]) if chain_type == Parameters.default_chain_type), None),
        help="The way the snippets are fed into the LLM."
    )
    st.session_state.engine.params.search_type = st.selectbox(
        label="Search type",
        options=["similarity"],
        index=next((idx for idx, search_type in enumerate(["similarity", "mmr"]) if search_type == Parameters.default_search_type), None),
        help="The type of search that will be used to retrieve the snippets."
    )
    st.session_state.engine.params.search_k = st.slider(
        label="Number of snippets retrieved",
        min_value=1,
        max_value=100,
        value=Parameters.default_search_k,
        step=1,
        help="The number of snippets that will be retrieved."
    )
    if st.button("Run"):
        st.session_state.rerun_requested = True

if not st.session_state.engine.is_file_loaded():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # File upload
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    st.header("Retrieval Augmented Search")
    st.markdown("")
    file = st.file_uploader("Upload a PDF file from your computer", type="pdf")
    if file is not None:
        progress_bar = st.progress(0, text="Uploading file...")
        st.session_state.engine.save_file(file)
        progress_bar.progress(100, text=f"File {file.name} is successfully saved.")
        progress_bar.empty()
        st.experimental_rerun()
    with st.expander("You can use this file as an example"):
        st.download_button(
            label="Download Niagara Water Quality Report 2022",
            file_name="Niagara-WQR-English-2022_053122-HR.pdf",
            data=open("examples/Niagara-WQR-English-2022_053122-HR.pdf", "rb").read(),
            mime="application/pdf"
        )
        
        
elif not st.session_state.engine.is_vector_store_loaded():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Load vector store
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    st.header(st.session_state.engine.filename)
    st.markdown("")
    progress_bar = st.progress(0, text="Loading vector store...")
    st.session_state.engine.load_data()
    progress_bar.progress(20, text=f"The data has been injested.")
    time.sleep(0.1)
    st.session_state.engine.chunk_data()
    progress_bar.progress(60, text=f"The data has been chunked.")
    time.sleep(0.1)
    st.session_state.engine.create_vector_store()
    progress_bar.progress(100, text=f"You can now ask questions.")
    time.sleep(1)
    progress_bar.empty()
    st.experimental_rerun()
else:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Chat
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    st.header(st.session_state.engine.filename)
    st.markdown("")
    query = st.text_input(
        label="Ask your question",
        value="",
        placeholder=""
    )
    if query and (query != st.session_state.previous_query or st.session_state.rerun_requested):
        with st.spinner("Searching..."):
            st.markdown("")
            st.session_state.rerun_requested = False
            st.session_state.previous_query = query
            answer = st.session_state.engine.query(query)
            st.write(answer)
    if st.session_state.engine.filename == "Niagara-WQR-English-2022_053122-HR.pdf":
        with st.expander("Example of questions"):
            st.markdown("""
                <div style="display: flex; flex-direction: column; margin-bottom: 20px">
                    <div style="font-weight: 600">Q: What's the story of Niagara?</div>
                    <div style="font-weight: 400 color:#000000">A: This query extracts the history section from page 4.</div>
                </div>
            """,
            unsafe_allow_html=True)
            st.markdown("""
                <div style="display: flex; flex-direction: column; margin-bottom: 20px">
                    <div style="font-weight: 600">Q: Summarize the story of Niagara</div>
                    <div style="font-weight: 400 color:#000000">A: This query summarizes the main highlights of the company from all over the document.</div>
                </div>
            """,
            unsafe_allow_html=True)
            st.markdown("""
                <div style="display: flex; flex-direction: column; margin-bottom: 20px">
                    <div style="font-weight: 600">Q: Explain how Niagara managed to save millions pounds of CO2 over the last 10 years</div>
                    <div style="font-weight: 400 color:#000000">A: The answer is extracted from page 5.</div>
                </div>
            """,
            unsafe_allow_html=True)



