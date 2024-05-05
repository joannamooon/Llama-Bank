import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

name = "meta-llama/Meta-Llama-3-8B"
auth_token = ""

@st.cache_resource
def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True) 

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

system_prompt = """<s>[INST] <<SYS>>
Provide Financial Performace Report<</SYS>>
"""
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
set_global_service_context(service_context)

PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()
documents = loader.load(file_path=Path('./HanakBank Report.pdf'), metadata=True)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

st.title('Llama Bank')
prompt = st.text_input('Input your prompt here')

if prompt:
    response = query_engine.query(prompt)
    st.write(response)

    with st.expander('Response Object'):
        st.write(response)
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())

