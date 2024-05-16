import os
import gc
import re
import uuid
import subprocess
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from rag_101.retriever import (
    load_embedding_model,
    load_reranker_model
)

# Setup environment variables
os.environ["HF_HOME"] = "/teamspace/studios/this_studio/weights"
os.environ["TORCH_HOME"] = "/teamspace/studios/this_studio/weights"

def initialize_session_state():
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}
        st.session_state.messages = []
        st.session_state.context = None

initialize_session_state()

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# LLM setup
llm = Ollama(model="llama3", request_timeout=60.0)

# Embedding model setup
lc_embedding_model = load_embedding_model()
embed_model = LangchainEmbedding(lc_embedding_model)

# Utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

with st.sidebar:
    github_url = st.text_input("GitHub Repository URL")
    process_button = st.button("Load")
    message_container = st.empty()

    if process_button and github_url:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            with st.spinner(f"Loading {repo} repository by {owner}..."):
                try:
                    input_dir_path = f"/teamspace/studios/this_studio/{repo}"
                    
                    if not os.path.exists(input_dir_path):
                        subprocess.run(["git", "clone", github_url], check=True, text=True, capture_output=True)

                    if os.path.exists(input_dir_path):
                        loader = SimpleDirectoryReader(
                            input_dir=input_dir_path,
                            required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
                            recursive=True
                        )
                        docs = loader.load_data()
                        
                        Settings.embed_model = embed_model
                        index = VectorStoreIndex.from_documents(docs)
                        Settings.llm = llm
                        query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
                        
                        qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "You are llama3, a large language model developed by Meta AI. Given the context information above I want you to think step by step to answer the query in a crisp manner, in case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                        )
                        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
                        
                        message_container.success("Data loaded successfully!")
                        st.session_state.query_engine = query_engine
                    else:
                        message_container.error('Error occurred while cloning the repository, carefully check the url')
                        st.stop()
                except Exception as e:
                    message_container.error(f"An error occurred: {e}")
                    st.stop()
        else:
            message_container.error('Invalid owner or repository')
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your code! Powered by Llama3 🦙🚀")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        query_engine = st.session_state.query_engine
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
