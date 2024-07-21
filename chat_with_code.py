import os
import ast

os.environ["HF_HOME"] = "/teamspace/studios/this_studio/weights"
os.environ["TORCH_HOME"] = "/teamspace/studios/this_studio/weights"

import gc
import re
import uuid
import textwrap
import subprocess
import nest_asyncio
from dotenv import load_dotenv
from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from rag_101.retriever import (
    load_embedding_model,
    load_reranker_model
)

# allows nested access to the event loop
nest_asyncio.apply()

# setting up the llm
llm=Ollama(model="llama3", request_timeout=60.0)

# setting up the embedding model
lc_embedding_model = load_embedding_model()
embed_model = LangchainEmbedding(lc_embedding_model)

# utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def clone_github_repo(repo_url):    
    try:
        print('Cloning the repo ...')
        result = subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return None

# utility function to validate owner and repo
def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def generate_repo_ast(repo_path):
    repo_summary = {}
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        # Count the number of each type of AST node
                        node_counts = {}
                        for node in ast.walk(tree):
                            node_type = type(node).__name__
                            node_counts[node_type] = node_counts.get(node_type, 0) + 1
                        repo_summary[file_path] = node_counts
                    except SyntaxError:
                        repo_summary[file_path] = "Unable to parse file"
    return repo_summary

# Setup a query engine

def setup_query_engine(github_url):
    
    owner, repo = parse_github_url(github_url)
    
    if validate_owner_repo(owner, repo):
        # Clone the GitHub repo & save it in a directory
        input_dir_path = f"/teamspace/studios/this_studio/{repo}"

        if os.path.exists(input_dir_path):
            pass
        else:
            clone_github_repo(github_url)
        
        loader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
            recursive=True
        )

        try:
            docs = loader.load_data()

            repo_ast = generate_repo_ast(input_dir_path)
            ast_text = json.dumps(repo_ast, indent=2)
            docs.append(Document(page_content=f"Repository AST:\n{ast_text}", metadata={"source": "repo_ast"}))

             # ====== Create vector store and upload data ======
            Settings.embed_model = embed_model
            index = VectorStoreIndex.from_documents(docs, show_progress=True)
            # ====== Setup a query engine ======
            Settings.llm = llm
            query_engine = index.as_query_engine(similarity_top_k=4)
            
             # ====== Customise prompt template ======
            qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Repository AST:\n"
            "{repo_ast}\n"
            "---------------------\n"
            "You are llama3, a large language model developed by Meta AI. Surya has integrated you into this environment so you can answer any user's coding questions! Given the context information and repository AST above, I want you to think step by step to answer the query in a crisp manner, considering both the code content and the file structure. If you don't know the answer, say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

            if docs:
                print("Data loaded successfully!!")
                print("Ready to chat!!")
            else:
                print("No data found, check if the repository is not empty!")
            
            return query_engine

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print('Invalid github repo, try again!')
        return None

# Provide url to the repository you want to chat with
github_url = "https://github.com/Lightning-AI/lit-gpt"

query_engine = setup_query_engine(github_url=github_url)

response = query_engine.query(f'Given the repository AST:\n{json.dumps(repo_ast, indent=2)}\n\nCan you provide a step by step guide to finetuning an llm using lit-gpt, considering the file structure?')
print(response)