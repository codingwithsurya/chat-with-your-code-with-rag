import os

os.environ["HF_HOME"] = "/teamspace/studios/this_studio/weights"
os.environ["TORCH_HOME"] = "/teamspace/studios/this_studio/weights"

from typing import List, Optional, Union

from langchain.callbacks import FileCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from loguru import logger
from rich import print
from sentence_transformers import CrossEncoder
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

logfile = "log/output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)


persist_directory = None


class RAGException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)


def load_pdf(
    files: Union[str, List[str]] = "example_data/2401.08406.pdf"
) -> List[Document]:
    if isinstance(files, str):
        loader = UnstructuredFileLoader(
            files,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        return loader.load()

    loaders = [
        UnstructuredFileLoader(
            file,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        for file in files
    ]
    docs = []
    for loader in loaders:
        docs.extend(
            loader.load(),
        )
    return docs


def create_parent_retriever(
    docs: List[Document], embeddings_model: HuggingFaceBgeEmbeddings()
):
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n"],
        chunk_size=2000,
        length_function=len,
        is_separator_regex=False,
    )

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n"],
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="split_documents",
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=10,
    )
    retriever.add_documents(docs)
    return retriever


def retrieve_context(query, retriever, reranker_model):
    retrieved_docs = retriever.get_relevant_documents(query)

    if len(retrieved_docs) == 0:
        raise RAGException(
            f"Couldn't retrieve any relevant document with the query `{query}`. Try modifying your question!"
        )
    reranked_docs = rerank_docs(
        query=query, retrieved_docs=retrieved_docs, reranker_model=reranker_model
    )
    return reranked_docs


def load_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"
) -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model


def load_reranker_model(
    reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cuda"
) -> CrossEncoder:
    reranker_model = CrossEncoder(
        model_name=reranker_model_name, max_length=512, device=device
    )
    return reranker_model


def main(
    file: str = "example_data/2401.08406.pdf",
    query: Optional[str] = None,
    llm_name="mistral",
):
    docs = load_pdf(files=file)

    embedding_model = load_embedding_model()
    retriever = create_parent_retriever(docs, embedding_model)
    reranker_model = load_reranker_model()

    context = retrieve_context(
        query, retriever=retriever, reranker_model=reranker_model
    )[0]
    print("context:\n", context, "\n", "=" * 50, "\n")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
