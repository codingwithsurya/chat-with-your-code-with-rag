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



def load_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"
) -> HuggingFaceBgeEmbeddings:
    """
    Loads and returns a HuggingFaceBgeEmbeddings model for embedding text. This embedding model is used to encode the documents and queries for retrieval.

    Args:
        model_name (str): The name or path of the pre-trained model to load. Defaults to "BAAI/bge-large-en-v1.5".
        device (str): The device to use for model inference. Defaults to "cuda".

    Returns:
        HuggingFaceBgeEmbeddings: The loaded HuggingFaceBgeEmbeddings model.
    """
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
    """
    Loads a reranker model from the specified model name and device. We will use CrossEncoder model as defined in https://arxiv.org/abs/2005.11401 for reranking the retrieved documents.

    Args:
        reranker_model_name (str): The name of the reranker model to load.
        device (str): The device to use for running the model (default: "cuda").

    Returns:
        CrossEncoder: The loaded reranker model.

    """
    reranker_model = CrossEncoder(
        model_name=reranker_model_name, max_length=512, device=device
    )
    return reranker_model


def load_pdf(
    files: Union[str, List[str]] = "example_data/2401.08406.pdf"
) -> List[Document]:
    """
    Load PDF files and return a list of Document objects. This is useful because the RAG model requires a list of Document objects as input.

    Args:
        files (Union[str, List[str]], optional): Path(s) to the PDF file(s) to load. Defaults to "example_data/2401.08406.pdf".

    Returns:
        List[Document]: A list of Document objects representing the loaded PDF files.
    """

    # If a single file is provided, load it and return the Document object
    if isinstance(files, str):
        loader = UnstructuredFileLoader(
            files,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        return loader.load()

    # If multiple files are provided, load each file and return a list of Document objects
    loaders = [
        UnstructuredFileLoader(
            file,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        for file in files
    ]

    # Load the documents from all the loaders
    docs = []
    for loader in loaders:
        docs.extend(
            loader.load(),
        )
    return docs

def rerank_docs(reranker_model, query, retrieved_docs):
    '''
    Reranks the retrieved documents based on a given query and a reranker model.

    Args:
        reranker_model (object): The reranker model used for reranking the documents.
        query (str): The query used for reranking.
        retrieved_docs (list): A list of retrieved documents.

    Returns:
        list: A sorted list of tuples containing the retrieved documents and their corresponding scores, 
              sorted in descending order of scores.

    '''
    # Create a list of tuples containing the query and the retrieved documents
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    # use the reranker model to rerank the retrieved documents
    scores = reranker_model.predict(query_and_docs)
    # sort the retrieved documents based on the scores
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)



def create_parent_retriever(
    docs: List[Document], embeddings_model: HuggingFaceBgeEmbeddings()
):
    """
    Create a parent document retriever. A parent document retriever is a document retrieval system that allows you to search for relevant documents from a list of documents based on a given query.
    It uses an embeddings model to convert the documents into vectors that capture their semantic meaning. These vectors are then indexed using a vectorstore, which allows for efficient retrieval based on similarity. 
    The parent document retriever supports searching for relevant documents by comparing the query vector with the indexed document vectors and returning the most similar documents.

    Args:
        docs (List[Document]): A list of documents to be added to the parent document retriever.
        embeddings_model (HuggingFaceBgeEmbeddings): The embeddings model used for vectorization of the documents.

    Returns:
        ParentDocumentRetriever: The created parent document retriever.

    """

    # This text splitter is used to create the parent documents. Parent documents are typically longer and provide broader context.
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n"],
        chunk_size=2000,
        length_function=len,
        is_separator_regex=False,
    )

    # This text splitter is used to create the child documents, which are smaller segments of text split from the parent documents.
    # They are more granular and help in capturing fine-grained details.
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n"],
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    # The vectorstore to use to index the child chunks. Chroma is a vector store that allows for efficient retrieval based on similarity.
    vectorstore = Chroma(
        collection_name="split_documents",
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    store = InMemoryStore() # stores the parent documents in memory

    # Use langchain to create the parent document retriever and pass in params
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=10,
    )

    # Add the documents to the parent document retriever
    retriever.add_documents(docs)
    return retriever


# Rerank the retrieved documents based on the query and reranker model
def retrieve_context(query, retriever, reranker_model):

    # Get the relevant documents for the query. the get_relevant_documents method is part of the parent document retriever in langchain.
    retrieved_docs = retriever.get_relevant_documents(query)

    # If no relevant documents are found, raise an exception
    if len(retrieved_docs) == 0:
        raise RAGException(
            f"Couldn't retrieve any relevant document with the query `{query}`. Try modifying your question!"
        )
    
    # Rerank the retrieved documents based on the query and reranker model
    reranked_docs = rerank_docs(
        query=query, retrieved_docs=retrieved_docs, reranker_model=reranker_model
    )
    return reranked_docs

# main function to run the program
def main(
    file: str = "example_data/2401.08406.pdf",
    query: Optional[str] = None,
    llm_name="llama3",
):
    """Main function that performs the retrieval and printing of the context.

    Args:
        file (str, optional): The file path of the PDF document to load. Defaults to "example_data/2401.08406.pdf".
        query (str, optional): The query string to search for in the document. Defaults to None.
        llm_name (str, optional): The name of the llama model to use. Defaults to "llama3".
    """
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