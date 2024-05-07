from langchain.callbacks import FileCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from rag_101.retriever import (
    RAGException,
    create_parent_retriever,
    load_embedding_model,
    load_pdf,
    load_reranker_model,
    retrieve_context,
)


class RAGClient:
    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()

    def __init__(self, files, model="mistral"):
        docs = load_pdf(files=files)
        self.retriever = create_parent_retriever(docs, self.embedding_model)

        llm = ChatOllama(model=model)
        prompt_template = ChatPromptTemplate.from_template(
            (
                "Please answer the following question based on the provided `context` that follows the question.\n"
                "Think step by step before coming to answer. If you do not know the answer then just say 'I do not know'\n"
                "question: {question}\n"
                "context: ```{context}```\n"
            )
        )
        self.chain = prompt_template | llm | StrOutputParser()

    def stream(self, query: str) -> dict:
        try:
            context, similarity_score = self.retrieve_context(query)[0]
            context = context.page_content
            if similarity_score < 0.005:
                context = "This context is not confident. " + context
        except RAGException as e:
            context, similarity_score = e.args[0], 0
        logger.info(context)
        for r in self.chain.stream({"context": context, "question": query}):
            yield r

    def retrieve_context(self, query: str):
        return retrieve_context(
            query, retriever=self.retriever, reranker_model=self.reranker_model
        )

    def generate(self, query: str) -> dict:
        contexts = self.retrieve_context(query)

        return {
            "contexts": contexts,
            "response": self.chain.invoke(
                {"context": contexts[0][0].page_content, "question": query}
            ),
        }
