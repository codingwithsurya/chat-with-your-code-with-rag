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

# main function to run the program and generate the response with prompt
class RAGClient:
    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()

    def __init__(self, files, model="llama3"):
        docs = load_pdf(files=files)
        self.retriever = create_parent_retriever(docs, self.embedding_model)

        llm = ChatOllama(model=model)
        prompt_template = ChatPromptTemplate.from_template(
            (
                "You are llama3, a large language model developed by Meta AI. Surya has integrated you into this environment so you can answer any user's coding questions! Please answer the following question based on the provided `context` that follows the question.\n"
                "Think step by step before coming to answer. If you do not know the answer then just say 'I do not know'\n"
                "question: {question}\n"
                "context: ```{context}```\n"
            )
        )
        # Pipeline: format query -> generate response -> parse to string
        self.chain = prompt_template | llm | StrOutputParser()

    
    def stream(self, query: str) -> dict:
        """
        This method generates a stream of responses for a given query.

        Args:
            query (str): The query to be streamed.

        Yields:
            dict: The response for each iteration.

        Returns:
            None
        """
        # Retrieve the most relevant context for the given query
        try:
            context, similarity_score = self.retrieve_context(query)[0]
            # Extract the content of the page from the context
            context = context.page_content
            # If the similarity score is low, add a warning to the context
            if similarity_score < 0.005:
                context = "This context is not confident. " + context
        # If no relevant context is found, a RAGException is raised
        except RAGException as e:
            # In this case, use the exception message as the context and set the similarity score to 0
            context, similarity_score = e.args[0], 0
        # Log the context for debugging purposes
        logger.info(context)
        # Generate a stream of responses using the retrieved context and the original query
        for r in self.chain.stream({"context": context, "question": query}):
            yield r

    def retrieve_context(self, query: str):
        return retrieve_context(
            query, retriever=self.retriever, reranker_model=self.reranker_model
        )

   
    def generate(self, query: str) -> dict:
        """
            Generate a response for the given query.

            Args:
                query (str): The input query.

            Returns:
                dict: A dictionary containing the generated response and associated contexts.
                    The dictionary has the following structure:
                    {
                        "contexts": List[Tuple[Context, float]],
                        "response": str
                    }
                    - "contexts" is a list of tuples, where each tuple contains a Context object and its relevance score.
                    - "response" is the generated response string.
            """
        contexts = self.retrieve_context(query)

        return {
            "contexts": contexts,
            "response": self.chain.invoke(
                {"context": contexts[0][0].page_content, "question": query}
            ),
        }