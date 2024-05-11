import time
from typing import List, Optional, Union

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from retriever import (
    create_parent_retriever,
    load_embedding_model,
    load_pdf,
    load_reranker_model,
    retrieve_context,
)


def main(
    file: str = "example_data/2401.08406.pdf",
    llm_name="llama3",
):
    docs = load_pdf(files=file)

    embedding_model = load_embedding_model()
    retriever = create_parent_retriever(docs, embedding_model)
    reranker_model = load_reranker_model()

    llm = ChatOllama(model=llm_name)
    prompt_template = ChatPromptTemplate.from_template(
        (
            "You are llama3, a large language model developed by Meta AI. Surya has integrated you into this environment so you can answer any user's coding questions! Please answer the following question based on the provided `context` that follows the question.\n"
            "If you do not know the answer then just say 'I do not know'\n"
            "question: {question}\n"
            "context: ```{context}```\n"
        )
    )
    chain = prompt_template | llm | StrOutputParser()

    while True:
        query = input("Ask question: ")
        context = retrieve_context(
            query, retriever=retriever, reranker_model=reranker_model
        )[0]
        print("LLM Response: ", end="")
        for e in chain.stream({"context": context[0].page_content, "question": query}):
            print(e, end="")
        print()
        time.sleep(0.1)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
