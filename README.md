# Chat with Your Code with RAG! Powered by Llama3 🦙🚀

This repoistory enables you to have a conversational interface with your GitHub codebase. By integrating the leading open-source model, Llama3-70b, fine-tuned with RAG, this project allows you to search and understand any codebase in natural language.

## About RAG

The RAG model combines the best of retrieval and generative response capabilities to provide contextually appropriate responses based on a retrieved set of documents (or code in this case!). More details on the RAG approach can be found in the [original RAG paper](https://arxiv.org/abs/2005.11401). In this project, RAG is fine-tuned with Llama3 to specifically adapt to the domain of code, enhancing its ability to understand and generate technical content accurately.

## Production Environment Setup

I am currently working on setting up a production environment so this project is accessible to everyone. Please bear with me as I try to find more GPUs! 🫡

## Repository Structure
- **`chat_with_code.py`**: This module interacts with a GitHub repository, cloning it, loading its data, and setting up a query engine for conversational interactions.
- **`rag_101/`**: Contains the implementation of the RAG model utilized in this project.
- **`architecture-diagram.png`**: A bird's eye view on the project!


## Dependencies

This project utilizes several key libraries:

- **`llama_index`**: For indexing and querying the codebase efficiently.
- **`langchain`**: Utilized for embedding the codebase to improve retrieval performance.
- **`ollama`**: Helps us incorporate llama3
