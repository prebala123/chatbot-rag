# Local LLM

This folder contains code for running the LLM locally. The requirements.txt file in the repository contains all of the dependencies. The get-data.ipynb file is a notebook that reads in all the important data that needs to be fed into the LLM. It then converts them into a Document type from langchain before inserting them into a Chroma vectorstore. 

The ChatbotLLM.py file contains the code for the RAG implementation as well as a user interface. It loads in the vectorstore and sets up a pipeline to retrieve the relevant documents based on the question and formulate an answer. Users can run this file to interact with the Chatbot which contains chat history and RAG database, making for easy question answering.