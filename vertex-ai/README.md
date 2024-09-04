# Vertex AI

This folder contains code for running the LLM in Google Cloud's Vertex AI. The create-faq-csv.ipynb can be run locally as its main function is to format all the data. Once it gives the csv file, that should be inserted into a cloud storage bucket to be put into the RAG data store.

The chatbot.ipynb file contains the actual RAG implementation in Vertex AI. It shows how to use the Vector Search feature of Vertex AI to set up the vectorstore. It also has the pipeline for getting the user's question, retrieving the right documents with the retriever tool, then outputting the answer. The last cell in the notebook contains a location to input the question, so that it outputs a relevant and detailed response.